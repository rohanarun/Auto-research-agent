# okc: problem=muse_attention_decode
"""Fused QK normalization/cache update, dense split attention, and gated reduction."""
import math
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _prepare(X, POS, LOC, KC, VC, KS, VS, Q,
             HQ:tl.constexpr, HK:tl.constexpr, THETA:tl.constexpr, LOGTHETA:tl.constexpr, FACTOR:tl.constexpr,
             HH:tl.constexpr):
    t=tl.program_id(0)
    h=tl.arange(0,HH)
    d=tl.arange(0,64)
    x=tl.load(X+t*(2*HQ+2*HK)*64+h[:,None]*64+d[None,:],h[:,None]<HQ+HK,0).to(tl.float32)
    x=x*tl.rsqrt(tl.sum(x*x,1)[:,None]/64+1.e-5)
    x=x*tl.where(h[:,None]<HQ,FACTOR,1.)
    if THETA != 0.0:
        # An explicit partner load preserves the interleaved GPT-J convention.
        y=tl.load(X+t*(2*HQ+2*HK)*64+h[:,None]*64+(d[None,:]^1),h[:,None]<HQ+HK,0).to(tl.float32)
        z=tl.load(X+t*(2*HQ+2*HK)*64+h[:,None]*64+d[None,:],h[:,None]<HQ+HK,0).to(tl.float32)
        y=y*tl.rsqrt(tl.sum(z*z,1)[:,None]/64+1.e-5)*tl.where(h[:,None]<HQ,FACTOR,1.)
        p=tl.load(POS+t).to(tl.float32)
        a=p*tl.exp((-LOGTHETA/64)*(d//2*2).to(tl.float32))
        x=x*tl.cos(a)[None,:]+y*tl.sin(a)[None,:]*tl.where(d%2==0,-1.,1.)[None,:]
    tl.store(Q+t*HQ*64+h[:,None]*64+d[None,:],x,h[:,None]<HQ)
    slot=tl.load(LOC+t)
    ks=tl.load(KS); vs=tl.load(VS)
    tl.store(KC+slot*HK*64+(h[:,None]-HQ)*64+d[None,:],x/ks,(h[:,None]>=HQ)&(h[:,None]<HQ+HK))
    v=tl.load(X+t*(2*HQ+2*HK)*64+(HQ+HK)*64+h[:,None]*64+d[None,:],h[:,None]<HK,0).to(tl.float32)
    tl.store(VC+slot*HK*64+h[:,None]*64+d[None,:],v/vs,h[:,None]<HK)

@tr.jit
def _attention(Q,KC,VC,KS,VS,TABLE,SEQ,POS,P,M,Z,
               HQ:tl.constexpr,HK:tl.constexpr,L:tl.constexpr,PAGE:tl.constexpr,NP:tl.constexpr,
               WINDOW:tl.constexpr,SM:tl.constexpr,SPLITS:tl.constexpr,CHUNK:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr):
    b=tl.program_id(0); kh=tl.program_id(1); split=tl.program_id(2)
    G:tl.constexpr=HQ//HK
    r=tl.arange(0,BM); d=tl.arange(0,64); n=tl.arange(0,BN)
    qt=r//G; qh=kh*G+r%G
    q=tl.load(Q+((b*L+qt[:,None])*HQ+qh[:,None])*64+d[None,:],r[:,None]<L*G,0)
    pos=tl.load(POS+b*L+qt,qt<L,0)
    seq=tl.load(SEQ+b)
    start=tl.maximum(0,seq-WINDOW+1) if WINDOW else 0
    lo=start+split*CHUNK
    hi=tl.minimum(lo+CHUNK,seq+L)
    mx=tl.full((BM,),-1.e30,tl.float32)
    den=tl.zeros((BM,),tl.float32)
    acc=tl.zeros((BM,64),tl.float32)
    ks=tl.load(KS); vs=tl.load(VS)
    for at in range(lo,hi,BN):
        ix=at+n
        pg=tl.load(TABLE+b*NP+ix//PAGE,ix<seq+L,0)
        rows=pg*PAGE+ix%PAGE
        k=tl.load(KC+(rows[None,:]*HK+kh)*64+d[:,None],ix[None,:]<hi,0.0).to(tl.bfloat16)
        v=tl.load(VC+(rows[:,None]*HK+kh)*64+d[None,:],ix[:,None]<hi,0.0).to(tl.bfloat16)
        score=tl.dot(q,k)*(SM*1.4426950408889634)*ks
        valid=(ix[None,:]<hi)&(ix[None,:]<=pos[:,None])&(r[:,None]<L*G)
        if WINDOW:
            valid=valid&(ix[None,:]>pos[:,None]-WINDOW)
        score=tl.where(valid,score,-1.e30)
        newmx=tl.maximum(mx,tl.max(score,1))
        alpha=tl.exp2(mx-newmx)
        prob=tl.where(valid,tl.exp2(score-newmx[:,None]),0.)
        den=den*alpha+tl.sum(prob,1)
        acc=acc*alpha[:,None]
        acc=tl.dot(prob.to(tl.bfloat16),v,acc)
        mx=newmx
    base=((b*HK+kh)*SPLITS+split)*L*G+r
    tl.store(P+base[:,None]*64+d[None,:],acc*vs,r[:,None]<L*G)
    tl.store(M+base,mx,r<L*G)
    tl.store(Z+base,den,r<L*G)

@tr.jit
def _decode_attention(X,LOC,KC,VC,KS,VS,TABLE,SEQ,POS,P,M,Z,
               HQ:tl.constexpr,HK:tl.constexpr,L:tl.constexpr,PAGE:tl.constexpr,NP:tl.constexpr,
               WINDOW:tl.constexpr,SM:tl.constexpr,SPLITS:tl.constexpr,CHUNK:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr,THETA:tl.constexpr,LOGTHETA:tl.constexpr,FACTOR:tl.constexpr):
    b=tl.program_id(0); kh=tl.program_id(1); split=tl.program_id(2)
    G:tl.constexpr=HQ//HK
    r=tl.arange(0,BM); d=tl.arange(0,64); n=tl.arange(0,BN)
    qt=r//G; qh=kh*G+r%G
    stride:tl.constexpr=(2*HQ+2*HK)*64
    raw=tl.load(X+b*stride+qh[:,None]*64+d[None,:],r[:,None]<G,0.0).to(tl.float32)
    qnorm=tl.rsqrt(tl.sum(raw*raw,1)/64+1.e-5)
    q=raw*qnorm[:,None]*FACTOR
    pos=tl.load(POS+b)
    seq=tl.load(SEQ+b)
    knew=tl.load(X+b*stride+(HQ+kh)*64+d).to(tl.float32)
    knorm=tl.rsqrt(tl.sum(knew*knew,0)/64+1.e-5)
    knew=knew*knorm
    if THETA != 0.0:
        angle=pos.to(tl.float32)*tl.exp((-LOGTHETA/64)*(d//2*2).to(tl.float32))
        co=tl.cos(angle); si=tl.sin(angle)*tl.where(d%2==0,-1.,1.)
        partner=tl.load(X+b*stride+qh[:,None]*64+(d[None,:]^1),r[:,None]<G,0.0).to(tl.float32)
        q=q*co[None,:]+partner*qnorm[:,None]*FACTOR*si[None,:]
        kp=tl.load(X+b*stride+(HQ+kh)*64+(d^1)).to(tl.float32)*knorm
        knew=knew*co+kp*si
    q=q.to(tl.bfloat16)
    ks=tl.load(KS); vs=tl.load(VS)
    kquant=(knew/ks).to(KC.dtype.element_ty)
    vnew=tl.load(X+b*stride+(HQ+HK+kh)*64+d).to(tl.float32)
    vquant=(vnew/vs).to(VC.dtype.element_ty)
    if split == 0:
        slot=tl.load(LOC+b)
        tl.store(KC+(slot*HK+kh)*64+d,kquant)
        tl.store(VC+(slot*HK+kh)*64+d,vquant)
    kinline=kquant.to(tl.bfloat16)
    vinline=vquant.to(tl.bfloat16)
    start=tl.maximum(0,seq-WINDOW+1) if WINDOW else 0
    lo=start+split*CHUNK
    hi=tl.minimum(lo+CHUNK,seq+L)
    mx=tl.full((BM,),-1.e30,tl.float32)
    den=tl.zeros((BM,),tl.float32)
    acc=tl.zeros((BM,64),tl.float32)
    for at in range(lo,hi,BN):
        ix=at+n
        pg=tl.load(TABLE+b*NP+ix//PAGE,ix<seq+L,0)
        rows=pg*PAGE+ix%PAGE
        k=tl.load(KC+(rows[None,:]*HK+kh)*64+d[:,None],(ix[None,:]<hi)&(ix[None,:]<seq),0.0).to(tl.bfloat16)
        v=tl.load(VC+(rows[:,None]*HK+kh)*64+d[None,:],(ix[:,None]<hi)&(ix[:,None]<seq),0.0).to(tl.bfloat16)
        k=tl.where(ix[None,:]==seq,kinline[:,None],k)
        v=tl.where(ix[:,None]==seq,vinline[None,:],v)
        score=tl.dot(q,k)*(SM*1.4426950408889634)*ks
        valid=(ix[None,:]<hi)&(ix[None,:]<=pos)&(r[:,None]<L*G)
        if WINDOW:
            valid=valid&(ix[None,:]>pos-WINDOW)
        score=tl.where(valid,score,-1.e30)
        newmx=tl.maximum(mx,tl.max(score,1))
        alpha=tl.exp2(mx-newmx)
        prob=tl.where(valid,tl.exp2(score-newmx[:,None]),0.)
        den=den*alpha+tl.sum(prob,1)
        acc=acc*alpha[:,None]
        acc=tl.dot(prob.to(tl.bfloat16),v,acc)
        mx=newmx
    base=((b*HK+kh)*SPLITS+split)*L*G+r
    tl.store(P+base[:,None]*64+d[None,:],acc*vs,r[:,None]<L*G)
    tl.store(M+base,mx,r<L*G)
    tl.store(Z+base,den,r<L*G)

@tr.jit
def _finish(P,M,Z,X,OUT,HQ:tl.constexpr,HK:tl.constexpr,L:tl.constexpr,SPLITS:tl.constexpr,SS:tl.constexpr):
    b=tl.program_id(0); h=tl.program_id(1); j=tl.program_id(2)
    G:tl.constexpr=HQ//HK
    s=tl.arange(0,SS); d=tl.arange(0,64)
    ix=((b*HK+h//G)*SPLITS+s)*L*G+j*G+h%G
    m=tl.load(M+ix,s<SPLITS,-1.e30)
    z=tl.load(Z+ix,s<SPLITS,0.)
    w=tl.exp2(m-tl.max(m,0))
    v=tl.load(P+ix[:,None]*64+d[None,:],s[:,None]<SPLITS,0.)
    a=tl.sum(v*w[:,None],0)/tl.maximum(tl.sum(z*w,0),1.e-30)
    # SGLang retains its attention activation in BF16.
    a=a.to(tl.bfloat16).to(tl.float32)
    a=a*tl.rsqrt(tl.sum(a*a,0)/64+1.e-5)
    gate=tl.load(X+(b*L+j)*(2*HQ+2*HK)*64+(HQ+2*HK+h)*64+d).to(tl.float32)
    out=a/(1.+tl.exp(-gate))
    tl.store(OUT+((b*L+j)*HQ+h)*64+d,out)

def kernel_fn(data):
    c=data['config']; b=c['batch']; l=c['q_len']; h=c['num_q_heads']; k=c['num_kv_heads']
    x=data['qkv']; np=data['page_table'].shape[1]; window=c['window']
    if window:
        chunk=64 if b == 1 else 256
        splits=tr.cdiv(window+l-1,chunk)
    else:
        desired=max(32,128//b)
        chunk=tr.cdiv(tr.cdiv(np*c['page_size'],desired),128)*128
        splits=tr.cdiv(np*c['page_size'],chunk)
    p=torch.empty((b,k,splits,l*(h//k),64),device=x.device,dtype=torch.float32)
    m=torch.empty((b,k,splits,l*(h//k)),device=x.device,dtype=torch.float32)
    z=torch.empty_like(m)
    out=torch.empty((b,l,h*64),device=x.device,dtype=torch.bfloat16)
    if l == 1:
        _decode_attention[(b,k,splits)](x,data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],data['page_table'],data['seq_lens'],data['positions'],p,m,z,h,k,l,c['page_size'],np,window,c['sm_scale'],splits,chunk,max(16,tr.next_power_of_2(h//k)),64 if window else 128,c['rope_theta'],math.log(c['rope_theta']) if c['rope_theta'] else 0.0,c['qk_scale_factor'],num_warps=4)
    else:
        q=torch.empty((b*l,h,64),device=x.device,dtype=torch.bfloat16)
        _prepare[(b*l,)](x,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],q,h,k,c['rope_theta'],math.log(c['rope_theta']) if c['rope_theta'] else 0.0,c['qk_scale_factor'],tr.next_power_of_2(h+k),num_warps=4)
        _attention[(b,k,splits)](q,data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],data['page_table'],data['seq_lens'],data['positions'],p,m,z,h,k,l,c['page_size'],np,window,c['sm_scale'],splits,chunk,max(16,tr.next_power_of_2(l*h//k)),64 if window else 128,num_warps=4)
    _finish[(b,h,l)](p,m,z,x,out,h,k,l,splits,tr.next_power_of_2(splits),num_warps=4)
    return out
