# okc: problem=muse_attention_decode
"""Dynamic paged GQA: split attention followed by merge, gate and cache update."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _attention(X,KC,VC,PT,POS,KS,VS,A,S,
               HQ:tl.constexpr,HK:tl.constexpr,L:tl.constexpr,PAGE:tl.constexpr,NP:tl.constexpr,
               WIN:tl.constexpr,SM:tl.constexpr,NS:tl.constexpr,CHUNK:tl.constexpr,
               M:tl.constexpr,N:tl.constexpr,STR:tl.constexpr,TH:tl.constexpr,SCALE:tl.constexpr):
    b=tl.program_id(0); kh=tl.program_id(1); split=tl.program_id(2)
    r=tl.arange(0,M); d=tl.arange(0,64); n=tl.arange(0,N)
    token=r//8; head=kh*8+r%8
    pos=tl.load(POS+b*L+token,token<L,0)
    first=tl.load(POS+b*L)
    end=first+L
    if WIN:
        begin=tl.maximum(0,first-WIN+1)
    else:
        begin=0
    low=begin+split*CHUNK; high=tl.minimum(low+CHUNK,end)
    qp=X+(b*L+token[:,None])*STR+head[:,None]*64
    raw=tl.load(qp+d[None,:],token[:,None]<L,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(raw*raw,1)/64+1.e-5)
    if TH>0:
        paired=tl.load(qp+(d[None,:]^1),token[:,None]<L,0).to(tl.float32)
        angle=pos[:,None].to(tl.float32)*tl.exp(-tl.log(TH)*(d[None,:]//2*2).to(tl.float32)/64)
        raw=raw*tl.cos(angle)+tl.where(d[None,:]%2==0,-paired,paired)*tl.sin(angle)
    q=(raw*inv[:,None]*SCALE).to(tl.bfloat16)
    ks=tl.load(KS); vs=tl.load(VS)
    maximum=tl.full((M,),-float('inf'),tl.float32)
    mass=tl.zeros((M,),tl.float32)
    accum=tl.zeros((M,64),tl.float32)
    for at in range(low,high,N):
        t=at+n
        within=t<high
        physical=tl.load(PT+b*NP+t//PAGE,within,0)
        row=physical*PAGE+t%PAGE
        old=within&(t<first)
        k=tl.load(KC+(row[None,:]*HK+kh)*64+d[:,None],old[None,:],0.).to(tl.bfloat16)
        v=tl.load(VC+(row[:,None]*HK+kh)*64+d[None,:],old[:,None],0.).to(tl.bfloat16)
        # New cache rows are computed directly, so attention never races a cache writer.
        if at+N>first:
            fresh=within&(t>=first)
            nt=tl.minimum(tl.maximum(t-first,0),L-1)
            kp=X+(b*L+nt[:,None])*STR+HQ*64+kh*64
            nk=tl.load(kp+d[None,:],fresh[:,None],0).to(tl.float32)
            ni=tl.rsqrt(tl.sum(nk*nk,1)/64+1.e-5)
            if TH>0:
                partner=tl.load(kp+(d[None,:]^1),fresh[:,None],0).to(tl.float32)
                np=tl.load(POS+b*L+nt,fresh,0).to(tl.float32)
                key_angle=np[:,None]*tl.exp(-tl.log(TH)*(d[None,:]//2*2).to(tl.float32)/64)
                nk=nk*tl.cos(key_angle)+tl.where(d[None,:]%2==0,-partner,partner)*tl.sin(key_angle)
            nk=(nk*ni[:,None]/ks).to(KC.dtype.element_ty).to(tl.bfloat16)
            nv=tl.load(X+(b*L+nt[:,None])*STR+(HQ+HK)*64+kh*64+d[None,:],fresh[:,None],0).to(tl.float32)
            nv=(nv/vs).to(VC.dtype.element_ty).to(tl.bfloat16)
            k=tl.where(fresh[None,:],tl.trans(nk),k)
            v=tl.where(fresh[:,None],nv,v)
        score=tl.dot(q,k)*(SM*1.4426950408889634)*ks
        visible=within[None,:]&(t[None,:]<=pos[:,None])&(token[:,None]<L)
        if WIN:
            visible=visible&(t[None,:]>pos[:,None]-WIN)
        score=tl.where(visible,score,-float('inf'))
        newmax=tl.maximum(maximum,tl.max(score,1))
        safe=tl.where(newmax==-float('inf'),0.,newmax)
        rescale=tl.exp2(maximum-safe)
        probability=tl.exp2(score-safe[:,None])
        accum=accum*rescale[:,None]+tl.dot(probability.to(tl.bfloat16),v)
        mass=mass*rescale+tl.sum(probability,1)
        maximum=newmax
    index=(((b*HK+kh)*NS+split)*L*8+r)
    tl.store(A+index[:,None]*64+d[None,:],accum*vs,token[:,None]<L)
    tl.store(S+index*2,maximum,token<L)
    tl.store(S+index*2+1,mass,token<L)

@tr.jit
def _merge(A,S,X,OUT,POS,LOC,KC,VC,KS,VS,
           HQ:tl.constexpr,HK:tl.constexpr,L:tl.constexpr,NS:tl.constexpr,SS:tl.constexpr,
           STR:tl.constexpr,TH:tl.constexpr):
    token=tl.program_id(0); head=tl.program_id(1)
    b=token//L; j=token%L; kh=head//8; group=head%8
    split=tl.arange(0,SS); d=tl.arange(0,64)
    index=(((b*HK+kh)*NS+split)*L*8+j*8+group)
    maximum=tl.load(S+index*2,split<NS,-float('inf'))
    mass=tl.load(S+index*2+1,split<NS,0)
    weights=tl.exp2(maximum-tl.max(maximum,0))
    partial=tl.load(A+index[:,None]*64+d[None,:],split[:,None]<NS,0)
    result=tl.sum(partial*weights[:,None],0)/tl.sum(mass*weights,0)
    result=result.to(tl.bfloat16).to(tl.float32)
    result=result*tl.rsqrt(tl.sum(result*result,0)/64+1.e-5)
    gate=tl.load(X+token*STR+(HQ+2*HK)*64+head*64+d).to(tl.float32)
    tl.store(OUT+(token*HQ+head)*64+d,result/(1+tl.exp(-gate)))
    if group==0:
        kp=X+token*STR+HQ*64+kh*64
        k=tl.load(kp+d).to(tl.float32)
        inv=tl.rsqrt(tl.sum(k*k,0)/64+1.e-5)
        if TH>0:
            paired=tl.load(kp+(d^1)).to(tl.float32)
            p=tl.load(POS+token).to(tl.float32)
            angle=p*tl.exp(-tl.log(TH)*(d//2*2).to(tl.float32)/64)
            k=k*tl.cos(angle)+tl.where(d%2==0,-paired,paired)*tl.sin(angle)
        ks=tl.load(KS); vs=tl.load(VS)
        v=tl.load(X+token*STR+(HQ+HK)*64+kh*64+d).to(tl.float32)
        slot=tl.load(LOC+token)
        tl.store(KC+(slot*HK+kh)*64+d,k*inv/ks)
        tl.store(VC+(slot*HK+kh)*64+d,v/vs)

def kernel_fn(data):
    c=data['config']; batch=c['batch']; length=c['q_len']
    hq=c['num_q_heads']; hk=c['num_kv_heads']
    assert hq==8*hk and c['head_dim']==64
    x=data['qkv']; stride=x.shape[-1]; pages=data['page_table'].shape[1]
    output=torch.empty((batch,length,hq*64),device=x.device,dtype=torch.bfloat16)
    capacity=pages*c['page_size']
    if c['window']:
        capacity=min(capacity,c['window']+length-1)
        chunk=32 if batch<=4 else (128 if batch<=16 else 256)
    else:
        chunk=512 if batch<=2 else (1024 if batch<=8 else 2048)
    chunk=max(chunk,tr.next_power_of_2(tr.cdiv(capacity,128)))
    splits=tr.cdiv(capacity,chunk)
    partial=torch.empty((batch,hk,splits,length*8,64),device=x.device,dtype=torch.float32)
    stats=torch.empty((batch,hk,splits,length*8,2),device=x.device,dtype=torch.float32)
    _attention[(batch,hk,splits)](x,data['k_cache'],data['v_cache'],data['page_table'],data['positions'],data['k_scale'],data['v_scale'],partial,stats,hq,hk,length,c['page_size'],pages,c['window'],c['sm_scale'],splits,chunk,16 if length==1 else 64,32 if c['window'] and batch<=4 else 64,stride,c['rope_theta'],c['qk_scale_factor'],num_warps=4)
    _merge[(batch*length,hq)](partial,stats,x,output,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],hq,hk,length,splits,tr.next_power_of_2(splits),stride,c['rope_theta'],num_warps=4)
    return output
