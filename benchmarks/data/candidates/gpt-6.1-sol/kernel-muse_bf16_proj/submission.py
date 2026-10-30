# okc: problem=muse_bf16_proj
"""Original-precision projections; configuration selection uses synthetic inputs only."""
import time
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _mm(X,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,DIRECT:tl.constexpr):
    pid=tl.program_id(0); si=tl.program_id(1)
    nm=tr.cdiv(M,BM)
    mi=pid%nm; ni=pid//nm
    rr=mi*BM+tl.arange(0,BM); cc=ni*BN+tl.arange(0,BN)
    kk=si*(K//S)+tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for step in range(tr.cdiv(K//S,BK)):
        k=kk+step*BK
        a=tl.load(X+rr[:,None]*K+k[None,:],rr[:,None]<M,0)
        b=tl.load(W+cc[None,:]*K+k[:,None],cc[None,:]<N,0)
        acc=tl.dot(a,b,acc)
    tl.store(P+si*M*N+rr[:,None]*N+cc[None,:],acc,(rr[:,None]<M)&(cc[None,:]<N))

@tr.jit
def _reduce(P,Y,M:tl.constexpr,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B); ss=tl.arange(0,S)
    a=tl.load(P+ss[:,None]*M*N+t[None,:],t[None,:]<M*N,0)
    tl.store(Y+t,tl.sum(a,0),t<M*N)

@tr.jit
def _gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    ns=tl.program_id(0)*BN+tl.arange(0,BN)
    ks=tl.arange(0,BK)
    a=tl.full((BN,BK),0,tl.float32)
    for step in range(tr.cdiv(K,BK)):
        kk=ks+step*BK
        x=tl.load(X+kk,kk<K,0).to(tl.float32)
        w=tl.load(W+ns[:,None]*K+kk[None,:],(ns[:,None]<N)&(kk[None,:]<K),0).to(tl.float32)
        a=tl.fma(w,x[None,:],a)
    tl.store(Y+ns,tl.sum(a,1),ns<N)

@tr.jit
def _router(X,H,L,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr):
    pid=tl.program_id(0); si=tl.program_id(1); nm=tr.cdiv(M,BM)
    mi=pid%nm; ni=pid//nm
    rr=mi*BM+tl.arange(0,BM); cc=ni*BN+tl.arange(0,BN)
    kk=si*(K//S)+tl.arange(0,BK)
    ah=tl.full((BM,BN),0,tl.float32); al=tl.full((BM,BN),0,tl.float32)
    for step in range(tr.cdiv(K//S,BK)):
        k=kk+step*BK
        a=tl.load(X+rr[:,None]*K+k[None,:],rr[:,None]<M,0)
        h=tl.load(H+cc[None,:]*K+k[:,None],cc[None,:]<N,0)
        l=tl.load(L+cc[None,:]*K+k[:,None],cc[None,:]<N,0)
        ah=tl.dot(a,h,ah); al=tl.dot(a,l,al)
    tl.store(P+si*M*N+rr[:,None]*N+cc[None,:],ah+al,(rr[:,None]<M)&(cc[None,:]<N))


@tr.jit
def _packed_mm(X,W,LOW,HIGH,BASE,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr):
    pid=tl.program_id(0); si=tl.program_id(1); nm=tr.cdiv(M,BM)
    mi=pid%nm; ni=pid//nm
    rr=mi*BM+tl.arange(0,BM); cc=ni*BN+tl.arange(0,BN)
    kk=si*(K//S)+tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for step in range(tr.cdiv(K//S,BK)):
        k=kk+step*BK
        base=tl.load(BASE+cc[None,:]*(K//128)+k[:,None]//128,cc[None,:]<N,255).to(tl.uint16)
        packed=base!=255
        lo=tl.load(LOW+cc[None,:]*K+k[:,None],(cc[None,:]<N)&packed,0).to(tl.uint16)
        h=tl.load(HIGH+cc[None,:]*(K//2)+k[:,None]//2,(cc[None,:]<N)&packed,0).to(tl.uint16)
        nib=(h>>((k[:,None]%2)*4))&15
        hi=(base+(nib&7))|((nib&8)<<4)
        bits=lo|(hi<<8)
        decoded=bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)
        full=tl.load(W+cc[None,:]*K+k[:,None],(cc[None,:]<N)&(~packed),0)
        b=tl.where(packed,decoded,full)
        a=tl.load(X+rr[:,None]*K+k[None,:],rr[:,None]<M,0)
        acc=tl.dot(a,b,acc)
    tl.store(P+si*M*N+rr[:,None]*N+cc[None,:],acc,(rr[:,None]<M)&(cc[None,:]<N))

@tr.jit
def _packed_gemv(X,W,LOW,HIGH,BASE,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    ns=tl.program_id(0)*BN+tl.arange(0,BN); ks=tl.arange(0,BK)
    acc=tl.full((BN,BK),0,tl.float32)
    for step in range(tr.cdiv(K,BK)):
        kk=ks+step*BK
        base=tl.load(BASE+ns[:,None]*(K//128)+kk[None,:]//128,(ns[:,None]<N)&(kk[None,:]<K),255).to(tl.uint16)
        packed=base!=255
        lo=tl.load(LOW+ns[:,None]*K+kk[None,:],(ns[:,None]<N)&(kk[None,:]<K)&packed,0).to(tl.uint16)
        h=tl.load(HIGH+ns[:,None]*(K//2)+kk[None,:]//2,(ns[:,None]<N)&(kk[None,:]<K)&packed,0).to(tl.uint16)
        nib=(h>>((kk[None,:]%2)*4))&15
        hi=(base+(nib&7))|((nib&8)<<4)
        decoded=(lo|(hi<<8)).to(tl.uint16).to(tl.bfloat16,bitcast=True)
        full=tl.load(W+ns[:,None]*K+kk[None,:],(ns[:,None]<N)&(kk[None,:]<K)&(~packed),0)
        w=tl.where(packed,decoded,full).to(tl.float32)
        x=tl.load(X+kk,kk<K,0).to(tl.float32)
        acc=tl.fma(w,x[None,:],acc)
    tl.store(Y+ns,tl.sum(acc,1),ns<N)

@tr.jit
def _flush(A,N:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(A+t,t.to(tl.float32),t<N)

def _run(x,state,layer,n,cfg):
    m,k=x.shape; w=state[layer]; router=layer=='router'
    if cfg[0]=='torch':
        return x.float()@w[:n].T if router else x@w[:n].T
    y=torch.empty((m,n),device=x.device,dtype=torch.float32 if router else torch.bfloat16)
    if cfg[0]=='packed_gemv':
        low,high,base=state['packed'][layer]
        _packed_gemv[(tr.cdiv(n,cfg[1]),)](x,w,low,high,base,y,n,k,cfg[1],cfg[2],num_warps=cfg[3])
        return y
    if cfg[0]=='gemv':
        _gemv[(tr.cdiv(n,cfg[1]),)](x,w,y,n,k,cfg[1],cfg[2],num_warps=cfg[3])
        return y
    _,bm,bn,bk,s,warps=cfg
    p=y if s==1 else torch.empty((s,m,n),device=x.device,dtype=torch.float32)
    grid=(tr.cdiv(m,bm)*tr.cdiv(n,bn),s)
    if cfg[0]=='packed':
        low,high,base=state['packed'][layer]
        _packed_mm[grid](x,w,low,high,base,p,m,n,k,bm,bn,bk,s,num_warps=warps)
    elif router:
        _router[grid](x,state['router_hi'],state['router_lo'],p,m,n,k,bm,bn,bk,s,num_warps=warps)
    else:
        _mm[grid](x,w,p,m,n,k,bm,bn,bk,s,s==1,num_warps=warps)
    if s!=1:
        _reduce[(tr.cdiv(m*n,256),)](p,y,m,n,s,256,num_warps=4)
    return y

def _configs(m,k,router):
    # Put substantially different algorithms early so the bounded search
    # evaluates GEMV, compressed weights, and direct output before refinements.
    if router:
        bm=16 if m<=16 else 32
        configs=[('mm',bm,32,64,8,4),('mm',bm,32,64,16,4),
                 ('mm',bm,32,64,4,4),('mm',32,64,64,4,4)]
        if m==1:
            configs=[('gemv',1,512,4),('gemv',1,8192,8),
                     ('gemv',2,4096,4)]+configs
        return configs+[('torch',)]
    bm=16 if m<=16 else 32
    if m==1:
        configs=[('gemv',4,1024,4),('packed_gemv',4,1024,4),
                 ('gemv',2,4096,4),('packed_gemv',2,4096,4),
                 ('gemv',1,8192,8),('packed_gemv',1,8192,8),
                 ('mm',16,64,64,8 if k>=4096 else 4,4)]
    elif m<=32:
        splits=8 if k>=4096 else 4
        configs=[('mm',bm,64,64,splits,4),
                 ('packed',bm,64,64,splits,4),
                 ('mm',bm,32,64,1,4),('packed',bm,32,64,1,4),
                 ('mm',bm,64,64,1,4),('packed',bm,64,64,1,4)]
    else:
        configs=[('mm',128,64,64,4,8),('packed',128,64,64,4,8),
                 ('mm',128,64,64,1,8),('packed',128,64,64,1,8),
                 ('mm',128,128,64,2,8),('packed',128,128,64,2,8),
                 ('mm',64,64,64,2,4),('packed',64,64,64,2,4)]
    refinements=[('mm',bm,64,64,s,4) for s in (4,8,16)]
    if m>=128:
        refinements += [('mm',m,64,64,s,8) for s in (2,4,8)]
        refinements += [('mm',64,128,64,2,4),('mm',128,128,64,1,8)]
    refinements += [('packed',)+c[1:] for c in refinements]
    configs += refinements+[('torch',)]
    return list(dict.fromkeys(configs))

def prepare(weights):
    deadline=time.monotonic()+240.0
    state={k:v.contiguous() for k,v in weights.items()}
    h=state['router'].to(torch.bfloat16)
    state['router_hi']=h
    state['router_lo']=(state['router']-h.float()).to(torch.bfloat16)
    state['configs']={}
    state['packed']={}
    # Each group either has an exactly representable 12-bit encoding or
    # retains the original BF16 bits through the fallback load.
    for name,w in weights.items():
        if name=='router': continue
        n,k=w.shape
        bits=w.view(torch.int16).to(torch.int32)&65535
        upper=(bits>>8)&127
        grouped=upper.reshape(n,k//128,128)
        hi=grouped.amax(-1); lo=grouped.amin(-1)
        base=(hi-7).clamp(min=0)
        valid=(hi-lo)<=7
        offsets=upper-base.repeat_interleave(128,dim=1)
        nib=(offsets&7)|((bits>>12)&8)
        packed=(nib[:,0::2]|(nib[:,1::2]<<4)).to(torch.uint8).contiguous()
        low=(bits&255).to(torch.uint8).contiguous()
        metadata=torch.where(valid,base,255).to(torch.uint8).contiguous()
        state['packed'][name]=(low,packed,metadata)
    # Round-robin search gives every shape candidates before refining any one.
    # Synthetic activations are independent of data passed to kernel_fn.
    flush=torch.empty((32*1024*1024,),device=h.device,dtype=torch.float32)
    _flush[(tr.cdiv(flush.numel(),1024),)](flush,flush.numel(),1024)
    jobs=[]
    for layer in ('router','pooler','unpooler','qkv_gate','o'):
        for m in (1,8,16,32,128,256):
            w=state[layer]
            n=1024 if layer=='unpooler' and m<=8 else w.shape[0]
            k=w.shape[1]
            candidates=_configs(m,k,layer=='router')
            # Establish a library fallback before compiling custom alternatives.
            candidates=[('torch',)]+[c for c in candidates if c!=('torch',)]
            state['configs'][(layer,m)]=('torch',)
            jobs.append((layer,m,n,k,candidates))
    scores={(layer,m):float('inf') for layer,m,_,_,_ in jobs}
    for round_index in range(max(len(j[4]) for j in jobs)):
        for layer,m,n,k,candidates in jobs:
            if time.monotonic()>=deadline:
                return state
            if round_index>=len(candidates):
                continue
            cfg=candidates[round_index]
            x=torch.randn((m,k),device=h.device,dtype=torch.bfloat16)
            try:
                _run(x,state,layer,n,cfg)
                torch.cuda.synchronize()
                start=torch.cuda.Event(enable_timing=True,external=True)
                end=torch.cuda.Event(enable_timing=True,external=True)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _flush[(tr.cdiv(flush.numel(),1024),)](flush,flush.numel(),1024)
                    start.record()
                    y=_run(x,state,layer,n,cfg)
                    end.record()
                times=[]
                for _ in range(5):
                    graph.replay()
                    end.synchronize()
                    times.append(start.elapsed_time(end))
                score=sorted(times)[2]
                if score<scores[(layer,m)]:
                    scores[(layer,m)]=score
                    state['configs'][(layer,m)]=cfg
                del graph,y
            except Exception:
                continue
    return state

def kernel_fn(data,state):
    x=data['x']; c=data['config']; layer=c['layer']; m=x.shape[0]
    cfg=state['configs'].get((layer,m))
    if cfg is None:
        cfg=('mm',16 if m<=16 else 32,32,64,8,4) if layer=='router' else ('torch',)
    return _run(x,state,layer,c['n'],cfg)
