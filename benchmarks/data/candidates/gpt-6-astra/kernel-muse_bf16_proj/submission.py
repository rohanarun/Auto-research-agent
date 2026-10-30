# okc: problem=muse_bf16_proj
"""Weight-streaming projections with FP32 partial sums."""
from typing import TypedDict
import torch
import triton as tr
import triton.language as tl

class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor
class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str
class Inputs(TypedDict):
    x: torch.Tensor
    config: Config

@tr.jit
def _gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rn=tl.program_id(0)*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    x=tl.load(X+rk,rk<K,0).to(tl.float32)
    w=tl.load(W+rn[:,None]*K+rk[None,:],(rn[:,None]<N)&(rk[None,:]<K),0).to(tl.float32)
    acc=tl.sum(w*x[None,:],1)
    tl.store(Y+rn,acc,rn<N)

@tr.jit
def _mm(X,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr):
    im=tl.program_id(0); jn=tl.program_id(1); sk=tl.program_id(2)
    rm=im*BM+tl.arange(0,BM)
    rn=jn*BN+tl.arange(0,BN)
    rk=sk*(tr.cdiv(K,SPLIT*BK)*BK)+tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for t in range(tr.cdiv(K,SPLIT*BK)):
        kk=rk+t*BK
        a=tl.load(X+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0)
        b=tl.load(W+rn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0)
        acc=tl.dot(a,b,acc)
    tl.store(P+sk*M*N+rm[:,None]*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))

@tr.jit
def _reduce(P,Y,SIZE:tl.constexpr,SPLIT:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0)*B+tl.arange(0,B)
    acc=tl.full((B,),0,tl.float32)
    for s in range(SPLIT):
        acc+=tl.load(P+s*SIZE+r,r<SIZE,0)
    tl.store(Y+r,acc,r<SIZE)

@tr.jit
def _router(X,H,L,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr):
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=tl.program_id(1)*BN+tl.arange(0,BN)
    rk=tl.program_id(2)*(tr.cdiv(K,SPLIT*BK)*BK)+tl.arange(0,BK)
    hi=tl.zeros((BM,BN),tl.float32)
    lo=tl.zeros((BM,BN),tl.float32)
    for t in range(tr.cdiv(K,SPLIT*BK)):
        kk=rk+t*BK
        a=tl.load(X+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0)
        h=tl.load(H+rn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0)
        l=tl.load(L+rn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0)
        hi=tl.dot(a,h,hi)
        lo=tl.dot(a,l,lo)
    tl.store(P+tl.program_id(2)*M*N+rm[:,None]*N+rn[None,:],hi+lo,(rm[:,None]<M)&(rn[None,:]<N))

def prepare(weights):
    state={key:w.contiguous() for key,w in weights.items()}
    hi=state['router'].to(torch.bfloat16)
    state['router_hi']=hi
    state['router_lo']=(state['router']-hi.float()).to(torch.bfloat16)
    return state

def kernel_fn(data,state):
    x=data['x']; c=data['config']; layer=c['layer']
    m,n,k=x.shape[0],c['n'],x.shape[1]
    router=layer=='router'
    # The measured 128-token custom path was slower than cuBLAS.
    if not router and m>32:
        return torch.mm(x,state[layer][:n].T)
    y=torch.empty((m,n),device=x.device,dtype=torch.float32 if router else torch.bfloat16)
    if m==1:
        bn=max(1,8192//k)
        _gemv[(tr.cdiv(n,bn),)](x,state[layer],y,n,k,bn,tr.next_power_of_2(k),num_warps=4)
        return y
    if router:
        bm=16 if m<=16 else 32
        bn=32
        split=max(1,min(32,tr.next_power_of_2(tr.cdiv(256,tr.cdiv(m,bm)*tr.cdiv(n,bn)))))
        p=torch.empty((split,m,n),device=x.device,dtype=torch.float32)
        _router[(tr.cdiv(m,bm),tr.cdiv(n,bn),split)](x,state['router_hi'],state['router_lo'],p,m,n,k,bm,bn,64,split,num_warps=4,num_stages=3)
    elif m<=32:
        bm=16 if m<=16 else 32
        bn=64
        split=max(1,min(16,tr.next_power_of_2(tr.cdiv(256,tr.cdiv(n,bn)))))
        p=torch.empty((split,m,n),device=x.device,dtype=torch.float32) if split>1 else y
        _mm[(tr.cdiv(m,bm),tr.cdiv(n,bn),split)](x,state[layer],p,m,n,k,bm,bn,128,split,num_warps=4,num_stages=3)
        if split==1:
            return y
    _reduce[(tr.cdiv(m*n,512),)](p,y,m*n,split,512,num_warps=4)
    return y
