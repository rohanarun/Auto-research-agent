# okc: problem=muse_attention_decode
"""Fused preparation around FA3 for Muse attention decode."""
from typing import TypedDict
import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.kvcache.kvcache import store_cache

FP8 = torch.float8_e4m3fn
RMS_EPS = 1e-5
ROPE_POSITIONS = 1 << 21
_ROPE = {}
_NORMS = {}

class Config(TypedDict):
    batch:int; q_len:int; num_q_heads:int; num_kv_heads:int; head_dim:int
    page_size:int; window:int; rope_theta:float; qk_scale_factor:float; sm_scale:float; kv_scale:float
class Inputs(TypedDict):
    qkv:torch.Tensor; positions:torch.Tensor; cache_loc:torch.Tensor; cache_seqlens:torch.Tensor
    cu_seqlens_q:torch.Tensor; scheduler_metadata:torch.Tensor|None; k_cache:torch.Tensor
    v_cache:torch.Tensor; k_scale:torch.Tensor; v_scale:torch.Tensor; page_table:torch.Tensor
    seq_lens:torch.Tensor; config:Config

def _cos_sin(theta, d, device):
    key=(theta,d,device)
    x=_ROPE.get(key)
    if x is None:
        inv=1.0/(theta**(torch.arange(0,d,2,dtype=torch.float32,device=device)/d))
        f=torch.einsum('i,j->ij',torch.arange(ROPE_POSITIONS,dtype=torch.float32,device=device),inv)
        x=torch.cat((f.cos(),f.sin()),dim=-1)
        _ROPE[key]=x
    return x

@triton.jit
def _prep(qkv, pos, slots, cs, qo, kc, vc,
          stride_t:tl.constexpr, HQ:tl.constexpr, HK:tl.constexpr, D:tl.constexpr,
          SCALE:tl.constexpr, ROPE:tl.constexpr):
    t=tl.program_id(0)
    # All query heads are normalized independently in one CTA.
    qh=tl.arange(0,HQ)[:,None]
    di=tl.arange(0,D)[None,:]
    x=tl.load(qkv+t*stride_t+qh*D+di).to(tl.float32)
    rr=tl.rsqrt(tl.sum(x*x,axis=1)[:,None]*(1.0/D)+1.0e-5)
    x=x*rr
    if ROPE:
        p=tl.load(pos+t)
        pair=di//2
        c=tl.load(cs+p*D+pair)
        s=tl.load(cs+p*D+D//2+pair)
        other=tl.load(qkv+t*stride_t+qh*D+(di^1)).to(tl.float32)*rr
        sg=tl.where((di&1)==0,-1.0,1.0)
        x=x*c+sg*other*s
    tl.store(qo+t*HQ*D+qh*D+di,x)
    # New K and V are packed directly into their final FP8 cache rows.
    kh=tl.arange(0,HK)[:,None]
    z=tl.load(qkv+t*stride_t+HQ*D+kh*D+di).to(tl.float32)
    kr=tl.rsqrt(tl.sum(z*z,axis=1)[:,None]*(1.0/D)+1.0e-5)
    z=z*kr
    if ROPE:
        p=tl.load(pos+t)
        pair=di//2
        c=tl.load(cs+p*D+pair)
        s=tl.load(cs+p*D+D//2+pair)
        other=tl.load(qkv+t*stride_t+HQ*D+kh*D+(di^1)).to(tl.float32)*kr
        sg=tl.where((di&1)==0,-1.0,1.0)
        z=z*c+sg*other*s
    v=tl.load(qkv+t*stride_t+(HQ+HK)*D+kh*D+di).to(tl.float32)
    slot=tl.load(slots+t)
    tl.store(kc+slot*HK*D+kh*D+di,z/SCALE)
    tl.store(vc+slot*HK*D+kh*D+di,v/SCALE)

def _norm(hq,hkv,d):
    key=(hq,hkv,d)
    if key not in _NORMS: _NORMS[key]=FusedQKNorm(head_dim=d,num_q_heads=hq,num_kv_heads=hkv,eps=RMS_EPS)
    return _NORMS[key]


@triton.jit
def _attn_split(qkv, positions, slots, seq_lens, page_table, kc, vc, cos_sin, partial, lse,
                STRIDE:tl.constexpr, MAXP:tl.constexpr, B:tl.constexpr, L:tl.constexpr,
                HQ:tl.constexpr, HK:tl.constexpr, D:tl.constexpr, GROUP:tl.constexpr,
                SPLITS:tl.constexpr, CHUNK:tl.constexpr, BN:tl.constexpr,
                WINDOW:tl.constexpr, ROPE:tl.constexpr, SM:tl.constexpr):
    pid=tl.program_id(0)
    sp=pid%SPLITS
    z=pid//SPLITS
    kh=z%HK
    z=z//HK
    qi=z%L
    b=z//L
    mm=tl.arange(0,GROUP)[:,None]
    dd=tl.arange(0,D)[None,:]
    valid_m=mm<GROUP
    qhead=kh*GROUP+mm
    qptr=qkv+(b*L+qi)*STRIDE+qhead*D+dd
    q=tl.load(qptr,mask=valid_m,other=0.0).to(tl.float32)
    qr=tl.rsqrt(tl.sum(q*q,axis=1)[:,None]*(1.0/D)+1.0e-5)
    q=q*qr
    pos=tl.load(positions+b*L+qi)
    if ROPE:
        pair=dd//2
        c=tl.load(cos_sin+pos*D+pair)
        sn=tl.load(cos_sin+pos*D+D//2+pair)
        other=tl.load(qkv+(b*L+qi)*STRIDE+qhead*D+(dd^1),mask=valid_m,other=0.0).to(tl.float32)*qr
        q=q*c+tl.where((dd&1)==0,-1.0,1.0)*other*sn
    q=q.to(tl.float8e4nv).to(tl.float16)
    m=tl.full((16,),-float('inf'),tl.float32)
    den=tl.zeros((16,),tl.float32)
    acc=tl.zeros((16,D),tl.float32)
    old=tl.load(seq_lens+b)
    first=pos-WINDOW+1 if WINDOW else 0
    nn=tl.arange(0,BN)[None,:]
    dk=tl.arange(0,D)[:,None]
    for st in range(0,CHUNK,BN):
        idx=first+sp*CHUNK+st+nn
        mask=(idx>=0)&(idx<old)&(idx<MAXP)
        phys=tl.load(page_table+b*MAXP+idx,mask=mask,other=0)
        kval=tl.load(kc+phys*HK*D+kh*D+dk,mask=mask,other=0.0).to(tl.float16)
        score=tl.dot(q,kval)*SM
        score=tl.where(mask,score,-float('inf'))
        mb=tl.max(score,axis=1)
        mn=tl.maximum(m,mb)
        alpha=tl.exp2((m-mn)*1.4426950408889634)
        prob=tl.exp2((score-mn[:,None])*1.4426950408889634)
        vval=tl.load(vc+phys.T*HK*D+kh*D+tl.arange(0,D)[None,:],mask=mask.T,other=0.0).to(tl.float16)
        acc=acc*alpha[:,None]+tl.dot(prob.to(tl.float16),vval)
        den=den*alpha+tl.sum(prob,axis=1)
        m=mn
    # Split zero incorporates the causal tokens from this verify block directly.  This
    # avoids a launch-wide dependency on the simultaneous FP8 cache writes.
    nj=tl.arange(0,16)[:,None]
    kd=tl.arange(0,D)[None,:]
    nmask=nj<L
    kin=tl.load(qkv+(b*L+nj)*STRIDE+HQ*D+kh*D+kd,mask=nmask,other=0.0).to(tl.float32)
    kr=tl.rsqrt(tl.sum(kin*kin,axis=1)[:,None]*(1.0/D)+1.0e-5)
    kn=kin*kr
    npos=old+nj
    if ROPE:
        pair=kd//2
        c=tl.load(cos_sin+npos*D+pair,mask=nmask,other=1.0)
        sn=tl.load(cos_sin+npos*D+D//2+pair,mask=nmask,other=0.0)
        ko=tl.load(qkv+(b*L+nj)*STRIDE+HQ*D+kh*D+(kd^1),mask=nmask,other=0.0).to(tl.float32)*kr
        kn=kn*c+tl.where((kd&1)==0,-1.0,1.0)*ko*sn
    vin=tl.load(qkv+(b*L+nj)*STRIDE+(HQ+HK)*D+kh*D+kd,mask=nmask,other=0.0).to(tl.float32)
    # Exactly one split/query program per KV head performs the required cache update.
    do_store=nmask&(sp==0)&(qi==0)
    sl=tl.load(slots+b*L+nj,mask=nmask,other=0)
    tl.store(kc+sl*HK*D+kh*D+kd,kn,mask=do_store)
    tl.store(vc+sl*HK*D+kh*D+kd,vin,mask=do_store)
    if sp==0:
        knq=kn.to(tl.float8e4nv).to(tl.float16)
        ns=tl.dot(q,tl.trans(knq))*SM
        causal=(tl.arange(0,16)[None,:]<=qi)&(tl.arange(0,16)[None,:]<L)
        ns=tl.where(causal,ns,-float('inf'))
        mb=tl.max(ns,axis=1)
        mn=tl.maximum(m,mb)
        alpha=tl.exp2((m-mn)*1.4426950408889634)
        prob=tl.exp2((ns-mn[:,None])*1.4426950408889634)
        acc=acc*alpha[:,None]+tl.dot(prob.to(tl.float16),vin.to(tl.float16))
        den=den*alpha+tl.sum(prob,axis=1)
        m=mn
    base=pid*GROUP*D
    gm=tl.arange(0,GROUP)[:,None]
    gd=tl.arange(0,D)[None,:]
    tl.store(partial+base+gm*D+gd,acc/den[:,None])
    tl.store(lse+pid*GROUP+tl.arange(0,GROUP),m+tl.log(den))

@triton.jit
def _attn_reduce(partial,lse,qkv,out,STRIDE:tl.constexpr,L:tl.constexpr,HQ:tl.constexpr,
                 HK:tl.constexpr,D:tl.constexpr,GROUP:tl.constexpr,SPLITS:tl.constexpr):
    pid=tl.program_id(0)
    h=pid%HQ
    z=pid//HQ
    qi=z%L
    b=z//L
    kh=h//GROUP
    gh=h%GROUP
    ss=tl.arange(0,SPLITS)[:,None]
    dd=tl.arange(0,D)[None,:]
    base=(((b*L+qi)*HK+kh)*SPLITS)
    ls=tl.load(lse+(base+ss)*GROUP+gh)
    mx=tl.max(ls,axis=0)
    w=tl.exp2((ls-mx)*1.4426950408889634)
    vals=tl.load(partial+((base+ss)*GROUP+gh)*D+dd)
    x=tl.sum(vals*w,axis=0)/tl.sum(w,axis=0)
    rr=tl.rsqrt(tl.sum(x*x,axis=0)*(1.0/D)+1.0e-5)
    gate=tl.load(qkv+(b*L+qi)*STRIDE+(HQ+2*HK)*D+h*D+tl.arange(0,D)).to(tl.float32)
    x=x*rr/(1.0+tl.exp(-gate))
    tl.store(out+(b*L+qi)*HQ*D+h*D+tl.arange(0,D),x)

def kernel_fn(data:Inputs)->torch.Tensor:
    cfg=data['config']; B=cfg['batch']; L=cfg['q_len']; hq=cfg['num_q_heads']; hkv=cfg['num_kv_heads']; d=cfg['head_dim']; T=B*L
    flat=data['qkv'].view(T,-1)
    maxp=data['page_table'].shape[1]
    tuned=(hq==16 and hkv==2 and d==64 and cfg['page_size']==1 and cfg['kv_scale']==1.0
           and B in (1,16,32) and 70000<=maxp<=90000)
    if tuned:
        if cfg['window']:
            splits=8 if B==1 else (4 if B==16 and L==1 else (2 if B==32 and L==1 else 1))
            extent=cfg['window']
        else:
            splits=(64 if L==1 else 8) if B==1 else (4 if B==16 and L==1 else (2 if B==32 and L==1 else 1))
            extent=maxp
        chunk=(extent+splits-1)//splits
        chunk=((chunk+127)//128)*128
        programs=B*L*hkv*splits
        partial=torch.empty((programs,8,d),dtype=torch.float32,device=flat.device)
        lse=torch.empty((programs,8),dtype=torch.float32,device=flat.device)
        out=torch.empty((B,L,hq*d),dtype=torch.bfloat16,device=flat.device)
        cs=_cos_sin(cfg['rope_theta'],d,flat.device) if cfg['rope_theta'] else flat
        _attn_split[(programs,)](flat,data['positions'],data['cache_loc'],data['seq_lens'],data['page_table'],
            data['k_cache'],data['v_cache'],cs,partial,lse,flat.shape[1],maxp,B,L,hq,hkv,d,8,
            splits,chunk,128,cfg['window'],bool(cfg['rope_theta']),cfg['sm_scale']*cfg['qk_scale_factor'],num_warps=8)
        _attn_reduce[(B*L*hq,)](partial,lse,flat,out,flat.shape[1],L,hq,hkv,d,8,splits,num_warps=4)
        return out
    if cfg['rope_theta']:
        q,k=_norm(hq,hkv,d)(flat,data['positions'],_cos_sin(cfg['rope_theta'],d,flat.device))
    else: q,k=_norm(hq,hkv,d)(flat)
    v=flat[:,(hq+hkv)*d:(hq+2*hkv)*d]
    if cfg['kv_scale']!=1.0: k,v=k/data['k_scale'],v/data['v_scale']
    store_cache(k.to(FP8).reshape(T,hkv*d).view(torch.uint8),v.to(FP8).reshape(T,hkv*d).view(torch.uint8),
                data['k_cache'].view(-1,hkv*d).view(torch.uint8),data['v_cache'].view(-1,hkv*d).view(torch.uint8),
                data['cache_loc'],reserved_skip_index=-1)
    desc={}
    if cfg['kv_scale']!=1.0:
        desc={'k_descale':data['k_scale'].reshape(1,1).expand(B,hkv),'v_descale':data['v_scale'].reshape(1,1).expand(B,hkv)}
    z=flash_attn_with_kvcache(q=q.view(T,hq,d).to(FP8),k_cache=data['k_cache'],v_cache=data['v_cache'],
        page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],
        max_seqlen_q=L,softmax_scale=cfg['sm_scale']*cfg['qk_scale_factor'],causal=True,
        window_size=(cfg['window']-1,0) if cfg['window'] else (-1,-1),scheduler_metadata=data['scheduler_metadata'],**desc)
    if isinstance(z,tuple): z=z[0]
    gate=flat[:,(hq+2*hkv)*d:]
    z=rms_norm_gate(z.reshape(T,hq*d).to(torch.bfloat16),gate,None,head_dim=d,eps=RMS_EPS,has_norm=True)
    return z.view(B,L,hq*d)
