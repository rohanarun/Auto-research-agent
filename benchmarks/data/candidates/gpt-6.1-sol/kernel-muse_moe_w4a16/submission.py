# okc: problem=muse_moe_w4a16
"""Dynamic local-expert alignment with the public Marlin W4A16 GEMMs.

Weights retain the baseline lossless NVFP4 repack. Routing, activation,
normalization, and weighting are recomputed on every graph replay.
The corrected final implementation has not been measured on the H200.
"""

from typing import TypedDict

import types

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }



import triton as tr
import triton.language as tl

@tr.jit
def _single_align(Ids, Local, Sorted, Experts, Padded,
                  T:tl.constexpr, E:tl.constexpr, START:tl.constexpr, BM:tl.constexpr):
    slot=tl.arange(0,T)
    ids=tl.load(Ids+slot)
    valid=(ids>=START)&(ids<START+E)
    rank=tl.cumsum(valid.to(tl.int32))-1
    lane=tl.arange(0,BM)
    value=tl.where(lane[None,:]==0,slot[:,None],T)
    tl.store(Sorted+rank[:,None]*BM+lane[None,:],value,valid[:,None])
    tl.store(Experts+rank,ids-START,valid)
    tl.store(Local+slot,tl.where(valid,ids-START,-1))
    tl.store(Padded,tl.sum(valid.to(tl.int32),0)*BM)

@tr.jit
def _collect(Ids, Local, Pairs, Counts, M:tl.constexpr, T:tl.constexpr, E:tl.constexpr, START:tl.constexpr, R:tl.constexpr):
    e=tl.program_id(0)
    r=tl.arange(0,R)
    ids=tl.load(Ids+r,r<M*T,other=-1)
    valid=(ids==e+START)&(r<M*T)
    rank=tl.cumsum(valid.to(tl.int32))-1
    tl.store(Pairs+e*M+rank,r,valid)
    tl.store(Counts+e,tl.sum(valid.to(tl.int32),0))
    if e==0:
        local=tl.where((ids>=START)&(ids<START+E),ids-START,-1)
        tl.store(Local+r,local,r<M*T)

@tr.jit
def _align(Pairs, Counts, Sorted, Experts, Padded, M:tl.constexpr, T:tl.constexpr, E:tl.constexpr, BM:tl.constexpr, R:tl.constexpr, RE:tl.constexpr, RB:tl.constexpr):
    e=tl.program_id(0)
    ee=tl.arange(0,RE)
    c=tl.load(Counts+ee,ee<E,other=0)
    padded=tl.cdiv(c,BM)*BM
    offset=tl.sum(tl.where(ee<e,padded,0),0)
    count=tl.load(Counts+e)
    r=tl.arange(0,R)
    pair=tl.load(Pairs+e*M+r,r<count,other=M*T)
    tl.store(Sorted+offset+r,pair,r<tl.cdiv(count,BM)*BM)
    b=tl.arange(0,RB)
    tl.store(Experts+offset//BM+b,e,b<tl.cdiv(count,BM))
    if e==0:
        tl.store(Padded,tl.sum(padded,0))

@tr.jit
def _norm_sum(Y, Ids, Gate, Norm, Out, H:tl.constexpr, T:tl.constexpr, BH:tl.constexpr):
    token=tl.program_id(0)
    h=tl.arange(0,BH)
    slot=tl.arange(0,T)
    ids=tl.load(Ids+token*T+slot)
    g=tl.load(Gate+token*T+slot)
    n=tl.load(Norm+h,h<H,other=0).to(tl.float32)
    y=tl.load(Y+(token*T+slot[:,None])*H+h[None,:],(ids[:,None]>=0)&(h[None,:]<H),other=0).to(tl.float32)
    y=(y*n[None,:]).to(tl.bfloat16).to(tl.float32)
    inv=tl.rsqrt(tl.sum(y*y,1)/H+1.e-8)
    z=tl.sum(y*(inv*g)[:,None],0)
    tl.store(Out+token*H+h,z,h<H)

def kernel_fn(data, state):
    cfg=data['config']
    bank=state['banks'][(cfg['local_start'],cfg['local_count'])]
    x=data['hidden_states']
    m,h=x.shape
    n=bank['size_n'];t=cfg['top_k'];e=cfg['local_count']
    bm=8 if m<=64 else 16
    dev=x.device
    local=torch.empty((m,t),device=dev,dtype=torch.int32)
    capacity=t*bm if m==1 else tr.cdiv(m*t+e*(bm-1),bm)*bm
    sorted_ids=torch.empty((capacity,),device=dev,dtype=torch.int32)
    block_experts=torch.empty((capacity//bm,),device=dev,dtype=torch.int32)
    n_padded=torch.empty((),device=dev,dtype=torch.int32)
    if m==1:
        _single_align[(1,)](data['topk_ids'],local,sorted_ids,block_experts,n_padded,
                           t,e,cfg['local_start'],bm,num_warps=4)
    else:
        pairs=torch.empty((e*m,),device=dev,dtype=torch.int32)
        counts=torch.empty((e,),device=dev,dtype=torch.int32)
        _collect[(e,)](data['topk_ids'],local,pairs,counts,m,t,e,cfg['local_start'],tr.next_power_of_2(m*t))
        _align[(e,)](pairs,counts,sorted_ids,block_experts,n_padded,m,t,e,bm,
                     tr.next_power_of_2(tr.cdiv(m,bm)*bm),tr.next_power_of_2(e),
                     tr.next_power_of_2(tr.cdiv(m,bm)),num_warps=4)
    gate=data['topk_weights']
    c1=torch.empty((m*t,2*n),device=dev,dtype=x.dtype)
    c2=torch.empty((m*t,n),device=dev,dtype=x.dtype)
    c3=torch.empty((m*t,h),device=dev,dtype=x.dtype)
    common=dict(moe_block_size=bm,mul_topk_weights=False,is_ep=True,
                b_q_type=scalar_types.float4_e2m1f,is_k_full=True,use_atomic_add=False,
                use_fp32_reduce=True,is_zp_float=False)
    ws=state['workspace']
    c1=moe_wna16_marlin_gemm(x,c1,bank['w13'],None,bank['w13_s'],bank['w13_gs'],None,None,None,ws,
        sorted_ids,block_experts,n_padded,gate,top_k=t,size_m=m,size_n=2*n,size_k=h,**common)
    silu_and_mul(c1,c2,expert_ids=local.view(-1),expert_step=1)
    y=moe_wna16_marlin_gemm(c2,c3,bank['w2'],None,bank['w2_s'],bank['w2_gs'],None,None,None,ws,
        sorted_ids,block_experts,n_padded,gate,top_k=1,size_m=m*t,size_n=h,size_k=n,**common)
    out=torch.empty_like(x)
    _norm_sum[(m,)](y,local,gate,state['norm_w'],out,h,t,tr.next_power_of_2(h),num_warps=8)
    return out
