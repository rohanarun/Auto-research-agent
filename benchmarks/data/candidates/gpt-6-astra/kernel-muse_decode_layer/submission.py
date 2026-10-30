# okc: problem=muse_decode_layer
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


import triton
import triton.language as tl

@triton.jit
def _small_mm(X, W, P, M:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
              BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr, SPLIT:tl.constexpr):
    n = tl.program_id(0)*BN+tl.arange(0,BN)
    m = tl.program_id(1)*BM+tl.arange(0,BM)
    sk = tl.program_id(2)
    k = sk*(K//SPLIT)+tl.arange(0,BK)
    acc = tl.full((BM,BN),0,tl.float32)
    for i in range(K//SPLIT//BK):
        a = tl.load(X+m[:,None]*K+k[None,:],m[:,None]<M,0)
        b = tl.load(W+n[None,:]*K+k[:,None],n[None,:]<N,0)
        acc = tl.dot(a,b,acc)
        k += BK
    tl.store(P+sk*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _mm_finish(P,Y,MN:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    s=tl.arange(0,S)
    v=tl.load(P+s[:,None]*MN+i[None,:],i[None,:]<MN,0)
    tl.store(Y+i,tl.sum(v,0),i<MN)

@triton.jit
def _gemv_split(X,W,P,N:tl.constexpr,K:tl.constexpr,S:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.program_id(1)*BK+tl.arange(0,BK)
    x=tl.load(X+k)
    w=tl.load(W+n[:,None]*K+k[None,:],n[:,None]<N,0).to(tl.float32)
    acc=tl.sum(w*x[None,:].to(tl.float32),1)
    tl.store(P+tl.program_id(1)*N+n,acc,n<N)

def _fast_gemm(x,w):
    m,k=x.shape;n=w.shape[0]
    if m>32:
        return _gemm(x,w)
    split=8
    p=torch.empty((split,m,n),device=x.device,dtype=torch.float32)
    y=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    if m==1:
        _gemv_split[(triton.cdiv(n,4),split)](x,w,p,n,k,split,4,k//split,num_warps=4)
    else:
        bm=16 if m<=16 else 32
        _small_mm[(triton.cdiv(n,64),triton.cdiv(m,bm),split)](x,w,p,m,n,k,bm,64,64,split,num_warps=4)
    _mm_finish[(triton.cdiv(m*n,256),)](p,y,m*n,split,256,num_warps=4)
    return y

@triton.jit
def _qkv_prep(A,Q,KC,VC,LOC,POS,CS,KS,VS,ROPE:tl.constexpr):
    t=tl.program_id(0)
    h=tl.arange(0,32)
    d=tl.arange(0,64)
    idx=tl.where(h<16,h*64,1024+(h-16)*64)
    a=tl.load(A+t*2304+idx[:,None]+d[None,:],h[:,None]<18,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(a*a,1)/64+1.e-5)
    a=a*inv[:,None]
    if ROPE:
        p=tl.load(POS+t)
        c=tl.load(CS+p*64+d//2)
        s=tl.load(CS+p*64+d//2+32)
        other=tl.load(A+t*2304+idx[:,None]+(d[None,:]^1),h[:,None]<18,0).to(tl.float32)*inv[:,None]
        a=a*c[None,:]+tl.where((d[None,:]&1)==0,-other,other)*s[None,:]
    a=a.to(tl.bfloat16).to(tl.float32)
    tl.store(Q+t*1024+h[:,None]*64+d[None,:],a,h[:,None]<16)
    slot=tl.load(LOC+t)
    ks=tl.load(KS);vs=tl.load(VS)
    tl.store(KC+slot*128+(h[:,None]-16)*64+d[None,:],(a/ks).to(tl.bfloat16),(h[:,None]>=16)&(h[:,None]<18))
    v=tl.load(A+t*2304+1152+(h[:,None]-16)*64+d[None,:],(h[:,None]>=16)&(h[:,None]<18),0).to(tl.float32)
    tl.store(VC+slot*128+(h[:,None]-16)*64+d[None,:],(v/vs).to(tl.bfloat16),(h[:,None]>=16)&(h[:,None]<18))

@triton.jit
def _norm_gate(A,QKV,O):
    t=tl.program_id(0);h=tl.arange(0,16);d=tl.arange(0,64)
    v=tl.load(A+t*1024+h[:,None]*64+d[None,:]).to(tl.float32)
    g=tl.load(QKV+t*2304+1280+h[:,None]*64+d[None,:]).to(tl.float32)
    v=v*tl.rsqrt(tl.sum(v*v,1)/64+1.e-5)[:,None]*tl.sigmoid(g)
    tl.store(O+t*1024+h[:,None]*64+d[None,:],v)

@triton.jit
def _prenorm(X,W,Y):
    t=tl.program_id(0);j=tl.arange(0,4096)
    x=tl.load(X+t*4096+j).to(tl.float32)
    w=tl.load(W+j).to(tl.float32)
    y=x*tl.rsqrt(tl.sum(x*x,0)/4096+1.e-5)*(1.+w)
    tl.store(Y+t*4096+j,y)

_old_prepare=prepare
def prepare(weights):
    state=_old_prepare(weights)
    state['pre_expert_raw']=weights['pre_expert_norm']
    return state

def kernel_fn(data,state):
    cfg=data['config'];tokens=cfg['tokens'];batch=cfg['batch'];length=cfg['q_len']
    dense=state['dense']
    qkv=_fast_gemm(data['attn_input'],dense['qkv_gate'])
    q=torch.empty((tokens,16,64),device=qkv.device,dtype=FP8)
    _qkv_prep[(tokens,)](qkv,q,data['k_cache'],data['v_cache'],data['cache_loc'],data['positions'],state['cos_sin'],data['k_scale'],data['v_scale'],bool(cfg['rope_theta']),num_warps=4)
    descale={}
    if cfg['kv_scale']!=1.:
        descale=dict(k_descale=data['k_scale'].reshape(1,1).expand(batch,2),v_descale=data['v_scale'].reshape(1,1).expand(batch,2))
    window=cfg['window']
    if window and length<=8:
        out=_sliding_attention(data,q)
    else:
        out=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],max_seqlen_q=length,softmax_scale=cfg['sm_scale']*cfg['qk_scale_factor'],causal=True,window_size=(window-1,0) if window else (-1,-1),scheduler_metadata=data['scheduler_metadata'],**descale)
    if isinstance(out,tuple):out=out[0]
    o=torch.empty((tokens,1024),device=qkv.device,dtype=torch.bfloat16)
    _norm_gate[(tokens,)](out,qkv,o,num_warps=4)
    attn_output=_fast_gemm(o,dense['o'])
    split_router=tokens>8
    ffn_input,stream,router_input=state['attn_boundary'](data['stream'],attn_output,branch_dtype=torch.bfloat16,router_dtype=None if split_router else torch.float32)
    main_stream,side_stream=torch.cuda.current_stream(),state['side_stream']
    side_stream.wait_stream(main_stream)
    expert_input=_fast_gemm(ffn_input,dense['pooler'])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits=torch.mm(ffn_input,state['router_hi_t'],out_dtype=torch.float32)
            logits+=torch.mm(ffn_input,state['router_lo_t'],out_dtype=torch.float32)
        else:
            logits=F.linear(router_input,state['router'])
        topk_weights,topk_ids=state['routing'](None,logits,TOP_K,True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights,topk_ids):tensor.record_stream(main_stream)
    p=torch.empty_like(expert_input)
    _prenorm[(tokens,)](expert_input,state['pre_expert_raw'],p,num_warps=4)
    if tokens==1:
        expert_output=_single_experts(p,topk_weights,topk_ids,data['local_start'],state)
    else:
        expert_output=_compact_experts(p,topk_weights,topk_ids,data['local_start'],state)
    if tokens<=8:
        unpooled=_fast_gemm(expert_output,dense['unpooler'][:1024])
        ffn_output=F.pad(unpooled,(0,7168))
    else:ffn_output=_fast_gemm(expert_output,dense['unpooler'])
    next_input,stream,_=state['ffn_boundary'](stream,ffn_output,branch_dtype=torch.bfloat16)
    return next_input,stream,topk_ids

# Compact split-window attention. Every visible position is visited, including
# newly written cache entries; all addresses are computed from replay inputs.
@triton.jit
def _window_attention(Q,KC,VC,PT,POS,KS,VS,PART,LSE,
                      L:tl.constexpr, PAGES:tl.constexpr, PAGE:tl.constexpr,
                      SCALE:tl.constexpr, BM:tl.constexpr, BN:tl.constexpr):
    b=tl.program_id(0); hk=tl.program_id(1); split=tl.program_id(2)
    r=tl.arange(0,BM); d=tl.arange(0,64); j=tl.arange(0,BN)
    token=b*L+r//8
    last=tl.load(POS+b*L+L-1)
    first=tl.maximum(0,last-(2048+L-2))
    kp=first+split*BN+j
    pos=tl.load(POS+token,r<L*8,0)
    page=tl.load(PT+b*PAGES+kp//PAGE,kp<=last,0)
    slot=page*PAGE+kp%PAGE
    q=tl.load(Q+token[:,None]*1024+hk*512+(r[:,None]%8)*64+d[None,:],r[:,None]<L*8,0.0).to(tl.bfloat16)
    k=tl.load(KC+slot[None,:]*128+hk*64+d[:,None],(kp[None,:]<=last)&(page[None,:]>=0),0.0).to(tl.bfloat16)
    scores=tl.dot(q,k)*SCALE*tl.load(KS)
    visible=(kp[None,:]<=pos[:,None])&(kp[None,:]>pos[:,None]-2048)&(r[:,None]<L*8)&(kp[None,:]<=last)
    scores=tl.where(visible,scores,-float('inf'))
    m=tl.max(scores,1)
    safe=tl.where(m==-float('inf'),0.,m)
    prob=tl.exp(scores-safe[:,None])
    denom=tl.sum(prob,1)
    v=tl.load(VC+slot[:,None]*128+hk*64+d[None,:],(kp[:,None]<=last)&(page[:,None]>=0),0.0).to(tl.bfloat16)
    acc=tl.dot(prob.to(tl.bfloat16),v)*tl.load(VS)
    ns=triton.cdiv(2048+L-1,BN)
    offset=(((b*2+hk)*ns+split)*(L*8)+r)
    tl.store(PART+offset[:,None]*64+d[None,:],acc/tl.maximum(denom[:,None],1.e-30),r[:,None]<L*8)
    tl.store(LSE+offset,safe+tl.log(denom),r<L*8)

@triton.jit
def _window_merge(PART,LSE,OUT,L:tl.constexpr,NS:tl.constexpr,BS:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1)
    b=t//L;r=t%L*8+h%8;hk=h//8
    s=tl.arange(0,BS);d=tl.arange(0,64)
    off=((b*2+hk)*NS+s)*(L*8)+r
    l=tl.load(LSE+off,s<NS,-float('inf'))
    p=tl.exp(l-tl.max(l,0)); p=p/tl.sum(p,0)
    a=tl.load(PART+off[:,None]*64+d[None,:],s[:,None]<NS,0)
    v=tl.sum(a*p[:,None],0)
    tl.store(OUT+t*1024+h*64+d,v)

def _sliding_attention(data,q):
    c=data['config'];l=c['q_len'];b=c['batch'];ns=triton.cdiv(2048+l-1,128)
    part=torch.empty((b,2,ns,l*8,64),device=q.device,dtype=torch.float32)
    lse=torch.empty((b,2,ns,l*8),device=q.device,dtype=torch.float32)
    out=torch.empty((b*l,16,64),device=q.device,dtype=torch.bfloat16)
    _window_attention[(b,2,ns)](q,data['k_cache'],data['v_cache'],data['page_table'],data['positions'],data['k_scale'],data['v_scale'],part,lse,l,data['page_table'].shape[1],c['page_size'],c['sm_scale']*c['qk_scale_factor'],max(16,l*8),128,num_warps=4)
    _window_merge[(b*l,16)](part,lse,out,l,ns,triton.next_power_of_2(ns),num_warps=4)
    return out

@triton.jit
def _single_align(IDS,START,LOCAL,SORTED,EXPERTS,COUNT):
    r=tl.arange(0,8)
    e=(tl.load(IDS+r)-tl.load(START)+256)%256
    valid=e<32
    rank=tl.cumsum(valid.to(tl.int32),0)-1
    tl.store(LOCAL+r,tl.where(valid,e,-1))
    # Each selected expert occurs once for a single token. One Marlin M=8
    # tile per local route; other rows are the ordinary out-of-range sentinel.
    j=tl.arange(0,8)
    tl.store(SORTED+rank[:,None]*8+j[None,:],tl.where(j[None,:]==0,r[:,None],8),valid[:,None])
    tl.store(EXPERTS+rank,e,valid)
    tl.store(COUNT,tl.sum(valid.to(tl.int32),0)*8)

def _single_experts(x,tw,ids,start,state):
    bank=state['bank'];device=x.device
    local=torch.empty((1,8),device=device,dtype=torch.int32)
    sorted_ids=torch.empty((64,),device=device,dtype=torch.int32)
    block_experts=torch.empty((8,),device=device,dtype=torch.int32)
    n_padded=torch.empty((1,),device=device,dtype=torch.int32)
    _single_align[(1,)](ids,start,local,sorted_ids,block_experts,n_padded,num_warps=4)
    a=torch.empty((8,8192),device=device,dtype=x.dtype)
    inter=torch.empty((8,4096),device=device,dtype=x.dtype)
    y=torch.empty((8,4096),device=device,dtype=x.dtype)
    common=dict(moe_block_size=8,mul_topk_weights=False,is_ep=True,b_q_type=scalar_types.float4_e2m1f,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
    a=moe_wna16_marlin_gemm(x,a,bank['w13'],None,bank['w13_s'],bank['w13_gs'],None,None,None,bank['workspace'],sorted_ids,block_experts,n_padded,tw,top_k=8,size_m=1,size_n=8192,size_k=4096,**common)
    silu_and_mul(a,inter,expert_ids=local.view(-1),expert_step=1)
    y=moe_wna16_marlin_gemm(inter,y,bank['w2'],None,bank['w2_s'],bank['w2_gs'],None,None,None,bank['workspace'],sorted_ids,block_experts,n_padded,tw,top_k=1,size_m=8,size_n=4096,size_k=4096,**common)
    return fused_post_expert_norm_weighted_sum(y.view(1,8,4096),tw,norm_weight=state['post_expert_w'],eps=1.e-8,weight_first=True,slot_expert_ids=local)

@triton.jit
def _route_counts(IDS,START,COUNTS,R:tl.constexpr,BR:tl.constexpr,BM:tl.constexpr):
    e=tl.program_id(0)
    r=tl.arange(0,BR)
    ids=tl.load(IDS+r,r<R,-256)
    local=(ids-tl.load(START)+256)%256
    n=tl.sum(((local==e)&(r<R)).to(tl.int32),0)
    tl.store(COUNTS+e,(n+BM-1)//BM)

@triton.jit
def _route_scatter(IDS,START,COUNTS,LOCAL,SORTED,EXPERTS,COUNT,
                   R:tl.constexpr,BR:tl.constexpr,BM:tl.constexpr,BB:tl.constexpr):
    e=tl.program_id(0)
    es=tl.arange(0,32)
    counts=tl.load(COUNTS+es)
    first_block=tl.sum(tl.where(es<e,counts,0),0)
    blocks=tl.sum(tl.where(es==e,counts,0),0)
    r=tl.arange(0,BR)
    ids=tl.load(IDS+r,r<R,-256)
    local=(ids-tl.load(START)+256)%256
    valid=(r<R)&(local==e)
    rank=tl.cumsum(valid.to(tl.int32),0)-1
    n=tl.sum(valid.to(tl.int32),0)
    tl.store(SORTED+first_block*BM+rank,r,valid)
    j=tl.arange(0,BM)
    tl.store(SORTED+first_block*BM+n+j,R,j<blocks*BM-n)
    bs=tl.arange(0,BB)
    tl.store(EXPERTS+first_block+bs,e,bs<blocks)
    if e==0:
        tl.store(LOCAL+r,tl.where(local<32,local,-1),r<R)
        tl.store(COUNT,tl.sum(counts,0)*BM)

def _compact_experts(x,tw,ids,start,state):
    bank=state['bank'];m=x.shape[0];r=m*8;device=x.device
    for bm in BLOCK_SIZE_M_CANDIDATES:
        if r/32/bm<0.9:break
    # Triton arange requires a power of two. Marlin also accepts M=32/64;
    # the only non-power-of-two baseline choice is replaced with M=32.
    if bm==48:bm=32
    capacity=r+32*(bm-1)
    local=torch.empty((m,8),device=device,dtype=torch.int32)
    counts=torch.empty((32,),device=device,dtype=torch.int32)
    sorted_ids=torch.empty((capacity,),device=device,dtype=torch.int32)
    block_experts=torch.empty((triton.cdiv(capacity,bm),),device=device,dtype=torch.int32)
    n_padded=torch.empty((1,),device=device,dtype=torch.int32)
    br=triton.next_power_of_2(r)
    _route_counts[(32,)](ids,start,counts,r,br,bm,num_warps=4)
    _route_scatter[(32,)](ids,start,counts,local,sorted_ids,block_experts,n_padded,r,br,bm,triton.next_power_of_2(triton.cdiv(r,bm)),num_warps=4)
    a=torch.empty((r,8192),device=device,dtype=x.dtype)
    inter=torch.empty((r,4096),device=device,dtype=x.dtype)
    y=torch.empty((r,4096),device=device,dtype=x.dtype)
    common=dict(moe_block_size=bm,mul_topk_weights=False,is_ep=True,b_q_type=scalar_types.float4_e2m1f,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
    a=moe_wna16_marlin_gemm(x,a,bank['w13'],None,bank['w13_s'],bank['w13_gs'],None,None,None,bank['workspace'],sorted_ids,block_experts,n_padded,tw,top_k=8,size_m=m,size_n=8192,size_k=4096,**common)
    silu_and_mul(a,inter,expert_ids=local.view(-1),expert_step=1)
    y=moe_wna16_marlin_gemm(inter,y,bank['w2'],None,bank['w2_s'],bank['w2_gs'],None,None,None,bank['workspace'],sorted_ids,block_experts,n_padded,tw,top_k=1,size_m=r,size_n=4096,size_k=4096,**common)
    return fused_post_expert_norm_weighted_sum(y.view(m,8,4096),tw,norm_weight=state['post_expert_w'],eps=1.e-8,weight_first=True,slot_expert_ids=local)
