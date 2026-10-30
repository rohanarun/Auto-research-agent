# okc: problem=muse_prefill_layer
"""Muse Spark prefill layer: cuBLAS dense GEMMs, FA3 FP8 attention, fused Triton kernels for the
norms/RoPE/cache store/routing/combine, and a Hopper grouped expert GEMM (NVFP4 weights dequantized
in registers as the wgmma A operand, BF16 activations from shared memory, fused SiLU*up epilogue)."""

import hashlib

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
from triton.language.extra import libdevice

HIDDEN = 8192
D_EXPERT = 4096
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
HQ, HKV, HD = 16, 2, 64
Q_SIZE, KV_SIZE = HQ * HD, HKV * HD
QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
ROUTER_MULT = 0.1767766952966369
ROPE_THETA = 500_000.0
FP8 = torch.float8_e4m3fn
P_GEMM1 = 8  # tile slots per weight n-tile in the (non-persistent) expert GEMMs
P_GEMM2 = 16
# expert GEMM launch mode per token count: "fused" (both GEMMs in one persistent launch), "persist" (two persistent
# launches), "p816" (two per-tile-grid launches with P_GEMM1/P_GEMM2 slots)
MOE_LARGE_T = 8192
MOE_MODE_SMALL = "fused"
MOE_MODE_LARGE = "fused"
# tile order for the persistent work queue: 0 expert order, 1 biggest first, 2 big first + small tiles spread between
MOE_ORDER_SMALL = 1  # 3: expert groups walked n-tile-major (weights shared through L2)
MOE_ORDER_LARGE = 1

try:  # FA3 (Hopper) from the judge image
    from sgl_kernel.flash_attn import flash_attn_with_kvcache as _fa3
except Exception:  # local testing
    _fa3 = None

# ----------------------------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------------------------


@triton.jit
def _to_e4m3(x):
    """fp32 -> float8_e4m3fn bits (uint8), round-to-nearest-even, saturating at 448."""
    a = tl.abs(x)
    u = a.to(tl.int32, bitcast=True)
    normal = ((u + 0x7FFFF + ((u >> 20) & 1)) >> 20) - 960
    sub = libdevice.rint(a * 512.0).to(tl.int32)
    code = tl.where(a < 0.015625, sub, normal)
    code = tl.minimum(code, 126)
    code = tl.where(x < 0, code | 128, code)
    return code.to(tl.uint8)


@triton.jit
def _e4m3_to_f32(b):
    """float8_e4m3fn bits (int32 0..255) -> fp32, exact."""
    e = (b >> 3) & 15
    normal = (((b & 0x7F) << 20) + (120 << 23)).to(tl.float32, bitcast=True)
    sub = (b & 7).to(tl.float32) * 0.001953125
    v = tl.where(e == 0, sub, normal)
    return tl.where((b & 0x80) != 0, -v, v)

@triton.jit
def _qk_prep_kernel(qkv_ptr, pos_ptr, loc_ptr, invf_ptr, q8_ptr, kc_ptr, vc_ptr, ksc_ptr, vsc_ptr,
                    ROPE: tl.constexpr, SCALED: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:  # programmatic dependent launch: let the successor launch, wait for the predecessor's results
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    row = qkv_ptr + t.to(tl.int64) * 2304
    h = tl.arange(0, 16)[:, None]
    i = tl.arange(0, 32)[None, :]
    qe = tl.load(row + h * 64 + 2 * i).to(tl.float32)
    qo = tl.load(row + h * 64 + 2 * i + 1).to(tl.float32)
    rq = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / 64.0 + 1e-5)[:, None]
    qe = qe * rq
    qo = qo * rq
    kh = tl.arange(0, 2)[:, None]
    ke = tl.load(row + 1024 + kh * 64 + 2 * i).to(tl.float32)
    ko = tl.load(row + 1024 + kh * 64 + 2 * i + 1).to(tl.float32)
    rk = tl.rsqrt((tl.sum(ke * ke, 1) + tl.sum(ko * ko, 1)) / 64.0 + 1e-5)[:, None]
    ke = ke * rk
    ko = ko * rk
    if ROPE:
        pos = tl.load(pos_ptr + t).to(tl.float32)
        ang = pos * tl.load(invf_ptr + i)
        c = libdevice.cos(ang)
        s = libdevice.sin(ang)
        qe, qo = qe * c - qo * s, qe * s + qo * c
        ke, ko = ke * c - ko * s, ke * s + ko * c
    qrow = q8_ptr + t.to(tl.int64) * 1024
    tl.store(qrow + h * 64 + 2 * i, _to_e4m3(qe))
    tl.store(qrow + h * 64 + 2 * i + 1, _to_e4m3(qo))
    loc = tl.load(loc_ptr + t).to(tl.int64)
    j = tl.arange(0, 64)[None, :]
    v = tl.load(row + 1152 + kh * 64 + j).to(tl.float32)
    if SCALED:
        ks = tl.load(ksc_ptr)
        vs = tl.load(vsc_ptr)
        ke = ke / ks
        ko = ko / ks
        v = v / vs
    tl.store(kc_ptr + loc * 128 + kh * 64 + 2 * i, _to_e4m3(ke))
    tl.store(kc_ptr + loc * 128 + kh * 64 + 2 * i + 1, _to_e4m3(ko))
    tl.store(vc_ptr + loc * 128 + kh * 64 + j, _to_e4m3(v))


@triton.jit
def _norm_gate_kernel(o_ptr, qkv_ptr, out_ptr):
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]
    j = tl.arange(0, 64)[None, :]
    o = tl.load(o_ptr + t.to(tl.int64) * 1024 + h * 64 + j).to(tl.float32)
    g = tl.load(qkv_ptr + t.to(tl.int64) * 2304 + 1280 + h * 64 + j).to(tl.float32)
    r = tl.rsqrt(tl.sum(o * o, 1) / 64.0 + 1e-5)[:, None]
    y = o * r * tl.sigmoid(g)
    tl.store(out_ptr + t.to(tl.int64) * 1024 + h * 64 + j, y.to(tl.bfloat16))


@triton.jit
def _boundary_kernel(branch_ptr, stream_ptr, gpost_ptr, alpha_ptr, beta_ptr, gpre_ptr, sout_ptr, nout_ptr,
                     HAS_GPOST: tl.constexpr, H: tl.constexpr, STORE_S: tl.constexpr = True):
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    base = t.to(tl.int64) * H
    b = tl.load(branch_ptr + base + offs).to(tl.float32)
    if HAS_GPOST:
        b = b * (1.0 + tl.load(gpost_ptr + offs).to(tl.float32))
    r1 = tl.rsqrt(tl.sum(b * b, 0) / H + 1e-8)
    s = tl.load(alpha_ptr + offs) * tl.load(stream_ptr + base + offs) + tl.load(beta_ptr + offs) * (b * r1)
    if STORE_S:
        tl.store(sout_ptr + base + offs, s)
    r2 = tl.rsqrt(tl.sum(s * s, 0) / H + 1e-5)
    tl.store(nout_ptr + base + offs, (s * r2 * (1.0 + tl.load(gpre_ptr + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _boundary2_kernel(a_ptr, stream_ptr, f_ptr, a1_ptr, b1_ptr, gpost_ptr, a2_ptr, b2_ptr, gnext_ptr, sout_ptr, nout_ptr,
                      H: tl.constexpr):
    """Layer-end boundary that recomputes the attention boundary's stream (alpha1*stream + beta1*norm(a)) instead of
    reading it back: 2 bytes/element less traffic than storing and reloading it in fp32."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    base = t.to(tl.int64) * H
    a = tl.load(a_ptr + base + offs).to(tl.float32)
    ra = tl.rsqrt(tl.sum(a * a, 0) / H + 1e-8)
    s1 = tl.load(a1_ptr + offs) * tl.load(stream_ptr + base + offs) + tl.load(b1_ptr + offs) * (a * ra)
    b = tl.load(f_ptr + base + offs).to(tl.float32) * (1.0 + tl.load(gpost_ptr + offs).to(tl.float32))
    rb = tl.rsqrt(tl.sum(b * b, 0) / H + 1e-8)
    s = tl.load(a2_ptr + offs) * s1 + tl.load(b2_ptr + offs) * (b * rb)
    tl.store(sout_ptr + base + offs, s)
    r2 = tl.rsqrt(tl.sum(s * s, 0) / H + 1e-5)
    tl.store(nout_ptr + base + offs, (s * r2 * (1.0 + tl.load(gnext_ptr + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _route_kernel(logit_ptr, bias_ptr, lstart_ptr, ids_ptr, w_ptr, lid_ptr, pos_ptr, cnt_ptr):
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    lg = tl.load(logit_ptr + t.to(tl.int64) * 512 + e) + tl.load(logit_ptr + t.to(tl.int64) * 512 + 256 + e)
    s = tl.sigmoid(lg * 0.1767766952966369)
    b = s + tl.load(bias_ptr + e)
    ls = tl.load(lstart_ptr)
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    sel = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        m = tl.max(b, 0)
        idx = tl.min(tl.where(b == m, e, 256), 0)
        sk = tl.sum(tl.where(e == idx, s, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        sel = tl.where(k8 == k, sk, sel)
        b = tl.where(e == idx, -float("inf"), b)
    w = sel / (tl.sum(sel, 0) + 1e-15)
    tl.store(ids_ptr + t * 8 + k8, ids)
    tl.store(w_ptr + t * 8 + k8, w)
    loc = (ids - ls + 256) % 256
    is_local = loc < 32
    lid = tl.where(is_local, loc, -1)
    pos = tl.atomic_add(cnt_ptr + loc, tl.full([8], 1, tl.int32), mask=is_local)
    tl.store(lid_ptr + t * 8 + k8, lid)
    tl.store(pos_ptr + t * 8 + k8, tl.where(is_local, pos, 0))


@triton.jit
def _combine_kernel(y_ptr, rrow_ptr, lid_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, H: tl.constexpr, PDL: tl.constexpr = False):
    offs = tl.arange(0, H)
    if COMBINE_PW_EARLY:  # the norm weight is a constant: load it before waiting for the expert kernel
        pw = 1.0 + tl.load(pw_ptr + offs).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    if not COMBINE_PW_EARLY:
        pw = 1.0 + tl.load(pw_ptr + offs).to(tl.float32)  # BF16 weight, applied as 1 + w
    acc = tl.zeros([H], dtype=tl.float32)
    for k in tl.static_range(8):
        lid = tl.load(lid_ptr + t * 8 + k)
        if lid >= 0:
            row = tl.load(rrow_ptr + t * 8 + k)
            y = tl.load(y_ptr + row.to(tl.int64) * H + offs).to(tl.float32) * (tl.load(gs_ptr + lid) * pw)
            rr = tl.rsqrt(tl.sum(y * y, 0) / H + 1e-8)
            acc += y * (rr * tl.load(w_ptr + t * 8 + k))
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(tl.bfloat16))


@triton.jit
def _rowq8_kernel(x_ptr, out_ptr, sc_ptr, H: tl.constexpr):
    offs = tl.arange(0, H)
    t = tl.program_id(0)
    x = tl.load(x_ptr + t.to(tl.int64) * H + offs).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), 0), 1e-30) * (1.0 / 448.0)
    tl.store(sc_ptr + t, s)
    tl.store(out_ptr + t.to(tl.int64) * H + offs, (x * (1.0 / s)).to(tl.float8e4nv))


@triton.jit
def _combine8_kernel(y_ptr, rrow_ptr, lid_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, sc_ptr, H: tl.constexpr):
    """_combine_kernel with an E4M3 output row and its scale (row amax / 448) for the FP8 unpooler GEMM."""
    offs = tl.arange(0, H)
    t = tl.program_id(0)
    pw = 1.0 + tl.load(pw_ptr + offs).to(tl.float32)
    acc = tl.zeros([H], dtype=tl.float32)
    for k in tl.static_range(8):
        lid = tl.load(lid_ptr + t * 8 + k)
        if lid >= 0:
            row = tl.load(rrow_ptr + t * 8 + k)
            y = tl.load(y_ptr + row.to(tl.int64) * H + offs).to(tl.float32) * (tl.load(gs_ptr + lid) * pw)
            rr = tl.rsqrt(tl.sum(y * y, 0) / H + 1e-8)
            acc += y * (rr * tl.load(w_ptr + t * 8 + k))
    s = tl.maximum(tl.max(tl.abs(acc), 0), 1e-30) * (1.0 / 448.0)
    tl.store(sc_ptr + t, s)
    tl.store(out_ptr + t.to(tl.int64) * H + offs, (acc * (1.0 / s)).to(tl.float8e4nv))


# ----------------------------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------------------------

def _residual(gate):
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha.contiguous(), beta.contiguous()


def prepare(weights):
    dev = weights["qkv_gate"].device
    st = {}
    for n in ("qkv_gate", "o", "pooler", "unpooler"):
        st[n] = weights[n].contiguous()
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    st["router_hilo"] = torch.cat([hi, lo], 0).contiguous()  # [512, 8192]: one GEMM gives both halves
    st["router_bias"] = weights["router_bias"].float().contiguous()
    st["a1"], st["b1"] = _residual(weights["attn_residual_gate"])
    st["a2"], st["b2"] = _residual(weights["ffn_residual_gate"])
    # norm weights stay BF16 exactly as given (applied as 1 + w): the kernels form 1 + w in fp32 when they load them
    for n in ("pre_ffn_norm", "pre_expert_norm", "post_expert_norm", "post_ffn_norm", "next_pre_attn_norm"):
        st["g_" + n] = weights[n].contiguous()
    st["inv_freq"] = (1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2, device=dev).float() / HD))).contiguous()
    # the fast block-scale decode is exact only for positive normal E4M3 scales (bytes 0x08..0x7E: no zero, subnormal,
    # negative or NaN encodings); otherwise build the exact decode
    fast_scales = all(bool(((weights[k].view(torch.uint8) >= 0x08) & (weights[k].view(torch.uint8) <= 0x7E)).all())
                      for k in ("w13_sf", "w2_sf"))
    st["mod"] = _build(fast_scales)
    st["w13r"] = _repack(weights["w13"], weights["w13_sf"], True)
    st["w2r"] = _repack(weights["w2"], weights["w2_sf"], False)
    st["w13_gs"] = weights["w13_gs"].float().contiguous()
    st["w2_gs"] = weights["w2_gs"].float().contiguous()
    st["ones32"] = torch.ones(LOCAL_EXPERTS, dtype=torch.float32, device=dev)
    if UNPOOL_FP8:  # E4M3 unpooler with one scale per output channel
        _w = weights["unpooler"].float()
        _ws = _w.abs().amax(1, keepdim=True).clamp(min=1e-30) / 448.0
        st["unpooler8"] = (_w / _ws).to(FP8).contiguous()
        st["unpooler_ws"] = _ws.view(1, -1).contiguous()
        _w = weights["pooler"].float()
        _ws = _w.abs().amax(1, keepdim=True).clamp(min=1e-30) / 448.0
        st["pooler8"] = (_w / _ws).to(FP8).contiguous()
        st["pooler_ws"] = _ws.view(1, -1).contiguous()
        del _w, _ws
    st["no_zero"] = torch.empty((0,), dtype=torch.int32, device=dev)
    st["no_zero_f"] = torch.empty((0,), dtype=torch.float32, device=dev)
    st["no_zero_b"] = torch.empty((0,), dtype=torch.bfloat16, device=dev)
    props = torch.cuda.get_device_properties(dev)
    st["smem"] = 227 * 1024 if props.major >= 9 else 163 * 1024
    if SKINNY_TEST:  # feasibility test build (MOE_SKINNY_TEST): 2 CTAs per SM
        st["smem"] = 100 * 1024
    st["mod"].moe_init(st["smem"])
    st["attn_smem"] = ATTN_SMEM + (0 if props.major >= 9 else 3 * 64 * 128)  # sm_80 test path stages P in smem
    st["mod"].attn_init(st["attn_smem"])
    st["no_pf"] = torch.empty((0,), dtype=torch.uint8, device=dev)
    st["nsm"] = props.multi_processor_count
    _NSM[0] = st["nsm"]
    st["moe_dbg"] = torch.empty((0,), dtype=torch.int64, device=dev)  # a timeline buffer here turns on item tracing
    st["moe_flags"] = 0  # experiments only
    if not _SIDE_STREAM:
        _SIDE_STREAM.append(torch.cuda.Stream(device=dev))  # kept out of the state (only tensors/plain data there)
    st["lt_choice"] = None
    try:
        st["lt"] = _build_lt()
        st["lt_ws"] = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
        st["lt_ws_side"] = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
        st["lt_choice"] = _tune_lt(st)
    except Exception as exc:  # noqa: BLE001
        print("LT unavailable:", repr(exc)[:300], flush=True)
        st["lt_choice"] = None
    torch.cuda.synchronize()
    st["attn_choice"] = None
    if props.major >= 9 or OWN_ATTN_LOCAL:
        try:
            ok = _validate_attn(st, list(ATTN_VARIANTS) if props.major >= 9 else [24])
            if ok:
                if props.major >= 9 and ATTN_FIXED is not None and all(v in ok for v in ATTN_FIXED.values()):
                    st["attn_choice"] = dict(ATTN_FIXED)  # a prepare-time race between near-equal options adds noise
                else:
                    st["attn_choice"] = _tune_attn(st, ok) if props.major >= 9 else {k: ok[0] for k in ("slide", "full_long", "full_short", "full_batch")}
        except Exception as exc:  # noqa: BLE001
            print("ATTN validation/tuning failed:", repr(exc)[:300], flush=True)
            st["attn_choice"] = None
    if SELFTEST:
        try:
            _selftest(st, weights)
        except Exception as exc:  # noqa: BLE001
            print("SELFTEST failed:", repr(exc), flush=True)
        for fn in (_selftest_attn2, _selftest_boundary):
            try:
                fn(st)
            except Exception as exc:  # noqa: BLE001
                print("SELFTEST", fn.__name__, "failed:", repr(exc), flush=True)
    return st


# ----------------------------------------------------------------------------------------------
# expert GEMM (CUDA)
# ----------------------------------------------------------------------------------------------

KS, WBLK = 64, 4608
CPP = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
void k_moe_plan(const int* counts, int* offsets, int* tiles, int* ntiles, int* rows, int order, cudaStream_t st);
void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                 int T, cudaStream_t st, int* tok_left, int* cq);
void k_moe_gather(const void* p, const void* gw, const int* lid, const int* pos, const int* offsets, const int* counts,
                  void* xt, int* rrow, int T, cudaStream_t st);
void k_moe_gather_plan(const void* p, const void* gw, const int* lid, const int* pos, const int* counts, void* xt, int* rrow,
                       int* offsets, int* tiles, int* ntiles, int* rows, int order, int T, cudaStream_t st,
                       int* row_tok, int* cq, int* qtail);
void k_moe_combine(const void* y, const int* rrow, const int* lid, const float* topw, const void* pw, void* out, int T,
                   cudaStream_t st);
void k_moe_gemm(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                int P, bool gated, int smem, cudaStream_t st);
void k_moe_gemm_p(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                  int nctas, bool gated, int smem, int* counter, cudaStream_t st);
void k_boundary(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                const void* gpre, float* sout, void* nout, int T, bool has_gpost, cudaStream_t st, int* zp, int zn);
void k_qk_prep(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* invf, void* q8, void* kc,
               void* vc, const float* ksc, const float* vsc, int T, bool rope, bool scaled, cudaStream_t st);
void k_boundary_rc(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                   const void* gpre, float* sout, void* nout, int T, int mode, cudaStream_t st, int* zp, int zn,
                   const void* br1, float* rio, const float* alpha1, const float* beta1);
void k_boundary2(const void* a, const float* stream, const void* f, const float* a1, const float* b1, const void* gpost,
                 const float* a2, const float* b2, const void* gnext, float* sout, void* nout, int T, cudaStream_t st);
void k_kv_gather(const void* kc, const void* vc, const int* page_table, const int* seq_lens, void* ko, void* vo, int B, int max_pages,
                 int page, int L, int window, int Smax, cudaStream_t st);
void k_moe_init(int smem);
void k_moe_fused(const void* w13, const void* w2, const void* xt, void* inter, void* y, const int* tiles, const int* ntiles,
                 const float* gs13, const float* gs2, int nctas, int smem, int* counter, int* done, void* dbg, int flags,
                 cudaStream_t st, int* cit, int* qctl, int* tok_left, int* cq, const int* row_tok, const int* rrow,
                 const int* lid, const float* topw, const void* pw, void* out, int T);
void k_attn_prep(const void* kc, const void* vc, const int* page_table, int max_pages, int page, const int* seq_lens, int B, int L,
                 int window, int ntk, void* kt, void* vt, cudaStream_t st, int part);
void k_attn_fwd(const void* q, const void* kt, const void* vt, const int* seq_lens, const void* qkv, void* out, int B, int L,
                int window, int ntk, float c, float v_scale, int smem, int nsplit, float* ws, int variant, cudaStream_t st,
                const void* pf, unsigned long long pf_bytes);
void k_attn_init(int smem);
void k_attn_set_prof(void* p);
void k_attn_set_cnt(void* p);

static cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream(); }

void moe_route(torch::Tensor logits, torch::Tensor bias, torch::Tensor lstart, torch::Tensor ids, torch::Tensor w,
               torch::Tensor lid, torch::Tensor pos, torch::Tensor cnt, torch::Tensor tok_left, torch::Tensor cq) {
  k_moe_route(logits.data_ptr<float>(), bias.data_ptr<float>(), lstart.data_ptr<int>(), ids.data_ptr<int>(), w.data_ptr<float>(),
              lid.data_ptr<int>(), pos.data_ptr<int>(), cnt.data_ptr<int>(), (int)logits.size(0), cur(),
              tok_left.numel() ? tok_left.data_ptr<int>() : nullptr, cq.numel() ? cq.data_ptr<int>() : nullptr);
}
void moe_plan(torch::Tensor counts, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor rows,
              int64_t order) {
  k_moe_plan(counts.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
             (int)order, cur());
}
void moe_gather_plan(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor counts,
                     torch::Tensor xt, torch::Tensor rrow, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles,
                     torch::Tensor rows, int64_t order, torch::Tensor cit, torch::Tensor tok) {
  // CIT (cit.numel() > 0): queue tail = cit[numel - 3]; tok = [tok_left (T) | cq (T) | row_tok]
  const bool on = cit.numel() > 0;
  const int T = (int)p.size(0);
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_gather_plan(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), counts.data_ptr<int>(), xt.data_ptr(),
                    rrow.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
                    (int)order, T, cur(), on ? tk + 2 * T : nullptr, on ? tk + T : nullptr,
                    on ? cit.data_ptr<int>() + (cit.numel() - 3) : nullptr);
}
void moe_gather(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor offsets,
                torch::Tensor counts, torch::Tensor xt, torch::Tensor rrow) {
  k_moe_gather(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), offsets.data_ptr<int>(),
               counts.data_ptr<int>(), xt.data_ptr(), rrow.data_ptr<int>(), (int)p.size(0), cur());
}
void moe_combine(torch::Tensor y, torch::Tensor rrow, torch::Tensor lid, torch::Tensor topw, torch::Tensor pw, torch::Tensor out) {
  k_moe_combine(y.data_ptr(), rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), pw.data_ptr(),
                out.data_ptr(), (int)out.size(0), cur());
}
void moe_gemm(torch::Tensor wrep, torch::Tensor xt, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor gscale,
              torch::Tensor out, int64_t NT, int64_t P, bool gated, int64_t smem) {
  k_moe_gemm(wrep.data_ptr(), xt.data_ptr(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), gscale.data_ptr<float>(), out.data_ptr(),
             (int)NT, (int)P, gated, (int)smem, cur());
}
void moe_gemm_p(torch::Tensor wrep, torch::Tensor xt, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor gscale,
                torch::Tensor out, int64_t NT, int64_t nctas, bool gated, int64_t smem, torch::Tensor counter) {
  k_moe_gemm_p(wrep.data_ptr(), xt.data_ptr(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), gscale.data_ptr<float>(),
               out.data_ptr(), (int)NT, (int)nctas, gated, (int)smem, counter.data_ptr<int>(), cur());
}
void boundary_rc(torch::Tensor branch, torch::Tensor stream, torch::Tensor gpost, torch::Tensor alpha, torch::Tensor beta,
                 torch::Tensor gpre, torch::Tensor sout, torch::Tensor nout, int64_t mode, torch::Tensor zero,
                 torch::Tensor br1, torch::Tensor rio, torch::Tensor alpha1, torch::Tensor beta1) {
  k_boundary_rc(branch.data_ptr(), stream.data_ptr<float>(), gpost.data_ptr(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
                gpre.data_ptr(), sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), (int)mode, cur(),
                zero.numel() ? zero.data_ptr<int>() : nullptr, (int)zero.numel(), br1.data_ptr(), rio.data_ptr<float>(),
                alpha1.data_ptr<float>(), beta1.data_ptr<float>());
}
void qk_prep(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor invf, torch::Tensor q8, torch::Tensor kc,
             torch::Tensor vc, torch::Tensor ksc, torch::Tensor vsc, bool rope, bool scaled) {
  k_qk_prep(qkv.data_ptr(), pos.data_ptr(), pos.element_size() == 8, loc.data_ptr(), loc.element_size() == 8, invf.data_ptr<float>(),
            q8.data_ptr(), kc.data_ptr(), vc.data_ptr(), ksc.data_ptr<float>(), vsc.data_ptr<float>(), (int)qkv.size(0), rope, scaled, cur());
}
void boundary(torch::Tensor branch, torch::Tensor stream, torch::Tensor gpost, torch::Tensor alpha, torch::Tensor beta,
              torch::Tensor gpre, torch::Tensor sout, torch::Tensor nout, bool has_gpost, torch::Tensor zero) {
  k_boundary(branch.data_ptr(), stream.data_ptr<float>(), gpost.data_ptr(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
             gpre.data_ptr(), sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), has_gpost, cur(),
             zero.numel() ? zero.data_ptr<int>() : nullptr, (int)zero.numel());
}
void boundary2(torch::Tensor a, torch::Tensor stream, torch::Tensor f, torch::Tensor a1, torch::Tensor b1, torch::Tensor gpost,
               torch::Tensor a2, torch::Tensor b2, torch::Tensor gnext, torch::Tensor sout, torch::Tensor nout) {
  k_boundary2(a.data_ptr(), stream.data_ptr<float>(), f.data_ptr(), a1.data_ptr<float>(), b1.data_ptr<float>(), gpost.data_ptr(),
              a2.data_ptr<float>(), b2.data_ptr<float>(), gnext.data_ptr(), sout.data_ptr<float>(), nout.data_ptr(),
              (int)a.size(0), cur());
}
void kv_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor page_table, torch::Tensor seq_lens, torch::Tensor ko,
               torch::Tensor vo, int64_t page, int64_t L, int64_t window, int64_t Smax) {
  k_kv_gather(kc.data_ptr(), vc.data_ptr(), page_table.data_ptr<int>(), seq_lens.data_ptr<int>(), ko.data_ptr(), vo.data_ptr(),
              (int)seq_lens.size(0), (int)page_table.size(1), (int)page, (int)L, (int)window, (int)Smax, cur());
}
void moe_init(int64_t smem) { k_moe_init((int)smem); }
void moe_fused(torch::Tensor w13, torch::Tensor w2, torch::Tensor xt, torch::Tensor inter, torch::Tensor y, torch::Tensor tiles,
               torch::Tensor ntiles, torch::Tensor gs13, torch::Tensor gs2, int64_t nctas, int64_t smem, torch::Tensor counter,
               torch::Tensor done, torch::Tensor dbg, int64_t flags, torch::Tensor cit, torch::Tensor tok, torch::Tensor rrow,
               torch::Tensor lid, torch::Tensor topw, torch::Tensor pw, torch::Tensor out) {
  // CIT (cit.numel() > 0): cit = [max tiles GEMM2 counters | head, tail, error, pad]; tok = [tok_left (T) | cq (T) | row_tok]
  const bool on = cit.numel() > 0;
  const int T = (int)rrow.size(0);
  int* c = on ? cit.data_ptr<int>() : nullptr;
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_fused(w13.data_ptr(), w2.data_ptr(), xt.data_ptr(), inter.data_ptr(), y.data_ptr(), tiles.data_ptr<int>(),
              ntiles.data_ptr<int>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), (int)nctas, (int)smem, counter.data_ptr<int>(),
              done.data_ptr<int>(), dbg.numel() ? dbg.data_ptr() : nullptr, (int)flags, cur(),
              c, on ? c + (cit.numel() - 4) : nullptr, tk, on ? tk + T : nullptr, on ? tk + 2 * T : nullptr,
              rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), pw.data_ptr(), on ? out.data_ptr() : nullptr, T);
}
void attn_prep(torch::Tensor kc, torch::Tensor vc, torch::Tensor page_table, torch::Tensor seq_lens, int64_t page, int64_t L,
               int64_t window, int64_t ntk, torch::Tensor kt, torch::Tensor vt, int64_t part) {
  k_attn_prep(kc.data_ptr(), vc.data_ptr(), page_table.data_ptr<int>(), (int)page_table.size(1), (int)page, seq_lens.data_ptr<int>(),
              (int)seq_lens.size(0), (int)L, (int)window, (int)ntk, kt.data_ptr(), vt.data_ptr(), cur(), (int)part);
}
void attn_fwd(torch::Tensor q, torch::Tensor kt, torch::Tensor vt, torch::Tensor seq_lens, torch::Tensor qkv, torch::Tensor out,
              int64_t L, int64_t window, int64_t ntk, double c, double v_scale, int64_t smem, int64_t nsplit, torch::Tensor ws,
              int64_t variant, torch::Tensor pf) {
  k_attn_fwd(q.data_ptr(), kt.data_ptr(), vt.data_ptr(), seq_lens.data_ptr<int>(), qkv.data_ptr(), out.data_ptr(),
             (int)seq_lens.size(0), (int)L, (int)window, (int)ntk, (float)c, (float)v_scale, (int)smem, (int)nsplit,
             nsplit != 1 ? ws.data_ptr<float>() : nullptr, (int)variant, cur(), pf.numel() ? pf.data_ptr() : nullptr,
             (unsigned long long)(pf.numel() * pf.element_size()));
}
void attn_init(int64_t smem) { k_attn_init((int)smem); }
void attn_set_prof(torch::Tensor p) { k_attn_set_prof(p.numel() ? p.data_ptr() : nullptr); }
void attn_set_cnt(torch::Tensor p) { k_attn_set_cnt(p.numel() ? p.data_ptr() : nullptr); }
'''

GLUE = r'''
#include <cuda_runtime.h>
using namespace okc_moe;

#ifndef OKC_PDL
#define OKC_PDL 0
#endif
// launch with the programmatic-stream-serialization attribute (OKC_PDL), else a plain launch
template <typename... KArgs, typename... Args>
static inline void okc_launch(void (*k)(KArgs...), dim3 g, dim3 b, size_t smem, cudaStream_t st, Args... args) {
#if OKC_PDL
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, ((KArgs)args)...);
#else
  k<<<g, b, smem, st>>>(((KArgs)args)...);
#endif
}
void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                 int T, cudaStream_t st, int* tok_left, int* cq) {
  okc_launch(moe_route_kernel, dim3((T + ROUTE_WARPS - 1) / ROUTE_WARPS), dim3(ROUTE_WARPS * 32), 0, st, logits, bias, lstart, ids, w, lid, pos, cnt, T,
             tok_left, cq);
}
void k_moe_plan(const int* counts, int* offsets, int* tiles, int* ntiles, int* rows, int order, cudaStream_t st) {
  okc_launch(moe_plan_kernel, dim3(1), dim3(32), 0, st, counts, offsets, (int4*)tiles, ntiles, rows, order);
}
static int g_attn_merge_host = 0;  // set by k_attn_set_cnt: attention splits merged in-kernel
void k_moe_gather(const void* p, const void* gw, const int* lid, const int* pos, const int* offsets, const int* counts,
                  void* xt, int* rrow, int T, cudaStream_t st) {
#ifndef MOE_PADFILL
#define MOE_PADFILL 1
#endif
  okc_launch(moe_gather_kernel<false>, dim3(T + (MOE_PADFILL ? 32 * MOE_PADFILL_PARTS : 0)), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, offsets, counts, (uint8_t*)xt, rrow, T, (int*)nullptr, (int4*)nullptr, (int*)nullptr, (int*)nullptr, 0,
             (int*)nullptr, (int*)nullptr, (int*)nullptr);
}
void k_moe_gather_plan(const void* p, const void* gw, const int* lid, const int* pos, const int* counts, void* xt, int* rrow,
                       int* offsets, int* tiles, int* ntiles, int* rows, int order, int T, cudaStream_t st,
                       int* row_tok, int* cq, int* qtail) {
  okc_launch(moe_gather_kernel<true>, dim3(T + (MOE_PADFILL ? 32 * MOE_PADFILL_PARTS : 0)), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, (const int*)nullptr, counts, (uint8_t*)xt, rrow, T,
             offsets, (int4*)tiles, ntiles, rows, order, row_tok, cq, qtail);
}
void k_moe_combine(const void* y, const int* rrow, const int* lid, const float* topw, const void* pw, void* out, int T,
                   cudaStream_t st) {
  okc_launch(moe_combine_kernel, dim3(T), dim3(256), 0, st, (const __nv_bfloat16*)y, rrow, lid, topw, (const __nv_bfloat16*)pw, (__nv_bfloat16*)out);
}
static Args make_args(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out,
                      int NT, int smem, int* counter) {
  Args a;
  a.wrep = (const uint8_t*)wrep; a.xt = (const uint8_t*)xt; a.tiles = (const int4*)tiles; a.ntiles = ntiles;
  a.gscale = gscale; a.out = (uint8_t*)out; a.NT = NT; a.smem_bytes = smem; a.counter = counter; a.flags = 0; a.mul16 = 16u; a.mul64 = 64u;
  return a;
}
void k_moe_gemm(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                int P, bool gated, int smem, cudaStream_t st) {
  Args a = make_args(wrep, xt, tiles, ntiles, gscale, out, NT, smem, nullptr);
  dim3 grid((unsigned)NT, (unsigned)P);
  if (gated) moe_gemm_kernel<true><<<grid, 384, smem, st>>>(a);
  else moe_gemm_kernel<false><<<grid, 384, smem, st>>>(a);
}
void k_moe_gemm_p(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                  int nctas, bool gated, int smem, int* counter, cudaStream_t st) {
  Args a = make_args(wrep, xt, tiles, ntiles, gscale, out, NT, smem, counter);
  if (gated) moe_gemm_persistent<true><<<(unsigned)nctas, 384, smem, st>>>(a);
  else moe_gemm_persistent<false><<<(unsigned)nctas, 384, smem, st>>>(a);
}
#ifndef BND_TPB
#define BND_TPB 512
#endif
#ifndef BND_HINT
#define BND_HINT false
#endif
template <bool R, bool S, typename PT, typename LT>
static void qk_prep_launch(const void* qkv, const void* pos, const void* loc, const float* invf, void* q8, void* kc, void* vc,
                           const float* ksc, const float* vsc, int T, cudaStream_t st) {
  okc_launch(qk_prep_kernel<R, S, PT, LT>, dim3(T), dim3(128), 0, st, (const __nv_bfloat16*)qkv, (const PT*)pos, (const LT*)loc, invf,
             (uint8_t*)q8, (uint8_t*)kc, (uint8_t*)vc, ksc, vsc);
}
void k_qk_prep(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* invf, void* q8, void* kc,
               void* vc, const float* ksc, const float* vsc, int T, bool rope, bool scaled, cudaStream_t st) {
#define QKP(R, S) \
  if (pos64 && loc64) qk_prep_launch<R, S, long long, long long>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st); \
  else if (pos64) qk_prep_launch<R, S, long long, int>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st); \
  else if (loc64) qk_prep_launch<R, S, int, long long>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st); \
  else qk_prep_launch<R, S, int, int>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st);
  if (rope && !scaled) { QKP(true, false) }
  else if (rope) { QKP(true, true) }
  else if (!scaled) { QKP(false, false) }
  else { QKP(false, true) }
#undef QKP
}
void k_boundary(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                const void* gpre, float* sout, void* nout, int T, bool has_gpost, cudaStream_t st, int* zp, int zn) {
  if (has_gpost)
    okc_launch(boundary_kernel<BND_TPB, true, BND_HINT, 0>, dim3(T), dim3(BND_TPB), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn, (const __nv_bfloat16*)nullptr, (float*)nullptr, (const float*)nullptr, (const float*)nullptr);
  else
    okc_launch(boundary_kernel<BND_TPB, false, BND_HINT, 0>, dim3(T), dim3(BND_TPB), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn, (const __nv_bfloat16*)nullptr, (float*)nullptr, (const float*)nullptr, (const float*)nullptr);
}
void k_boundary_rc(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                   const void* gpre, float* sout, void* nout, int T, int mode, cudaStream_t st, int* zp, int zn,
                   const void* br1, float* rio, const float* alpha1, const float* beta1) {
  if (mode == 1)
    boundary_kernel<BND_TPB, false, BND_HINT, 1><<<T, BND_TPB, 0, st>>>((const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn,
                                                                        (const __nv_bfloat16*)br1, rio, alpha1, beta1);
  else
    boundary_kernel<BND_TPB, true, BND_HINT, 2><<<T, BND_TPB, 0, st>>>((const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn,
                                                                       (const __nv_bfloat16*)br1, rio, alpha1, beta1);
}
void k_boundary2(const void* a, const float* stream, const void* f, const float* a1, const float* b1, const void* gpost,
                 const float* a2, const float* b2, const void* gnext, float* sout, void* nout, int T, cudaStream_t st) {
  boundary2_kernel<<<T, 512, 0, st>>>((const __nv_bfloat16*)a, stream, (const __nv_bfloat16*)f, a1, b1, (const __nv_bfloat16*)gpost, a2, b2, (const __nv_bfloat16*)gnext, sout,
                                      (__nv_bfloat16*)nout);
}
void k_kv_gather(const void* kc, const void* vc, const int* page_table, const int* seq_lens, void* ko, void* vo, int B, int max_pages,
                 int page, int L, int window, int Smax, cudaStream_t st) {
  dim3 grid((unsigned)((Smax + 31) / 32), (unsigned)B);
  kv_gather_kernel<<<grid, 256, 0, st>>>((const uint8_t*)kc, (const uint8_t*)vc, page_table, seq_lens, (uint8_t*)ko, (uint8_t*)vo,
                                         max_pages, page, L, window, Smax);
}
void k_moe_fused(const void* w13, const void* w2, const void* xt, void* inter, void* y, const int* tiles, const int* ntiles,
                 const float* gs13, const float* gs2, int nctas, int smem, int* counter, int* done, void* dbg, int flags,
                 cudaStream_t st, int* cit, int* qctl, int* tok_left, int* cq, const int* row_tok, const int* rrow,
                 const int* lid, const float* topw, const void* pw, void* out, int T) {
  FusedArgs fa;
  fa.cit = cit; fa.qctl = qctl; fa.tok_left = tok_left; fa.cq = cq; fa.row_tok = row_tok; fa.rrow = rrow; fa.lid = lid;
  fa.topw = topw; fa.pw = (const __nv_bfloat16*)pw; fa.out = (__nv_bfloat16*)out; fa.T = T;
  fa.dbg = (unsigned long long*)dbg;
  fa.flags = flags;
  fa.g1 = make_args(w13, xt, tiles, ntiles, gs13, inter, 64, smem, counter);
  fa.g2 = make_args(w2, inter, tiles, ntiles, gs2, y, 32, smem, counter);
  fa.g1.flags = flags;
  fa.g2.flags = flags;
  fa.done = done;
  if ((fa.flags >> 8) & 1)  // bit 8: fine remainder tiles (the plan used the same flag)
    okc_launch(moe_fused_persistent<1>, dim3((unsigned)nctas), dim3(MOE_FUSED_THREADS), (size_t)smem, st, fa);
  else
    okc_launch(moe_fused_persistent<0>, dim3((unsigned)nctas), dim3(MOE_FUSED_THREADS), (size_t)smem, st, fa);
}
void k_moe_init(int smem) {
  cudaFuncSetAttribute(moe_fused_persistent<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_fused_persistent<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_persistent<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_persistent<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}
// ---- attention ----
#ifndef ATTN_HALF_MAX
#define ATTN_HALF_MAX 640  // per-head prep grids below this use half-tile blocks
#endif
void k_attn_prep(const void* kc, const void* vc, const int* page_table, int max_pages, int page, const int* seq_lens, int B, int L,
                 int window, int ntk, void* kt, void* vt, cudaStream_t st, int part) {
  if (ntk * B >= 264) {  // long contexts: both heads per block (full cache rows, one page-table read per key)
    dim3 grid((unsigned)ntk, 1, (unsigned)B);
    okc_launch(okc_attn::attn_prep2_kernel, grid, dim3(256), 0, st, (const uint8_t*)kc, (const uint8_t*)vc, page_table, max_pages, page, seq_lens, L,
               window, ntk, (uint8_t*)kt, (uint8_t*)vt, part);
  } else if (2 * ntk * B >= ATTN_HALF_MAX) {  // one head per block
    dim3 grid((unsigned)ntk, 2, (unsigned)B);
    okc_launch(okc_attn::attn_prep_kernel<1>, grid, dim3(256), 0, st, (const uint8_t*)kc, (const uint8_t*)vc, page_table, max_pages, page, seq_lens, L,
               window, ntk, (uint8_t*)kt, (uint8_t*)vt, part);
  } else {  // few tiles (short sliding windows): one head and half a tile per block
    dim3 grid((unsigned)(2 * ntk), 2, (unsigned)B);
    okc_launch(okc_attn::attn_prep_kernel<2>, grid, dim3(256), 0, st, (const uint8_t*)kc, (const uint8_t*)vc, page_table, max_pages, page, seq_lens, L,
               window, ntk, (uint8_t*)kt, (uint8_t*)vt, part);
  }
}
#ifndef ATTN_CR
#define ATTN_CR 2  // (token, head) rows per warp in the split merge (Oct 26 search: 2 +0.03..0.06% over 4)
#endif
void k_attn_fwd(const void* q, const void* kt, const void* vt, const int* seq_lens, const void* qkv, void* out, int B, int L,
                int window, int ntk, float c, float v_scale, int smem, int nsplit, float* ws, int variant, cudaStream_t st,
                const void* pf, unsigned long long pf_bytes) {
  okc_attn::AttnArgs a;
  a.q = (const uint8_t*)q; a.kt = (const uint8_t*)kt; a.vt = (const uint8_t*)vt; a.seq_lens = seq_lens;
  a.qkv = (const __nv_bfloat16*)qkv; a.out = (__nv_bfloat16*)out;
  const int nwg = variant == 99 ? 1 : (variant == 64 ? 4 : ((variant & 8) ? 3 : 2));
  a.L = L; a.window = window; a.ntk = ntk; a.nqb = (L + 8 * nwg - 1) / (8 * nwg); a.c = c; a.v_scale = v_scale;
  a.zero = 0;
  a.pf = (const uint8_t*)pf; a.pf_bytes = pf_bytes;
  a.sk = 0; a.nb = B; a.skbuf = nullptr;
  if (nsplit < 0) {  // stream-K: grid -nsplit; ws = [int header: W, pair prefix, first pairs | partials]
    const int G = -nsplit, npairs = B * a.nqb * 2;
    const int hdr = ((2 + npairs + 1 + G) + 63) & ~63;
    a.sk = G; a.skbuf = reinterpret_cast<int*>(ws); a.ws = ws + hdr; a.nsplit = 1;
    if (variant == 43) okc_launch(okc_attn::attn_fwd_sk_kernel<3, 0, true, true>, dim3((unsigned)G), dim3(512), (size_t)smem, st, a);
    else okc_launch(okc_attn::attn_fwd_sk_kernel<3, 0, false, false>, dim3((unsigned)G), dim3(512), (size_t)smem, st, a);
    const int T = B * L;
    okc_launch(okc_attn::attn_combine_sk_kernel<ATTN_CR, 24>, dim3((T * (16 / ATTN_CR) + 7) / 8), dim3(256), 0, st, (const float*)(ws + hdr),
               (const int*)ws, T, L, a.nqb, (const __nv_bfloat16*)qkv, (__nv_bfloat16*)out, c, v_scale);
    return;
  }
  a.nsplit = nsplit; a.ws = nsplit > 1 ? ws : nullptr;
  dim3 grid((unsigned)a.nqb, (unsigned)(2 * nsplit), (unsigned)B);
  switch (variant) {
    case 0: okc_launch(okc_attn::attn_fwd_kernel<2, 0, false>, grid, dim3(384), (size_t)smem, st, a); break;
    case 1: okc_launch(okc_attn::attn_fwd_kernel<2, 16, false>, grid, dim3(384), (size_t)smem, st, a); break;
    case 8: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false>, grid, dim3(512), (size_t)smem, st, a); break;
    case 9: okc_launch(okc_attn::attn_fwd_kernel<3, 16, false>, grid, dim3(512), (size_t)smem, st, a); break;
    case 16: okc_launch(okc_attn::attn_fwd_kernel<2, 0, true>, grid, dim3(384), (size_t)smem, st, a); break;
    case 64: okc_launch(okc_attn::attn_fwd_kernel<4, 0, false, false, false, false, true>, grid, dim3(640), (size_t)smem, st, a); break;
    case 43: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, grid, dim3(512), (size_t)smem, st, a); break;
#ifdef ATTN_NWG1
    // timing experiment: one consumer warpgroup per CTA and one CTA per SM (200 KB smem): a lone warpgroup's chain
    case 99: okc_launch(okc_attn::attn_fwd_kernel<1, 0, false>, grid, dim3(256), (size_t)(200 * 1024), st, a); break;
#endif
#ifdef ATTN_V52
    case 52: okc_launch(okc_attn::attn_fwd_kernel<2, 0, false, false, true>, grid, dim3(384), (size_t)smem, st, a); break;
#endif
#ifdef ATTN_V53
    // 2 consumer warpgroups at 240 registers: two S sets (QK(t+1) during softmax(t)), exp turns
    case 53: okc_launch(okc_attn::attn_fwd_kernel<2, 0, false, false, false, false, false, false, true>, grid, dim3(384), (size_t)smem, st, a); break;
#endif
#ifdef ATTN_PIPE_NORR
    case 40: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false, true, false, false>, grid, dim3(512), (size_t)smem, st, a); break;  // pipelined, no exp turns
#endif
#ifdef ATTN_RR_STEP_VARIANTS
    case 41: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false, false, false, true>, grid, dim3(512), (size_t)smem, st, a); break;  // RR turns, no pipeline
    case 42: okc_launch(okc_attn::attn_fwd_kernel<3, 0, true, false, false, true>, grid, dim3(512), (size_t)smem, st, a); break;   // RR turns + QK overlap
#endif
    default: okc_launch(okc_attn::attn_fwd_kernel<3, 0, true>, grid, dim3(512), (size_t)smem, st, a); break;
  }
  if (nsplit > 1 && !g_attn_merge_host) {  // (with the in-kernel merge the last split's CTA writes the output)
    const int T = B * L;
    okc_launch(okc_attn::attn_combine2_kernel<ATTN_CR>, dim3((T * (16 / ATTN_CR) + 7) / 8), dim3(256), 0, st, ws, nsplit, T, (const __nv_bfloat16*)qkv, (__nv_bfloat16*)out,
               c, v_scale);
  }
}
void k_attn_set_cnt(void* p) {
#if ATTN_MERGE
  cudaMemcpyToSymbol(okc_attn::g_attn_cnt, &p, sizeof(p));
  g_attn_merge_host = p != nullptr;
#endif
}
void k_attn_set_prof(void* p) {
#ifdef ATTN_PROF
  cudaMemcpyToSymbol(okc_attn::g_attn_prof, &p, sizeof(p));
#endif
}
void k_attn_init(int smem) {
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<4, 0, false, false, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_sk_kernel<3, 0, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_sk_kernel<3, 0, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
#ifdef ATTN_NWG1
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<1, 0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
#endif
#ifdef ATTN_V52
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 0, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
#endif
#ifdef ATTN_V53
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 0, false, false, false, false, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
#endif
#ifdef ATTN_PIPE_NORR
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false, true, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
#endif
#ifdef ATTN_RR_STEP_VARIANTS
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, true, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
#endif
}
'''



_CUDA_SRC = r'''#define OKC_PDL 1
// Grouped NVFP4 x BF16 expert GEMM for Muse Spark's MoE (one H200 = 32 local experts).
// out^T[w_rows, tokens] = W_e[w_rows, K] (NVFP4, dequantized in registers) x X^T[K, tokens] (BF16)
// sm_90: warp-specialized (1 producer warpgroup issuing cp.async.bulk, 2 consumer warpgroups
//        running wgmma m64nNk16 with the weights as the register A operand).
// sm_80: same data flow with cp.async + mma.sync (correctness testing only).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

// ---- programmatic dependent launch (OKC_PDL): every kernel lets its successor launch as soon as all of its blocks
// are running, and waits for its predecessor's results before touching global memory (a no-op when the kernel was not
// launched with the PDL attribute) ----
#ifndef OKC_PDL
#define OKC_PDL 0
#endif
__device__ __forceinline__ void pdl_enter() {
#if OKC_PDL && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
#ifndef MOE_PROXY_FENCE
#define MOE_PROXY_FENCE 0  // not needed for a WAR on generic reads (free either way)
#endif
// Before lane 0 hands a ring slot back, the whole warp's generic reads of it are ordered before the arrive (a warp
// barrier: the memory model orders the other lanes' LDS before lane 0's release only through it).
#ifndef MOE_SYNCWARP_RELEASE
#define MOE_SYNCWARP_RELEASE 0  // measured: a __syncwarp at the release point costs ~17% of the MoE span (wgmma pipeline)
#endif
// MOE_ALLLANE_RELEASE: every lane arrives on the slot's empty barrier (count 32 per warp), so each lane's own reads are
// ordered before its own arrive without a warp barrier.
#ifndef MOE_ALLLANE_RELEASE
#define MOE_ALLLANE_RELEASE 1  // v70: formally ordered slot release, measured free
#endif
#define REL_LANES (MOE_ALLLANE_RELEASE ? 32 : 1)

namespace okc_moe {

constexpr int KS = 64;        // k-stages of 64 elements (K = 4096)
constexpr int WBLK = 4608;    // bytes per (ntile, kstage) weight block: 4096 code bytes + 512 scale bytes
constexpr int MAX_STAGES = 24;

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEVI void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEVI void mbar_inval(uint32_t bar) { asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(bar) : "memory"); }
// Norm weights stay BF16 exactly as given; kernels form 1 + w in fp32 when they load them (the same value an fp32
// (1 + w) table would hold). g1p4: 4 consecutive weights (8-byte aligned) -> 1 + w.
DEVI float4 g1p4(const __nv_bfloat16* g) {
  const uint2 v = __ldg(reinterpret_cast<const uint2*>(g));
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
  return make_float4(1.f + a.x, 1.f + a.y, 1.f + b.x, 1.f + b.y);
}
DEVI void mbar_wait(uint32_t bar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n.reg .pred P1;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@!P1 bra WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
#else
  asm volatile(
      "{\n.reg .pred P1;\nWAIT_%=:\n"
      "mbarrier.test_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@!P1 bra WAIT_%=;\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
#endif
}
DEVI void mbar_arrive(uint32_t bar) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(bar) : "memory");
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
DEVI void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
// L2 prefetch of a weight block a few stages ahead of its TMA load (fire and forget: no smem, no barrier), so the
// weight stream keeps more bytes in flight than the 4-slot ring holds. MOE_L2PF = distance in stages (0: off).
#ifndef MOE_L2PF
#define MOE_L2PF 1  // Oct 26: 1 + ALL = +0.29% (search r21, x24)
#endif
#ifndef MOE_L2PF_ALL
#define MOE_L2PF_ALL 1  // 0: small token tiles only (2+ k-blocks per stage); 1: every item
#endif
DEVI void prefetch_l2(const void* src, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
#endif
}
// A ring slot whose weight codes and scales the consumers read with generic loads (LDS) is about to be overwritten by
// the async proxy (bulk copy): after the empty-barrier acquire, a proxy fence orders those generic reads before the
// async writes (PTX memory model; compute-sanitizer racecheck flags the slot reuse without it).
DEVI void rel_sync() {
#if MOE_SYNCWARP_RELEASE
  __syncwarp();
#endif
}
DEVI void slot_proxy_fence() {
#if MOE_PROXY_FENCE && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
// shared -> global bulk copy (async proxy), tracked per thread with bulk groups
DEVI void bulk_s2g(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes) : "memory");
}
DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
DEVI void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
#endif

DEVI void keep_alive(const uint32_t (&a)[4]) {
  asm volatile("" ::"r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]) : "memory");
}

DEVI void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// SwiGLU's x * sigmoid(x) as 0.5 x (1 + tanh(x / 2)): one MUFU op instead of exp + reciprocal (the epilogue is
// MUFU-bound and the tensor cores idle during it)
DEVI float silu_fast(float x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return 0.5f * x * (1.f + t);
#else
  return x / (1.f + __expf(-x));
#endif
}

// wgmma descriptor for a K-major, 128B-swizzled B tile (rows of 128 B, 8-row atoms of 1024 B).
DEVI uint64_t desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;           // LBO (unused for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32; // SBO: 8 rows x 128 B
  d |= (uint64_t)1 << 62;           // SWIZZLE_128B
  return d;
}

template <int N>
DEVI void wgmma_rs(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d);
template <> DEVI void wgmma_rs<32>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#endif
}

template <> DEVI void wgmma_rs<64>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#endif
}

template <> DEVI void wgmma_rs<128>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#endif
}

template <> DEVI void wgmma_rs<256>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#endif
}

template <> DEVI void wgmma_rs<96>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%48, %49, %50, %51}, %52, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#endif
}

template <> DEVI void wgmma_rs<160>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %85, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n160k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79}, {%80, %81, %82, %83}, %84, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#endif
}

template <> DEVI void wgmma_rs<192>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %101, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, {%96, %97, %98, %99}, %100, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#endif
}

template <> DEVI void wgmma_rs<224>(float (&d)[128], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %117, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n224k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111}, {%112, %113, %114, %115}, %116, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
#endif
}

#ifndef MOE_XPF256
#define MOE_XPF256 0
#endif
#ifndef MOE_XPFSMALL
#define MOE_XPFSMALL 0
#endif
#ifndef MOE_FINE_TILES
#define MOE_FINE_TILES 1
#endif
#ifndef MOE_FINE_MASK
#define MOE_FINE_MASK 6  // extra widths: 1 = 96, 2 = 160, 4 = 192, 8 = 224 (3+ of them: ptxas C7512, wgmma serialized)
#endif

DEVI uint32_t bf162_u32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }
// mma.sync m16n8k16 (bf16 x bf16 -> fp32): the A fragment layout equals one warp's share of the wgmma RS m64 A operand
DEVI void mma16816(float* c, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEVI __nv_bfloat162 u32_bf162(uint32_t v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }

// Two E4M3 block scales (low byte: row g, high byte: row g+8) -> bf16x2 of scale * 2^119.
// MOE_FAST_SCALE (prepare checks that no block scale is zero or subnormal): for a normal E4M3 byte b the bf16 bits of
// value * 2^119 are exactly (b << 4) + 0x7780 (exponent rebias 120 + 119, mantissa bits moved up 4): one byte permute
// and one IMAD for the pair.
#ifndef MOE_FAST_SCALE
#define MOE_FAST_SCALE 1
#endif
DEVI uint32_t scale_pair(uint32_t s2) {
#if MOE_FAST_SCALE
  return __byte_perm(s2, 0u, 0x4140) * 16u + 0x77807780u;  // [b0, 0, b1, 0] -> (b << 4) + 0x7780 per half
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t h2;
  asm("{\n.reg .b16 t;\ncvt.u16.u32 t, %1;\ncvt.rn.f16x2.e4m3x2 %0, t;\n}\n" : "=r"(h2) : "r"(s2));
  // f16 (exact for every e4m3 value) -> bf16 with the exponent rebiased by 112 + 119
  return ((h2 >> 3) & 0x0FFF0FFFu) + 0x73807380u;
#else
  uint32_t r = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint32_t b = (s2 >> (8 * i)) & 0xFF;
    uint32_t e = (b >> 3) & 15, m = b & 7;
    float v = e ? __uint_as_float(((b & 0x7F) << 20) + (120u << 23)) : (float)m * 0.001953125f;
    v *= 6.646139978924579e+35f;  // 2^119
    __nv_bfloat16 h = __float2bfloat16_rn(v);
    r |= (uint32_t)(*reinterpret_cast<uint16_t*>(&h)) << (16 * i);
  }
  return r;
#endif
}

// The scale pair of one k16 step straight from the 32-bit word holding two steps' pairs: the byte permute selects
// bytes 0,1 (hi = 0) or 2,3 (hi = 1) itself, so no shift or mask (hi is a compile-time constant in the unrolled loops).
DEVI uint32_t scale_pair_sel(uint32_t w, int hi, uint32_t m16 = 16u) {
#if MOE_FAST_SCALE
  return __byte_perm(w, 0u, hi ? 0x4342 : 0x4140) * m16 + 0x77807780u;  // m16 opaque: IMAD on the FMA pipe
#else
  return scale_pair((w >> (16 * hi)) & 0xFFFF);
#endif
}

// 8 E2M1 codes -> 4 bf16x2 A-fragment registers (scaled). The repack places each code's bits so that one rotate and
// one mask put it in place: a register half holds (sign << 15 | magnitude << 6), which as bf16 bits is exactly
// e2m1 * 2^-126 (subnormal for magnitude 1), times scale * 2^119. Bit positions in q (sign, m2, m1, m0):
//   a0 (row g):    code A (15, 8, 7, 6),   code B (31, 24, 23, 22)    q & M
//   a1 (row g+8):  code A (21, 14, 13, 12), code B (5, 30, 29, 28)    rotr(q, 6) & M
//   a2 (row g):    code A (25, 18, 17, 16), code B (9, 2, 1, 0)       rotr(q, 10) & M
//   a3 (row g+8):  code A (11, 4, 3, 10),  code B (27, 20, 19, 26)    rotl(q, 4) & 0x81808180 | rotr(q, 4) & 0x00400040
// (9 ALU ops for 8 weights, no IMAD shifts; the repack in the template writes this layout.)
// MOE_DQ7 (moe agent's layout, via the scout): 7 integer-pipe ops + 2 IMAD shifts per 8 weights. Two of the shifts
// become multiplies by 16 / 64 held in registers (m16 = 16 + opaque 0, m64 = 64 + opaque 0), which run on the FMA pipe
// as IMAD instead of on the half-rate integer pipe. Bit positions in q (m0, m1, m2, sign):
//   a0: code A (6, 7, 8, 15),   code B (22, 23, 24, 31)   q & M
//   a1: code A (2, 3, 4, 11),   code B (18, 19, 20, 27)   (q * 16) & M
//   a2: code A (28, 29, 30, 5), code B (12, 13, 14, 21)   rotl(q, 10) & M
//   a3: code A (0, 1, 26, 9),   code B (16, 17, 10, 25)   (q * 64) & 0x80C080C0 | rotl(q, 14) & 0x01000100
#ifndef MOE_DQ7
#define MOE_DQ7 1
#endif
DEVI uint32_t opaque_zero() {
  uint32_t z;
  asm volatile("mov.u32 %0, 0;" : "=r"(z));
  return z;
}
DEVI void convert8(uint32_t q, uint32_t sb, uint32_t (&a)[4], uint32_t m16 = 16u, uint32_t m64 = 64u) {
  const __nv_bfloat162 s = u32_bf162(sb);
  const __nv_bfloat162 slo = __low2bfloat162(s), shi = __high2bfloat162(s);
  constexpr uint32_t M = 0x81C081C0u;
#if MOE_DQ7
  const uint32_t r0 = q & M;
  const uint32_t r1 = (q * m16) & M;
  const uint32_t r2 = __funnelshift_l(q, q, 10) & M;
  const uint32_t r3 = ((q * m64) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
#else
  const uint32_t r0 = q & M;
  const uint32_t r1 = __funnelshift_r(q, q, 6) & M;
  const uint32_t r2 = __funnelshift_r(q, q, 10) & M;
  const uint32_t r3 = (__funnelshift_l(q, q, 4) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
#endif
  a[0] = bf162_u32(__hmul2(u32_bf162(r0), slo));
  a[1] = bf162_u32(__hmul2(u32_bf162(r1), shi));
  a[2] = bf162_u32(__hmul2(u32_bf162(r2), slo));
  a[3] = bf162_u32(__hmul2(u32_bf162(r3), shi));
}

#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
// sm_80 reference MMA: one warp's 16 rows x N tokens, B fragments read straight from the swizzled tile.
template <int N>
DEVI void mma_sm80(float (&d)[128], const uint32_t (&a)[4], const uint8_t* xs, int j, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int jj = 0; jj < N / 8; ++jj) {
    const int tok = jj * 8 + g;
    const uint8_t* row = xs + tok * 128;
    uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t);
    uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[4 * jj]), "+f"(d[4 * jj + 1]), "+f"(d[4 * jj + 2]), "+f"(d[4 * jj + 3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
}
#endif

struct Args {
  const uint8_t* wrep;   // [E][NT][KS][WBLK]
  const uint8_t* xt;     // activations in tile layout (byte base; a tile starting at row r0 is at r0 * 8192)
  const int4* tiles;     // (expert, row0, n, 0)
  const int* ntiles;     // device scalar
  const float* gscale;   // [E]
  uint8_t* out;          // GATED: tile layout (bytes); else row-major bf16 [rows, 4096]
  int NT;
  int smem_bytes;
  int* counter;          // persistent kernel work queue (zeroed before launch)
  int flags;             // experiments only (0 in production): bit 2 = skip the NVFP4 conversion
  uint32_t mul16, mul64;  // 16 and 64, from the host: multiplies by them stay IMAD (FMA-heavy pipe), since ptxas sees
                          // through an inline-asm zero and turns `x * (16 + 0)` back into an integer-pipe shift
};
#ifndef MOE_OPAQUE_MUL
#define MOE_OPAQUE_MUL 1
#endif

template <int N, bool GATED>
DEVI void consumer_tile(const Args& p, uint8_t* ring, uint32_t bars, int stride, int nst, int e, int row0,
                        int ntile, int* done = nullptr) {
  const uint32_t zz_ = opaque_zero(), m16 = 16u + zz_, m64 = 64u + zz_;  // opaque multipliers (IMAD shifts)
  const int ctid = threadIdx.x - 128;  // 0..255
  // warp-uniform by construction (shuffled from lane 0), so ptxas keeps the smem descriptor math on the uniform datapath
#ifndef MOE_UNIFORM_WG
#define MOE_UNIFORM_WG 0  // measured: no change in the MoE spans
#endif
  const int wg = MOE_UNIFORM_WG ? __shfl_sync(0xffffffffu, ctid >> 7, 0) : ctid >> 7;
  const int w = MOE_UNIFORM_WG ? __shfl_sync(0xffffffffu, (ctid >> 5) & 3, 0) : (ctid >> 5) & 3;
  const int lane = ctid & 31;
  const int g = lane >> 2, t = lane & 3;
  float acc[128];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  uint32_t afrag[2][4];
  int s = 0, prev_s = 0;
  uint32_t ph = 0;
  for (int ks = 0; ks < KS; ++ks) {
    mbar_wait(bars + 8 * s, ph);
    uint8_t* st = ring + s * stride;
    const uint4 wq = *reinterpret_cast<const uint4*>(st + N * 128 + ((wg * 4 + w) * 32 + lane) * 16);
    const uint2 sc = *reinterpret_cast<const uint2*>(st + N * 128 + 4096 + ((wg * 4 + w) * 8 + g) * 8);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint64_t desc = desc_sw128(smem_u32(st));
#endif
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
      const uint32_t sw = j < 2 ? sc.x : sc.y;  // scales of k16 steps (j & ~1, j | 1): bytes 0,1 and 2,3
      convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j & 1], m16, m64);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      wg_fence();
      wgmma_rs<N>(acc, afrag[j & 1], desc + 2 * j, 1);
      wg_commit();
      wg_wait<1>();
#else
      mma_sm80<N>(acc, afrag[j & 1], st, j, lane);
#endif
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // steps 0-2 of this stage are complete: every wgmma reading the previous stage is done
    if (ks > 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * MAX_STAGES + 8 * prev_s); }
#else
    __syncwarp();
    if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * MAX_STAGES + 8 * s);
#endif
    prev_s = s;
    if (++s == nst) { s = 0; ph ^= 1; }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  wg_wait<0>();
#endif
  // ---- epilogue ----
  const float cs = p.gscale[e] * 128.f;  // weights were scaled by 2^-7
  named_bar_sync(1, 256);               // every consumer is done reading the ring
  uint8_t* stg = ring;
  if constexpr (GATED) {
    // rows g / g+8 of each 16-row slice are gate / up of inter column c = 8*slice + g
    const int q = wg * 4 + w;
#pragma unroll
    for (int jj = 0; jj < N / 8; ++jj) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const float gt = acc[4 * jj + b] * cs, up = acc[4 * jj + 2 + b] * cs;
        const float h = silu_fast(gt) * up;
        const int tok = 8 * jj + 2 * t + b;
        *reinterpret_cast<__nv_bfloat16*>(stg + tok * 128 + ((q ^ (tok & 7)) << 4) + g * 2) = __float2bfloat16_rn(h);
      }
    }
    named_bar_sync(1, 256);
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)row0 * 8192 + (size_t)ntile * N * 128);
    const uint4* src = reinterpret_cast<const uint4*>(stg);
    for (int i = ctid; i < N * 8; i += 256) dst[i] = src[i];
    if (done) {  // fused kernel: publish this n-tile of the tile's intermediate rows
      __threadfence();
      named_bar_sync(1, 256);
      if (ctid == 0) atomicAdd(done, 1);
    }
  } else {
    // output columns (within the 128-wide ntile) c0 = 16*slice + g, c1 = c0 + 8; staging rows of 272 B
    const int c0 = (wg * 4 + w) * 16 + g;
#pragma unroll
    for (int jj = 0; jj < N / 8; ++jj) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const int tok = 8 * jj + 2 * t + b;
        __nv_bfloat16* r = reinterpret_cast<__nv_bfloat16*>(stg + tok * 272);
        r[c0] = __float2bfloat16_rn(acc[4 * jj + b] * cs);
        r[c0 + 8] = __float2bfloat16_rn(acc[4 * jj + 2 + b] * cs);
      }
    }
    named_bar_sync(1, 256);
    for (int i = ctid; i < N * 16; i += 256) {
      const int tok = i >> 4, c = i & 15;
      *reinterpret_cast<uint4*>(p.out + ((size_t)(row0 + tok) * 4096 + ntile * 128) * 2 + c * 16) =
          *reinterpret_cast<const uint4*>(stg + tok * 272 + c * 16);
    }
  }
}

template <bool GATED>
__global__ void __launch_bounds__(384, 1) moe_gemm_kernel(Args p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);  // full[MAX_STAGES], empty[MAX_STAGES]
  uint8_t* ring = base + 1024;
  const int ring_bytes = p.smem_bytes - 1024 - (int)pad;
  const int ntile = blockIdx.x;
  const int ntiles = *p.ntiles;
  if (threadIdx.x < 128) {
    // ---------------- producer warpgroup ----------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
#endif
    for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      const int stride = (n * 128 + WBLK + 1023) & ~1023;
      int nst = ring_bytes / stride;
      nst = nst > MAX_STAGES ? MAX_STAGES : nst;
      if (threadIdx.x == 0) {
        for (int s = 0; s < nst; ++s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
          mbar_init(bars + 8 * s, 1);
#else
          mbar_init(bars + 8 * s, 128);
#endif
          mbar_init(bars + 8 * MAX_STAGES + 8 * s, 8 * REL_LANES);
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
      }
      __syncthreads();
      const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
      const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
      const uint32_t xbytes = n * 128;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if (threadIdx.x == 0) {
        for (int ks = 0; ks < KS; ++ks) {
          const int s = ks % nst;
          const uint32_t ph = (ks / nst) & 1;
          mbar_wait(bars + 8 * MAX_STAGES + 8 * s, ph ^ 1);
          slot_proxy_fence();
          const uint32_t full = bars + 8 * s;
          mbar_expect_tx(full, xbytes + WBLK);
          const uint32_t dst = smem_u32(ring + s * stride);
          bulk_g2s(dst, xsrc + (size_t)ks * xbytes, xbytes, full);
          bulk_g2s(dst + xbytes, wsrc + (size_t)ks * WBLK, WBLK, full);
        }
      }
#else
      const int nchunk = (xbytes + WBLK) / 16;
      for (int ks = 0; ks < KS; ++ks) {
        const int s = ks % nst;
        const uint32_t ph = (ks / nst) & 1;
        mbar_wait(bars + 8 * MAX_STAGES + 8 * s, ph ^ 1);
        const uint32_t dst = smem_u32(ring + s * stride);
        for (int i = threadIdx.x; i < nchunk; i += 128) {
          const uint8_t* src = i * 16 < (int)xbytes ? xsrc + (size_t)ks * xbytes + i * 16
                                                    : wsrc + (size_t)ks * WBLK + (i * 16 - xbytes);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + i * 16), "l"(src) : "memory");
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(bars + 8 * s) : "memory");
      }
      asm volatile("cp.async.wait_all;" ::: "memory");
#endif
      __syncthreads();
      if (threadIdx.x == 0) {
        for (int s = 0; s < nst; ++s) {
          mbar_inval(bars + 8 * s);
          mbar_inval(bars + 8 * MAX_STAGES + 8 * s);
        }
      }
      __syncthreads();
    }
  } else {
    // ---------------- consumer warpgroups ----------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
#endif
    for (int tile = blockIdx.y; tile < ntiles; tile += gridDim.y) {
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      const int stride = (n * 128 + WBLK + 1023) & ~1023;
      int nst = ring_bytes / stride;
      nst = nst > MAX_STAGES ? MAX_STAGES : nst;
      __syncthreads();
      switch (n) {
        case 32: consumer_tile<32, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 64: consumer_tile<64, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
        case 128: consumer_tile<128, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
#if MOE_FINE_TILES
#if MOE_FINE_MASK & 1
        case 96: consumer_tile<96, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
#endif
#if MOE_FINE_MASK & 2
        case 160: consumer_tile<160, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
#endif
#if MOE_FINE_MASK & 4
        case 192: consumer_tile<192, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
#endif
#if MOE_FINE_MASK & 8
        case 224: consumer_tile<224, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
#endif
#endif
        default: consumer_tile<256, GATED>(p, ring, bars, stride, nst, e, row0, ntile); break;
      }
      __syncthreads();
      __syncthreads();
    }
  }
}

// ---- persistent variant: grid = #SMs, items (tile, ntile) round-robin, fixed slot ring + separate staging ----
constexpr int SLOT = 43008;     // bytes per ring slot (1024-aligned): fits kps k-stages of any tile size
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
constexpr int NSLOT = 2;        // sm_80 testing: 163 KB of shared memory
#else
constexpr int NSLOT = 4;
#endif
constexpr int STG_BYTES = 45056; // epilogue staging

// k-blocks per ring stage by token-tile size (a stage must fit SLOT: N=32 allows 1/2/4, N=64 and 128 1/2, N=256 1)
#ifndef MOE_KPS32
#define MOE_KPS32 4
#endif
#ifndef MOE_KPS64
#define MOE_KPS64 2
#endif
#ifndef MOE_KPS128
#define MOE_KPS128 2
#endif
#ifndef MOE_GROUP4_MAX
#define MOE_GROUP4_MAX 128  // token tiles up to this size commit one wgmma group per stage (see GROUP4)
#endif
// tile widths: 32, 64 (small: weight-conversion-bound), 96, 128 (two k-blocks a stage), 160..256 in steps of 32 (one
// k-block a stage; MOE_FINE_TILES: an expert's last tile is its remaining rows rounded up to 32, not to 128/256)
DEVI int kps_for(int n) { return n >= 160 ? 1 : (n <= 32 ? MOE_KPS32 : (n <= 64 ? MOE_KPS64 : MOE_KPS128)); }
// a stage (kps k-blocks of X and W) must fit its ring slot and kps must divide the 64 k-blocks (Oct 26: the search's
// "kps64_4 +0.16%" was a kernel writing 8 KB past each slot: 4 x (64 x 128 + 4608) = 51200 > SLOT)
static_assert(MOE_KPS32 * (32 * 128 + WBLK) <= SLOT && 64 % MOE_KPS32 == 0, "MOE_KPS32: stage does not fit a ring slot");
static_assert(MOE_KPS64 * (64 * 128 + WBLK) <= SLOT && 64 % MOE_KPS64 == 0, "MOE_KPS64: stage does not fit a ring slot");
static_assert(MOE_KPS128 * (128 * 128 + WBLK) <= SLOT && 64 % MOE_KPS128 == 0, "MOE_KPS128: stage does not fit a ring slot");
static_assert(256 * 128 + WBLK <= SLOT, "N=256 stage does not fit a ring slot");

// Copy-out queue in shared memory (fused sm_90 kernel): pubq = base + 256, 8 entries of 32 bytes {dst, done, kind,
// bytes or tokens}; base + 240 = jobs published (st.release by consumer thread 0; negative = -(count + 1) at exit),
// base + 244 = consumer thread 0's own count, base + 248 = jobs whose staging reads the publisher has finished.
DEVI void enqueue_out(volatile long long* pubq, void* dst, int* done, int kind, int amount, int row0 = 0, int nreal = 0) {
  volatile int* cnt = reinterpret_cast<volatile int*>(pubq) - 3;  // base + 244
  const int n = *cnt;
  volatile long long* e = pubq + 4 * (n & 7);
  e[0] = (long long)dst;
  e[1] = (long long)done;
  reinterpret_cast<volatile int*>(e)[4] = kind;
  reinterpret_cast<volatile int*>(e)[5] = amount;
  reinterpret_cast<volatile int*>(e)[6] = row0;   // CIT (GEMM2): the tile's first row and real row count
  reinterpret_cast<volatile int*>(e)[7] = nreal;
  *cnt = n + 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32((const void*)(pubq - 2))), "r"(n + 1) : "memory");
#endif
}
// consumer thread 0: wait until the publisher has read every queued staging buffer
DEVI void wait_staging_free(volatile long long* pubq) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const int n = *(reinterpret_cast<volatile int*>(pubq) - 3);
  const uint32_t a = smem_u32((const void*)(reinterpret_cast<volatile int*>(pubq) - 2));  // base + 248
  while (true) {
    int f;
    asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(f) : "r"(a) : "memory");
    if (f >= n) break;
  }
#endif
}

// pubq (fused sm_90 kernel): instead of fencing and bumping `done` itself, the consumers hand the GEMM1 n-tile's
// completion to the producer warpgroup's publisher warp (smem queue at pubq = base + 256, published count at base + 240,
// consumer thread 0's running count at base + 244), so the next item's main loop does not wait for the store fence.
#ifndef MOE_CONS_FIRST
#define MOE_CONS_FIRST 0  // fused kernel: consumers at threads 0..255, producer + publisher warps at 256..319 (320
                          // threads: 200 registers for every thread at launch, no setmaxnreg)
#endif
#define MOE_CB (MOE_CONS_FIRST ? 0 : 128)  // first consumer thread
#define MOE_PB (MOE_CONS_FIRST ? 256 : 0)  // first producer thread
#define MOE_FUSED_THREADS (MOE_CONS_FIRST ? 320 : 384)
template <int N, int CB = 128>
DEVI void consumer_item(const Args& p, uint8_t* ring, uint8_t* stg, uint32_t bars, int& slot, uint32_t& ph,
                        int e, int row0, int ntile, bool GATED, int* done = nullptr, unsigned long long* t_mid = nullptr,
                        volatile long long* pubq = nullptr, int nreal = 256) {
#ifndef MOE_IMAD_MAXN
#define MOE_IMAD_MAXN 64  // IMAD shifts only for small token tiles (N=128 measured 2-3% slower with them)
#endif
  const uint32_t zz_ = N <= MOE_IMAD_MAXN ? opaque_zero() : 0u;
  // small tiles: multipliers ptxas cannot see (host values), so the shifts run as IMAD on the FMA-heavy pipe
  const uint32_t m16 = (N <= MOE_IMAD_MAXN && MOE_OPAQUE_MUL) ? p.mul16 : 16u + zz_;
  const uint32_t m64 = (N <= MOE_IMAD_MAXN && MOE_OPAQUE_MUL) ? p.mul64 : 64u + zz_;
  constexpr int KPS = N >= 160 ? 1 : (N <= 32 ? MOE_KPS32 : (N <= 64 ? MOE_KPS64 : MOE_KPS128));
  constexpr int XB = N * 128;  // X bytes per k-stage
  const int ctid = threadIdx.x - CB;
  // warp-uniform by construction (shuffled from lane 0), so ptxas keeps the smem descriptor math on the uniform datapath
#ifndef MOE_UNIFORM_WG
#define MOE_UNIFORM_WG 0  // measured: no change in the MoE spans
#endif
  const int wg = MOE_UNIFORM_WG ? __shfl_sync(0xffffffffu, ctid >> 7, 0) : ctid >> 7;
  const int w = MOE_UNIFORM_WG ? __shfl_sync(0xffffffffu, (ctid >> 5) & 3, 0) : (ctid >> 5) & 3;
  const int lane = ctid & 31;
  const int g = lane >> 2, t = lane & 3;
  // small token tiles: the wgmma is short, so one commit group carries a whole k-stage (4 x k16) and the next
  // stage's conversions overlap it (one group in flight, double-buffered A fragments)
#ifndef MOE_G4_96
#define MOE_G4_96 0  // 96-row tiles in the grouped path compile to wgmma serialization for the whole kernel (C7512)
#endif
  constexpr bool GROUP4 = N <= MOE_GROUP4_MAX && (N != 96 || MOE_G4_96);
  // small token tiles: a commit group's MMAs are shorter than the wgmma latency, so consecutive groups accumulate into
  // two independent register sets (summed before the epilogue) instead of one dependency chain
#ifndef MOE_NACC2
#define MOE_NACC2 1
#endif
  // MOE_NACC2 = 0: one accumulator everywhere, so a row's sums do not depend on its tile size (bitwise-deterministic
  // outputs even though the router places rows with atomics)
  // MOE_NACC4 (tiles up to MOE_NACC4_MAXN tokens): the k16 steps of a group also alternate between two sets, so each
  // group is two short dependency chains instead of one of four (four accumulator sets in all)
#ifndef MOE_NACC4
#define MOE_NACC4 1
#endif
#ifndef MOE_NACC4_MAXN
#define MOE_NACC4_MAXN 32
#endif
  // MOE_TRI (N=32 tiles): three A-fragment buffers and three accumulator sets rotate with the k-block index, so a
  // k-block's conversion runs while the two previous k-blocks' MMA groups are in flight (wait<2> instead of wait<1>)
#ifndef MOE_TRI
#define MOE_TRI 0  // 13:50: C7511 (wgmma serialized, register pressure in the fused kernel) -> 15.5 vs 12.2 us per N=32 item
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MOE_TRI_ARCH 1
#else
#define MOE_TRI_ARCH 0
#endif
  constexpr bool TRI = MOE_TRI && MOE_TRI_ARCH && MOE_NACC2 && GROUP4 && N == 32 && KPS == 4;
  constexpr int NACC = TRI ? 3 : ((GROUP4 && MOE_NACC2) ? ((MOE_NACC4 && N <= MOE_NACC4_MAXN) ? 4 : 2) : 1);
  float acc[128];
  float acc2[128];  // only the first N/2 are live (and only when NACC > 1)
  float acc3[128], acc4[128];  // NACC == 4 only
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (NACC > 1 ? N / 2 : 0); ++i) acc2[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (NACC > 2 ? N / 2 : 0); ++i) acc3[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (NACC > 3 ? N / 2 : 0); ++i) acc4[i] = 0.f;
  // N=32 tiles may keep more MMA groups in flight (MOE_SMALL_WAIT groups behind; 4 A-fragment buffers indexed by the
  // k-block within the 4-k-block stage); otherwise two buffers (k-block parity) and one group behind
#ifndef MOE_SMALL_WAIT
#define MOE_SMALL_WAIT 1
#endif
  constexpr bool DEEP = GROUP4 && N == 32 && KPS == 4 && MOE_SMALL_WAIT > 1;
#ifndef MOE_MMA32
#define MOE_MMA32 0  // N=32 tiles: mma.sync m16n8k16 per warp instead of wgmma, skipping the 8-token groups past the
                     // tile's real rows (tiles[].w): the tensor work follows the tokens, not the 32-row tile
#endif
  constexpr bool MMA32 = MOE_MMA32 && GROUP4 && !DEEP && N == 32;
  const int nblk = MMA32 ? min(4, max(1, (nreal + 7) >> 3)) : 4;
  constexpr int AFB = DEEP ? 4 : 2;
  uint32_t afrag[GROUP4 ? 4 * AFB : 2][4];
  int prev = -1;
  const int woff = ((wg * 4 + w) * 32 + lane) * 16, soff = 4096 + ((wg * 4 + w) * 8 + g) * 8;
  // packed weights/scales are loaded one k-block ahead, across stage boundaries too
  mbar_wait(bars + 8 * slot, ph);
  uint4 nwq = *reinterpret_cast<const uint4*>(ring + slot * SLOT + KPS * XB + woff);
  uint2 nsc = *reinterpret_cast<const uint2*>(ring + slot * SLOT + KPS * XB + soff);
  constexpr bool XPF = N >= 160 ? MOE_XPF256 : MOE_XPFSMALL;
  if constexpr (TRI) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t af3[3][4][4];
    // one 4-k-block stage; P = (global k-block index of kk = 0) mod 3, so buffer / accumulator b = (P + kk) % 3 is a
    // compile-time index once the kk loop is unrolled
    auto stage = [&]<int P>(int it) {
      uint8_t* st = ring + slot * SLOT;
      if (it > 0) {
        mbar_wait(bars + 8 * slot, ph);
        nwq = *reinterpret_cast<const uint4*>(st + KPS * XB + woff);
        nsc = *reinterpret_cast<const uint2*>(st + KPS * XB + soff);
      }
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int b = (P + kk) % 3;
        const uint4 wq = nwq;
        const uint2 sc = nsc;
        if (kk + 1 < 4) {
          nwq = *reinterpret_cast<const uint4*>(st + KPS * XB + (kk + 1) * WBLK + woff);
          nsc = *reinterpret_cast<const uint2*>(st + KPS * XB + (kk + 1) * WBLK + soff);
        }
        const uint64_t desc = desc_sw128(smem_u32(st + kk * XB));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;
          convert8(q, scale_pair_sel(sw, j & 1, m16), af3[b][j], m16, m64);  // buffer b's last group (3 k-blocks ago) is complete
        }
        wg_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j) wgmma_rs<N>(b == 0 ? acc : (b == 1 ? acc2 : acc3), af3[b][j], desc + 2 * j, 1);
        wg_commit();
        wg_wait<2>();
        // after the wait at kk = 1 only this stage's k-blocks 0 and 1 may still run: the previous slot is free
        if (kk == 1 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
      }
      prev = slot;
      if (++slot == NSLOT) { slot = 0; ph ^= 1; }
    };
    for (int it = 0; it < KS / KPS; ++it) {
      const int ph3 = (it * 4) % 3;
      if (ph3 == 0) stage.template operator()<0>(it);
      else if (ph3 == 1) stage.template operator()<1>(it);
      else stage.template operator()<2>(it);
    }
    wg_wait<0>();
    rel_sync();
    if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev);
#endif
  } else {
#ifndef MOE_HOIST
#define MOE_HOIST 1  // per-stage descriptor base and weight/scale base pointers (constant per-k-block offsets)
#endif
  for (int it = 0; it < KS / KPS; ++it) {
    uint8_t* st = ring + slot * SLOT;
    const uint8_t* wbase = st + KPS * XB + woff;
    const uint8_t* sbase = st + KPS * XB + soff;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint64_t dbase = desc_sw128(smem_u32(st));
#endif
    if (!XPF && it > 0) {
      mbar_wait(bars + 8 * slot, ph);
      nwq = *reinterpret_cast<const uint4*>(wbase);
      nsc = *reinterpret_cast<const uint2*>(sbase);
    }
#pragma unroll
    for (int kk = 0; kk < KPS; ++kk) {
      const uint4 wq = nwq;
      const uint2 sc = nsc;
      if (kk + 1 < KPS) {
#ifdef MOE_DBG_NOLDS  // timing experiment only (wrong results): no per-k-block weight/scale loads inside a stage
        if (false) {
#else
        if (MOE_HOIST) {
#endif
          nwq = *reinterpret_cast<const uint4*>(wbase + (kk + 1) * WBLK);
          nsc = *reinterpret_cast<const uint2*>(sbase + (kk + 1) * WBLK);
        } else {
          nwq = *reinterpret_cast<const uint4*>(st + KPS * XB + (kk + 1) * WBLK + woff);
          nsc = *reinterpret_cast<const uint2*>(st + KPS * XB + (kk + 1) * WBLK + soff);
        }
      } else if (XPF && it + 1 < KS / KPS) {
        const int ns = slot + 1 == NSLOT ? 0 : slot + 1;
        mbar_wait(bars + 8 * ns, slot + 1 == NSLOT ? ph ^ 1 : ph);
        nwq = *reinterpret_cast<const uint4*>(ring + ns * SLOT + KPS * XB + woff);
        nsc = *reinterpret_cast<const uint2*>(ring + ns * SLOT + KPS * XB + soff);
      }

      uint8_t* xs = st + kk * XB;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      // the k-block's X tile sits kk * XB bytes into the slot: its descriptor is the slot's plus kk * XB / 16
      const uint64_t desc = MOE_HOIST ? dbase + (uint64_t)((kk * XB) >> 4) : desc_sw128(smem_u32(xs));
#endif
      if constexpr (GROUP4) {
#ifdef MOE_DBG_NOCONV  // timing experiment only (wrong results): A fragments from the codes without conversion
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          uint32_t* af = afrag[(DEEP ? kk : (kk & 1)) * 4 + j];
          af[0] = q; af[1] = q ^ sc.x; af[2] = q ^ sc.y; af[3] = q + 1;
        }
#else
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#ifdef MOE_DBG_NOLDS
          const uint32_t q = (j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w) ^ (uint32_t)(kk * 0x01010101);
#else
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
#endif
          const uint32_t sw = j < 2 ? sc.x : sc.y;  // scales of k16 steps (j & ~1, j | 1): bytes 0,1 and 2,3
#ifdef MOE_DBG_CONSTSCALE  // timing experiment only (wrong results): no scale decode
          (void)sw;
          convert8(q, 0x77807780u + (uint32_t)(kk * 4 + j), afrag[(DEEP ? kk : (kk & 1)) * 4 + j], m16, m64);
#else
          convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[(DEEP ? kk : (kk & 1)) * 4 + j], m16, m64);
#endif
        }
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if constexpr (MMA32) {
          // B fragments from the 128B-swizzled token rows: ldmatrix.x4 = (group 2q, k lo), (2q, k hi), (2q+1, lo),
          // (2q+1, hi); lane l addresses row 8*(2q + (l>>4)) + (l&7), 16-byte chunk (2j + ((l>>3)&1)) ^ (l&7)
          const uint32_t xbase = smem_u32(st + kk * XB);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float* ca = (NACC > 1 && (j & 1)) ? acc2 : acc;
#pragma unroll
            for (int q2 = 0; q2 < 2; ++q2) {
              if (2 * q2 < nblk) {
                const int mrow = 8 * (2 * q2 + (lane >> 4)) + (lane & 7);
                const int chunk = (2 * j + ((lane >> 3) & 1)) ^ (lane & 7);
                uint32_t bfr[4];
                ldsm_x4(bfr, xbase + mrow * 128 + (chunk << 4));
                mma16816(ca + 8 * q2, afrag[(kk & 1) * 4 + j], bfr[0], bfr[1]);
                if (2 * q2 + 1 < nblk) mma16816(ca + 8 * q2 + 4, afrag[(kk & 1) * 4 + j], bfr[2], bfr[3]);
              }
            }
          }
          if (kk == 0 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
        } else {
        wg_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j)
          wgmma_rs<N>(NACC > 2 ? ((kk & 1) ? ((j & 1) ? acc4 : acc2) : ((j & 1) ? acc3 : acc))
                               : ((NACC > 1 && (kk & 1)) ? acc2 : acc),
                      afrag[(DEEP ? kk : (kk & 1)) * 4 + j], desc + 2 * j, 1);
        wg_commit();
        if constexpr (DEEP) {
          wg_wait<MOE_SMALL_WAIT>();
          // the previous stage's last group completes once MOE_SMALL_WAIT - 1 k-blocks of this stage are issued
          if (kk == MOE_SMALL_WAIT - 1 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
        } else {
          wg_wait<1>();
          // the previous stage's MMAs are complete after the first wait of this stage: hand its slot back early
          if (kk == 0 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
        }
        }  // !MMA32
#else
#pragma unroll
        for (int j = 0; j < 4; ++j) mma_sm80<N>((NACC > 1 && (kk & 1)) ? acc2 : acc, afrag[(kk & 1) * 4 + j], xs, j, lane);
#endif
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = j == 0 ? wq.x : j == 1 ? wq.y : j == 2 ? wq.z : wq.w;
          const uint32_t sw = j < 2 ? sc.x : sc.y;  // scales of k16 steps (j & ~1, j | 1): bytes 0,1 and 2,3
          convert8(q, scale_pair_sel(sw, j & 1, m16), afrag[j & 1], m16, m64);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
          wg_fence();
          wgmma_rs<N>(acc, afrag[j & 1], desc + 2 * j, 1);
          wg_commit();
          wg_wait<1>();
          if (kk == 0 && j == 0 && prev >= 0) { rel_sync(); if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev); }
#else
          mma_sm80<N>(acc, afrag[j & 1], xs, j, lane);
#endif
        }
      }
    }
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    __syncwarp();
    if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * slot);
#endif
    prev = slot;
    if (++slot == NSLOT) { slot = 0; ph ^= 1; }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  wg_wait<0>();
  rel_sync();
  if (MOE_ALLLANE_RELEASE || lane == 0) mbar_arrive(bars + 8 * NSLOT + 8 * prev);
#endif
  }  // !TRI
  if (t_mid) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *t_mid = t;
  }
  if constexpr (NACC > 3) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[i] = (acc[i] + acc3[i]) + (acc2[i] + acc4[i]);
  } else if constexpr (NACC > 2) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[i] = (acc[i] + acc2[i]) + acc3[i];
  } else if constexpr (NACC > 1) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[i] += acc2[i];
  }
  // ---- epilogue through the dedicated staging buffer ----
  const float cs = p.gscale[e] * 128.f;
  if (pubq && ctid == 0) wait_staging_free(pubq);
  named_bar_sync(1, 256);  // previous item's copy-out is done with the staging buffer
  if (GATED) {
    const int q = wg * 4 + w;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // stmatrix.trans: the accumulator fragment of an 8-token group is an 8x8 matrix [row g][token]; stored transposed,
    // each token's 8 rows (16 bytes) land in its tile-layout chunk in one instruction per 4 groups
#pragma unroll
    for (int j4 = 0; j4 < N / 32; ++j4) {
      uint32_t r[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int jj = 4 * j4 + u;
        const float h0 = silu_fast(acc[4 * jj] * cs) * (acc[4 * jj + 2] * cs);
        const float h1 = silu_fast(acc[4 * jj + 1] * cs) * (acc[4 * jj + 3] * cs);
        r[u] = bf162_u32(__floats2bfloat162_rn(h0, h1));
      }
      const int rr = lane & 7, tokr = 8 * (4 * j4 + (lane >> 3)) + rr;
      const uint32_t addr = smem_u32(stg + tokr * 128 + ((q ^ rr) << 4));
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r[0]),
                   "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
    }
#else
#pragma unroll
    for (int jj = 0; jj < N / 8; ++jj) {
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const float gt = acc[4 * jj + b] * cs, up = acc[4 * jj + 2 + b] * cs;
        const float h = silu_fast(gt) * up;
        const int tok = 8 * jj + 2 * t + b;
        *reinterpret_cast<__nv_bfloat16*>(stg + tok * 128 + ((q ^ (tok & 7)) << 4) + g * 2) = __float2bfloat16_rn(h);
      }
    }
#endif
    if (pubq) {  // fused sm_90: the publisher warp copies the staged tile out (bulk copy) and then publishes `done`
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
      named_bar_sync(1, 256);
      if (ctid == 0) enqueue_out(pubq, p.out + (size_t)row0 * 8192 + (size_t)ntile * N * 128, done, 0, N * 128);
    } else {
      named_bar_sync(1, 256);
      uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)row0 * 8192 + (size_t)ntile * N * 128);
      const uint4* src = reinterpret_cast<const uint4*>(stg);
      for (int i = ctid; i < N * 8; i += 256) dst[i] = src[i];
      if (done) {  // fused kernel: publish this n-tile of the tile's intermediate rows
        __threadfence();
        named_bar_sync(1, 256);
        if (ctid == 0) atomicAdd(done, 1);
      }
    }
  } else {
    const int c0 = (wg * 4 + w) * 16 + g;
    // halves of up to 128 tokens through the 128-row staging buffer: half 1 (tiles over 128 rows) holds N - 128 rows
    auto gemm2_half = [&]<int half, int NJ>() {
      if (half) named_bar_sync(1, 256);  // (half 0 was copied out by the consumers themselves)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      // stmatrix.trans: per 8-token group, two 8x8 matrices (this warp's columns 16w..16w+7 and 16w+8..16w+15)
      if constexpr (NJ >= 2) {
#pragma unroll
        for (int j2 = 0; j2 < NJ / 2; ++j2) {
          uint32_t r[4];
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            const int ai = 4 * (2 * j2 + u + half * 16);
            r[2 * u] = bf162_u32(__floats2bfloat162_rn(acc[ai] * cs, acc[ai + 1] * cs));
            r[2 * u + 1] = bf162_u32(__floats2bfloat162_rn(acc[ai + 2] * cs, acc[ai + 3] * cs));
          }
          const int m = lane >> 3, rr = lane & 7;
          const int tl = 8 * (2 * j2 + (m >> 1)) + rr;
          const uint32_t addr = smem_u32(stg + tl * 272 + ((wg * 4 + w) * 16 + 8 * (m & 1)) * 2);
          asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(r[0]),
                       "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
        }
      } else {
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
          for (int b = 0; b < 2; ++b) {
            const int tl = 8 * jj + 2 * t + b;
            const int ai = 4 * (jj + half * 16) + b;
            __nv_bfloat16* r = reinterpret_cast<__nv_bfloat16*>(stg + tl * 272);
            r[c0] = __float2bfloat16_rn(acc[ai] * cs);
            r[c0 + 8] = __float2bfloat16_rn(acc[ai + 2] * cs);
          }
        }
      }
#else
#pragma unroll
      for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          const int tl = 8 * jj + 2 * t + b;  // token within this half
          const int ai = 4 * (jj + half * 16) + b;
          __nv_bfloat16* r = reinterpret_cast<__nv_bfloat16*>(stg + tl * 272);
          r[c0] = __float2bfloat16_rn(acc[ai] * cs);
          r[c0 + 8] = __float2bfloat16_rn(acc[ai + 2] * cs);
        }
      }
#endif
      constexpr int ntok = 8 * NJ;
      // the last half goes to the publisher warp (its copy loop overlaps the next item); a first half is copied by the
      // consumers, since the second half needs the staging buffer right away
      if (pubq && half == (N > 128 ? 1 : 0)) {
        named_bar_sync(1, 256);
        if (ctid == 0)
          enqueue_out(pubq, p.out + ((size_t)(row0 + half * 128) * 4096 + ntile * 128) * 2, done, 1, ntok, row0, nreal);
      } else {
        named_bar_sync(1, 256);
        for (int i = ctid; i < ntok * 16; i += 256) {
          const int tl = i >> 4, c = i & 15;
          *reinterpret_cast<uint4*>(p.out + ((size_t)(row0 + half * 128 + tl) * 4096 + ntile * 128) * 2 + c * 16) =
              *reinterpret_cast<const uint4*>(stg + tl * 272 + c * 16);
        }
      }
    };
    gemm2_half.template operator()<0, (N > 128 ? 16 : N / 8)>();
    if constexpr (N > 128) gemm2_half.template operator()<1, (N - 128) / 8>();
  }
}

template <bool GATED>
__global__ void __launch_bounds__(384, 1) moe_gemm_persistent(Args p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);  // full[NSLOT], empty[NSLOT], ifull[2], iempty[2]
  const uint32_t ibar = bars + 16 * NSLOT;
  volatile int* iring = reinterpret_cast<volatile int*>(base + 512);
  uint8_t* ring = base + 1024;
  uint8_t* stg = ring + NSLOT * SLOT;
  const int nitems = *p.ntiles * p.NT;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NSLOT; ++s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      mbar_init(bars + 8 * s, 1);
#else
      mbar_init(bars + 8 * s, 128);
#endif
      mbar_init(bars + 8 * NSLOT + 8 * s, 8 * REL_LANES);
    }
    for (int k = 0; k < 2; ++k) {
      mbar_init(ibar + 8 * k, 1);
      mbar_init(ibar + 16 + 8 * k, MOE_ALLLANE_RELEASE ? 256 : 1);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
  }
  __syncthreads();
  int slot = 0;
  uint32_t ph = 0;
  if (threadIdx.x < 128) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (threadIdx.x == 0) {
      int k = 0;
      uint32_t iph = 0;
      while (true) {
        mbar_wait(ibar + 16 + 8 * k, iph ^ 1);
        const int item = atomicAdd(p.counter, 1);
        iring[k] = item;
        mbar_arrive(ibar + 8 * k);
        if (item >= nitems) break;
        const int tile = item / p.NT, ntile = item - tile * p.NT;
        const int4 ti = p.tiles[tile];
        const int e = ti.x, row0 = ti.y, n = ti.z;
        const int kps = kps_for(n);
        const uint32_t xb = n * 128;
        const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
        const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
        for (int it = 0; it < KS / kps; ++it) {
          mbar_wait(bars + 8 * NSLOT + 8 * slot, ph ^ 1);
          slot_proxy_fence();
          const uint32_t full = bars + 8 * slot;
          mbar_expect_tx(full, kps * (xb + WBLK));
          const uint32_t dst = smem_u32(ring + slot * SLOT);
          bulk_g2s(dst, xsrc + (size_t)it * kps * xb, kps * xb, full);
          bulk_g2s(dst + kps * xb, wsrc + (size_t)it * kps * WBLK, kps * WBLK, full);
          if (++slot == NSLOT) { slot = 0; ph ^= 1; }
        }
        k ^= 1;
        if (k == 0) iph ^= 1;
      }
    }
#else
    int k = 0;
    uint32_t iph = 0;
    while (true) {
      mbar_wait(ibar + 16 + 8 * k, iph ^ 1);
      if (threadIdx.x == 0) iring[k] = atomicAdd(p.counter, 1);
      named_bar_sync(3, 128);
      const int item = iring[k];
      named_bar_sync(3, 128);
      if (threadIdx.x == 0) mbar_arrive(ibar + 8 * k);
      if (item >= nitems) break;
      const int tile = item / p.NT, ntile = item - tile * p.NT;
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      const int kps = kps_for(n);
      const int xb = n * 128;
      const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
      const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
      for (int it = 0; it < KS / kps; ++it) {
        mbar_wait(bars + 8 * NSLOT + 8 * slot, ph ^ 1);
        const uint32_t dst = smem_u32(ring + slot * SLOT);
        const int nx = kps * xb / 16, nw = kps * WBLK / 16;
        for (int i = threadIdx.x; i < nx + nw; i += 128) {
          const uint8_t* src = i < nx ? xsrc + (size_t)it * kps * xb + i * 16 : wsrc + (size_t)it * kps * WBLK + (i - nx) * 16;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + i * 16), "l"(src) : "memory");
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(bars + 8 * slot) : "memory");
        if (++slot == NSLOT) { slot = 0; ph ^= 1; }
      }
      k ^= 1;
      if (k == 0) iph ^= 1;
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
#endif
    int k = 0;
    uint32_t iph = 0;
    while (true) {
      mbar_wait(ibar + 8 * k, iph);
      const int item = iring[k];
      named_bar_sync(1, 256);  // every consumer has read the item
      if (MOE_ALLLANE_RELEASE || threadIdx.x == 128) mbar_arrive(ibar + 16 + 8 * k);  // all: each reader orders its own read
      if (item >= nitems) break;
      k ^= 1;
      if (k == 0) iph ^= 1;
      const int tile = item / p.NT, ntile = item - tile * p.NT;
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      switch (n) {
        case 32: consumer_item<32>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 64: consumer_item<64>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
        case 128: consumer_item<128>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
#if MOE_FINE_TILES
#if MOE_FINE_MASK & 1
        case 96: consumer_item<96>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
#endif
#if MOE_FINE_MASK & 2
        case 160: consumer_item<160>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
#endif
#if MOE_FINE_MASK & 4
        case 192: consumer_item<192>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
#endif
#if MOE_FINE_MASK & 8
        case 224: consumer_item<224>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
#endif
#endif
        default: consumer_item<256>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED); break;
      }
    }
  }
}

// ---- both expert GEMMs in one persistent launch: the queue holds every GEMM1 item, then every GEMM2 item; a GEMM2
// item's loads wait until all 64 GEMM1 n-tiles of its token tile have been written (per-tile counters) ----
struct FusedArgs {
  Args g1, g2;     // GEMM1 (gated, writes inter in tile layout) and GEMM2 (reads inter, writes y)
  int* done;       // [max tiles] GEMM1 n-tiles finished per token tile (zeroed before launch)
  unsigned long long* dbg;  // optional per-item timeline: [item][start ns, end ns, smid, cta] (nullptr: off)
  int flags;       // bit 8: fine remainder tiles (FINE instantiation); experiments: bit 0 = load X only for an item's first stage, bit 1 = same for W
  // CIT (combine in the tail; cit == nullptr: off): the last GEMM2 n-tile item of a tile counts its tokens' routes down
  // (tok_left, set by the route kernel); a token whose routes are all done goes into the combine queue cq (cleared by the
  // route kernel; no-route tokens queued by the gather); CTAs out of expert items pop tokens and write their combined
  // rows (the separate combine launch is gone). All counters are zeroed per call inside the graph (boundary kernel).
  int* cit;        // [max tiles] GEMM2 n-tiles finished per tile
  int* qctl;       // [0] queue head (pops), [1] queue tail (pushes), [2] error flag (a spin limit hit)
  int* tok_left;   // [T] local routes of each token not yet finished
  int* cq;         // [T] combine queue: token + 1 (0: not yet written)
  const int* row_tok;  // [rows] token of each routed row
  const int* rrow; const int* lid; const float* topw; const __nv_bfloat16* pw;  // the combine's inputs
  __nv_bfloat16* out;  // [T, 4096] combined expert output
  int T;
};

#ifndef MOE_FINE_NOINLINE
#define MOE_FINE_NOINLINE 0  // 1: the extra widths as separate functions -> ptxas C7510 (wgmma pipeline crossing a call): whole kernel serialized
#endif
template <int N>
__noinline__ __device__ void consumer_item_ni(const Args& p, uint8_t* ring, uint8_t* stg, uint32_t bars, int& slot, uint32_t& ph,
                                              int e, int row0, int ntile, bool GATED, int* done, unsigned long long* t_mid,
                                              volatile long long* pubq, int nreal = 256) {
  consumer_item<N>(p, ring, stg, bars, slot, ph, e, row0, ntile, GATED, done, t_mid, pubq, nreal);
}

// ---- CIT: combine in the tail of the fused expert kernel ----
// The 256 consumer threads of a CTA with no expert items left pop tokens from the combine queue and write each one's
// combined row, with the CUDA combine's arithmetic (moe_combine_kernel): every local route's y row times (1 + w),
// RMS-normalized on its own (eps 1e-8), times its routing weight, summed in route order k = 0..7 (two routes' rows in
// flight, one block reduction per pair). The order is fixed per token, so the result does not depend on which CTA
// combines it or when (deterministic). y rows were written by other CTAs during this kernel: L2 loads (ld.global.cg)
// after the acquire chain (item releases -> done2 RMW -> tok_left acq_rel RMWs -> queue store release -> acquire here).
#ifndef MOE_CIT
#define MOE_CIT 0  // combine in the tail (Oct 26: measured 2.5-3x slower than the combine kernel; TRICKS 5.16): compiled out
#endif
#ifndef CIT_BACKOFF_NS
#define CIT_BACKOFF_NS 500
#endif
#ifndef CIT_SPIN_NS
#define CIT_SPIN_NS 200000000ull  // 200 ms: a queue slot never written sets the error flag (fa.qctl[2]) instead of hanging
#endif
// a completed GEMM2 tile counts down the routes of its rows' tokens; a token whose last route this was is queued
__noinline__ __device__ void cit_release(const int* row_tok, int* tok_left, int* qctl, int* cq, int r0, int nreal, int lane) {
  // (the caller's fence.acq_rel.gpu after acquiring the tile's items is the release side; the counts go down with
  // relaxed RMWs, issued together (up to 8 per lane, 256 rows); a token's last decrementer fences again (acquire side)
  // before the queue store with release: fence + relaxed RMW patterns on both ends)
  int t[8], prev[8];
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    const int lr = lane + 32 * u;
    t[u] = lr < nreal ? row_tok[r0 + lr] : -1;
  }
#pragma unroll
  for (int u = 0; u < 8; ++u) prev[u] = t[u] >= 0 ? atomicAdd(tok_left + t[u], -1) : 0;
  bool any = false;
#pragma unroll
  for (int u = 0; u < 8; ++u) any |= prev[u] == 1;
  if (any) {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      if (prev[u] == 1) {
        const int i = atomicAdd(qctl + 1, 1);
        asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(cq + i), "r"(t[u] + 1) : "memory");
      }
    }
  }
}
DEVI uint4 ldcg16(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
DEVI void cit_combine_token(const FusedArgs& fa, int tok, const float (&g)[16], float (*red)[8], int ctid) {
  // routes in order k = 0..7 (compile-time indices: no local-memory arrays), one double-buffered block reduction each
  const int lane = ctid & 31, warp = ctid >> 5;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
  int b = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int l = fa.lid[tok * 8 + k];
    if (l >= 0) {  // uniform across the CTA (one token)
      const int row = fa.rrow[tok * 8 + k];
      const float wt = fa.topw[tok * 8 + k];
      const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(fa.g2.out) + (size_t)row * 4096 + ctid * 16;
      const uint4 v0 = ldcg16(src), v1 = ldcg16(src + 8);
      const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
      const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
      float x[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h0[i]), q = __bfloat1622float2(h1[i]);
        x[2 * i] = f.x * g[2 * i]; x[2 * i + 1] = f.y * g[2 * i + 1];
        x[8 + 2 * i] = q.x * g[8 + 2 * i]; x[8 + 2 * i + 1] = q.y * g[8 + 2 * i + 1];
      }
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (lane == 0) red[b][warp] = ss;
      named_bar_sync(1, 256);  // (double buffer: a buffer is rewritten only after the next route's barrier)
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) tot += red[b][w];
      const float sc = rsqrtf(tot / 4096.f + 1e-8f) * wt;
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[i] += x[i] * sc;
      b ^= 1;
    }
  }
  uint4 o0, o1;
  __nv_bfloat162* b0 = reinterpret_cast<__nv_bfloat162*>(&o0);
  __nv_bfloat162* b1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    b0[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    b1[i] = __floats2bfloat162_rn(acc[8 + 2 * i], acc[8 + 2 * i + 1]);
  }
  uint4* dst = reinterpret_cast<uint4*>(fa.out + (size_t)tok * 4096 + ctid * 16);
  dst[0] = o0;
  dst[1] = o1;
}
DEVI void cit_combine(const FusedArgs& fa, int ctid, uint8_t* base) {
  // warp 0 lane 0 is the CTA's only poller of the global queue (with backoff: thousands of pollers slowed the last
  // expert items down); it hands tokens to warps 1..7 through shared-memory slots (token + 1, 0 = empty, -1 = done);
  // each combining warp takes one token at a time
  volatile int* slot = reinterpret_cast<volatile int*>(base + 768);  // [8]
  const int lane = ctid & 31, warp = ctid >> 5;
  if (lane == 0) slot[warp] = 0;
  named_bar_sync(1, 256);
  if (warp == 0) {
    if (lane == 0) {
      unsigned long long t0, now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      bool done = false;
      while (!done) {
        bool idle = true;
        for (int w = 1; w < 8 && !done; ++w) {
          if (slot[w] != 0) continue;
          int head, tail;
          asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(head) : "l"(fa.qctl) : "memory");
          if (head >= fa.T) { done = true; break; }
          asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(tail) : "l"(fa.qctl + 1) : "memory");
          if (tail <= head) break;
          const int idx = atomicAdd(fa.qctl, 1);
          if (idx >= fa.T) { done = true; break; }
          int v;
          while (true) {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(fa.cq + idx) : "memory");
            if (v != 0) break;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            if (now - t0 > CIT_SPIN_NS) { atomicExch(fa.qctl + 2, 1); break; }
            __nanosleep(100);
          }
          if (v != 0)  // release (CTA scope): the combining warp's acquire of the slot orders its row loads after ours
            asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32((const void*)(slot + w))), "r"(v) : "memory");
          idle = false;
        }
        if (done) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (now - t0 > CIT_SPIN_NS) { atomicExch(fa.qctl + 2, 1); break; }
        if (idle) __nanosleep(CIT_BACKOFF_NS);
      }
      for (int w = 1; w < 8; ++w) {  // done: each warp gets -1 once its slot is free
        while (slot[w] != 0) __nanosleep(64);
        slot[w] = -1;
      }
    }
    __syncwarp();
    return;
  }
  while (true) {
    int v = 0;
    if (lane == 0) {
      while (true) {
        asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32((const void*)(slot + warp))) : "memory");
        if (v != 0) break;
        __nanosleep(32);
      }
    }
    v = __shfl_sync(0xffffffffu, v, 0);
    if (v < 0) break;
    __syncwarp();
    const int tok = v - 1;
    // local routes in order, two at a time; pass 1 issues all 16 chunk loads of both rows at once (L1-allocating: the
    // rows were never read on this SM, and L1 starts clean each launch) for their RMS, pass 2 re-reads them from L1
    int rows[8], nl = 0;
    float wts[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int l = fa.lid[tok * 8 + k];
      rows[k] = -1;
      wts[k] = 0.f;
      if (l >= 0) {
        rows[k] = fa.rrow[tok * 8 + k];
        wts[k] = fa.topw[tok * 8 + k];
        ++nl;
      }
    }
    float sc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      sc[k] = 0.f;
      if (rows[k] >= 0) {
        const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(fa.g2.out) + (size_t)rows[k] * 4096 + lane * 8;
        uint4 q[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(q[j].x), "=r"(q[j].y), "=r"(q[j].z), "=r"(q[j].w) : "l"(src + 256 * j) : "memory");
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&q[j]);
          const float4 g0 = g1p4(fa.pw + lane * 8 + 256 * j), g1 = g1p4(fa.pw + lane * 8 + 256 * j + 4);
          const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(h[i]);
            const float a = f.x * gv[2 * i], b = f.y * gv[2 * i + 1];
            ss += a * a + b * b;
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        sc[k] = rsqrtf(ss / 4096.f + 1e-8f) * wts[k];
      }
    }
    (void)nl;
#pragma unroll 4
    for (int j = 0; j < 16; ++j) {
      const float4 g0 = g1p4(fa.pw + lane * 8 + 256 * j), g1 = g1p4(fa.pw + lane * 8 + 256 * j + 4);
      const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if (rows[k] >= 0) {
          uint4 q;
          asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w)
                       : "l"(reinterpret_cast<const __nv_bfloat16*>(fa.g2.out) + (size_t)rows[k] * 4096 + lane * 8 + 256 * j) : "memory");
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&q);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(h[i]);
            acc[2 * i] += (f.x * gv[2 * i]) * sc[k];
            acc[2 * i + 1] += (f.y * gv[2 * i + 1]) * sc[k];
          }
        }
      }
      uint4 o;
      __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) b[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
      *reinterpret_cast<uint4*>(fa.out + (size_t)tok * 4096 + lane * 8 + 256 * j) = o;
    }
    __syncwarp();
    if (lane == 0) slot[warp] = 0;
  }
}

template <int FINE>
#ifndef MOE_EARLY_INIT
#define MOE_EARLY_INIT 0  // 1: shared-memory barrier setup before the PDL wait (overlaps the gather's tail)
#endif
__global__ void __launch_bounds__(MOE_FUSED_THREADS, 1) moe_fused_persistent(FusedArgs fa) {
  if constexpr (!MOE_EARLY_INIT) pdl_enter();
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);
  const uint32_t ibar = bars + 16 * NSLOT;
  volatile int* iring = reinterpret_cast<volatile int*>(base + 512);
  uint8_t* ring = base + 1024;
  uint8_t* stg = ring + NSLOT * SLOT;
  // publisher queue (sm_90): count at base + 240, 32 done-pointers at base + 256
  volatile long long* pubq = reinterpret_cast<volatile long long*>(base + 256);
  if (threadIdx.x == 0) {
    *reinterpret_cast<volatile int*>(base + 240) = 0;
    *reinterpret_cast<volatile int*>(base + 244) = 0;
    *reinterpret_cast<volatile int*>(base + 248) = 0;
    for (int s = 0; s < NSLOT; ++s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      mbar_init(bars + 8 * s, 1);
#else
      mbar_init(bars + 8 * s, 128);
#endif
      mbar_init(bars + 8 * NSLOT + 8 * s, 8 * REL_LANES);
    }
    for (int k = 0; k < 2; ++k) {
      mbar_init(ibar + 8 * k, 1);
      mbar_init(ibar + 16 + 8 * k, MOE_ALLLANE_RELEASE ? 256 : 1);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
  }
  __syncthreads();
  if constexpr (MOE_EARLY_INIT) pdl_enter();
  const int nt = *fa.g1.ntiles;
  const int n1 = nt * fa.g1.NT, nitems = n1 + nt * fa.g2.NT;
  int slot = 0;
  uint32_t ph = 0;
  if (MOE_CONS_FIRST ? threadIdx.x >= 256 : threadIdx.x < 128) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (!MOE_CONS_FIRST) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (threadIdx.x == MOE_PB) {
      int k = 0;
      uint32_t iph = 0;
      while (true) {
        mbar_wait(ibar + 16 + 8 * k, iph ^ 1);
        const int item = atomicAdd(fa.g1.counter, 1);
        iring[k] = item;
        mbar_arrive(ibar + 8 * k);
        if (item >= nitems) break;
        const bool second = item >= n1;
        const Args& p = second ? fa.g2 : fa.g1;
        const int it2 = second ? item - n1 : item;
        const int tile = it2 / p.NT, ntile = it2 - tile * p.NT;
        const int4 ti = p.tiles[tile];
        const int e = ti.x, row0 = ti.y, n = ti.z;
        if (second) {
          const int need = fa.g1.NT;
          int v;
          while (true) {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(fa.done + tile) : "memory");
            if (v >= need) break;
            __nanosleep(64);
          }
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        const int kps = kps_for(n);
        const uint32_t xb = n * 128;
        const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
        const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
        const bool pf = MOE_L2PF > 0 && (MOE_L2PF_ALL || kps >= 2);
        if (pf) {  // the item's first stages: a head start while the previous item's last stages are consumed
          for (int j = 0; j < MOE_L2PF && j < KS / kps; ++j) prefetch_l2(wsrc + (size_t)j * kps * WBLK, kps * WBLK);
        }
        for (int it = 0; it < KS / kps; ++it) {
          if (pf && it + MOE_L2PF < KS / kps) prefetch_l2(wsrc + (size_t)(it + MOE_L2PF) * kps * WBLK, kps * WBLK);
          mbar_wait(bars + 8 * NSLOT + 8 * slot, ph ^ 1);
          slot_proxy_fence();
          const uint32_t full = bars + 8 * slot;
          const bool lx = !(fa.flags & 1) || it < NSLOT, lw = !(fa.flags & 2) || it < NSLOT;
          mbar_expect_tx(full, (lx ? kps * xb : 0) + (lw ? kps * WBLK : 0) + (!lx && !lw ? 16 : 0));
          const uint32_t dst = smem_u32(ring + slot * SLOT);
          if (lx) bulk_g2s(dst, xsrc + (size_t)it * kps * xb, kps * xb, full);
          if (lw) bulk_g2s(dst + kps * xb, wsrc + (size_t)it * kps * WBLK, kps * WBLK, full);
          if (!lx && !lw) bulk_g2s(dst, xsrc, 16, full);
          if (++slot == NSLOT) { slot = 0; ph ^= 1; }
        }
        k ^= 1;
        if (k == 0) iph ^= 1;
      }
    } else if (threadIdx.x >= MOE_PB + 32 && threadIdx.x < MOE_PB + 64) {
      // publisher warp: copies each staged epilogue tile out with bulk copies (the consumers go on with their next
      // item), frees the staging buffer once read, and for GEMM1 tiles waits for the writes, then a GPU-scope release
      // and the tile's done counter (read with acquire by other CTAs' producers)
      const int lane = threadIdx.x - (MOE_PB + 32);
      const uint32_t stg_s = smem_u32(stg);
      int seen = 0;
      while (true) {
        int r = 0;
#if MOE_ALLLANE_RELEASE  // every lane acquires the count itself, so its own entry and staging reads are ordered after it
        asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(r) : "r"(smem_u32(base + 240)) : "memory");
        r = __shfl_sync(0xffffffffu, r, 0);  // one value for the warp (a later lane may see a newer count)
#else
        if (lane == 0) asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(r) : "r"(smem_u32(base + 240)) : "memory");
        r = __shfl_sync(0xffffffffu, r, 0);
#endif
        const int avail = r < 0 ? -r - 1 : r;
        while (seen < avail) {
          volatile long long* e = pubq + 4 * (seen & 7);
          uint8_t* dst = reinterpret_cast<uint8_t*>(e[0]);
          int* d = reinterpret_cast<int*>(e[1]);
          const int kind = reinterpret_cast<volatile int*>(e)[4], amount = reinterpret_cast<volatile int*>(e)[5];
          if (kind == 0) {  // one contiguous tile-layout block: a single bulk copy
            if (lane == 0) {
              bulk_s2g(dst, stg_s, amount);
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
              asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            }
          } else {  // token rows of 256 bytes (272-byte staging pitch): the warp copies them, 16 bytes per lane
            for (int i = lane; i < amount * 16; i += 32) {
              const int tl = i >> 4, c = i & 15;
              *reinterpret_cast<uint4*>(dst + (size_t)tl * 8192 + c * 16) = *reinterpret_cast<const uint4*>(stg + tl * 272 + c * 16);
            }
          }
          __syncwarp();  // after lane 0's bulk-copy read of the staging buffer has completed
#if MOE_ALLLANE_RELEASE  // every lane releases the staging buffer itself (same value), after its own reads
          asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32(base + 248)), "r"(seen + 1) : "memory");
#endif
          int old = -1;
          if (lane == 0) {
#if !MOE_ALLLANE_RELEASE
            asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32(base + 248)), "r"(seen + 1) : "memory");
#endif
            if (d) {
              if (kind == 0) {  // a bulk (async-proxy) copy: wait for its writes first
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                asm volatile("fence.proxy.async.global;" ::: "memory");
              }
              asm volatile("fence.acq_rel.gpu;" ::: "memory");
              old = atomicAdd(d, 1);
            }
          }
          if (MOE_CIT && kind == 1 && d != nullptr) {  // CIT: a GEMM2 item; the tile's last one counts its tokens' routes down
            old = __shfl_sync(0xffffffffu, old, 0);
            __syncwarp();  // (memory ordering: lane 0's acquire of the other items' releases, before every lane's release)
            if (old == fa.g2.NT - 1) {
              asm volatile("fence.acq_rel.gpu;" ::: "memory");
              // (entry seen & 7 is rewritten only for job seen + 8, after this job: safe to read its rows here)
              cit_release(fa.row_tok, fa.tok_left, fa.qctl, fa.cq, reinterpret_cast<volatile int*>(e)[6],
                          reinterpret_cast<volatile int*>(e)[7], lane);
            }
          }
          ++seen;
        }
        if (r < 0) break;
        __nanosleep(64);
      }
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
#else
    int k = 0;
    uint32_t iph = 0;
    while (true) {
      mbar_wait(ibar + 16 + 8 * k, iph ^ 1);
      if (threadIdx.x == 0) iring[k] = atomicAdd(fa.g1.counter, 1);
      named_bar_sync(3, 128);
      const int item = iring[k];
      named_bar_sync(3, 128);
      if (threadIdx.x == 0) mbar_arrive(ibar + 8 * k);
      if (item >= nitems) break;
      const bool second = item >= n1;
      const Args& p = second ? fa.g2 : fa.g1;
      const int it2 = second ? item - n1 : item;
      const int tile = it2 / p.NT, ntile = it2 - tile * p.NT;
      const int4 ti = p.tiles[tile];
      const int e = ti.x, row0 = ti.y, n = ti.z;
      if (second) {
        if (threadIdx.x == 0) {
          while (atomicAdd(fa.done + tile, 0) < fa.g1.NT) __nanosleep(64);
          __threadfence();
        }
        named_bar_sync(3, 128);
      }
      const int kps = kps_for(n);
      const int xb = n * 128;
      const uint8_t* wsrc = p.wrep + ((size_t)(e * p.NT + ntile) * KS) * WBLK;
      const uint8_t* xsrc = p.xt + (size_t)row0 * 8192;
      for (int it = 0; it < KS / kps; ++it) {
        mbar_wait(bars + 8 * NSLOT + 8 * slot, ph ^ 1);
        const uint32_t dst = smem_u32(ring + slot * SLOT);
        const int nx = kps * xb / 16, nw = kps * WBLK / 16;
        for (int i = threadIdx.x; i < nx + nw; i += 128) {
          const uint8_t* src = i < nx ? xsrc + (size_t)it * kps * xb + i * 16 : wsrc + (size_t)it * kps * WBLK + (i - nx) * 16;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + i * 16), "l"(src) : "memory");
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(bars + 8 * slot) : "memory");
        if (++slot == NSLOT) { slot = 0; ph ^= 1; }
      }
      k ^= 1;
      if (k == 0) iph ^= 1;
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (!MOE_CONS_FIRST) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
#endif
    int k = 0;
    uint32_t iph = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    volatile long long* pq = pubq;
#else
    volatile long long* pq = nullptr;
#endif
    while (true) {
      mbar_wait(ibar + 8 * k, iph);
      const int item = iring[k];
      named_bar_sync(1, 256);
      if (MOE_ALLLANE_RELEASE || threadIdx.x == MOE_CB) mbar_arrive(ibar + 16 + 8 * k);  // all: each reader orders its own read
      if (item >= nitems) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if (threadIdx.x == MOE_CB) {  // the publisher drains the queue and exits
          const int pubn = *reinterpret_cast<volatile int*>(base + 244);
          asm volatile("st.release.cta.shared.b32 [%0], %1;" ::"r"(smem_u32(base + 240)), "r"(-(pubn + 1)) : "memory");
        }
#endif
#ifndef CIT_DBG_NOCOMBINE
#define CIT_DBG_NOCOMBINE 0  // timing experiment only: the per-item release path without the tail combine
#endif
        if (MOE_CIT && fa.cit != nullptr && !CIT_DBG_NOCOMBINE) cit_combine(fa, threadIdx.x - MOE_CB, base);  // CIT: combine finished tokens
        break;
      }
      k ^= 1;
      if (k == 0) iph ^= 1;
      unsigned long long t_start = 0;
      if (fa.dbg && threadIdx.x == MOE_CB) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
      const bool first = item < n1;
      const Args& p = first ? fa.g1 : fa.g2;
      const int it2 = first ? item : item - n1;
      const int tile = it2 / p.NT, ntile = it2 - tile * p.NT;
      const int4 ti = p.tiles[tile];
      int* dn = first ? fa.done + tile : ((MOE_CIT && fa.cit != nullptr) ? fa.cit + tile : nullptr);
      unsigned long long t_mid = 0;
      unsigned long long* tm = (fa.dbg && threadIdx.x == MOE_CB) ? &t_mid : nullptr;
      switch (ti.z) {
        case 32: consumer_item<32, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w); break;
        case 64: consumer_item<64, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w); break;
#ifndef MOE_ONLY_SMALL  // experiment: a fused kernel with only the small-tile paths (registers for N <= 64 only)
        case 128: consumer_item<128, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w); break;
#if MOE_FINE_TILES
#if MOE_FINE_MASK & 1
        case 96:
          if constexpr (FINE != 0) {
            if constexpr (MOE_FINE_NOINLINE) consumer_item_ni<96>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
            else consumer_item<96, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
          }
          break;
#endif
#if MOE_FINE_MASK & 2
        case 160:
          if constexpr (FINE != 0) {
            if constexpr (MOE_FINE_NOINLINE) consumer_item_ni<160>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
            else consumer_item<160, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
          }
          break;
#endif
#if MOE_FINE_MASK & 4
        case 192:
          if constexpr (FINE != 0) {
            if constexpr (MOE_FINE_NOINLINE) consumer_item_ni<192>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
            else consumer_item<192, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
          }
          break;
#endif
#if MOE_FINE_MASK & 8
        case 224:
          if constexpr (FINE != 0) {
            if constexpr (MOE_FINE_NOINLINE) consumer_item_ni<224>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
            else consumer_item<224, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w);
          }
          break;
#endif
#endif
        default: consumer_item<256, MOE_CB>(p, ring, stg, bars, slot, ph, ti.x, ti.y, ntile, first, dn, tm, pq, ti.w); break;
#endif
      }
      if (fa.dbg && threadIdx.x == MOE_CB) {
        unsigned long long t_end;
        unsigned int smid;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        fa.dbg[(size_t)item * 4 + 0] = t_start;
        fa.dbg[(size_t)item * 4 + 1] = t_end;
        fa.dbg[(size_t)item * 4 + 2] = smid | ((unsigned long long)(t_mid - t_start) << 16);
        fa.dbg[(size_t)item * 4 + 3] = blockIdx.x;
      }
    }
  }
}

// ---- routing plan: per-expert tiles of 256 rows plus one remainder tile of 32/64/128/256 ----
// an expert's last (remainder) tile: 32 or 64 rows when it fits, else (MOE_FINE_TILES) the rows rounded up to 32
DEVI int rem_tile(int r, int fine) {
  if (r == 0) return 0;
  if (r <= 32) return 32;
  if (r <= 64) return 64;
#if MOE_FINE_TILES
  if (!fine) return r <= 128 ? 128 : 256;
  if ((MOE_FINE_MASK & 1) && r <= 96) return 96;
  if (r <= 128) return 128;
  if ((MOE_FINE_MASK & 2) && r <= 160) return 160;
  if ((MOE_FINE_MASK & 4) && r <= 192) return 192;
  if ((MOE_FINE_MASK & 8) && r <= 224) return 224;
  return 256;
#else
  return r <= 128 ? 128 : 256;
#endif
}
#ifndef MOE_SPLIT_REM
#define MOE_SPLIT_REM 1
#endif
// an expert's remainder rows as one or two tiles: without fine widths, 129..192 rows become 128 + 32/64 (cheaper than
// one 256-row tile: ~22 + 11..16 vs ~41 us per weight n-tile item)
DEVI void rem_tiles(int r, int fine, int& na, int& nb) {
  if (MOE_SPLIT_REM && !fine && r > 128 && r <= 192) {
    na = 128;
    nb = rem_tile(r - 128, 0);
  } else {
    na = rem_tile(r, fine);
    nb = 0;
  }
}
DEVI void plan_warp(const int* __restrict__ counts, int* __restrict__ offsets, int4* __restrict__ tiles,
                    int* __restrict__ ntiles, int* __restrict__ rows_per_expert, int order, int e) {
  const int fine = order >> 8;  // bit 8: fine remainder tiles (the expert kernel must be the FINE instantiation)
  order &= 255;
  // Tile list order (the persistent kernel takes items from it through an atomic counter):
  //   0: expert order; 1: biggest first (all 256-row tiles, then 128, 64, 32: longest-processing-time);
  //   2: the 256/128-row (compute-bound) tiles biggest first with the 64/32-row (weight-streaming) tiles spread
  //      between them, the last fifth of the small tiles at the end.
  // Each expert's rows stay contiguous (offsets) in tile-row order.
  const int c = counts[e];  // one warp, lane e = expert e
  const int full = c >> 8, r = c & 255;
  int nr, nb;  // remainder tile widths (nb: second remainder tile or 0)
  rem_tiles(r, fine, nr, nb);
  const int rows = full * 256 + nr + nb;
  const int nt = full + (nr > 0) + (nb > 0);
  // size classes, widest first: 256, 224, 192, 160, 128, 96 ("big", tensor-bound), 64, 32 ("small")
  // cnt: [0] rows, [1 + k] tiles of class k (k = 0..7), [9] tiles
  int cnt[10];
  cnt[0] = rows;
#pragma unroll
  for (int k = 0; k < 8; ++k) cnt[1 + k] = (k == 0 ? full : 0) + (nr == 256 - 32 * k) + (nb == 256 - 32 * k);
  cnt[9] = nt;
  int incl[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) {
    int v = cnt[k];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a2 = __shfl_up_sync(0xffffffffu, v, o);
      if (e >= o) v += a2;
    }
    incl[k] = v;
  }
  int tot[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) tot[k] = __shfl_sync(0xffffffffu, incl[k], 31);
  const int r0 = incl[0] - rows;
  offsets[e] = r0;
  rows_per_expert[e] = rows;
  if (e == 31) *ntiles = tot[9];
  if (order == 0) {
    const int t0 = incl[9] - nt;
    for (int i = 0; i < full; ++i) tiles[t0 + i] = make_int4(e, r0 + 256 * i, 256, 256);
    if (nr) tiles[t0 + full] = make_int4(e, r0 + 256 * full, nr, nb ? nr : r);
    if (nb) tiles[t0 + full + 1] = make_int4(e, r0 + 256 * full + nr, nb, r - nr);
    return;
  }
  // class bases in widest-first order; big ranks k (classes 0..5), small ranks j (classes 6, 7)
  int cbase[8];
  {
    int acc = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) { cbase[k] = (k == 6) ? 0 : acc; acc = (k == 5) ? 0 : acc + tot[1 + k]; }
  }
  const int B = tot[1] + tot[2] + tot[3] + tot[4] + tot[5] + tot[6], S = tot[7] + tot[8];
  auto pos_big = [&](int k) -> int {
    if (order == 1 || B == 0) return k;
    return k + min(S, (4 * k * S) / (5 * B));
  };
  auto pos_small = [&](int j) -> int {
    if (order == 1) return B + j;
    if (B == 0) return j;
    const int lim = (4 * S) / 5;
    const int bb = j < lim ? (5 * B * (j + 1) + 4 * S - 1) / (4 * S) : B;
    return j + bb;
  };
  // class 0 (256): this expert's full tiles and a 256-row remainder, at in-class ranks incl - cnt ..
  const int p256 = cbase[0] + incl[1] - cnt[1];
  for (int i = 0; i < full; ++i) tiles[pos_big(p256 + i)] = make_int4(e, r0 + 256 * i, 256, 256);
  if (nr == 256) tiles[pos_big(p256 + full)] = make_int4(e, r0 + 256 * full, 256, r);
#pragma unroll
  for (int k = 1; k < 6; ++k)
    if (nr == 256 - 32 * k) tiles[pos_big(cbase[k] + incl[1 + k] - 1)] = make_int4(e, r0 + 256 * full, nr, nb ? nr : r);
  if (nr == 64) tiles[pos_small(cbase[6] + incl[7] - 1)] = make_int4(e, r0 + 256 * full, 64, r);
  if (nr == 32) tiles[pos_small(cbase[7] + incl[8] - 1)] = make_int4(e, r0 + 256 * full, 32, r);
  // a split remainder's second tile (32 or 64 rows; the first one, 128 rows, was placed above)
  if (nb == 64) tiles[pos_small(cbase[6] + incl[7] - 1)] = make_int4(e, r0 + 256 * full + nr, 64, r - nr);
  if (nb == 32) tiles[pos_small(cbase[7] + incl[8] - 1)] = make_int4(e, r0 + 256 * full + nr, 32, r - nr);
}

__global__ void moe_plan_kernel(const int* __restrict__ counts, int* __restrict__ offsets, int4* __restrict__ tiles,
                                int* __restrict__ ntiles, int* __restrict__ rows_per_expert, int order) {
  pdl_enter();
  plan_warp(counts, offsets, tiles, ntiles, rows_per_expert, order, threadIdx.x);  // 32 threads
}

// ---- normalize the pooler output and scatter each token's row into its local routes' tile slots ----
// p: [T, 4096] bf16; lid/pos: [T, 8]; writes xt (tile layout); zero-fills pad rows; rrow[t*8+k] = row (or -1)
#ifndef MOE_GATHER_MINB
#define MOE_GATHER_MINB 1
#endif
#ifndef MOE_PADFILL_PARTS
#define MOE_PADFILL_PARTS 8  // pad-fill blocks per expert (fewer blocks: the gather grid fits one wave at small T)
#endif
// PLAN: the block works out the expert offsets itself (one warp scan over the 32 counts) and block 0 also writes the
// whole plan (offsets, tile list, tile count, rows per expert) for the expert kernel: no separate plan launch.
template <bool PLAN>
__global__ void __launch_bounds__(256, MOE_GATHER_MINB) moe_gather_kernel(const __nv_bfloat16* __restrict__ p,
                                                         const __nv_bfloat16* __restrict__ gw, const int* __restrict__ lid,
                                                         const int* __restrict__ pos, const int* __restrict__ offsets_in,
                                                         const int* __restrict__ counts, uint8_t* __restrict__ xt,
                                                         int* __restrict__ rrow, int T, int* __restrict__ offsets_out = nullptr,
                                                         int4* __restrict__ tiles = nullptr, int* __restrict__ ntiles = nullptr,
                                                         int* __restrict__ rows_out = nullptr, int order = 0,
                                                         int* __restrict__ row_tok = nullptr, int* __restrict__ cq = nullptr,
                                                         int* __restrict__ qtail = nullptr) {
#ifndef MOE_GATHER_GPRE
#define MOE_GATHER_GPRE 0  // 1: the norm weight (1 + w, a constant) is loaded before the PDL wait
#endif
#if MOE_GATHER_GPRE
  float4 gpre[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) gpre[i] = g1p4(gw + threadIdx.x * 16 + 4 * i);
#endif
  pdl_enter();
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  const int fine = order >> 8;
  __shared__ float red[8];
  __shared__ int off_s[32];
  const int* offsets = offsets_in;
  if constexpr (PLAN) {
    if (tid < 32) {
      const int c = counts[tid];
      const int full = c >> 8, r = c & 255;
      int nr, nb;
      rem_tiles(r, fine, nr, nb);
      const int rows = full * 256 + nr + nb;
      int v = rows;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int a2 = __shfl_up_sync(0xffffffffu, v, o);
        if (tid >= o) v += a2;
      }
      off_s[tid] = v - rows;
      if (blockIdx.x == 0) plan_warp(counts, offsets_out, tiles, ntiles, rows_out, order, tid);
    }
    offsets = off_s;
  }
  if (tok >= T) {
    if constexpr (PLAN) __syncthreads();
    // pad-row zero fill: block (T + e*PARTS + part) zeroes part of expert e's pad rows
    const int idx = tok - T, e = idx / MOE_PADFILL_PARTS, part = idx % MOE_PADFILL_PARTS;
    const int c = counts[e];
    const int full = c >> 8, r = c & 255;
    int nr, nb;
    rem_tiles(r, fine, nr, nb);
    // remainder tile(s): pad rows [used, width) of each; tile layout: 64 stage blocks of width * 128 B
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int n = q ? nb : nr;
      if (n == 0) continue;
      const int r0 = offsets[e] + 256 * full + (q ? nr : 0);
      const int used = q ? r - nr : (r < nr ? r : nr);
      for (int lr = used + part; lr < n; lr += MOE_PADFILL_PARTS) {
        for (int i = tid; i < 64 * 8; i += 256) {
          const int stg = i >> 3, ch = i & 7;
          *reinterpret_cast<uint4*>(xt + (size_t)r0 * 8192 + (size_t)stg * n * 128 + lr * 128 + ch * 16) =
              make_uint4(0, 0, 0, 0);
        }
      }
    }
    return;
  }
  // thread handles elements [16*tid, 16*tid+16): stage tid/4, chunks 2*(tid%4), 2*(tid%4)+1
  const uint4* src = reinterpret_cast<const uint4*>(p + (size_t)tok * 4096 + tid * 16);
  uint4 v0 = src[0], v1 = src[1];
  // the token's routes (row base, tile rows, row in tile), worked out by 8 threads while the row load is in flight
  __shared__ long long mrow[8];
  __shared__ int mn[8], mlr[8];
  if constexpr (PLAN) __syncwarp();
  if (tid < 8) {
    const int k = tid;
    const int l = lid[tok * 8 + k];
    if (l < 0) {
      rrow[tok * 8 + k] = -1;
      mn[k] = 0;
    } else {
      const int ps = pos[tok * 8 + k];
      const int c = counts[l];
      const int full = c >> 8, r = c & 255;
      int nr, nb;
      rem_tiles(r, fine, nr, nb);
      const int tl = ps >> 8;
      int row0 = offsets[l] + 256 * tl, lr = ps & 255, n = tl < full ? 256 : nr;
      if (tl >= full && nb && lr >= nr) {  // a split remainder's second tile
        row0 += nr;
        lr -= nr;
        n = nb;
      }
      rrow[tok * 8 + k] = row0 + lr;
      if (row_tok != nullptr) row_tok[row0 + lr] = tok;  // CIT: which token a GEMM2 row belongs to
      mrow[k] = (long long)row0 * 8192 + lr * 128;
      mn[k] = n;
      mlr[k] = lr;
    }
    if (cq != nullptr) {  // CIT: a token without local routes is ready now (its combine writes zeros)
      const unsigned lm = __ballot_sync(0xffu, l >= 0);
      if (k == 0 && lm == 0) {
        const int i = atomicAdd(qtail, 1);
        asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(cq + i), "r"(tok + 1) : "memory");
      }
    }
  }
  float f[16];
  const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
  const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 a = __bfloat1622float2(h0[i]), b = __bfloat1622float2(h1[i]);
    f[2 * i] = a.x; f[2 * i + 1] = a.y; f[8 + 2 * i] = b.x; f[8 + 2 * i + 1] = b.y;
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) ss += f[i] * f[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) red[tid >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float rr = rsqrtf(tot / 4096.f + 1e-5f);
  uint32_t outw[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#if MOE_GATHER_GPRE
    const float4 gg = gpre[i];
#else
    const float4 gg = g1p4(gw + tid * 16 + 4 * i);
#endif
    __nv_bfloat162 a = __floats2bfloat162_rn(f[4 * i] * rr * gg.x, f[4 * i + 1] * rr * gg.y);
    __nv_bfloat162 b = __floats2bfloat162_rn(f[4 * i + 2] * rr * gg.z, f[4 * i + 3] * rr * gg.w);
    outw[2 * i] = bf162_u32(a);
    outw[2 * i + 1] = bf162_u32(b);
  }
  const int stg = tid >> 2, ch0 = 2 * (tid & 3);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int n = mn[k];  // (written before the reduction's __syncthreads)
    if (n == 0) continue;
    const int lr = mlr[k];
    uint8_t* rowp = xt + mrow[k] + (size_t)stg * n * 128;
    const int x = lr & 7;
    *reinterpret_cast<uint4*>(rowp + ((ch0 ^ x) << 4)) = make_uint4(outw[0], outw[1], outw[2], outw[3]);
    *reinterpret_cast<uint4*>(rowp + (((ch0 + 1) ^ x) << 4)) = make_uint4(outw[4], outw[5], outw[6], outw[7]);
  }
}

// ---- routing: top-8 experts per token from the router logits (hi + lo halves), one warp per token ----
// Same math and tie-break as the Triton reference version (sigmoid(logit / sqrt(32)), + bias for selection, the
// highest biased score first, lowest expert index on ties; weights = selected scores / their sum), but each lane
// holds 8 experts and the 8 rounds of arg-max are warp shuffles (no block barriers). Local routes (experts
// local_start .. +31) get a row position from a per-expert counter: positions are first taken from block-local
// shared-memory counters, then each block reserves its range with one global atomic per expert (hot experts take
// thousands of routes, and per-route global atomics on one address serialize).
#ifndef ROUTE_WARPS
#define ROUTE_WARPS 16  // v69: 16 measured +0.06% over 32
#endif
__global__ void __launch_bounds__(ROUTE_WARPS * 32) moe_route_kernel(const float* __restrict__ logits,
                                                                     const float* __restrict__ bias,
                                                                     const int* __restrict__ lstart, int* __restrict__ ids,
                                                                     float* __restrict__ wout, int* __restrict__ lid,
                                                                     int* __restrict__ pos, int* __restrict__ cnt, int T,
                                                                     int* __restrict__ tok_left = nullptr, int* __restrict__ cq = nullptr) {
  pdl_enter();
  __shared__ int s_cnt[32], s_base[32];
  const int t = blockIdx.x * ROUTE_WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (threadIdx.x < 32) s_cnt[threadIdx.x] = 0;
  __syncthreads();
  int e_mine = 0, loc = 0, lpos = 0;
  bool local = false;
  float w_mine = 0.f;
  if (t < T) {
    const float* lg = logits + (size_t)t * 512;
    float sv[8], bv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = lane + 32 * j;
      const float x = (lg[e] + lg[256 + e]) * 0.1767766952966369f;
      sv[j] = 1.f / (1.f + expf(-x));
      bv[j] = sv[j] + bias[e];
    }
    int sel_e[8];
    float sel_s[8];
#ifndef ROUTE_REDUX
#define ROUTE_REDUX 1  // (Oct 26: bit-identical, route 7.6 -> 6.4 us at 753) 1: warp argmax by two REDUX reductions (max of order-preserving keys, then the lowest expert)
#endif
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bvv = bv[0];
      int be = lane;
#pragma unroll
      for (int j = 1; j < 8; ++j) {  // ascending expert index within the lane: strict > keeps the lowest on ties
        if (bv[j] > bvv) { bvv = bv[j]; be = lane + 32 * j; }
      }
#if ROUTE_REDUX
      {
        const uint32_t u = __float_as_uint(bvv);
        const uint32_t key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // unsigned order == float order
        const uint32_t mk = __reduce_max_sync(0xffffffffu, key);
        be = (int)__reduce_min_sync(0xffffffffu, key == mk ? (uint32_t)be : 0xffffffffu);  // lowest expert on ties
      }
#else
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bvv, off);
        const int oe = __shfl_xor_sync(0xffffffffu, be, off);
        if (ov > bvv || (ov == bvv && oe < be)) { bvv = ov; be = oe; }
      }
#endif
      float mys = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j == (be >> 5)) mys = sv[j];
      }
      sel_s[k] = __shfl_sync(0xffffffffu, mys, be & 31);
      sel_e[k] = be;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (lane + 32 * j == be) bv[j] = -INFINITY;
      }
    }
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) sum += sel_s[k];
    e_mine = sel_e[0];
    float sk = sel_s[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) {
      if (lane == k) { e_mine = sel_e[k]; sk = sel_s[k]; }
    }
    w_mine = sk / (sum + 1e-15f);
    loc = (e_mine - *lstart + 256) % 256;
    local = lane < 8 && loc < 32;
    if (local) lpos = atomicAdd(&s_cnt[loc], 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int c = s_cnt[threadIdx.x];
    s_base[threadIdx.x] = c ? atomicAdd(cnt + threadIdx.x, c) : 0;
  }
  __syncthreads();
  if (t < T && tok_left != nullptr) {  // CIT: the token's local route count, and its combine-queue slot cleared
    const unsigned lm = __ballot_sync(0xffffffffu, local);
    if (lane == 0) {
      tok_left[t] = __popc(lm);
      cq[t] = 0;
    }
  }
  if (t < T && lane < 8) {
    ids[t * 8 + lane] = e_mine;
    wout[t * 8 + lane] = w_mine;
    lid[t * 8 + lane] = local ? loc : -1;
    pos[t * 8 + lane] = local ? s_base[loc] + lpos : 0;
  }
}

// ---- combine: out[t] = sum over t's local routes of norm(y[row] * pw) * w, one block of 256 threads per token ----
// (the post-expert norm: scale by pw = 1 + w first, then RMSNorm with eps 1e-8). The token's local routes (about two of
// its eight) are compacted first; their sums of squares share one block reduction, and the second pass re-reads each
// route row from L1/L2. Few registers, so several blocks per SM keep enough loads in flight.
__global__ void __launch_bounds__(256) moe_combine_kernel(const __nv_bfloat16* __restrict__ y, const int* __restrict__ rrow,
                                                          const int* __restrict__ lid, const float* __restrict__ topw,
                                                          const __nv_bfloat16* __restrict__ pw, __nv_bfloat16* __restrict__ out) {
  // one token per block, 16 elements per thread: each local route's row post-normed on its own (rsqrt of its mean
  // square + 1e-8, times g_post_expert_norm), weighted by its routing weight, summed (as the Triton _combine_kernel);
  // up to 4 routes' rows in flight at once, one block reduction per group of 4
  pdl_enter();
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float red[4][8];
  int rows[8];
  float wts[8];
  int nl = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int l = lid[tok * 8 + k];
    if (l >= 0) {
      rows[nl] = rrow[tok * 8 + k];
      wts[nl] = topw[tok * 8 + k];
      ++nl;
    }
  }
  float g[16];
  {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 q = g1p4(pw + tid * 16 + 4 * i);
      g[4 * i] = q.x; g[4 * i + 1] = q.y; g[4 * i + 2] = q.z; g[4 * i + 3] = q.w;
    }
  }
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#ifndef COMB_GROUP
#define COMB_GROUP 2
#endif
  for (int k0 = 0; k0 < nl; k0 += COMB_GROUP) {
    float x[COMB_GROUP][16];
    float ss[COMB_GROUP];
#pragma unroll
    for (int u = 0; u < COMB_GROUP; ++u) {
      ss[u] = 0.f;
      if (k0 + u < nl) {
        const uint4* src = reinterpret_cast<const uint4*>(y + (size_t)rows[k0 + u] * 4096 + tid * 16);
        const uint4 v0 = src[0], v1 = src[1];
        const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
        const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 f = __bfloat1622float2(h0[i]), q = __bfloat1622float2(h1[i]);
          x[u][2 * i] = f.x * g[2 * i]; x[u][2 * i + 1] = f.y * g[2 * i + 1];
          x[u][8 + 2 * i] = q.x * g[8 + 2 * i]; x[u][8 + 2 * i + 1] = q.y * g[8 + 2 * i + 1];
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) ss[u] += x[u][i] * x[u][i];
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) x[u][i] = 0.f;
      }
    }
#pragma unroll
    for (int u = 0; u < COMB_GROUP; ++u) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss[u] += __shfl_xor_sync(0xffffffffu, ss[u], o);
    }
    if (k0) __syncthreads();  // the previous group's red[] has been read
    if (lane == 0) {
#pragma unroll
      for (int u = 0; u < COMB_GROUP; ++u) red[u][warp] = ss[u];
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < COMB_GROUP; ++u) {
      if (k0 + u < nl) {
        float tot = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) tot += red[u][w];
        const float sc = rsqrtf(tot / 4096.f + 1e-8f) * wts[k0 + u];
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] += x[u][i] * sc;
      }
    }
  }
  uint4 o0, o1;
  __nv_bfloat162* b0 = reinterpret_cast<__nv_bfloat162*>(&o0);
  __nv_bfloat162* b1 = reinterpret_cast<__nv_bfloat162*>(&o1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    b0[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    b1[i] = __floats2bfloat162_rn(acc[8 + 2 * i], acc[8 + 2 * i + 1]);
  }
  uint4* dst = reinterpret_cast<uint4*>(out + (size_t)tok * 4096 + tid * 16);
  dst[0] = o0;
  dst[1] = o1;
}

}  // namespace okc_moe

namespace okc_moe {
// ---- residual boundary: s_out = alpha*s + beta*rmsnorm(branch*gpost, 1e-8); n_out = bf16(rmsnorm(s_out, 1e-5)*gpre) ----
// Layer-end boundary that recomputes the attention boundary's residual stream (alpha1 * stream + beta1 * norm(a)) instead
// of storing it in fp32 and reading it back: per row, a (bf16) + stream (fp32) + f (bf16) in, stream2 (fp32) + hidden
// (bf16) out. 512 threads x 16 elements per row of 8192.
__global__ void __launch_bounds__(512) boundary2_kernel(const __nv_bfloat16* __restrict__ a, const float* __restrict__ stream,
                                                        const __nv_bfloat16* __restrict__ f, const float* __restrict__ a1,
                                                        const float* __restrict__ b1, const __nv_bfloat16* __restrict__ gpost,
                                                        const float* __restrict__ a2, const float* __restrict__ b2,
                                                        const __nv_bfloat16* __restrict__ gnext, float* __restrict__ sout,
                                                        __nv_bfloat16* __restrict__ nout) {
  constexpr int H = 8192;
  const size_t row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float red[3][16];
  float av[16], sv[16], fv[16];
  const __nv_bfloat16* ar = a + row * H;
  const __nv_bfloat16* fr = f + row * H;
  const float* sr = stream + row * H;
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int off = c * 4096 + tid * 8;
    const uint4 va = __ldcs(reinterpret_cast<const uint4*>(ar + off));
    const uint4 vf = __ldcs(reinterpret_cast<const uint4*>(fr + off));
    const float4 s0 = __ldcs(reinterpret_cast<const float4*>(sr + off));
    const float4 s1 = __ldcs(reinterpret_cast<const float4*>(sr + off + 4));
    const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&va);
    const __nv_bfloat162* hf = reinterpret_cast<const __nv_bfloat162*>(&vf);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 x = __bfloat1622float2(ha[i]);
      const float2 y = __bfloat1622float2(hf[i]);
      av[c * 8 + 2 * i] = x.x; av[c * 8 + 2 * i + 1] = x.y;
      fv[c * 8 + 2 * i] = y.x; fv[c * 8 + 2 * i + 1] = y.y;
    }
    sv[c * 8 + 0] = s0.x; sv[c * 8 + 1] = s0.y; sv[c * 8 + 2] = s0.z; sv[c * 8 + 3] = s0.w;
    sv[c * 8 + 4] = s1.x; sv[c * 8 + 5] = s1.y; sv[c * 8 + 6] = s1.z; sv[c * 8 + 7] = s1.w;
  }
  float ssa = 0.f, ssb = 0.f;
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int off = c * 4096 + tid * 8;
    const float4 g0 = g1p4(gpost + off);
    const float4 g1 = g1p4(gpost + off + 4);
    const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      fv[c * 8 + i] *= gv[i];
      ssa += av[c * 8 + i] * av[c * 8 + i];
      ssb += fv[c * 8 + i] * fv[c * 8 + i];
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ssa += __shfl_xor_sync(0xffffffffu, ssa, o);
    ssb += __shfl_xor_sync(0xffffffffu, ssb, o);
  }
  if ((tid & 31) == 0) { red[0][tid >> 5] = ssa; red[1][tid >> 5] = ssb; }
  __syncthreads();
  float ta = 0.f, tb = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) { ta += red[0][i]; tb += red[1][i]; }
  const float ra = rsqrtf(ta / H + 1e-8f), rb = rsqrtf(tb / H + 1e-8f);
  float ss2 = 0.f;
  float* so = sout + row * H;
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int off = c * 4096 + tid * 8;
    const float4 p0 = *reinterpret_cast<const float4*>(a1 + off), p1 = *reinterpret_cast<const float4*>(a1 + off + 4);
    const float4 q0 = *reinterpret_cast<const float4*>(b1 + off), q1 = *reinterpret_cast<const float4*>(b1 + off + 4);
    const float4 u0 = *reinterpret_cast<const float4*>(a2 + off), u1 = *reinterpret_cast<const float4*>(a2 + off + 4);
    const float4 v0 = *reinterpret_cast<const float4*>(b2 + off), v1 = *reinterpret_cast<const float4*>(b2 + off + 4);
    const float pa[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
    const float qa[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    const float ua[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    const float va[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float s1v = pa[i] * sv[c * 8 + i] + qa[i] * (av[c * 8 + i] * ra);
      const float x = ua[i] * s1v + va[i] * (fv[c * 8 + i] * rb);
      sv[c * 8 + i] = x;
      ss2 += x * x;
    }
    __stcs(reinterpret_cast<float4*>(so + off), make_float4(sv[c * 8 + 0], sv[c * 8 + 1], sv[c * 8 + 2], sv[c * 8 + 3]));
    __stcs(reinterpret_cast<float4*>(so + off + 4), make_float4(sv[c * 8 + 4], sv[c * 8 + 5], sv[c * 8 + 6], sv[c * 8 + 7]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss2 += __shfl_xor_sync(0xffffffffu, ss2, o);
  if ((tid & 31) == 0) red[2][tid >> 5] = ss2;
  __syncthreads();
  float t2 = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) t2 += red[2][i];
  const float r2 = rsqrtf(t2 / H + 1e-5f);
  __nv_bfloat16* no = nout + row * H;
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int off = c * 4096 + tid * 8;
    const float4 g0 = g1p4(gnext + off), g1 = g1p4(gnext + off + 4);
    const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    uint4 o;
    __nv_bfloat162* ho = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      ho[i] = __floats2bfloat162_rn(sv[c * 8 + 2 * i] * r2 * gv[2 * i], sv[c * 8 + 2 * i + 1] * r2 * gv[2 * i + 1]);
    __stcs(reinterpret_cast<uint4*>(no + off), o);
  }
}

// ---- q/k prep, CUDA version of the Triton _qk_prep_kernel (QK_PREP_CUDA): one block of 128 threads per token.
// Per-head RMSNorm (eps 1e-5) of q (16 heads) and k (2 heads), RoPE on interleaved (even, odd) pairs with the angle
// pos * inv_freq[i] rounded to fp32 as in the Triton kernel (cos/sin evaluated in double, then rounded), e4m3 (RNE,
// satfinite) q rows and k/v cache rows (k and v divided by their scales when SCALED). Thread layout: q head tid/8, dims
// 8*(tid%8)..+7; threads 0..15 also do the k heads (head tid/8), threads 16..31 the v row (dims 8*(tid-16)..+7).
DEVI uint32_t qk_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI uint2 qk_pack8(const float (&x)[8]) {
  return make_uint2(qk_e4m3x2(x[0], x[1]) | (qk_e4m3x2(x[2], x[3]) << 16), qk_e4m3x2(x[4], x[5]) | (qk_e4m3x2(x[6], x[7]) << 16));
}
DEVI void qk_load8(const __nv_bfloat16* p, float (&x)[8]) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) { const float2 f = __bfloat1622float2(h[i]); x[2 * i] = f.x; x[2 * i + 1] = f.y; }
}
template <bool ROPE>
DEVI void qk_norm_rope8(float (&x)[8], const float* cs, const float* sn, int d0) {
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += x[i] * x[i];
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  ss += __shfl_xor_sync(0xffffffffu, ss, 4);
  const float r = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] *= r;
  if constexpr (ROPE) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float c = cs[d0 / 2 + j], s = sn[d0 / 2 + j];
      const float e = x[2 * j], o = x[2 * j + 1];
      x[2 * j] = e * c - o * s;
      x[2 * j + 1] = e * s + o * c;
    }
  }
}
template <bool ROPE, bool SCALED, typename PT, typename LT>
__global__ void __launch_bounds__(128) qk_prep_kernel(const __nv_bfloat16* __restrict__ qkv, const PT* __restrict__ pos,
                                                      const LT* __restrict__ loc, const float* __restrict__ invf,
                                                      uint8_t* __restrict__ q8, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                                      const float* __restrict__ ksc, const float* __restrict__ vsc) {
  __shared__ float s_cos[32], s_sin[32];
  const int t = blockIdx.x, tid = threadIdx.x;
  float pf = 0.f;
  if (ROPE && tid < 32) pf = (float)pos[t];  // positions / cache rows are inputs, not the predecessor's results
  const long long lrow = tid < 32 ? (long long)loc[t] : 0;
  pdl_enter();
  if constexpr (ROPE) {
    if (tid < 32) {
      const float ang = __fmul_rn(pf, invf[tid]);
      double sd, cd;
      sincos((double)ang, &sd, &cd);
      s_cos[tid] = (float)cd;
      s_sin[tid] = (float)sd;
    }
    __syncthreads();
  }
  const __nv_bfloat16* row = qkv + (size_t)t * 2304;
  const int d0 = (tid & 7) * 8;
  {
    float x[8];
    qk_load8(row + (tid >> 3) * 64 + d0, x);
    qk_norm_rope8<ROPE>(x, s_cos, s_sin, d0);
    *reinterpret_cast<uint2*>(q8 + (size_t)t * 1024 + (tid >> 3) * 64 + d0) = qk_pack8(x);
  }
  if (tid < 16) {  // k heads 0, 1 (the 8-lane groups of the reduction stay within one head)
    float x[8];
    qk_load8(row + 1024 + (tid >> 3) * 64 + d0, x);
    qk_norm_rope8<ROPE>(x, s_cos, s_sin, d0);
    if constexpr (SCALED) {
      const float ks = *ksc;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], ks);
    }
    *reinterpret_cast<uint2*>(kc + lrow * 128 + (tid >> 3) * 64 + d0) = qk_pack8(x);
  } else if (tid < 32) {  // v: no norm
    float x[8];
    const int dv = (tid - 16) * 8;
    qk_load8(row + 1152 + dv, x);
    if constexpr (SCALED) {
      const float vs = *vsc;
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __fdiv_rn(x[i], vs);
    }
    *reinterpret_cast<uint2*>(vc + lrow * 128 + dv) = qk_pack8(x);
  }
}

// Residual boundary, one block per token row: stream = alpha * stream + beta * rms(branch [* gpost]); h = rms(stream)
// * gpre in bf16. TPB threads x 8 elements per chunk; used from 2048 tokens up (the Triton version wins below).
// MODE 0: stream' = alpha * stream + beta * rmsnorm(branch [* gpost]); writes stream' (fp32) and its rmsnorm * gpre (bf16).
// MODE 1 (first boundary, recompute scheme): the same, but stream' is not written; the branch's rms factor r1 goes to
//   rio[row] instead (4 bytes a row instead of 32 KB).
// MODE 2 (second boundary, recompute scheme): `stream` is the layer's input stream; the first boundary's stream' is
//   rebuilt on the fly as alpha1 * stream + beta1 * (br1 * rio[row]) (the same fp32 expression as MODE 1) instead of
//   being read back: 16 KB of bf16 branch read instead of 32 KB of fp32 stream written and read.
#ifndef BND_RC_MINB
#define BND_RC_MINB 2  // MODE 2 at 512 threads compiled to 74 registers (1 block/SM, half the bytes in flight)
#endif
template <int TPB, bool GPOST, bool STREAMHINT, int MODE = 0>
__global__ void __launch_bounds__(TPB, MODE == 2 ? BND_RC_MINB : (TPB <= 512 ? 2 : 1)) boundary_kernel(const __nv_bfloat16* __restrict__ branch, const float* __restrict__ stream,
                                           const __nv_bfloat16* __restrict__ gpost, const float* __restrict__ alpha,
                                           const float* __restrict__ beta, const __nv_bfloat16* __restrict__ gpre,
                                           float* __restrict__ sout, __nv_bfloat16* __restrict__ nout, int* __restrict__ zp = nullptr, int zn = 0,
                                           const __nv_bfloat16* __restrict__ br1 = nullptr, float* __restrict__ rio = nullptr,
                                           const float* __restrict__ alpha1 = nullptr, const float* __restrict__ beta1 = nullptr) {
  pdl_enter();
  // block 0 also zeroes the MoE's per-call counters (counts | work counter | tile done): saves a fill kernel
  if (zn > 0 && blockIdx.x == 0)
    for (int i = threadIdx.x; i < zn; i += TPB) zp[i] = 0;
  constexpr int H = 8192, C = H / (TPB * 8), NW = TPB / 32;
  const size_t row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float red[2][NW];
  float b[C * 8], s[C * 8];
  const __nv_bfloat16* br = branch + row * H;
  const float* sr = stream + row * H;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    uint4 v; float4 s0, s1;
    if (STREAMHINT) { v = __ldcs(reinterpret_cast<const uint4*>(br + off)); s0 = __ldcs(reinterpret_cast<const float4*>(sr + off)); s1 = __ldcs(reinterpret_cast<const float4*>(sr + off + 4)); }
    else { v = *reinterpret_cast<const uint4*>(br + off); s0 = *reinterpret_cast<const float4*>(sr + off); s1 = *reinterpret_cast<const float4*>(sr + off + 4); }
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) { float2 f = __bfloat1622float2(h[i]); b[c * 8 + 2 * i] = f.x; b[c * 8 + 2 * i + 1] = f.y; }
    s[c * 8 + 0] = s0.x; s[c * 8 + 1] = s0.y; s[c * 8 + 2] = s0.z; s[c * 8 + 3] = s0.w;
    s[c * 8 + 4] = s1.x; s[c * 8 + 5] = s1.y; s[c * 8 + 6] = s1.z; s[c * 8 + 7] = s1.w;
    if constexpr (MODE == 2) {
      const uint4 u = *reinterpret_cast<const uint4*>(br1 + row * H + off);
      const __nv_bfloat162* hu = reinterpret_cast<const __nv_bfloat162*>(&u);
      const float r1p = rio[row];
      const float4 a0 = __ldg(reinterpret_cast<const float4*>(alpha1 + off)), a1 = __ldg(reinterpret_cast<const float4*>(alpha1 + off + 4));
      const float4 e0 = __ldg(reinterpret_cast<const float4*>(beta1 + off)), e1 = __ldg(reinterpret_cast<const float4*>(beta1 + off + 4));
      const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float bv[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(hu[i]);
        s[c * 8 + 2 * i] = __fmaf_rn(av[2 * i], s[c * 8 + 2 * i], bv[2 * i] * (f.x * r1p));
        s[c * 8 + 2 * i + 1] = __fmaf_rn(av[2 * i + 1], s[c * 8 + 2 * i + 1], bv[2 * i + 1] * (f.y * r1p));
      }
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    if constexpr (GPOST) {
      const float4 g0 = g1p4(gpost + off), g1 = g1p4(gpost + off + 4);
      b[c * 8 + 0] *= g0.x; b[c * 8 + 1] *= g0.y; b[c * 8 + 2] *= g0.z; b[c * 8 + 3] *= g0.w;
      b[c * 8 + 4] *= g1.x; b[c * 8 + 5] *= g1.y; b[c * 8 + 6] *= g1.z; b[c * 8 + 7] *= g1.w;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += b[c * 8 + i] * b[c * 8 + i];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) red[0][tid >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) tot += red[0][i];
  const float r1 = rsqrtf(tot / H + 1e-8f);
  if constexpr (MODE == 1) {
    if (tid == 0) rio[row] = r1;
  }
  float ss2 = 0.f;
  float* so = sout + row * H;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    const float4 a0 = __ldg(reinterpret_cast<const float4*>(alpha + off)), a1 = __ldg(reinterpret_cast<const float4*>(alpha + off + 4));
    const float4 e0 = __ldg(reinterpret_cast<const float4*>(beta + off)), e1 = __ldg(reinterpret_cast<const float4*>(beta + off + 4));
    const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bv[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (MODE == 0) s[c * 8 + i] = av[i] * s[c * 8 + i] + bv[i] * (b[c * 8 + i] * r1);
      else s[c * 8 + i] = __fmaf_rn(av[i], s[c * 8 + i], bv[i] * (b[c * 8 + i] * r1));  // MODE 2 rebuilds it identically
      ss2 += s[c * 8 + i] * s[c * 8 + i];
    }
    if constexpr (MODE != 1) {
      float4 o0 = make_float4(s[c * 8 + 0], s[c * 8 + 1], s[c * 8 + 2], s[c * 8 + 3]), o1 = make_float4(s[c * 8 + 4], s[c * 8 + 5], s[c * 8 + 6], s[c * 8 + 7]);
      if (STREAMHINT) { __stcs(reinterpret_cast<float4*>(so + off), o0); __stcs(reinterpret_cast<float4*>(so + off + 4), o1); }
      else { *reinterpret_cast<float4*>(so + off) = o0; *reinterpret_cast<float4*>(so + off + 4) = o1; }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss2 += __shfl_xor_sync(0xffffffffu, ss2, o);
  if ((tid & 31) == 0) red[1][tid >> 5] = ss2;
  __syncthreads();
  float tot2 = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) tot2 += red[1][i];
  const float r2 = rsqrtf(tot2 / H + 1e-5f);
  __nv_bfloat16* no = nout + row * H;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int off = c * TPB * 8 + tid * 8;
    const float4 g0 = g1p4(gpre + off), g1 = g1p4(gpre + off + 4);
    const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    uint4 o;
    __nv_bfloat162* ho = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) ho[i] = __floats2bfloat162_rn(s[c * 8 + 2 * i] * r2 * gv[2 * i], s[c * 8 + 2 * i + 1] * r2 * gv[2 * i + 1]);
    if (STREAMHINT) __stcs(reinterpret_cast<uint4*>(no + off), o); else *reinterpret_cast<uint4*>(no + off) = o;
  }
}
}  // namespace okc_moe

namespace okc_moe {
// ---- gather the attended keys/values of each request into contiguous [B, Smax, 2, 64] FP8 buffers ----
// keys at positions [first_b, seq_b + L) where first_b = window ? max(0, seq_b - window + 1) : 0; slot j <- position first_b + j
__global__ void __launch_bounds__(256) kv_gather_kernel(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                                        const int* __restrict__ page_table, const int* __restrict__ seq_lens,
                                                        uint8_t* __restrict__ ko, uint8_t* __restrict__ vo, int max_pages,
                                                        int page, int L, int window, int Smax) {
  const int b = blockIdx.y;
  const int seq = seq_lens[b];
  const int first = window ? max(0, seq - window + 1) : 0;
  const int n = seq + L - first;
  const int j = blockIdx.x * 32 + (threadIdx.x >> 3);  // 32 rows per block, 8 threads x 16 B per row (K), same for V
  const int c = threadIdx.x & 7;
  if (j >= n) return;
  const int pos = first + j;
  const long row = (long)page_table[(long)b * max_pages + pos / page] * page + pos % page;
  const uint4 kv = *reinterpret_cast<const uint4*>(kc + row * 128 + c * 16);
  const uint4 vv = *reinterpret_cast<const uint4*>(vc + row * 128 + c * 16);
  *reinterpret_cast<uint4*>(ko + ((long)b * Smax + j) * 128 + c * 16) = kv;
  *reinterpret_cast<uint4*>(vo + ((long)b * Smax + j) * 128 + c * 16) = vv;
}
}  // namespace okc_moe

// FP8 prefill attention for Muse Spark (hdim 64, 16 q heads / 2 kv heads), fused output RMSNorm * sigmoid(gate).
// Appended after moe_gemm.cu (uses its helpers).
//
// attn_prep_kernel gathers each request's attended keys/values from the paged cache into per-(request, kv head,
// 128-key tile) blocks: K tiles in the 64B-swizzled K-major layout; V tiles transposed (72 rows x 128 keys, 128B
// swizzle) with the keys permuted inside each 32-key chunk so that the QK^T accumulator registers are directly the
// FP8 A operand of the P.V wgmma, plus a row of ones (row 64) so the tensor core also sums P (the softmax
// denominator, exactly over the FP8-rounded P that multiplies V).
//
// attn_fwd_kernel: one CTA = 16 query tokens x the 8 query heads of one kv head (128 rows); 2 consumer warpgroups
// (64 rows each) + 1 producer warp streaming K/V tiles with bulk copies into a 4-stage ring. Per tile and
// warpgroup: S = Q K^T (FP8 wgmma) -> mask -> online softmax (P scaled by 2^8 like FA3) -> P.V (FP8 wgmma).
// EXPMODE n >= 0: n of each thread's 64 exponentials per tile as an FMA-pipe polynomial, the rest on the MUFU;
// -1: exp2 on f16 pairs converted straight to FP8 pairs. P is rounded to E4M3 for the P.V MMA in every mode.
DEVI void wgmma_fp8_ss_n128(float* d, uint64_t da, uint64_t db, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "l"(da), "l"(db), "r"(scale_d));
#endif
}

DEVI void wgmma_fp8_rs_n64(float* d, const uint32_t* a, uint64_t db, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(scale_d));
#endif
}

DEVI void wgmma_fp8_rs_n72(float* d, const uint32_t* a, uint64_t db, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, {%36, %37, %38, %39}, %40, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(scale_d));
#endif
}


DEVI void wgmma_fp8_ss_n64(float* d, uint64_t da, uint64_t db, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "l"(da), "l"(db), "r"(scale_d));
#endif
}

#ifndef ATTN_BISECT
#define ATTN_BISECT 0
#endif

namespace okc_attn {
using namespace okc_moe;

constexpr int NK = 128;                   // keys per tile
constexpr int TILE_K = NK * 64;           // 8 KB
constexpr int VROWS = 72;                 // 64 head dims + the ones row (+ 7 zero rows)
constexpr int TILE_V = VROWS * NK;        // 9 KB
constexpr int STAGE = TILE_K + TILE_V;    // 17 KB (1024-aligned)
#ifndef ATTN_ASTAGES
#define ATTN_ASTAGES 4
#endif
constexpr int ASTAGES = ATTN_ASTAGES;
#ifndef ATTN_VOTEMAX
#define ATTN_VOTEMAX 1  // exp-turn softmax: skip the row-max shuffles and rescale exps when a warp's maxima did not grow
#endif
#ifndef ATTN_VOTEMAX_ALL
#define ATTN_VOTEMAX_ALL 0  // the same in the other softmax paths (slower for the sliding-window variant 8)
#endif
#ifndef ATTN_GATE_PF
#define ATTN_GATE_PF 1
#endif
#ifndef ATTN_RR_HAND2
#define ATTN_RR_HAND2 48  // early hand-off point (exps per thread) of the non-pipelined RR step (variants 41/42: slower than 8 on sliding)
#endif

#ifndef ATTN_PREP_PRMT
#define ATTN_PREP_PRMT 1  // attn_prep2's V transpose by register 4x4 byte transposes (PRMT) instead of byte loads
#endif
DEVI int sigma32(int s) {
  const int t = (s & 15) >> 2, i = s & 3;
  return (s & 16) + ((i < 2) ? (2 * t + i) : (8 + 2 * t + i - 2));
}

// grid (ntk * HALVES, 2, B), block 256; HALVES = 2: each block stages one 64-key half of a tile (short windows: more
// blocks in flight)
// ---- programmatic dependent launch (OKC_PDL): every kernel lets its successor launch as soon as all of its blocks
// are running, and waits for its predecessor's results before touching global memory (a no-op when the kernel was not
// launched with the PDL attribute) ----
#ifndef OKC_PDL
#define OKC_PDL 0
#endif
__device__ __forceinline__ void pdl_enter_attn() {
#if OKC_PDL && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger_attn() {
#if OKC_PDL && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_wait_attn() {
#if OKC_PDL && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
// ATTN_EARLY_Q: the K/V staging kernels let the attention kernel launch only after their own wait (so qk_prep, which
// writes Q, is complete), and the attention kernel waits for the staged K/V in its producer only: barrier setup and the
// consumers' Q staging overlap the staging kernel.
#ifndef ATTN_EARLY_Q
#define ATTN_EARLY_Q 1  // (Oct 26: search +0.06..0.09% on v113; the staging kernel triggers only after its own wait, so Q is complete)
#endif
__device__ __forceinline__ void pdl_enter_prep() {
  if constexpr (ATTN_EARLY_Q) {
    pdl_wait_attn();
    pdl_trigger_attn();
  } else {
    pdl_enter_attn();
  }
}

template <int HALVES>
__global__ void __launch_bounds__(256) attn_prep_kernel(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                                        const int* __restrict__ page_table, int max_pages, int page,
                                                        const int* __restrict__ seq_lens, int L, int window, int ntk,
                                                        uint8_t* __restrict__ kt, uint8_t* __restrict__ vt, int part = 0) {
  pdl_enter_prep();
  const int tile = blockIdx.x / HALVES, hf = blockIdx.x % HALVES, h = blockIdx.y, b = blockIdx.z;
  constexpr int NKH = NK / HALVES;  // keys staged by this block
  const int seq = seq_lens[b];
  const int first = window ? max(0, seq - window + 1) : 0;
  const int nk = seq + L - first;
  if (tile * NK >= nk) return;
  // part 1: only tiles made entirely of keys already in the cache before this call (they need nothing from this
  // call's qk_prep, so they are staged on a side stream during the qkv GEMM); part 2: the rest; part 0: all
  if (part && ((part == 1) != (tile < (seq - first) / NK))) return;
#if ATTN_PREP_PRMT
  constexpr int VPS = 68;  // padded rows (17 words): the transposing LDS.32 reads of 4 keys hit different banks
#else
  constexpr int VPS = 64;
#endif
  __shared__ __align__(16) uint8_t vs[NK * VPS];
  const size_t tb = ((size_t)(b * 2 + h) * ntk + tile);
  uint8_t* ko = kt + tb * TILE_K;
  uint8_t* vo = vt + tb * TILE_V;
  for (int i = threadIdx.x; i < NKH * 4; i += 256) {
    const int r = hf * NKH + (i >> 2), c = i & 3;
    const int j = tile * NK + r;
    uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
    if (j < nk) {
      const int pos = first + j;
      const long row = (long)page_table[(long)b * max_pages + pos / page] * page + pos % page;
      kv = *reinterpret_cast<const uint4*>(kc + row * 128 + h * 64 + c * 16);
      vv = *reinterpret_cast<const uint4*>(vc + row * 128 + h * 64 + c * 16);
    }
    *reinterpret_cast<uint4*>(ko + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = kv;
#if ATTN_PREP_PRMT
    uint32_t* vw = reinterpret_cast<uint32_t*>(vs + r * VPS + c * 16);
    vw[0] = vv.x; vw[1] = vv.y; vw[2] = vv.z; vw[3] = vv.w;
#else
    *reinterpret_cast<uint4*>(vs + r * 64 + c * 16) = vv;
#endif
  }
  __syncthreads();
#if ATTN_PREP_PRMT
  // V^T by register 4x4 byte transposes (as attn_prep2): this block's chunks cc of 16 dim groups, then the ones and
  // zero rows
  for (int i = threadIdx.x; i < 16 * (8 / HALVES); i += 256) {
    const int dg = i / (8 / HALVES), cc = hf * (8 / HALVES) + i % (8 / HALVES);
    const int d0 = 4 * dg, kb = (cc >> 1) * 32 + (cc & 1) * 16;
    uint32_t w[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t* r0 = reinterpret_cast<const uint32_t*>(vs + (kb + 2 * q) * VPS + d0);
      const uint32_t a0 = r0[0], a1 = r0[VPS / 4], a2 = r0[8 * VPS / 4], a3 = r0[9 * VPS / 4];
      const uint32_t t0 = __byte_perm(a0, a1, 0x5140), t1 = __byte_perm(a2, a3, 0x5140);
      const uint32_t t2 = __byte_perm(a0, a1, 0x7362), t3 = __byte_perm(a2, a3, 0x7362);
      w[0][q] = __byte_perm(t0, t1, 0x5410);
      w[1][q] = __byte_perm(t0, t1, 0x7632);
      w[2][q] = __byte_perm(t2, t3, 0x5410);
      w[3][q] = __byte_perm(t2, t3, 0x7632);
    }
#pragma unroll
    for (int dd = 0; dd < 4; ++dd) {
      const int d = d0 + dd;
      *reinterpret_cast<uint4*>(vo + d * 128 + ((cc ^ (d & 7)) << 4)) = make_uint4(w[dd][0], w[dd][1], w[dd][2], w[dd][3]);
    }
  }
  for (int i = threadIdx.x; i < 8 * (8 / HALVES); i += 256) {
    const int d = 64 + i / (8 / HALVES), cc = hf * (8 / HALVES) + i % (8 / HALVES);
    const uint32_t one = d == 64 ? 0x38383838u : 0u;  // e4m3 1.0
    *reinterpret_cast<uint4*>(vo + d * 128 + ((cc ^ (d & 7)) << 4)) = make_uint4(one, one, one, one);
  }
}
#else
  for (int i = threadIdx.x; i < VROWS * (8 / HALVES); i += 256) {
    const int d = i / (8 / HALVES), cc = hf * (8 / HALVES) + i % (8 / HALVES);
    uint4 out;
    if (d < 64) {
      uint32_t w[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t v = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int s = cc * 16 + q * 4 + e;
          const int key = (s & ~31) + sigma32(s & 31);
          v |= (uint32_t)vs[key * 64 + d] << (8 * e);
        }
        w[q] = v;
      }
      out = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
      const uint32_t one = d == 64 ? 0x38383838u : 0u;  // e4m3 1.0
      out = make_uint4(one, one, one, one);
    }
    *reinterpret_cast<uint4*>(vo + d * 128 + ((cc ^ (d & 7)) << 4)) = out;
  }
}
#endif

// grid (ntk, 1, B), block 256: both heads of a 128-key tile per block (full 128-byte cache rows, one page-table read per
// key); V goes through shared memory with a 68-byte row pitch so the byte transpose reads are bank-conflict free
__global__ void __launch_bounds__(256) attn_prep2_kernel(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                                         const int* __restrict__ page_table, int max_pages, int page,
                                                         const int* __restrict__ seq_lens, int L, int window, int ntk,
                                                         uint8_t* __restrict__ kt, uint8_t* __restrict__ vt, int part = 0) {
  pdl_enter_prep();
  const int tile = blockIdx.x, b = blockIdx.z;
  const int seq = seq_lens[b];
  const int first = window ? max(0, seq - window + 1) : 0;
  const int nk = seq + L - first;
  if (tile * NK >= nk) return;
  if (part && ((part == 1) != (tile < (seq - first) / NK))) return;  // (see attn_prep_kernel)
  constexpr int VP = 68;
  __shared__ __align__(16) uint8_t vs[2][NK * VP];
  // 128 keys x 8 chunks of 16 bytes (chunks 0-3: head 0, 4-7: head 1)
  // (explicitly batching a thread's 4 page lookups and 8 row loads measured slower: 12.0 -> 13.4 us at turn-1 full)
  for (int i = threadIdx.x; i < NK * 8; i += 256) {
    const int r = i >> 3, c8 = i & 7, h = c8 >> 2, c = c8 & 3;
    const int j = tile * NK + r;
    uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
    if (j < nk) {
      const int pos = first + j;
      const long row = (long)page_table[(long)b * max_pages + pos / page] * page + pos % page;
      kv = *reinterpret_cast<const uint4*>(kc + row * 128 + c8 * 16);
      vv = *reinterpret_cast<const uint4*>(vc + row * 128 + c8 * 16);
    }
    uint8_t* ko = kt + ((size_t)(b * 2 + h) * ntk + tile) * TILE_K;
    *reinterpret_cast<uint4*>(ko + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = kv;
    uint32_t* vw = reinterpret_cast<uint32_t*>(vs[h] + r * VP + c * 16);  // 4-byte aligned (68 = 17 words)
    vw[0] = vv.x; vw[1] = vv.y; vw[2] = vv.z; vw[3] = vv.w;
  }
  __syncthreads();
#if ATTN_PREP_PRMT
  // V^T by 4x4 byte transposes in registers: thread (h, dims 4dg..4dg+3, chunk cc) reads, for each of the chunk's 4
  // words q, the 4 keys' 4-dim words (LDS.32) and transposes them with 8 PRMTs into 4 dims' words of 4 keys
  {
    const int h = threadIdx.x >> 7, dg = (threadIdx.x >> 3) & 15, cc = threadIdx.x & 7;
    const int d0 = 4 * dg;
    const int kb = (cc >> 1) * 32 + (cc & 1) * 16;
    uint32_t w[4][4];  // [dim dd][word q]
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t* r0 = reinterpret_cast<const uint32_t*>(vs[h] + (kb + 2 * q) * VP + d0);
      const uint32_t a0 = r0[0], a1 = r0[VP / 4], a2 = r0[8 * VP / 4], a3 = r0[9 * VP / 4];  // keys 2q, 2q+1, 2q+8, 2q+9
      const uint32_t t0 = __byte_perm(a0, a1, 0x5140), t1 = __byte_perm(a2, a3, 0x5140);
      const uint32_t t2 = __byte_perm(a0, a1, 0x7362), t3 = __byte_perm(a2, a3, 0x7362);
      w[0][q] = __byte_perm(t0, t1, 0x5410);
      w[1][q] = __byte_perm(t0, t1, 0x7632);
      w[2][q] = __byte_perm(t2, t3, 0x5410);
      w[3][q] = __byte_perm(t2, t3, 0x7632);
    }
    uint8_t* vo = vt + ((size_t)(b * 2 + h) * ntk + tile) * TILE_V;
#pragma unroll
    for (int dd = 0; dd < 4; ++dd) {
      const int d = d0 + dd;
      *reinterpret_cast<uint4*>(vo + d * 128 + ((cc ^ (d & 7)) << 4)) = make_uint4(w[dd][0], w[dd][1], w[dd][2], w[dd][3]);
    }
    // the ones row (d = 64) and 7 zero rows: 2 heads x 8 rows x 8 chunks = 128 stores
    if (threadIdx.x < 128) {
      const int hh = threadIdx.x >> 6, dr = 64 + ((threadIdx.x >> 3) & 7), c2 = threadIdx.x & 7;
      const uint32_t one = dr == 64 ? 0x38383838u : 0u;  // e4m3 1.0
      uint8_t* vo2 = vt + ((size_t)(b * 2 + hh) * ntk + tile) * TILE_V;
      *reinterpret_cast<uint4*>(vo2 + dr * 128 + ((c2 ^ (dr & 7)) << 4)) = make_uint4(one, one, one, one);
    }
  }
}
#else
  for (int i = threadIdx.x; i < 2 * VROWS * 8; i += 256) {
    const int h = i / (VROWS * 8), ii = i - h * (VROWS * 8);
    const int d = ii >> 3, cc = ii & 7;
    uint4 out;
    if (d < 64) {
      uint32_t w[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t v = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int s2 = cc * 16 + q * 4 + e;
          const int key = (s2 & ~31) + sigma32(s2 & 31);
          v |= (uint32_t)vs[h][key * VP + d] << (8 * e);
        }
        w[q] = v;
      }
      out = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
      const uint32_t one = d == 64 ? 0x38383838u : 0u;  // e4m3 1.0
      out = make_uint4(one, one, one, one);
    }
    uint8_t* vo = vt + ((size_t)(b * 2 + h) * ntk + tile) * TILE_V;
    *reinterpret_cast<uint4*>(vo + d * 128 + ((cc ^ (d & 7)) << 4)) = out;
  }
}
#endif

struct AttnArgs {
  const uint8_t* q;         // [T, 16, 64] e4m3
  const uint8_t* kt;
  const uint8_t* vt;
  const int* seq_lens;
  const __nv_bfloat16* qkv; // [T, 2304]: gate at 1280 + head*64
  __nv_bfloat16* out;       // [T, 1024]
  int L, window, ntk, nqb;
  float c;                  // softmax scale * k_scale * log2(e)
  float v_scale;
  int nsplit;               // key-range splits (grid.y = 2 * nsplit); > 1 writes partials to ws
  float* ws;                // [nsplit, T, 16, 66] fp32: O (64, unnormalized), m, l
  int zero;                 // always 0: the first QK k-step's scale-d, opaque to the compiler
  const uint8_t* pf;        // ATTN_PF: the next GEMM's weight, prefetched into L2 by the producers (nullptr: none)
  unsigned long long pf_bytes;
  int sk;                   // stream-K (attn_fwd_sk_kernel): the grid size G; 0 otherwise
  int nb;                   // stream-K: batch count
  int* skbuf;               // stream-K: [0] = W (tiles per CTA), [1 .. npairs+1] = pair prefix, then each CTA's first pair
};
#ifndef ATTN_PF
#define ATTN_PF 0  // (Oct 26 03:12: o-proj weight prefetch during the attention: no change, off)
#endif

DEVI uint64_t desc_sw64(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(512 >> 4) << 32;  // 8 rows x 64 B
  d |= (uint64_t)2 << 62;           // SWIZZLE_64B
  return d;
}

#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890)
DEVI uint32_t sw_e4m3(float x) {  // RTNE, satfinite (448)
  const float a = fabsf(x);
  const uint32_t u = __float_as_uint(a);
  int code;
  if (a < 0.015625f) code = (int)rintf(a * 512.f);
  else code = (int)(((u + 0x7FFFFu + ((u >> 20) & 1u)) >> 20) - 960u);
  code = min(code, 126);
  return (uint32_t)(x < 0.f ? (code | 128) : code);
}
#endif

DEVI uint32_t pack_e4m3x4(float p0, float p1, float p2, float p3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(p1), "f"(p0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(p3), "f"(p2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
#else
  return sw_e4m3(p0) | (sw_e4m3(p1) << 8) | (sw_e4m3(p2) << 16) | (sw_e4m3(p3) << 24);
#endif
}

// e4m3 bytes of 2^y0..2^y3 (low to high) via f16 pairs: one MUFU op per two exponentials
DEVI uint32_t exp2_e4m3x4(float y0, float y1, float y2, float y3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint32_t h01, h23;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h01) : "f"(y1), "f"(y0));
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h23) : "f"(y3), "f"(y2));
  asm("ex2.approx.f16x2 %0, %0;" : "+r"(h01));
  asm("ex2.approx.f16x2 %0, %0;" : "+r"(h23));
  uint16_t b01, b23;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(b01) : "r"(h01));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(b23) : "r"(h23));
  return (uint32_t)b01 | ((uint32_t)b23 << 16);
#else
  auto h = [](float y) { return __half2float(__float2half_rn(exp2f(__half2float(__float2half_rn(y))))); };
  return sw_e4m3(h(y0)) | (sw_e4m3(h(y1)) << 8) | (sw_e4m3(h(y2)) << 16) | (sw_e4m3(h(y3)) << 24);
#endif
}

// 2^y on the FMA pipe (FA4-style): round-to-nearest split, degree-3 minimax on [-0.5, 0.5], exponent add.
DEVI float exp2_poly(float y) {
  y = fmaxf(y, -127.f);
  const float jm = y + 12582912.f;  // 1.5 * 2^23: integer part lands in the mantissa
  const float j = jm - 12582912.f;
  const float f = y - j;            // [-0.5, 0.5]
  float pf = fmaf(f, 0.0555041086648216f, 0.2402264923172690f);
  pf = fmaf(pf, f, 0.6931472028550421f);
  pf = fmaf(pf, f, 1.0f);
  return __int_as_float(__float_as_int(pf) + (__float_as_int(jm) << 23));
}

DEVI float e4m3_to_float(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  const float v = e ? __uint_as_float(((uint32_t)(b & 0x7F) << 20) + (120u << 23)) : (float)m * 0.001953125f;
  return (b & 0x80) ? -v : v;
}


template <int N>
DEVI void fence_f(float* r) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N>
DEVI void fence_u(uint32_t* r) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}

// P = exp2(s * c - (m * c - ATTN_POFF)): P is 2^ATTN_POFF at the running maximum, rounded to e4m3 (max 448). Every
// softmax path (masked or not, split partials) must use the same offset: O and l carry the same 2^ATTN_POFF factor.
#ifndef ATTN_POFF
#define ATTN_POFF 8.f
#endif
// ATTN_LATEMAX (exp-turn path, unmasked tiles): no row maximum before the exps. P is computed against the running
// maximum of earlier tiles, packed, then tested on the packed bytes: a byte of 0x7E (P >= ~432, or saturated) means a
// score jumped past the headroom (448 / 2^ATTN_POFF); only then (and on the first tile, whose running maximum is
// -1e30) the warp takes the exact path: row maxima, rescale factors, and the tile's P again from the fp32 scores.
// P below 448 against a stale maximum is still exact softmax arithmetic (O and l share the reference).
#ifndef ATTN_LATEMAX
#define ATTN_LATEMAX 0
#endif
struct AttnState {
  float o[36];  // columns 0..63: O; column 64: the softmax denominator (ones row of V)
  float mA, mB;
  int nready;   // ATTN_EARLY_TEST: the next tile's K/V barrier already tested complete
#ifdef ATTN_PROF
  long long pw, pq, ps, pp, pt;
#endif
};
#ifndef ATTN_MERGE
#define ATTN_MERGE 0  // (1: +3..7 us slower, Oct 26 01:06) key splits merged by the last-finishing split's CTA (no attn_combine2 launch)
#endif
__device__ int* g_attn_cnt;  // per (batch, query block, kv head) arrival counters (zeroed at prepare; reset by the merger)
#ifdef ATTN_PROF
__device__ unsigned long long* g_attn_prof;
DEVI unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#endif
struct AttnCtx {
  const AttnArgs& p;
  uint8_t* qs;
  uint8_t* ring;
  uint32_t bars;
  int wg, w, lane, g, t4;
  int kmaxA, kmaxB, kminA, kminB, nk, L, q_first, q_last, t_begin, t_end;
  uint64_t dq, dk0, dv0;  // wgmma descriptors of this warpgroup's Q tile and of stage 0's K / V tiles (ATTN_TRIM)
  int t_ring;             // the tile that sits in ring position 0 (t_begin, or earlier for later stream-K segments)
};

#ifndef ATTN_TRIM
#define ATTN_TRIM 0  // bits: 1 unsigned stage math, 2 descriptors hoisted out of the tile loop, 4 immediate scale-d (exact)
#endif
#if ATTN_TRIM & 1
DEVI int stage_of(const AttnCtx& cx, int t) { return (int)((unsigned)(t - cx.t_ring) % ASTAGES); }
DEVI uint32_t phase_of(const AttnCtx& cx, int t) { return ((unsigned)(t - cx.t_ring) / ASTAGES) & 1; }
#else
DEVI int stage_of(const AttnCtx& cx, int t) { return (t - cx.t_ring) % ASTAGES; }
DEVI uint32_t phase_of(const AttnCtx& cx, int t) { return ((t - cx.t_ring) / ASTAGES) & 1; }
#endif
template <int SD>
DEVI void wgmma_fp8_ss_n128_c(float* d, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "l"(da), "l"(db), "n"(SD));
#endif
}

template <int SD>
DEVI void wgmma_fp8_rs_n72_c(float* d, const uint32_t* a, uint64_t db) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, {%36, %37, %38, %39}, %40, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(SD));
#endif
}


DEVI void issue_qk(const AttnCtx& cx, float* sacc, int stage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if ATTN_TRIM & 2
  const uint64_t dq = cx.dq, dk = cx.dk0 + (uint64_t)(stage * (STAGE >> 4));
#else
  uint8_t* kst = cx.ring + stage * STAGE;
  const uint64_t dq = desc_sw64(smem_u32(cx.qs + cx.wg * 64 * 64));
  const uint64_t dk = desc_sw64(smem_u32(kst));
#endif
  fence_f<64>(sacc);
  wg_fence();
#if ATTN_TRIM & 4
  wgmma_fp8_ss_n128_c<0>(sacc, dq, dk);
  wgmma_fp8_ss_n128_c<1>(sacc, dq + 2, dk + 2);
#else
  wgmma_fp8_ss_n128(sacc, dq, dk, cx.p.zero);
  wgmma_fp8_ss_n128(sacc, dq + 2, dk + 2, 1);
#endif
  wg_commit();
  fence_f<64>(sacc);
#endif
}

#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
DEVI void emul_qk(const AttnCtx& cx, float* sacc, int stage) {
  uint8_t* kst = cx.ring + stage * STAGE;
  const int rA = 64 * cx.wg + 16 * cx.w + cx.g;
#pragma unroll 1
  for (int jj = 0; jj < 16; ++jj) {
#pragma unroll 1
    for (int e = 0; e < 4; ++e) {
      const int r = rA + ((e >> 1) << 3);
      const int key = 8 * jj + 2 * cx.t4 + (e & 1);
      float acc = 0.f;
      for (int d = 0; d < 64; ++d) {
        const int qc = d >> 4;
        const float qv = e4m3_to_float(cx.qs[r * 64 + ((qc ^ ((r >> 1) & 3)) << 4) + (d & 15)]);
        const float kv = e4m3_to_float(kst[key * 64 + ((qc ^ ((key >> 1) & 3)) << 4) + (d & 15)]);
        acc += qv * kv;
      }
      sacc[4 * jj + e] = acc;
    }
  }
}
#endif

// one key tile, sequential within the warpgroup (QK -> softmax -> P.V); the two consumer warpgroups overlap each
// other's softmax with their MMAs.
// EXPMODE: -1 = exp2 on f16 pairs; 0..64 = that many of each thread's 64 exponentials per tile on the FMA pipe
DEVI void named_bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// RR: the consumer warpgroups take turns on the softmax (named barriers 4..4+NWG-1), so each softmax has the SM's
// exp2 units to itself while the other warpgroups run their QK^T / P.V on the tensor cores.
template <int EXPMODE, bool OVL, int RR = 0>
DEVI void attn_step(const AttnCtx& cx, AttnState& st, float* sacc, int t) {
  const AttnArgs& p = cx.p;
  const int scu = stage_of(cx, t);
#ifdef ATTN_PROF
  const long long c0 = clock64();
  long long c1 = c0, c2 = c0;
#endif
  if (!OVL) {  // OVL: S(t) was computed by the previous step (issued together with its P.V)
    mbar_wait(cx.bars + 8 * scu, phase_of(cx, t));
#ifdef ATTN_PROF
    c1 = clock64();
#endif
    issue_qk(cx, sacc, scu);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    wg_wait<0>();
#else
    emul_qk(cx, sacc, scu);
#endif
#ifdef ATTN_PROF
    c2 = clock64();
#endif
  }
  const int k0 = t * NK;
  const int L = cx.L, nk = cx.nk;
  const bool need_mask = (k0 + NK - 1 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
  if (need_mask) {
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = k0 + 8 * jj + 2 * cx.t4 + e;
        if (key > cx.kmaxA || key < cx.kminA) sacc[4 * jj + e] = -INFINITY;
        if (key > cx.kmaxB || key < cx.kminB) sacc[4 * jj + 2 + e] = -INFINITY;
      }
    }
  }
  const float c = p.c;
  // tree reductions (short dependency chains): row A = elements 4jj, 4jj+1; row B = 4jj+2, 4jj+3
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
  const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
  const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
  float tmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), tmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
  float nmA = st.mA, nmB = st.mB;
  float scA = 1.f, scB = 1.f;
  // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
  // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
  if (!ATTN_VOTEMAX_ALL || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
  uint32_t pa[4][4];
  if constexpr (EXPMODE < 0) {
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const float* x = sacc + 16 * cc;
      pa[cc][0] = exp2_e4m3x4(fmaf(x[0], c, -offA), fmaf(x[1], c, -offA), fmaf(x[4], c, -offA), fmaf(x[5], c, -offA));
      pa[cc][1] = exp2_e4m3x4(fmaf(x[2], c, -offB), fmaf(x[3], c, -offB), fmaf(x[6], c, -offB), fmaf(x[7], c, -offB));
      pa[cc][2] = exp2_e4m3x4(fmaf(x[8], c, -offA), fmaf(x[9], c, -offA), fmaf(x[12], c, -offA), fmaf(x[13], c, -offA));
      pa[cc][3] = exp2_e4m3x4(fmaf(x[10], c, -offB), fmaf(x[11], c, -offB), fmaf(x[14], c, -offB), fmaf(x[15], c, -offB));
    }
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // RR: the warpgroups take turns on the exp2 units for the exp loop only (the max before it and the FP8 packing
    // after it run outside the turn), so the SM's MUFU pipe always has one warpgroup's exps to issue
    if constexpr (RR > 1) {
      named_bar_sync(4 + cx.wg, 256);
      fence_f<64>(sacc);
    }
#endif
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      if (4 * jj < EXPMODE) {
        sacc[4 * jj] = exp2_poly(fmaf(sacc[4 * jj], c, -offA));
        sacc[4 * jj + 1] = exp2_poly(fmaf(sacc[4 * jj + 1], c, -offA));
        sacc[4 * jj + 2] = exp2_poly(fmaf(sacc[4 * jj + 2], c, -offB));
        sacc[4 * jj + 3] = exp2_poly(fmaf(sacc[4 * jj + 3], c, -offB));
      } else {
#if ATTN_BISECT == 9
        sacc[4 * jj] = fmaf(sacc[4 * jj], c, -offA);
        sacc[4 * jj + 1] = fmaf(sacc[4 * jj + 1], c, -offA);
        sacc[4 * jj + 2] = fmaf(sacc[4 * jj + 2], c, -offB);
        sacc[4 * jj + 3] = fmaf(sacc[4 * jj + 3], c, -offB);
#else
        sacc[4 * jj] = exp2f(fmaf(sacc[4 * jj], c, -offA));
        sacc[4 * jj + 1] = exp2f(fmaf(sacc[4 * jj + 1], c, -offA));
        sacc[4 * jj + 2] = exp2f(fmaf(sacc[4 * jj + 2], c, -offB));
        sacc[4 * jj + 3] = exp2f(fmaf(sacc[4 * jj + 3], c, -offB));
#endif
      }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if constexpr (RR > 1 && ATTN_RR_HAND2 < 64) {
        if (4 * jj + 4 == ATTN_RR_HAND2) {  // early hand-off: the next warpgroup's turn starts now
          fence_f<ATTN_RR_HAND2>(sacc);
          named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);
        }
      }
#endif
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (RR > 1) {
      if constexpr (ATTN_RR_HAND2 >= 64) {
        fence_f<64>(sacc);
        named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);  // the next warpgroup's turn
      }
    }
#endif
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const float* x = sacc + 16 * cc;
      pa[cc][0] = pack_e4m3x4(x[0], x[1], x[4], x[5]);
      pa[cc][1] = pack_e4m3x4(x[2], x[3], x[6], x[7]);
      pa[cc][2] = pack_e4m3x4(x[8], x[9], x[12], x[13]);
      pa[cc][3] = pack_e4m3x4(x[10], x[11], x[14], x[15]);
    }
  }
  if (!__all_sync(0xffffffffu, scA == 1.f && scB == 1.f)) {  // the running max rarely moves after a few tiles
#pragma unroll
    for (int jj = 0; jj < 9; ++jj) {
      st.o[4 * jj] *= scA; st.o[4 * jj + 1] *= scA;
      st.o[4 * jj + 2] *= scB; st.o[4 * jj + 3] *= scB;
    }
  }
  uint8_t* vst = cx.ring + scu * STAGE + TILE_K;
#ifdef ATTN_PROF
  const long long c3 = clock64();
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint64_t dv = desc_sw128(smem_u32(vst));
  fence_u<16>(&pa[0][0]);
  fence_f<36>(st.o);
  wg_fence();
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) wgmma_fp8_rs_n72(st.o, pa[cc], dv + 2 * cc, 1);
  wg_commit();
  fence_f<36>(st.o);
  if (OVL && t + 1 <= cx.t_end) {  // the next tile's QK^T runs on the tensor cores together with this P.V
    const int sn = stage_of(cx, t + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, t + 1));
    issue_qk(cx, sacc, sn);
  }
  wg_wait<0>();
#ifdef ATTN_PROF
  {
    const long long c4 = clock64();
    st.pw += c1 - c0; st.pq += c2 - c1; st.ps += c3 - c2; st.pp += c4 - c3;
#ifdef ATTN_TL
    const int k = t - cx.t_begin;
    if (k < 24 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && (threadIdx.x & 127) == 0 && g_attn_prof) {
      unsigned long long* tl = g_attn_prof + 400000 + (cx.wg * 24 + k) * 5;
      tl[0] = c0; tl[1] = c1; tl[2] = c2; tl[3] = c3; tl[4] = c4;
    }
#endif
  }
#endif
#else
  uint8_t* pw = cx.ring + ASTAGES * STAGE + cx.wg * 64 * 128;  // sm80 emulation staging
  const int rl = 16 * cx.w + cx.g;
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    *reinterpret_cast<uint32_t*>(pw + rl * 128 + 32 * cc + 4 * cx.t4) = pa[cc][0];
    *reinterpret_cast<uint32_t*>(pw + (rl + 8) * 128 + 32 * cc + 4 * cx.t4) = pa[cc][1];
    *reinterpret_cast<uint32_t*>(pw + rl * 128 + 32 * cc + 16 + 4 * cx.t4) = pa[cc][2];
    *reinterpret_cast<uint32_t*>(pw + (rl + 8) * 128 + 32 * cc + 16 + 4 * cx.t4) = pa[cc][3];
  }
  named_bar_sync(2 + cx.wg, 128);
#pragma unroll 1
  for (int jj = 0; jj < 9; ++jj) {
#pragma unroll 1
    for (int e = 0; e < 4; ++e) {
      const int r = rl + ((e >> 1) << 3);
      const int dcol = 8 * jj + 2 * cx.t4 + (e & 1);
      float acc = 0.f;
      for (int sl = 0; sl < 128; ++sl) {
        const float pv = e4m3_to_float(pw[r * 128 + sl]);
        const float vv = e4m3_to_float(vst[dcol * 128 + ((((sl >> 4) ^ (dcol & 7))) << 4) + (sl & 15)]);
        acc += pv * vv;
      }
      st.o[4 * jj + e] += acc;
    }
  }
  named_bar_sync(2 + cx.wg, 128);
  if (OVL && t + 1 <= cx.t_end) {
    const int sn = stage_of(cx, t + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, t + 1));
    emul_qk(cx, sacc, sn);
  }
#endif
  __syncwarp();
  if (cx.lane == 0) mbar_arrive(cx.bars + 8 * ASTAGES + 8 * scu);
}

// ---------------- half tiles: each 128-key stage in two 64-key steps ----------------
// S and P are half as large (32 + 8 registers), so four consumer warpgroups fit the register file: more warps per
// scheduler keep the exp2 units busy while other warpgroups wait on their MMAs.
DEVI void issue_qk_h(const AttnCtx& cx, float* sacc, int stage, int h) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint8_t* kst = cx.ring + stage * STAGE + h * 64 * 64;
  uint64_t dq0 = desc_sw64(smem_u32(cx.qs + cx.wg * 64 * 64));
  uint64_t dk0 = desc_sw64(smem_u32(kst));
  uint64_t dq1 = dq0 + 2, dk1 = dk0 + 2;
  // both k-steps' descriptors exist before the pipeline stage opens (a descriptor register rewritten between the two
  // wgmmas serializes every wgmma in the kernel, C7513)
  asm volatile("" : "+l"(dq0), "+l"(dk0), "+l"(dq1), "+l"(dk1));
  fence_f<32>(sacc);
  wg_fence();
#if ATTN_BISECT == 3
  wgmma_fp8_ss_n64(sacc, dq0, dk0, 0);
#else
  wgmma_fp8_ss_n64(sacc, dq0, dk0, cx.p.zero);
#endif
  wgmma_fp8_ss_n64(sacc, dq1, dk1, 1);
  wg_commit();
  fence_f<32>(sacc);
#endif
}

template <int EXPMODE>
DEVI void attn_step_h(const AttnCtx& cx, AttnState& st, float* sacc, int t) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const AttnArgs& p = cx.p;
  const int scu = stage_of(cx, t);
  mbar_wait(cx.bars + 8 * scu, phase_of(cx, t));
  uint8_t* vst = cx.ring + scu * STAGE + TILE_K;
  const uint64_t dv = desc_sw128(smem_u32(vst));
  const int L = cx.L, nk = cx.nk;
  const float c = p.c;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    issue_qk_h(cx, sacc, scu, h);
    wg_wait<0>();
    const int k0 = t * NK + 64 * h;
    const bool need_mask = (k0 + 63 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
    if (need_mask) {
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int key = k0 + 8 * jj + 2 * cx.t4 + e;
          if (key > cx.kmaxA || key < cx.kminA) sacc[4 * jj + e] = -INFINITY;
          if (key > cx.kmaxB || key < cx.kminB) sacc[4 * jj + 2 + e] = -INFINITY;
        }
      }
    }
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
    float tmA = fmaxf(fmaxf(MX4(0), MX4(8)), fmaxf(MX4(16), MX4(24)));
    float tmB = fmaxf(fmaxf(MX4(2), MX4(10)), fmaxf(MX4(18), MX4(26)));
#undef MX4
    float nmA = st.mA, nmB = st.mB;
    float scA = 1.f, scB = 1.f;
    // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
    // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
    if (!ATTN_VOTEMAX_ALL || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
      tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
      tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
      tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
      tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
      nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
      scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
      st.mA = nmA; st.mB = nmB;
    }
    const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      if (4 * jj < EXPMODE) {
        sacc[4 * jj] = exp2_poly(fmaf(sacc[4 * jj], c, -offA));
        sacc[4 * jj + 1] = exp2_poly(fmaf(sacc[4 * jj + 1], c, -offA));
        sacc[4 * jj + 2] = exp2_poly(fmaf(sacc[4 * jj + 2], c, -offB));
        sacc[4 * jj + 3] = exp2_poly(fmaf(sacc[4 * jj + 3], c, -offB));
      } else {
        sacc[4 * jj] = exp2f(fmaf(sacc[4 * jj], c, -offA));
        sacc[4 * jj + 1] = exp2f(fmaf(sacc[4 * jj + 1], c, -offA));
        sacc[4 * jj + 2] = exp2f(fmaf(sacc[4 * jj + 2], c, -offB));
        sacc[4 * jj + 3] = exp2f(fmaf(sacc[4 * jj + 3], c, -offB));
      }
    }
    uint32_t pa[2][4];
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const float* x = sacc + 16 * cc;
      pa[cc][0] = pack_e4m3x4(x[0], x[1], x[4], x[5]);
      pa[cc][1] = pack_e4m3x4(x[2], x[3], x[6], x[7]);
      pa[cc][2] = pack_e4m3x4(x[8], x[9], x[12], x[13]);
      pa[cc][3] = pack_e4m3x4(x[10], x[11], x[14], x[15]);
    }
    if (!__all_sync(0xffffffffu, scA == 1.f && scB == 1.f)) {
#pragma unroll
      for (int jj = 0; jj < 9; ++jj) {
        st.o[4 * jj] *= scA; st.o[4 * jj + 1] *= scA;
        st.o[4 * jj + 2] *= scB; st.o[4 * jj + 3] *= scB;
      }
    }
    fence_u<8>(&pa[0][0]);
    fence_f<36>(st.o);
    wg_fence();
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) wgmma_fp8_rs_n72(st.o, pa[cc], dv + 2 * (2 * h + cc), 1);
    wg_commit();
    fence_f<36>(st.o);
    wg_wait<0>();
  }
  __syncwarp();
  if (cx.lane == 0) mbar_arrive(cx.bars + 8 * ASTAGES + 8 * scu);
#else
  attn_step<EXPMODE, false>(cx, st, sacc, t);  // sm_80 test builds: the full-tile step (same results)
#endif
}

// ---------------- software-pipelined consumer (FA3-style intra-warpgroup overlap) ----------------
// Tile t's softmax runs while the tensor cores do tile t-1's P.V: per tile, issue QK(t) and P.V(t-1), wait for QK(t),
// softmax(t) into a second P register set, wait for P.V(t-1), rescale O, release tile t-1's stage.
// Reads S without writing it (S is the accumulator of a wgmma still inside an open pipeline stage: writing it in place
// makes ptxas serialize every wgmma, C7513); masking is applied on the fly.
template <int EXPMODE, bool MASK>
DEVI void softmax_core(const AttnCtx& cx, AttnState& st, const float* sacc, int k0, uint32_t (&pa)[4][4], float& scA, float& scB) {
  auto V = [&](int i) -> float {
    if constexpr (!MASK) {
      return sacc[i];
    } else {
      const int key = k0 + 8 * (i >> 2) + 2 * cx.t4 + (i & 1);
      const bool bad = (i & 2) ? (key > cx.kmaxB || key < cx.kminB) : (key > cx.kmaxA || key < cx.kminA);
      return bad ? -INFINITY : sacc[i];
    }
  };
  const float c = cx.p.c;
#define MX4(o) fmaxf(fmaxf(V(o), V((o) + 1)), fmaxf(V((o) + 4), V((o) + 5)))
  const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
  const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
  float tmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), tmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
  float nmA = st.mA, nmB = st.mB;
  scA = 1.f; scB = 1.f;
  // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
  // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
  if (!ATTN_VOTEMAX_ALL || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
  auto E = [&](int i) -> float {
    const float y = fmaf(V(i), c, (i & 2) ? -offB : -offA);
    if (i < EXPMODE) return exp2_poly(y);
    return exp2f(y);
  };
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int o = 16 * cc;
    pa[cc][0] = pack_e4m3x4(E(o + 0), E(o + 1), E(o + 4), E(o + 5));
    pa[cc][1] = pack_e4m3x4(E(o + 2), E(o + 3), E(o + 6), E(o + 7));
    pa[cc][2] = pack_e4m3x4(E(o + 8), E(o + 9), E(o + 12), E(o + 13));
    pa[cc][3] = pack_e4m3x4(E(o + 10), E(o + 11), E(o + 14), E(o + 15));
  }
}

#ifndef ATTN_RR_HAND
#define ATTN_RR_HAND 24  // v64: 24 measured +0.16% with the vote-skip
#endif
// RR > 1: the softmax with the warpgroups taking turns on the exp2 units (named barriers 4..4+RR-1) for the exp loop
// only; the max before it and the FP8 packing after it run outside the turn. Exps are written over S in place.
template <bool MASK, int RR, int EXPMODE = 0>
DEVI void softmax_core_rr(const AttnCtx& cx, AttnState& st, float* sacc, int k0, uint32_t (&pa)[4][4], float& scA, float& scB) {
  // EXPMODE > 0: that many of the 64 exps per thread run as a cubic polynomial on the FMA pipe (every 64/EXPMODE-th),
  // off the exp2 units that bound the turns
  auto EXP = [&](int i, float y) -> float {
    if constexpr (EXPMODE > 0) {
      if (i % (64 / EXPMODE) == 0) return exp2_poly(y);
    }
    #ifdef ATTN_DBG_NOEXP  // timing experiment only (wrong results): no exp2 units
    return fmaf(y, 0.0078125f, 1.0f);
#endif
    return exp2f(y);
  };
#if ATTN_LATEMAX
  if constexpr (!MASK) {
    const float c = cx.p.c;
    scA = 1.f; scB = 1.f;
    float offA = st.mA * c - ATTN_POFF, offB = st.mB * c - ATTN_POFF;
    // P register (cc, j) packs scores base, base+1, base+4, base+5 with base = 16cc + 8(j>>1) + 2(j&1); odd j: row B
    auto pack_p = [&](int r) -> uint32_t {
      const int cc = r >> 2, j = r & 3;
      const int base = 16 * cc + ((j & 2) ? 8 : 0) + ((j & 1) ? 2 : 0);
      const float off = (j & 1) ? offB : offA;
      const float e0 = EXP(base, fmaf(sacc[base], c, -off)), e1 = EXP(base + 1, fmaf(sacc[base + 1], c, -off));
      const float e2 = EXP(base + 4, fmaf(sacc[base + 4], c, -off)), e3 = EXP(base + 5, fmaf(sacc[base + 5], c, -off));
      return pack_e4m3x4(e0, e1, e2, e3);
    };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#ifdef ATTN_PROF
    const long long tq0 = clock64();
#endif
    named_bar_sync(4 + cx.wg, 256);  // my turn
#ifdef ATTN_PROF
    st.pt += clock64() - tq0;
#endif
    fence_f<64>(sacc);
#endif
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      pa[r >> 2][r & 3] = pack_p(r);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if (4 * (r + 1) == (ATTN_RR_HAND & ~3)) {
        fence_u<(ATTN_RR_HAND >> 2)>(&pa[0][0]);  // the P registers packed so far (row-major [4][4])
        named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);
      }
#endif
    }
    uint32_t fl = 0u;
#pragma unroll
    for (int r = 0; r < 16; ++r) fl |= pa[r >> 2][r & 3] + 0x02020202u;  // bytes <= 0x7E: no carry between bytes
#ifndef ATTN_LM_FORCE
#define ATTN_LM_FORCE -1  // timing experiments: 0 = never take the exact path, 1 = always
#endif
    const bool slow = ATTN_LM_FORCE == 0 ? false : (ATTN_LM_FORCE == 1 ? true : __any_sync(0xffffffffu, (fl & 0x80808080u) != 0u));
    if (slow) {
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
      const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
      const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
      float tmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), tmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
      tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
      tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
      tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
      tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
      const float nmA = fmaxf(st.mA, tmA), nmB = fmaxf(st.mB, tmB);
      scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
      st.mA = nmA; st.mB = nmB;
      offA = nmA * c - ATTN_POFF; offB = nmB * c - ATTN_POFF;
#pragma unroll
      for (int r = 0; r < 16; ++r) pa[r >> 2][r & 3] = pack_p(r);
    }
    return;
  }
#endif
#if ATTN_LATEMAX == 2
  // after the first tile (running maximum set): exps against the running maximum straight away, in place, the same
  // turn order as below; then the packed-P test. A flagged warp renormalises each row from its fp32 exps (exact
  // values; only the e4m3 packing saturates): factor f = 2^POFF / max P, applied to the row's P and returned as the
  // O rescale, and the running maximum moves by log2(max P / 2^POFF) / c.
  if (!MASK && st.mA > -1e29f) {
    const float c = cx.p.c;
    scA = 1.f; scB = 1.f;
    const float offA = st.mA * c - ATTN_POFF, offB = st.mB * c - ATTN_POFF;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#ifdef ATTN_PROF
    const long long tq0 = clock64();
#endif
    named_bar_sync(4 + cx.wg, 256);  // my turn
#ifdef ATTN_PROF
    st.pt += clock64() - tq0;
#endif
    fence_f<64>(sacc);
#endif
#pragma unroll
    for (int i = 0; i < ATTN_RR_HAND; ++i) sacc[i] = EXP(i, fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    fence_f<ATTN_RR_HAND>(sacc);
    named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);
#endif
#pragma unroll
    for (int i = ATTN_RR_HAND; i < 64; ++i) sacc[i] = EXP(i, fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
    auto pack_all = [&]() {
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        const float* x = sacc + 16 * cc;
        pa[cc][0] = pack_e4m3x4(x[0], x[1], x[4], x[5]);
        pa[cc][1] = pack_e4m3x4(x[2], x[3], x[6], x[7]);
        pa[cc][2] = pack_e4m3x4(x[8], x[9], x[12], x[13]);
        pa[cc][3] = pack_e4m3x4(x[10], x[11], x[14], x[15]);
      }
    };
    pack_all();
    uint32_t fl = 0u;
#pragma unroll
    for (int r = 0; r < 16; ++r) fl |= pa[r >> 2][r & 3] + 0x02020202u;  // bytes <= 0x7E: no carry between bytes
#ifndef ATTN_LM_FORCE
#define ATTN_LM_FORCE -1
#endif
    const bool slow = ATTN_LM_FORCE == 0 ? false : (ATTN_LM_FORCE == 1 ? true : __any_sync(0xffffffffu, (fl & 0x80808080u) != 0u));
    if (slow) {
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
      const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
      const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
      float pmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), pmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
      pmA = fmaxf(pmA, __shfl_xor_sync(0xffffffffu, pmA, 1));
      pmA = fmaxf(pmA, __shfl_xor_sync(0xffffffffu, pmA, 2));
      pmB = fmaxf(pmB, __shfl_xor_sync(0xffffffffu, pmB, 1));
      pmB = fmaxf(pmB, __shfl_xor_sync(0xffffffffu, pmB, 2));
      constexpr float PTOP = (float)(1 << (int)ATTN_POFF);
      const float fA = pmA > PTOP ? PTOP / pmA : 1.f, fB = pmB > PTOP ? PTOP / pmB : 1.f;
#pragma unroll
      for (int i = 0; i < 64; ++i) sacc[i] *= (i & 2) ? fB : fA;
      pack_all();
      if (pmA > PTOP) st.mA += __log2f(pmA / PTOP) / c;
      if (pmB > PTOP) st.mB += __log2f(pmB / PTOP) / c;
      scA = fA; scB = fB;
    }
    return;
  }
#endif
  if constexpr (MASK) {
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      const int key = k0 + 8 * (i >> 2) + 2 * cx.t4 + (i & 1);
      const bool bad = (i & 2) ? (key > cx.kmaxB || key < cx.kminB) : (key > cx.kmaxA || key < cx.kminA);
      if (bad) sacc[i] = -INFINITY;
    }
  }
  const float c = cx.p.c;
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
  const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
  const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
  float tmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), tmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
  float nmA = st.mA, nmB = st.mB;
  scA = 1.f; scB = 1.f;
#ifdef ATTN_DBG_NOMAX  // timing experiment only (wrong results after the first tile): no row maximum on later tiles
  if (st.mA < -1e29f) {
#else
  // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
  // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
  if (!ATTN_VOTEMAX || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
#endif
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#ifdef ATTN_PROF
  const long long tq0 = clock64();
#endif
  named_bar_sync(4 + cx.wg, 256);  // my turn
#ifdef ATTN_PROF
  st.pt += clock64() - tq0;
#endif
  fence_f<64>(sacc);
#endif
  // the next warpgroup's turn starts after the first ATTN_RR_HAND of my 64 exps: two warpgroups overlap on the exp2
  // units for the rest, which hides the hand-off latency
#pragma unroll
  for (int i = 0; i < ATTN_RR_HAND; ++i) sacc[i] = EXP(i, fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  fence_f<ATTN_RR_HAND>(sacc);
  named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);
#endif
#pragma unroll
  for (int i = ATTN_RR_HAND; i < 64; ++i) sacc[i] = EXP(i, fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const float* x = sacc + 16 * cc;
    pa[cc][0] = pack_e4m3x4(x[0], x[1], x[4], x[5]);
    pa[cc][1] = pack_e4m3x4(x[2], x[3], x[6], x[7]);
    pa[cc][2] = pack_e4m3x4(x[8], x[9], x[12], x[13]);
    pa[cc][3] = pack_e4m3x4(x[10], x[11], x[14], x[15]);
  }
}

template <int EXPMODE, int RR = 0>
DEVI void softmax_tile(const AttnCtx& cx, AttnState& st, float* sacc, int t, uint32_t (&pa)[4][4], float& scA, float& scB) {
  const AttnArgs& p = cx.p;
  const int k0 = t * NK;
  const int L = cx.L, nk = cx.nk;
  const bool need_mask = (k0 + NK - 1 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
  if constexpr (RR > 1) {
    if (need_mask) softmax_core_rr<true, RR, EXPMODE>(cx, st, sacc, k0, pa, scA, scB);
    else softmax_core_rr<false, RR, EXPMODE>(cx, st, sacc, k0, pa, scA, scB);
  } else {
    if (need_mask) softmax_core<EXPMODE, true>(cx, st, sacc, k0, pa, scA, scB);
    else softmax_core<EXPMODE, false>(cx, st, sacc, k0, pa, scA, scB);
  }
}

DEVI void issue_pv(const AttnCtx& cx, float* o, uint32_t (&pa)[4][4], int stage) {
  uint8_t* vst = cx.ring + stage * STAGE + TILE_K;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if ATTN_TRIM & 2
  const uint64_t dv = cx.dv0 + (uint64_t)(stage * (STAGE >> 4));
#else
  const uint64_t dv = desc_sw128(smem_u32(vst));
#endif
  fence_u<16>(&pa[0][0]);
  fence_f<36>(o);
  wg_fence();
#pragma unroll
#if ATTN_TRIM & 4
  for (int cc = 0; cc < 4; ++cc) wgmma_fp8_rs_n72_c<1>(o, pa[cc], dv + 2 * cc);
#else
  for (int cc = 0; cc < 4; ++cc) wgmma_fp8_rs_n72(o, pa[cc], dv + 2 * cc, 1);
#endif
  wg_commit();
  fence_f<36>(o);
#else
  uint8_t* pw = cx.ring + ASTAGES * STAGE + cx.wg * 64 * 128;  // sm80 emulation staging
  const int rl = 16 * cx.w + cx.g;
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    *reinterpret_cast<uint32_t*>(pw + rl * 128 + 32 * cc + 4 * cx.t4) = pa[cc][0];
    *reinterpret_cast<uint32_t*>(pw + (rl + 8) * 128 + 32 * cc + 4 * cx.t4) = pa[cc][1];
    *reinterpret_cast<uint32_t*>(pw + rl * 128 + 32 * cc + 16 + 4 * cx.t4) = pa[cc][2];
    *reinterpret_cast<uint32_t*>(pw + (rl + 8) * 128 + 32 * cc + 16 + 4 * cx.t4) = pa[cc][3];
  }
  named_bar_sync(2 + cx.wg, 128);
#pragma unroll 1
  for (int jj = 0; jj < 9; ++jj) {
#pragma unroll 1
    for (int e = 0; e < 4; ++e) {
      const int r = rl + ((e >> 1) << 3);
      const int dcol = 8 * jj + 2 * cx.t4 + (e & 1);
      float acc = 0.f;
      for (int sl = 0; sl < 128; ++sl) {
        const float pv = e4m3_to_float(pw[r * 128 + sl]);
        const float vv = e4m3_to_float(vst[dcol * 128 + ((((sl >> 4) ^ (dcol & 7))) << 4) + (sl & 15)]);
        acc += pv * vv;
      }
      o[4 * jj + e] += acc;
    }
  }
  named_bar_sync(2 + cx.wg, 128);
#endif
}

DEVI void rescale_o(AttnState& st, float scA, float scB) {
  if (!__all_sync(0xffffffffu, scA == 1.f && scB == 1.f)) {
#pragma unroll
    for (int jj = 0; jj < 9; ++jj) {
      st.o[4 * jj] *= scA; st.o[4 * jj + 1] *= scA;
      st.o[4 * jj + 2] *= scB; st.o[4 * jj + 3] *= scB;
    }
  }
}

DEVI void release_stage(const AttnCtx& cx, int stage) {
  __syncwarp();
  if (cx.lane == 0) mbar_arrive(cx.bars + 8 * ASTAGES + 8 * stage);
}

#ifndef ATTN_EARLY_TEST
#define ATTN_EARLY_TEST 0  // test the next tile's K/V barrier (non-blocking) before the P.V wait; skip the blocking wait
#endif
DEVI bool mbar_test(uint32_t bar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t r;
  asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(r) : "r"(bar), "r"(parity) : "memory");
  return r != 0;
#else
  return false;
#endif
}
template <int EXPMODE, int RR = 0>
DEVI void pipe_step(const AttnCtx& cx, AttnState& st, float* sacc, int t, uint32_t (&pin)[4][4], uint32_t (&pout)[4][4]) {
  const int s = stage_of(cx, t), sp = stage_of(cx, t - 1);
#ifdef ATTN_PROF
  const long long c0 = clock64();
#endif
  if (!ATTN_EARLY_TEST || !st.nready) mbar_wait(cx.bars + 8 * s, phase_of(cx, t));
#ifdef ATTN_PROF
  const long long c1 = clock64();
#endif
  issue_qk(cx, sacc, s);
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  emul_qk(cx, sacc, s);
#endif
  issue_pv(cx, st.o, pin, sp);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  wg_wait<1>();  // QK(t) is done; P.V(t-1) may still run
#endif
#ifdef ATTN_PROF
  const long long c2 = clock64();
#endif
  float scA, scB;
  softmax_tile<EXPMODE, RR>(cx, st, sacc, t, pout, scA, scB);
#ifdef ATTN_PROF
  const long long c3 = clock64();
#endif
  if (ATTN_EARLY_TEST) st.nready = t + 1 <= cx.t_end && mbar_test(cx.bars + 8 * stage_of(cx, t + 1), phase_of(cx, t + 1));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  wg_wait<0>();  // P.V(t-1) is done
#endif
  fence_u<16>(&pin[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, sp);
  rescale_o(st, scA, scB);
#ifdef ATTN_PROF
  st.pw += c1 - c0; st.pq += c2 - c1; st.ps += c3 - c2; st.pp += clock64() - c3;
#endif
}

DEVI void pipe_last(const AttnCtx& cx, AttnState& st, uint32_t (&pin)[4][4]) {
  const int sl = stage_of(cx, cx.t_end);
  issue_pv(cx, st.o, pin, sl);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  wg_wait<0>();
#endif
  fence_u<16>(&pin[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, sl);
}

// two P register sets alternate (no copies: a copy into an in-flight wgmma's operand serializes the pipeline)
template <int EXPMODE, int RR = 0>
DEVI void attn_pipe(const AttnCtx& cx, AttnState& st, float* sacc) {
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  uint32_t p0[4][4], p1[4][4];
  {
    const int s = stage_of(cx, tb);
    mbar_wait(cx.bars + 8 * s, phase_of(cx, tb));
    issue_qk(cx, sacc, s);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    wg_wait<0>();
#else
    emul_qk(cx, sacc, s);
#endif
    float scA, scB;
    softmax_tile<EXPMODE, RR>(cx, st, sacc, tb, p0, scA, scB);  // O is still zero: nothing to rescale
  }
  int t = tb + 1;
  for (; t + 1 <= te; t += 2) {
    pipe_step<EXPMODE, RR>(cx, st, sacc, t, p0, p1);
    pipe_step<EXPMODE, RR>(cx, st, sacc, t + 1, p1, p0);
  }
  if (t <= te) {
    pipe_step<EXPMODE, RR>(cx, st, sacc, t, p0, p1);
    pipe_last(cx, st, p1);
  } else {
    pipe_last(cx, st, p0);
  }
}

#ifndef ATTN_EARLY_QK
#define ATTN_EARLY_QK 0
#endif
// Early QK^T: QK(t+1) is issued as soon as softmax(t) has packed P(t) (S is dead then), so its latency hides behind the
// P.V(t-1) drain, the O rescale and the next step's P.V issue instead of opening the next step. On entry QK(t) is in
// flight.
template <int EXPMODE, int RR, bool MORE>
DEVI void pipe_step_eq(const AttnCtx& cx, AttnState& st, float* sacc, int t, uint32_t (&pin)[4][4], uint32_t (&pout)[4][4]) {
  // compile-time MORE: a wgmma issued under a runtime branch serializes the kernel (ptxas C7513)
  const int sp = stage_of(cx, t - 1);
  issue_pv(cx, st.o, pin, sp);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  wg_wait<1>();  // QK(t) is done; P.V(t-1) may still run
#endif
  float scA, scB;
  softmax_tile<EXPMODE, RR>(cx, st, sacc, t, pout, scA, scB);
  if constexpr (MORE) {
    const int sn = stage_of(cx, t + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, t + 1));
    issue_qk(cx, sacc, sn);
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    emul_qk(cx, sacc, sn);
#endif
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (MORE) wg_wait<1>();  // P.V(t-1) is done; QK(t+1) may still run
  else wg_wait<0>();
#endif
  fence_u<16>(&pin[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, sp);
  rescale_o(st, scA, scB);
}

template <int EXPMODE, int RR = 0>
DEVI void attn_pipe_eq(const AttnCtx& cx, AttnState& st, float* sacc) {
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  uint32_t p0[4][4], p1[4][4];
  {
    const int s = stage_of(cx, tb);
    mbar_wait(cx.bars + 8 * s, phase_of(cx, tb));
    issue_qk(cx, sacc, s);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    wg_wait<0>();
#else
    emul_qk(cx, sacc, s);
#endif
    float scA, scB;
    softmax_tile<EXPMODE, RR>(cx, st, sacc, tb, p0, scA, scB);  // O is still zero: nothing to rescale
  }
  {  // QK(tb + 1) runs while the first tile's tail drains. Issued even for a single-tile range (the producer loads a
     // dummy tile): a separate single-tile path makes ptxas serialize every wgmma (C7511)
    const int sn = stage_of(cx, tb + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, tb + 1));
    issue_qk(cx, sacc, sn);
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    emul_qk(cx, sacc, sn);
#endif
  }
  int t = tb + 1;
  for (; t + 2 <= te; t += 2) {
    pipe_step_eq<EXPMODE, RR, true>(cx, st, sacc, t, p0, p1);
    pipe_step_eq<EXPMODE, RR, true>(cx, st, sacc, t + 1, p1, p0);
  }
  if (t + 1 == te) {
    pipe_step_eq<EXPMODE, RR, true>(cx, st, sacc, t, p0, p1);
    pipe_step_eq<EXPMODE, RR, false>(cx, st, sacc, t + 1, p1, p0);
    pipe_last(cx, st, p0);
  } else if (t == te) {
    pipe_step_eq<EXPMODE, RR, false>(cx, st, sacc, t, p0, p1);
    pipe_last(cx, st, p1);
  } else {  // a single-tile range: P(tb) is in p0; pipe_last's wait also retires the dummy QK
    pipe_last(cx, st, p0);
  }
}

// Deeper pipeline: QK(t+1) and P.V(t-1) both run on the tensor cores during softmax(t) (two S and two P register sets).
template <int EXPMODE>
DEVI void pipe2_step(const AttnCtx& cx, AttnState& st, int t, float* s_cur, float* s_next, uint32_t (&pin)[4][4],
                     uint32_t (&pout)[4][4]) {
  const bool more = t + 1 <= cx.t_end;
  if (more) {
    const int sn = stage_of(cx, t + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, t + 1));
    issue_qk(cx, s_next, sn);
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    emul_qk(cx, s_next, sn);
#endif
  }
  const int sp = stage_of(cx, t - 1);
  issue_pv(cx, st.o, pin, sp);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (more) wg_wait<2>();  // QK(t) is done
  else wg_wait<1>();
#endif
  float scA, scB;
  softmax_tile<EXPMODE>(cx, st, s_cur, t, pout, scA, scB);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  wg_wait<0>();  // P.V(t-1) (and QK(t+1), issued before it) are done
#endif
  fence_u<16>(&pin[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, sp);
  rescale_o(st, scA, scB);
}

template <int EXPMODE>
DEVI void attn_pipe2(const AttnCtx& cx, AttnState& st, float* s0, float* s1) {
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  uint32_t p0[4][4], p1[4][4];
  {
    const int s = stage_of(cx, tb);
    mbar_wait(cx.bars + 8 * s, phase_of(cx, tb));
    issue_qk(cx, s0, s);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    wg_wait<0>();
#else
    emul_qk(cx, s0, s);
#endif
    if (tb + 1 <= te) {
      const int sn = stage_of(cx, tb + 1);
      mbar_wait(cx.bars + 8 * sn, phase_of(cx, tb + 1));
      issue_qk(cx, s1, sn);
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
      emul_qk(cx, s1, sn);
#endif
    }
    float scA, scB;
    softmax_tile<EXPMODE>(cx, st, s0, tb, p0, scA, scB);
  }
  int t = tb + 1;
  for (; t + 1 <= te; t += 2) {
    pipe2_step<EXPMODE>(cx, st, t, s1, s0, p0, p1);
    pipe2_step<EXPMODE>(cx, st, t + 1, s0, s1, p1, p0);
  }
  if (t <= te) {
    pipe2_step<EXPMODE>(cx, st, t, s1, s0, p0, p1);
    pipe_last(cx, st, p1);
  } else {
    pipe_last(cx, st, p0);
  }
}

// ---------------- variant 53: two consumer warpgroups (240 registers), two S register sets, one P set, exp turns ----------------
// Step t (QK(t) in flight on entry): issue P.V(t-1); issue QK(t+1) into the other S set; wait QK(t); softmax(t): row max
// (vote-skipped), my exp turn (the other warpgroup's turn starts after ATTN_RR_HAND exps), then wait P.V(t-1), rescale O,
// pack P(t). P.V(t-1) and QK(t+1) run on the tensor cores during the softmax; the two warpgroups alternate on the exp2
// units. MORE (another tile follows) is compile-time: a wgmma issued under a runtime branch serializes (C7513).
#ifndef ATTN_P3_HAND
#define ATTN_P3_HAND ATTN_RR_HAND
#endif
#ifndef ATTN_P3_DIRECT
#define ATTN_P3_DIRECT 0  // 1: exps packed straight into P (S only read); needs ATTN_P3_PVW >= 1
#endif
#ifndef ATTN_P3_PVW
#define ATTN_P3_PVW 0  // where softmax(t) waits for P.V(t-1): 0 after the exps, 1 after the row max, 2 before the row max
#endif
template <bool MASK, int RR, bool PVPEND, bool MORE>
DEVI void softmax_core_p3(const AttnCtx& cx, AttnState& st, float* sacc, int k0, uint32_t (&pa)[4][4]) {
  auto pv_wait = [&]() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (PVPEND) {
      if constexpr (MORE) wg_wait<1>();  // QK(t+1), issued after P.V(t-1), may still run
      else wg_wait<0>();
    }
#endif
    fence_u<16>(&pa[0][0]);
    fence_f<36>(st.o);
  };
  if constexpr (ATTN_P3_PVW == 2) pv_wait();
  if constexpr (MASK) {
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      const int key = k0 + 8 * (i >> 2) + 2 * cx.t4 + (i & 1);
      const bool bad = (i & 2) ? (key > cx.kmaxB || key < cx.kminB) : (key > cx.kmaxA || key < cx.kminA);
      if (bad) sacc[i] = -INFINITY;
    }
  }
  const float c = cx.p.c;
#define MX4(o) fmaxf(fmaxf(sacc[(o)], sacc[(o) + 1]), fmaxf(sacc[(o) + 4], sacc[(o) + 5]))
  const float a0 = fmaxf(MX4(0), MX4(8)), a1 = fmaxf(MX4(16), MX4(24)), a2 = fmaxf(MX4(32), MX4(40)), a3 = fmaxf(MX4(48), MX4(56));
  const float b0 = fmaxf(MX4(2), MX4(10)), b1 = fmaxf(MX4(18), MX4(26)), b2 = fmaxf(MX4(34), MX4(42)), b3 = fmaxf(MX4(50), MX4(58));
#undef MX4
  float tmA = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)), tmB = fmaxf(fmaxf(b0, b1), fmaxf(b2, b3));
  float nmA = st.mA, nmB = st.mB;
  float scA = 1.f, scB = 1.f;
  if (!ATTN_VOTEMAX || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
  if constexpr (ATTN_P3_PVW >= 1) {
    if constexpr (ATTN_P3_PVW == 1) pv_wait();
    rescale_o(st, scA, scB);
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#ifdef ATTN_PROF
  const long long tq0 = clock64();
#endif
  if constexpr (RR > 1) named_bar_sync(4 + cx.wg, 256);  // my turn
#ifdef ATTN_PROF
  st.pt += clock64() - tq0;
#endif
  fence_f<64>(sacc);
#endif
#if ATTN_P3_DIRECT
  // exps packed straight into P (S is only read: nothing written over a wgmma accumulator)
  static_assert(ATTN_P3_PVW >= 1, "direct packing overwrites P during the exps: wait for P.V(t-1) first");
  auto pk = [&](int r) -> uint32_t {
    const int cc = r >> 2, j = r & 3;
    const int base = 16 * cc + ((j & 2) ? 8 : 0) + ((j & 1) ? 2 : 0);
    const float off = (j & 1) ? offB : offA;
    return pack_e4m3x4(exp2f(fmaf(sacc[base], c, -off)), exp2f(fmaf(sacc[base + 1], c, -off)),
                       exp2f(fmaf(sacc[base + 4], c, -off)), exp2f(fmaf(sacc[base + 5], c, -off)));
  };
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    pa[r >> 2][r & 3] = pk(r);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (RR > 1) {
      if (4 * (r + 1) == (ATTN_P3_HAND & ~3)) {
        fence_u<(ATTN_P3_HAND >> 2)>(&pa[0][0]);
        named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);
      }
    }
#endif
  }
#else
#pragma unroll
  for (int i = 0; i < ATTN_P3_HAND; ++i) sacc[i] = exp2f(fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (RR > 1) {
    fence_f<ATTN_P3_HAND>(sacc);
    named_bar_arrive(4 + (cx.wg + 1 == RR ? 0 : cx.wg + 1), 256);
  }
#endif
#pragma unroll
  for (int i = ATTN_P3_HAND; i < 64; ++i) sacc[i] = exp2f(fmaf(sacc[i], c, (i & 2) ? -offB : -offA));
  // P.V(t-1) reads P and accumulates into O: it must be done before O is rescaled and P is overwritten
  if constexpr (ATTN_P3_PVW == 0) {
    pv_wait();
    rescale_o(st, scA, scB);
  }
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const float* x = sacc + 16 * cc;
    pa[cc][0] = pack_e4m3x4(x[0], x[1], x[4], x[5]);
    pa[cc][1] = pack_e4m3x4(x[2], x[3], x[6], x[7]);
    pa[cc][2] = pack_e4m3x4(x[8], x[9], x[12], x[13]);
    pa[cc][3] = pack_e4m3x4(x[10], x[11], x[14], x[15]);
  }
#endif
}

template <int RR, bool PVPEND, bool MORE>
DEVI void softmax_tile_p3(const AttnCtx& cx, AttnState& st, float* sacc, int t, uint32_t (&pa)[4][4]) {
  const AttnArgs& p = cx.p;
  const int k0 = t * NK;
  const int L = cx.L, nk = cx.nk;
  const bool need_mask = (k0 + NK - 1 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
#ifdef ATTN_P3_DBG_NOMASK  // compile experiment only (wrong results on boundary tiles)
  softmax_core_p3<false, RR, PVPEND, MORE>(cx, st, sacc, k0, pa);
#else
  if (need_mask) softmax_core_p3<true, RR, PVPEND, MORE>(cx, st, sacc, k0, pa);
  else softmax_core_p3<false, RR, PVPEND, MORE>(cx, st, sacc, k0, pa);
#endif
}

template <int RR, bool MORE>
DEVI void pipe3_step(const AttnCtx& cx, AttnState& st, int t, float* s_cur, float* s_next, uint32_t (&pa)[4][4]) {
  const int sp = stage_of(cx, t - 1);
#ifdef ATTN_PROF
  const long long c0 = clock64();
#endif
#ifndef ATTN_P3_DBG_NOPV  // compile experiment only: no P.V in the loop
  issue_pv(cx, st.o, pa, sp);  // P(t-1), V(t-1)
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  wg_fence(); wg_commit();
#endif
#ifdef ATTN_PROF
  const long long ca = clock64();
  long long cb = ca;
#endif
  if constexpr (MORE) {
    const int sn = stage_of(cx, t + 1);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, t + 1));
#ifdef ATTN_PROF
    cb = clock64();
#endif
    issue_qk(cx, s_next, sn);
  }
#ifdef ATTN_PROF
  const long long c1 = clock64();
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (MORE) wg_wait<2>();  // QK(t) is done; P.V(t-1) and QK(t+1) may still run
  else wg_wait<1>();
#endif
#ifdef ATTN_PROF
  const long long c2 = clock64();
#endif
  softmax_tile_p3<RR, true, MORE>(cx, st, s_cur, t, pa);
#ifdef ATTN_PROF
  const long long c3 = clock64();
#endif
  release_stage(cx, sp);
#ifdef ATTN_PROF
#ifdef ATTN_P3_PROF_ISSUE  // "wait" = P.V issue, "qk" = K/V barrier wait, "pv" = QK issue + QK wait + the tail
  st.pw += ca - c0; st.pq += cb - ca; st.ps += c3 - c2; st.pp += (c2 - cb) + (clock64() - c3);
#else
  st.pw += c1 - c0; st.pq += c2 - c1; st.ps += c3 - c2; st.pp += clock64() - c3;
#endif
#endif
}

template <int RR>
DEVI void attn_pipe3(const AttnCtx& cx, AttnState& st, float* s0, float* s1) {
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  uint32_t pa[4][4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  {
    // QK(tb + 1) is issued even for a single-tile range (the producer loads a dummy tile): one uniform prologue
    // (a separate single-tile path makes ptxas serialize every wgmma: C7511)
    const int s = stage_of(cx, tb), sn = stage_of(cx, tb + 1);
    mbar_wait(cx.bars + 8 * s, phase_of(cx, tb));
    issue_qk(cx, s0, s);
    mbar_wait(cx.bars + 8 * sn, phase_of(cx, tb + 1));
    issue_qk(cx, s1, sn);
    wg_wait<1>();  // QK(tb) is done; QK(tb+1) may still run
    softmax_tile_p3<RR, false, true>(cx, st, s0, tb, pa);  // O is still zero: nothing pending on it
  }
  int t = tb + 1;  // QK(t) in flight, into s1
  for (; t + 1 < te; t += 2) {
    pipe3_step<RR, true>(cx, st, t, s1, s0, pa);
    pipe3_step<RR, true>(cx, st, t + 1, s0, s1, pa);
  }
  if (t + 1 == te) {  // two tiles left
    pipe3_step<RR, true>(cx, st, t, s1, s0, pa);
    pipe3_step<RR, false>(cx, st, t + 1, s0, s1, pa);
  } else if (t == te) {  // one tile left
    pipe3_step<RR, false>(cx, st, t, s1, s0, pa);
  } else {  // a single-tile range: retire the dummy QK
    wg_wait<0>();
  }
  pipe_last(cx, st, pa);
#endif
}

// ---------------- half tiles, software-pipelined (QK of the next 64-key half and P.V of the previous one run on the
// tensor cores during this half's softmax; two S and two P register sets of half size fit three warpgroups) ----------------
template <bool MASK>
DEVI void softmax_half(const AttnCtx& cx, AttnState& st, const float* sacc, int k0, uint32_t (&pa)[2][4], float& scA, float& scB) {
  auto V = [&](int i) -> float {
    if constexpr (!MASK) {
      return sacc[i];
    } else {
      const int key = k0 + 8 * (i >> 2) + 2 * cx.t4 + (i & 1);
      const bool bad = (i & 2) ? (key > cx.kmaxB || key < cx.kminB) : (key > cx.kmaxA || key < cx.kminA);
      return bad ? -INFINITY : sacc[i];
    }
  };
  const float c = cx.p.c;
#define MX4(o) fmaxf(fmaxf(V(o), V((o) + 1)), fmaxf(V((o) + 4), V((o) + 5)))
  float tmA = fmaxf(fmaxf(MX4(0), MX4(8)), fmaxf(MX4(16), MX4(24)));
  float tmB = fmaxf(fmaxf(MX4(2), MX4(10)), fmaxf(MX4(18), MX4(26)));
#undef MX4
  float nmA = st.mA, nmB = st.mB;
  scA = 1.f; scB = 1.f;
  // exact skip: when no lane's local maximum exceeds its row's running maximum, every row of this warp keeps
  // its maximum (rescale 1): no shuffles and no rescale exps on the exp2 units
  if (!ATTN_VOTEMAX_ALL || __any_sync(0xffffffffu, (tmA > st.mA) || (tmB > st.mB))) {
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 1));
    tmA = fmaxf(tmA, __shfl_xor_sync(0xffffffffu, tmA, 2));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 1));
    tmB = fmaxf(tmB, __shfl_xor_sync(0xffffffffu, tmB, 2));
    nmA = fmaxf(st.mA, tmA); nmB = fmaxf(st.mB, tmB);
    scA = exp2f((st.mA - nmA) * c); scB = exp2f((st.mB - nmB) * c);
    st.mA = nmA; st.mB = nmB;
  }
  const float offA = nmA * c - ATTN_POFF, offB = nmB * c - ATTN_POFF;
  auto E = [&](int i) -> float { return exp2f(fmaf(V(i), c, (i & 2) ? -offB : -offA)); };
#pragma unroll
  for (int cc = 0; cc < 2; ++cc) {
    const int o = 16 * cc;
    pa[cc][0] = pack_e4m3x4(E(o + 0), E(o + 1), E(o + 4), E(o + 5));
    pa[cc][1] = pack_e4m3x4(E(o + 2), E(o + 3), E(o + 6), E(o + 7));
    pa[cc][2] = pack_e4m3x4(E(o + 8), E(o + 9), E(o + 12), E(o + 13));
    pa[cc][3] = pack_e4m3x4(E(o + 10), E(o + 11), E(o + 14), E(o + 15));
  }
}

DEVI void softmax_half_tile(const AttnCtx& cx, AttnState& st, const float* sacc, int u, uint32_t (&pa)[2][4], float& scA, float& scB) {
  const AttnArgs& p = cx.p;
  const int k0 = (u >> 1) * NK + 64 * (u & 1);
  const int L = cx.L, nk = cx.nk;
  const bool need_mask = (k0 + 63 > nk - L + cx.q_first) || (p.window && k0 < nk - L + cx.q_last - p.window + 1);
  if (need_mask) softmax_half<true>(cx, st, sacc, k0, pa, scA, scB);
  else softmax_half<false>(cx, st, sacc, k0, pa, scA, scB);
}

template <int H>
DEVI void issue_pv_h(const AttnCtx& cx, float* o, uint32_t (&pa)[2][4], int stage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint8_t* vst = cx.ring + stage * STAGE + TILE_K;
  uint64_t dv0 = desc_sw128(smem_u32(vst)) + 4 * H;
  uint64_t dv1 = dv0 + 2;
  asm volatile("" : "+l"(dv0), "+l"(dv1));
  fence_u<8>(&pa[0][0]);
  fence_f<36>(o);
  wg_fence();
  wgmma_fp8_rs_n72(o, pa[0], dv0, 1);
  wgmma_fp8_rs_n72(o, pa[1], dv1, 1);
  wg_commit();
  fence_f<36>(o);
#endif
}

// sub-step u = 2 t + h covers keys [128 t + 64 h, +64); H (= u & 1) is a compile-time constant at every call
template <int H>
DEVI void issue_qk_u(const AttnCtx& cx, float* s, int u) {
  const int t = u >> 1;
  const int st = stage_of(cx, t);
  if (H == 0) mbar_wait(cx.bars + 8 * st, phase_of(cx, t));
  issue_qk_h(cx, s, st, H);
}

// MORE: issue the next half's QK (every step but the last; the wgmma groups never depend on a runtime branch)
template <int HC, bool MORE>
DEVI void pipe2h_step(const AttnCtx& cx, AttnState& st, int u, float* s_cur, float* s_next, uint32_t (&pin)[2][4],
                      uint32_t (&pout)[2][4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (MORE) issue_qk_u<1 - HC>(cx, s_next, u + 1);
  const int up = u - 1;
  issue_pv_h<1 - HC>(cx, st.o, pin, stage_of(cx, up >> 1));
  if constexpr (MORE) wg_wait<2>();  // QK(u) is done
  else wg_wait<1>();
  float scA = 1.f, scB = 1.f;
#if ATTN_BISECT == 1
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) pout[i][j] = __float_as_uint(s_cur[4 * i + j]);
#else
  softmax_half_tile(cx, st, s_cur, u, pout, scA, scB);
#endif
  wg_wait<0>();  // P.V(u-1) (and QK(u+1), issued before it) are done
  fence_u<8>(&pin[0][0]);
  fence_f<36>(st.o);
  if (HC == 0) release_stage(cx, stage_of(cx, up >> 1));  // u-1 was the second half of its tile
#if ATTN_BISECT != 2
  rescale_o(st, scA, scB);
#endif
#endif
}

DEVI void attn_pipe2h(const AttnCtx& cx, AttnState& st, float* s0, float* s1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const int tb = cx.t_begin, te = cx.t_end;
  if (tb > te) return;
  const int u0 = 2 * tb, u_end = 2 * te + 1;
  uint32_t p0[2][4], p1[2][4];
  {
    issue_qk_u<0>(cx, s0, u0);
    wg_wait<0>();
    issue_qk_u<1>(cx, s1, u0 + 1);  // u_end >= u0 + 1 always (two halves per tile)
    float scA, scB;
    softmax_half_tile(cx, st, s0, u0, p0, scA, scB);
  }
  int u = u0 + 1;  // odd
  for (; u < u_end; u += 2) {
    pipe2h_step<1, true>(cx, st, u, s1, s0, p0, p1);
    pipe2h_step<0, true>(cx, st, u + 1, s0, s1, p1, p0);
  }
  // u == u_end here (odd: the last tile's second half)
  pipe2h_step<1, false>(cx, st, u, s1, s0, p0, p1);
  issue_pv_h<1>(cx, st.o, p1, stage_of(cx, te));
  wg_wait<0>();
  fence_u<8>(&p1[0][0]);
  fence_f<36>(st.o);
  release_stage(cx, stage_of(cx, te));
#endif
}

template <int NWG, int EXPMODE, bool OVL, bool PIPE = false, bool PIPE2 = false, bool RRS = false, bool HALF = false, bool PIPE2H = false,
          bool PIPE3 = false>
#ifdef ATTN_DBG_LB  // compile probe only: the register budget ptxas would need (the kernel cannot launch like this)
__global__ void __launch_bounds__(ATTN_DBG_LB, 1) attn_fwd_kernel(AttnArgs p) {
#else
__global__ void __launch_bounds__(128 * (NWG + 1), 1) attn_fwd_kernel(AttnArgs p) {
#endif
  if constexpr (ATTN_EARLY_Q) pdl_trigger_attn();
  else pdl_enter_attn();
  constexpr int QBT = 8 * NWG;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);           // full[4], empty[4]
  uint8_t* qs = base + 1024;                      // 64*NWG rows x 64 B (SW64)
  uint8_t* ring = qs + 4096 * NWG;                // ASTAGES x (K tile, V tile)
#ifdef ATTN_PROF
  const size_t cta_id = ((size_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0 && g_attn_prof) g_attn_prof[1000000 + cta_id * 4 + 0] = gtimer();
#endif
  const int qb = p.nqb - 1 - blockIdx.x;          // heaviest (latest) query blocks first
  const int h = blockIdx.y & 1, split = blockIdx.y >> 1, b = blockIdx.z;
  const int L = p.L;
  const int seq = p.seq_lens[b];
  const int first = p.window ? max(0, seq - p.window + 1) : 0;
  const int nk = seq + L - first;
  const int q_first = qb * QBT;
  const int q_last = min(q_first + QBT - 1, L - 1);
  const int kmax_last = nk - L + q_last;
  const int kmin_first = p.window ? max(0, nk - L + q_first - p.window + 1) : 0;
  int t_begin = kmin_first / NK, t_end = kmax_last / NK;
  if (p.nsplit > 1) {
    const int ntiles = t_end - t_begin + 1;
    const int per = (ntiles + p.nsplit - 1) / p.nsplit;
    const int tb = t_begin + split * per;
    t_end = min(t_end, tb + per - 1);
    t_begin = tb;  // may exceed t_end: empty split
  }
  const size_t tb0 = (size_t)(b * 2 + h) * p.ntk;
  if (threadIdx.x == 0) {
    for (int s = 0; s < ASTAGES; ++s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      mbar_init(bars + 8 * s, 1);
#else
      mbar_init(bars + 8 * s, 128);
#endif
      mbar_init(bars + 8 * ASTAGES + 8 * s, 4 * NWG);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    // ---------------- producer warpgroup (one thread issues the bulk copies) ----------------
    const int ptid = threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (NWG >= 3 || PIPE2 || PIPE3) asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
    else asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (ptid == 0) {
      if constexpr (ATTN_EARLY_Q) pdl_wait_attn();  // the staged K/V tiles
      if (ATTN_PF && p.pf != nullptr) {
        // the attention leaves DRAM mostly idle: warm L2 with this CTA's slice of the next GEMM's weight
        const unsigned long long nct = (unsigned long long)gridDim.x * gridDim.y * gridDim.z;
        const unsigned long long cid = ((unsigned long long)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        const unsigned long long per = (p.pf_bytes / nct + 255ull) & ~255ull;
        const unsigned long long b0 = cid * per, b1 = min(b0 + per, p.pf_bytes);
        for (unsigned long long off = b0; off < b1; off += 32768ull)
          prefetch_l2(p.pf + off, (uint32_t)min(32768ull, b1 - off) & ~15u);
      }
      int s = 0;
      uint32_t ph = 0;
      // PIPE3 always issues QK(t_begin + 1): a single-tile range gets a dummy second load (tile t_end again)
      constexpr bool DUMMY = PIPE3 || (PIPE && RRS && ATTN_EARLY_QK);  // always issue QK(t_begin + 1)
      const int t_last = (DUMMY && t_begin == t_end) ? t_end + 1 : t_end;
      for (int t = t_begin; t <= t_last; ++t) {
        mbar_wait(bars + 8 * ASTAGES + 8 * s, ph ^ 1);
        const uint32_t full = bars + 8 * s;
        mbar_expect_tx(full, STAGE);
        const uint32_t dst = smem_u32(ring + s * STAGE);
        const int tt = min(t, t_end);
        bulk_g2s(dst, p.kt + (tb0 + tt) * TILE_K, TILE_K, full);
        bulk_g2s(dst + TILE_K, p.vt + (tb0 + tt) * TILE_V, TILE_V, full);
        if (++s == ASTAGES) { s = 0; ph ^= 1; }
      }
    }
#else
    if constexpr (ATTN_EARLY_Q) pdl_wait_attn();
    int s = 0;
    uint32_t ph = 0;
    for (int t = t_begin; t <= t_end; ++t) {
      mbar_wait(bars + 8 * ASTAGES + 8 * s, ph ^ 1);
      const uint32_t dst = smem_u32(ring + s * STAGE);
      for (int i = ptid; i < STAGE / 16; i += 128) {
        const uint8_t* src = i < TILE_K / 16 ? p.kt + (tb0 + t) * TILE_K + i * 16 : p.vt + (tb0 + t) * TILE_V + (i * 16 - TILE_K);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + i * 16), "l"(src) : "memory");
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(bars + 8 * s) : "memory");
      if (++s == ASTAGES) { s = 0; ph ^= 1; }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
    return;
  }
  // ---------------- consumers (warpgroups 1 .. NWG) ----------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // setmaxnreg moves registers inside the CTA's launch allocation: 640 threads launch with 96 each, and the producer's
  // 72 x 128 freed registers cover at most +16 for each of the 512 consumer threads
  if constexpr (NWG == 4) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\n");
  else if constexpr (NWG == 3) asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n");
  else if constexpr (PIPE2 || PIPE3) asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
#endif
  const int ctid = threadIdx.x - 128;
  const int wg = ctid >> 7, w = (ctid >> 5) & 3, lane = ctid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  // stage Q: row r = (token q_first + r/8, head 8h + r%8) -> smem SW64
  for (int i = ctid; i < 64 * NWG * 4; i += 128 * NWG) {
    const int r = i >> 2, c = i & 3;
    const int tok = q_first + (r >> 3);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tok < L) {
      const size_t trow = (size_t)b * L + tok;
      v = *reinterpret_cast<const uint4*>(p.q + (trow * 16 + 8 * h + (r & 7)) * 64 + c * 16);
    }
    *reinterpret_cast<uint4*>(qs + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = v;
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
  named_bar_sync(1, 128 * NWG);
  // this thread's rows: A = (token q_first + 8wg + 2w, head g), B = token + 1
  const int tokA = q_first + 8 * wg + 2 * w, tokB = tokA + 1;
  const int kmaxA = nk - L + tokA, kmaxB = nk - L + tokB;
  const int kminA = p.window ? kmaxA - p.window + 1 : -1, kminB = p.window ? kmaxB - p.window + 1 : -1;
  AttnState stt;
#pragma unroll
  for (int i = 0; i < 36; ++i) stt.o[i] = 0.f;
  stt.mA = -1e30f; stt.mB = -1e30f; stt.nready = 0;
#ifdef ATTN_PROF
  stt.pw = stt.pq = stt.ps = stt.pp = stt.pt = 0;
  const long long c_start = clock64();
  if (ctid == 0 && g_attn_prof) g_attn_prof[1000000 + cta_id * 4 + 1] = gtimer();
#endif
  AttnCtx cx{p, qs, ring, bars, wg, w, lane, g, t4, kmaxA, kmaxB, kminA, kminB, nk, L, q_first, q_last, t_begin, t_end,
             desc_sw64(smem_u32(qs + wg * 64 * 64)), desc_sw64(smem_u32(ring)), desc_sw128(smem_u32(ring + TILE_K)), t_begin};
  // the output gate, loaded before the tile loop so that the epilogue does not wait on global loads (variants with
  // registers to spare only: the pipelined ones would spill)
  constexpr bool GPF = ATTN_GATE_PF && !PIPE && !PIPE2 && !PIPE2H && !PIPE3 && !HALF && NWG <= 3;
  uint32_t gpf[2][8];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int tok = half ? tokB : tokA;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) gpf[half][jj] = 0u;
    if (GPF && p.nsplit == 1 && tok < L) {
      const __nv_bfloat16* gp = p.qkv + ((size_t)b * L + tok) * 2304 + 1280 + (8 * h + g) * 64;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) gpf[half][jj] = *reinterpret_cast<const uint32_t*>(gp + 8 * jj + 2 * t4);
    }
  }
  float s0[64];
  if (OVL && t_begin <= t_end) {
    mbar_wait(bars, 0);
    issue_qk(cx, s0, 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    wg_wait<0>();
#else
    emul_qk(cx, s0, 0);
#endif
  }
  if constexpr (PIPE2H) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    float s1[32];
    attn_pipe2h(cx, stt, s0, s1);
#else
    for (int t = t_begin; t <= t_end; ++t) attn_step<EXPMODE, false>(cx, stt, s0, t);
#endif
  } else if constexpr (HALF) {
    for (int t = t_begin; t <= t_end; ++t) attn_step_h<EXPMODE>(cx, stt, s0, t);
  } else if constexpr (PIPE3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (wg == NWG - 1 && t_begin <= t_end) named_bar_arrive(4, 256);  // warpgroup 0 starts
    float s1[64];
#ifndef ATTN_P3_RR
#define ATTN_P3_RR NWG
#endif
    attn_pipe3<ATTN_P3_RR>(cx, stt, s0, s1);
#else
    for (int t = t_begin; t <= t_end; ++t) attn_step<EXPMODE, false>(cx, stt, s0, t);
#endif
  } else if constexpr (PIPE2) {
    float s1[64];
    attn_pipe2<EXPMODE>(cx, stt, s0, s1);
  } else if constexpr (PIPE && RRS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (wg == NWG - 1 && t_begin <= t_end) named_bar_arrive(4, 256);  // warpgroup 0 starts
#endif
    if constexpr (ATTN_EARLY_QK) attn_pipe_eq<EXPMODE, NWG>(cx, stt, s0);
    else attn_pipe<EXPMODE, NWG>(cx, stt, s0);
  } else if constexpr (PIPE) attn_pipe<EXPMODE>(cx, stt, s0);
  else if constexpr (RRS) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (wg == NWG - 1 && t_begin <= t_end) named_bar_arrive(4, 256);  // warpgroup 0 starts
#endif
    for (int t = t_begin; t <= t_end; ++t) attn_step<EXPMODE, OVL, NWG>(cx, stt, s0, t);
  } else
    for (int t = t_begin; t <= t_end; ++t) attn_step<EXPMODE, OVL>(cx, stt, s0, t);
#ifdef ATTN_PROF
  if (ctid == 0 && g_attn_prof) g_attn_prof[1000000 + cta_id * 4 + 2] = gtimer();
  if ((ctid & 127) == 0 && g_attn_prof) {
    const size_t cta = ((size_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    unsigned long long* pr = g_attn_prof + (cta * 4 + wg) * 8;
    pr[0] = stt.pw; pr[1] = stt.pq; pr[2] = stt.ps; pr[3] = stt.pp; pr[4] = clock64() - c_start;
    pr[5] = t_end - t_begin + 1;
    unsigned int sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); pr[6] = sm;
    pr[7] = stt.pt;
  }
#endif
  float* o = stt.o;
  const float mA = stt.mA, mB = stt.mB;
  // the denominator (column 64) sits in the t4 == 0 lane of each quad
  const float lA = __shfl_sync(0xffffffffu, o[32], lane & ~3), lB = __shfl_sync(0xffffffffu, o[34], lane & ~3);
  const int head = 8 * h + g;
  // ---- epilogue: partials for a split, else O / l, per-head RMSNorm, * sigmoid(gate) ----
  if (p.nsplit > 1) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int tok = half ? tokB : tokA;
      if (tok >= L) continue;
      const size_t trow = (size_t)b * L + tok;
      const size_t T = (size_t)gridDim.z * L;
      float* wp = p.ws + ((size_t)split * T * 16 + trow * 16 + head) * 66;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
        *reinterpret_cast<float2*>(wp + 8 * jj + 2 * t4) = make_float2(o[4 * jj + 2 * half], o[4 * jj + 2 * half + 1]);
      if (t4 == 0) {
        wp[64] = half ? mB : mA;
        wp[65] = half ? lB : lA;
      }
    }
#if ATTN_MERGE
    if (g_attn_cnt != nullptr) {
      // the last split to finish this (batch, query block, kv head) merges every split's partials and writes the output
      // (the threadfence-reduction pattern: each split's stores, fence, then one counter atomic per CTA)
      __shared__ int s_last;
      __threadfence();
      named_bar_sync(1, 128 * NWG);
      if (ctid == 0) {
        const int unit = (b * p.nqb + qb) * 2 + h;
        const int old = atomicAdd(g_attn_cnt + unit, 1);
        s_last = old == p.nsplit - 1;
        if (s_last) g_attn_cnt[unit] = 0;  // every split has arrived: ready for the next call
      }
      named_bar_sync(1, 128 * NWG);
      if (s_last) {
        __threadfence();
        const size_t T = (size_t)gridDim.z * L;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int tok = half ? tokB : tokA;
          if (tok >= L) continue;
          const size_t trow = (size_t)b * L + tok;
          float M = -1e30f;
          for (int s2 = 0; s2 < p.nsplit; ++s2) M = fmaxf(M, p.ws[((size_t)s2 * T * 16 + trow * 16 + head) * 66 + 64]);
          float om[16], lm = 0.f;
#pragma unroll
          for (int i = 0; i < 16; ++i) om[i] = 0.f;
          for (int s2 = 0; s2 < p.nsplit; ++s2) {
            const float* wq = p.ws + ((size_t)s2 * T * 16 + trow * 16 + head) * 66;
            const float ls = wq[65], ms = wq[64];
            if (ls != 0.f) {  // an empty split contributes nothing
              const float sc = exp2f((ms - M) * p.c);
#pragma unroll
              for (int jj = 0; jj < 8; ++jj) {
                const float2 v = *reinterpret_cast<const float2*>(wq + 8 * jj + 2 * t4);
                om[2 * jj] += sc * v.x;
                om[2 * jj + 1] += sc * v.y;
              }
              lm += sc * ls;
            }
          }
          const float inv = p.v_scale / lm;
          float ss = 0.f;
#pragma unroll
          for (int i = 0; i < 16; ++i) { om[i] *= inv; ss += om[i] * om[i]; }
          ss += __shfl_xor_sync(0xffffffffu, ss, 1);
          ss += __shfl_xor_sync(0xffffffffu, ss, 2);
          const float rr = rsqrtf(ss / 64.f + 1e-5f);
          const __nv_bfloat16* gp = p.qkv + trow * 2304 + 1280 + head * 64;
          __nv_bfloat16* op = p.out + trow * 1024 + head * 64;
#pragma unroll
          for (int jj = 0; jj < 8; ++jj) {
            const int d = 8 * jj + 2 * t4;
            const float2 gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + d));
            const float x0 = om[2 * jj] * rr, x1 = om[2 * jj + 1] * rr;
            const float y0 = x0 / (1.f + __expf(-gv.x)), y1 = x1 / (1.f + __expf(-gv.y));
            *reinterpret_cast<__nv_bfloat162*>(op + d) = __floats2bfloat162_rn(y0, y1);
          }
        }
      }
    }
#endif
#ifdef ATTN_PROF
    if (ctid == 0 && g_attn_prof) g_attn_prof[1000000 + cta_id * 4 + 3] = gtimer();
#endif
    return;
  }
  const float iA = p.v_scale / lA, iB = p.v_scale / lB;
  float ssA = 0.f, ssB = 0.f;
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) {
    o[4 * jj] *= iA; o[4 * jj + 1] *= iA; o[4 * jj + 2] *= iB; o[4 * jj + 3] *= iB;
    ssA += o[4 * jj] * o[4 * jj] + o[4 * jj + 1] * o[4 * jj + 1];
    ssB += o[4 * jj + 2] * o[4 * jj + 2] + o[4 * jj + 3] * o[4 * jj + 3];
  }
  ssA += __shfl_xor_sync(0xffffffffu, ssA, 1);
  ssA += __shfl_xor_sync(0xffffffffu, ssA, 2);
  ssB += __shfl_xor_sync(0xffffffffu, ssB, 1);
  ssB += __shfl_xor_sync(0xffffffffu, ssB, 2);
  const float rA = rsqrtf(ssA / 64.f + 1e-5f), rB = rsqrtf(ssB / 64.f + 1e-5f);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int tok = half ? tokB : tokA;
    if (tok >= L) continue;
    const size_t trow = (size_t)b * L + tok;
    const float rr = half ? rB : rA;
    const __nv_bfloat16* gp = p.qkv + trow * 2304 + 1280 + head * 64;
    __nv_bfloat16* op = p.out + trow * 1024 + head * 64;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      const int d = 8 * jj + 2 * t4;
      float2 gv;
      if constexpr (GPF) {
        const uint32_t gbits = gpf[half][jj];
        gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gbits));
      } else {
        gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + d));
      }
      const float x0 = o[4 * jj + 2 * half] * rr, x1 = o[4 * jj + 2 * half + 1] * rr;
      const float y0 = x0 / (1.f + __expf(-gv.x)), y1 = x1 / (1.f + __expf(-gv.y));
      *reinterpret_cast<__nv_bfloat162*>(op + d) = __floats2bfloat162_rn(y0, y1);
    }
  }
#ifdef ATTN_PROF
  if (ctid == 0 && g_attn_prof) g_attn_prof[1000000 + cta_id * 4 + 3] = gtimer();
#endif
}

// ---------------- stream-K attention ----------------
// The (pair, key tile) space, pair = (batch, query block, kv head) in that order, is cut into gridDim.x equal ranges of W
// tiles, one per CTA, so every CTA (and SM) gets the same work: a fixed grid of nqb x 2 x B CTAs leaves the last wave
// part-empty (T=753 full: 128 CTAs on 132 SMs; T=3012 full: 256 CTAs = 1.94 waves). A CTA runs its range as one segment
// per pair it touches, each writing a partial (O, m, l) for the pair's QBT x 8 rows to ws slot pair + cta (unique: every
// segment starts at a pair or a CTA boundary, so slots run 0 .. npairs + G - 2); attn_combine_sk_kernel merges them. The ring (stage, phase) continues across segments, and the exp-turn
// hand-off of the last warpgroup at a segment's end opens the next segment (no extra start arrive after segment 0).
#ifndef ATTN_SK_MAXSEG
#define ATTN_SK_MAXSEG 4
#endif
#ifndef ATTN_SK_MAXPAIRS
#define ATTN_SK_MAXPAIRS 1536
#endif
DEVI void sk_pair(const AttnArgs& p, int QBT, int pp, int& b, int& qb, int& h, int& tb, int& te, int& nk) {
  h = pp & 1;
  const int bq = pp >> 1;
  b = bq / p.nqb;
  qb = bq - b * p.nqb;
  const int L = p.L, seq = p.seq_lens[b];
  const int first = p.window ? max(0, seq - p.window + 1) : 0;
  nk = seq + L - first;
  const int q_first = qb * QBT, q_last = min(q_first + QBT - 1, L - 1);
  const int kmin_first = p.window ? max(0, nk - L + q_first - p.window + 1) : 0;
  tb = kmin_first / NK;
  te = (nk - L + q_last) / NK;
}

template <int NWG, int EXPMODE, bool PIPE, bool RRS>
__global__ void __launch_bounds__(128 * (NWG + 1), 1) attn_fwd_sk_kernel(AttnArgs p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  pdl_enter_attn();
  constexpr int QBT = 8 * NWG, NT = 128 * (NWG + 1);
  constexpr int PER = (ATTN_SK_MAXPAIRS + NT - 1) / NT;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = ((raw + 1023) & ~1023u) - raw;
  uint8_t* base = smem_raw + pad;
  const uint32_t bars = smem_u32(base);
  uint8_t* qs = base + 1024;
  uint8_t* ring = qs + 4096 * NWG;
  __shared__ int s_pre[ATTN_SK_MAXPAIRS + 1];
  __shared__ int s_wsum[NT / 32];
  const int npairs = p.nb * p.nqb * 2;
  // per-pair tile counts -> inclusive prefix s_pre[pp + 1] (thread i owns pairs [i * PER, i * PER + PER))
  {
    int cnt[PER];
    int sum = 0;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      const int pp = threadIdx.x * PER + k;
      int n = 0;
      if (pp < npairs) {
        int b, qb, h, tb, te, nk;
        sk_pair(p, QBT, pp, b, qb, h, tb, te, nk);
        n = max(0, te - tb + 1);
      }
      cnt[k] = n;
      sum += n;
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += v;
    }
    if (lane == 31) s_wsum[wid] = incl;
    __syncthreads();
    int wbase = 0;
    for (int i = 0; i < wid; ++i) wbase += s_wsum[i];
    int run = wbase + incl - sum;  // exclusive prefix of this thread's first pair
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      const int pp = threadIdx.x * PER + k;
      run += cnt[k];
      if (pp < npairs) s_pre[pp + 1] = run;
    }
    if (threadIdx.x == 0) s_pre[0] = 0;
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < ASTAGES; ++s) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * ASTAGES + 8 * s, 4 * NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int total = s_pre[npairs];
  const int W = (total + (int)gridDim.x - 1) / (int)gridDim.x;
  const int r0 = min((int)blockIdx.x * W, total), r1 = min(r0 + W, total);
  // first pair of the range: the last pp with s_pre[pp] <= r0 (binary search)
  int lo = 0, hi = npairs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (s_pre[mid] <= r0) lo = mid; else hi = mid - 1;
  }
  const int pp0 = lo;
  if (threadIdx.x == 0) {  // for attn_combine_sk_kernel: W, the prefix, each CTA's first pair
    p.skbuf[2 + npairs + blockIdx.x] = pp0;
    if (blockIdx.x == 0) p.skbuf[0] = W;
  }
  if (blockIdx.x == 0)
    for (int i = threadIdx.x; i <= npairs; i += NT) p.skbuf[1 + i] = s_pre[i];
  if (threadIdx.x < 128) {
    // ---------------- producer warpgroup ----------------
    if constexpr (NWG >= 3) asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
    else asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (threadIdx.x == 0) {
      int s = 0;
      uint32_t ph = 0;
      for (int pp = pp0; r0 < r1 && pp < npairs && s_pre[pp] < r1; ++pp) {
        int b, qb, h, tb, te, nk;
        sk_pair(p, QBT, pp, b, qb, h, tb, te, nk);
        const int l0 = max(r0, s_pre[pp]) - s_pre[pp], l1 = min(r1, s_pre[pp + 1]) - s_pre[pp];
        const size_t tb0 = (size_t)(b * 2 + h) * p.ntk;
        for (int t = tb + l0; t < tb + l1; ++t) {
          mbar_wait(bars + 8 * ASTAGES + 8 * s, ph ^ 1);
          const uint32_t full = bars + 8 * s;
          mbar_expect_tx(full, STAGE);
          const uint32_t dst = smem_u32(ring + s * STAGE);
          bulk_g2s(dst, p.kt + (tb0 + t) * TILE_K, TILE_K, full);
          bulk_g2s(dst + TILE_K, p.vt + (tb0 + t) * TILE_V, TILE_V, full);
          if (++s == ASTAGES) { s = 0; ph ^= 1; }
        }
      }
    }
    return;
  }
  // ---------------- consumers ----------------
  if constexpr (NWG == 3) asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  const int ctid = threadIdx.x - 128;
  const int wg = ctid >> 7, w = (ctid >> 5) & 3, lane = ctid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  // the consumers' loop state (pair, ring position) lives in shared memory, not in registers across the tile loop:
  // ctid 0 writes the next segment's values after the post-staging barrier; everyone reads them after the top barrier
  __shared__ int s_loop[2];
  if (ctid == 0) { s_loop[0] = pp0; s_loop[1] = 0; }
  for (int seg = 0;; ++seg) {
    named_bar_sync(1, 128 * NWG);  // (also: every consumer warpgroup is done with the previous segment's Q)
    const int pp = reinterpret_cast<volatile int*>(s_loop)[0], ring0 = reinterpret_cast<volatile int*>(s_loop)[1];
    if (!(r0 < r1 && pp < p.nb * p.nqb * 2 && s_pre[pp] < r1)) break;
    int b, qb, h, tb, te, nk;
    sk_pair(p, QBT, pp, b, qb, h, tb, te, nk);
    const int l0 = max(r0, s_pre[pp]) - s_pre[pp], l1 = min(r1, s_pre[pp + 1]) - s_pre[pp];
    const int stb = tb + l0, ste = tb + l1 - 1;
    const int L = p.L;
    const int q_first = qb * QBT, q_last = min(q_first + QBT - 1, L - 1);
    for (int i = ctid; i < 64 * NWG * 4; i += 128 * NWG) {
      const int r = i >> 2, c = i & 3;
      const int tok = q_first + (r >> 3);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (tok < L) {
        const size_t trow = (size_t)b * L + tok;
        v = *reinterpret_cast<const uint4*>(p.q + (trow * 16 + 8 * h + (r & 7)) * 64 + c * 16);
      }
      *reinterpret_cast<uint4*>(qs + r * 64 + ((c ^ ((r >> 1) & 3)) << 4)) = v;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    named_bar_sync(1, 128 * NWG);
    if (ctid == 0) { s_loop[0] = pp + 1; s_loop[1] = ring0 + ste - stb + 1; }  // read after the next top barrier
    const int tokA = q_first + 8 * wg + 2 * w, tokB = tokA + 1;
    const int kmaxA = nk - L + tokA, kmaxB = nk - L + tokB;
    const int kminA = p.window ? kmaxA - p.window + 1 : -1, kminB = p.window ? kmaxB - p.window + 1 : -1;
    AttnState stt;
#pragma unroll
    for (int i = 0; i < 36; ++i) stt.o[i] = 0.f;
    stt.mA = -1e30f; stt.mB = -1e30f; stt.nready = 0;
#ifdef ATTN_PROF
    stt.pw = stt.pq = stt.ps = stt.pp = stt.pt = 0;
#endif
    AttnCtx cx{p, qs, ring, bars, wg, w, lane, g, t4, kmaxA, kmaxB, kminA, kminB, nk, L, q_first, q_last, stb, ste,
               desc_sw64(smem_u32(qs + wg * 64 * 64)), desc_sw64(smem_u32(ring)), desc_sw128(smem_u32(ring + TILE_K)),
               stb - ring0};
    float s0[64];
    if constexpr (PIPE && RRS) {
      if (seg == 0 && wg == NWG - 1) named_bar_arrive(4, 256);  // warpgroup 0 starts (later segments: the last hand-off)
      attn_pipe<EXPMODE, NWG>(cx, stt, s0);
    } else {
      for (int t = stb; t <= ste; ++t) attn_step<EXPMODE, false>(cx, stt, s0, t);
    }
    // partial (O unnormalized, m, l) of this segment: slot blockIdx.x * MAXSEG + seg, row (token in block) * 8 + g
    float* o = stt.o;
    const float lA = __shfl_sync(0xffffffffu, o[32], lane & ~3), lB = __shfl_sync(0xffffffffu, o[34], lane & ~3);
    float* wslot = p.ws + ((size_t)pp + blockIdx.x) * (QBT * 8 * 66);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int tok = half ? tokB : tokA;
      if (tok >= L) continue;
      float* wp = wslot + ((tok - q_first) * 8 + g) * 66;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
        *reinterpret_cast<float2*>(wp + 8 * jj + 2 * t4) = make_float2(o[4 * jj + 2 * half], o[4 * jj + 2 * half + 1]);
      if (t4 == 0) {
        wp[64] = half ? stt.mB : stt.mA;
        wp[65] = half ? lB : lA;
      }
    }
  }
#else
  __trap();  // sm_90 only
#endif
}

// stream-K merge: one warp per R (token, head) rows (adjacent heads of one kv head), the pair's partials from the CTAs
// whose ranges cover it, then O / l, the per-head RMSNorm and * sigmoid(gate)
template <int R, int QBT>
__global__ void __launch_bounds__(256) attn_combine_sk_kernel(const float* __restrict__ ws, const int* __restrict__ skbuf,
                                                              int T, int L, int nqb, const __nv_bfloat16* __restrict__ qkv,
                                                              __nv_bfloat16* __restrict__ out, float c, float v_scale) {
  pdl_enter_attn();
  const int wid = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (wid >= T * (16 / R)) return;
  const int r0 = R * wid;
  const int tok = r0 >> 4, head = r0 & 15;
  const int b = tok / L, tq = tok - b * L, qb = tq / QBT;
  const int npairs = (T / L) * nqb * 2;
  const int pp = (b * nqb + qb) * 2 + (head >> 3);
  const int W = skbuf[0];
  const int* pre = skbuf + 1;
  const int i0 = pre[pp] / W, i1 = (pre[pp + 1] - 1) / W;
  const int rowp = (tq - qb * QBT) * 8 + (head & 7);
  float M[R], o0[R], o1[R], l[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { M[r] = -1e30f; o0[r] = 0.f; o1[r] = 0.f; l[r] = 0.f; }
  for (int i = i0; i <= i1; ++i) {
    const float* wp = ws + (((size_t)pp + i) * (QBT * 8) + rowp) * 66;
#pragma unroll
    for (int r = 0; r < R; ++r) M[r] = fmaxf(M[r], wp[66 * r + 64]);
  }
  for (int i = i0; i <= i1; ++i) {
    const float* wp = ws + (((size_t)pp + i) * (QBT * 8) + rowp) * 66;
    float ls[R], ms[R];
    float2 v[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      ls[r] = wp[66 * r + 65];
      ms[r] = wp[66 * r + 64];
      v[r] = *reinterpret_cast<const float2*>(wp + 66 * r + 2 * lane);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (ls[r] != 0.f) {
        const float sc = exp2f((ms[r] - M[r]) * c);
        o0[r] += sc * v[r].x; o1[r] += sc * v[r].y; l[r] += sc * ls[r];
      }
    }
  }
  float ss[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float inv = v_scale / l[r];
    o0[r] *= inv; o1[r] *= inv;
    ss[r] = o0[r] * o0[r] + o1[r] * o1[r];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int r = 0; r < R; ++r) ss[r] += __shfl_xor_sync(0xffffffffu, ss[r], off);
  }
  const __nv_bfloat16* gp = qkv + (size_t)tok * 2304 + 1280 + head * 64 + 2 * lane;
  __nv_bfloat16* op = out + (size_t)tok * 1024 + head * 64 + 2 * lane;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float rr = rsqrtf(ss[r] / 64.f + 1e-5f);
    const float2 gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + 64 * r));
    *reinterpret_cast<__nv_bfloat162*>(op + 64 * r) =
        __floats2bfloat162_rn(o0[r] * rr / (1.f + __expf(-gv.x)), o1[r] * rr / (1.f + __expf(-gv.y)));
  }
}

// merge key splits and apply the output norm * sigmoid(gate): one warp per (token, head)
__global__ void __launch_bounds__(256) attn_combine_kernel(const float* __restrict__ ws, int nsplit, int T,
                                                           const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out,
                                                           float c, float v_scale) {
  const int wid = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (wid >= T * 16) return;
  const int tok = wid >> 4, head = wid & 15;
  float M = -1e30f;
  for (int s = 0; s < nsplit; ++s) M = fmaxf(M, ws[((size_t)s * T * 16 + wid) * 66 + 64]);
  float o0 = 0.f, o1 = 0.f, l = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    const float* wp = ws + ((size_t)s * T * 16 + wid) * 66;
    const float ls = wp[65];
    if (ls == 0.f) continue;  // empty split
    const float sc = exp2f((wp[64] - M) * c);
    const float2 v = *reinterpret_cast<const float2*>(wp + 2 * lane);
    o0 += sc * v.x; o1 += sc * v.y; l += sc * ls;
  }
  const float inv = v_scale / l;
  o0 *= inv; o1 *= inv;
  float ss = o0 * o0 + o1 * o1;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float r = rsqrtf(ss / 64.f + 1e-5f);
  const float2 gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qkv + (size_t)tok * 2304 + 1280 + head * 64 + 2 * lane));
  const float y0 = o0 * r / (1.f + __expf(-gv.x)), y1 = o1 * r / (1.f + __expf(-gv.y));
  *reinterpret_cast<__nv_bfloat162*>(out + (size_t)tok * 1024 + head * 64 + 2 * lane) = __floats2bfloat162_rn(y0, y1);
}

// Same merge, R (token, head) rows per warp (adjacent heads of one token): all R rows' loads are in flight together.
template <int R>
__global__ void __launch_bounds__(256) attn_combine2_kernel(const float* __restrict__ ws, int nsplit, int T,
                                                            const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out,
                                                            float c, float v_scale) {
  pdl_enter_attn();
  const int wid = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (wid >= T * (16 / R)) return;
  const int r0 = R * wid;  // rows r0 .. r0 + R - 1: token r0 >> 4, heads (r0 & 15) ..
  const int tok = r0 >> 4, head = r0 & 15;
  float M[R], o0[R], o1[R], l[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { M[r] = -1e30f; o0[r] = 0.f; o1[r] = 0.f; l[r] = 0.f; }
  for (int s = 0; s < nsplit; ++s) {
    const float* wp = ws + ((size_t)s * T * 16 + r0) * 66;
#pragma unroll
    for (int r = 0; r < R; ++r) M[r] = fmaxf(M[r], wp[66 * r + 64]);
  }
  for (int s = 0; s < nsplit; ++s) {
    const float* wp = ws + ((size_t)s * T * 16 + r0) * 66;
    float ls[R], ms[R];
    float2 v[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      ls[r] = wp[66 * r + 65];
      ms[r] = wp[66 * r + 64];
      v[r] = *reinterpret_cast<const float2*>(wp + 66 * r + 2 * lane);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (ls[r] != 0.f) {  // an empty split contributes nothing
        const float sc = exp2f((ms[r] - M[r]) * c);
        o0[r] += sc * v[r].x; o1[r] += sc * v[r].y; l[r] += sc * ls[r];
      }
    }
  }
  float ss[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float inv = v_scale / l[r];
    o0[r] *= inv; o1[r] *= inv;
    ss[r] = o0[r] * o0[r] + o1[r] * o1[r];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int r = 0; r < R; ++r) ss[r] += __shfl_xor_sync(0xffffffffu, ss[r], off);
  }
  const __nv_bfloat16* gp = qkv + (size_t)tok * 2304 + 1280 + head * 64 + 2 * lane;
  __nv_bfloat16* op = out + (size_t)tok * 1024 + head * 64 + 2 * lane;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float rr = rsqrtf(ss[r] / 64.f + 1e-5f);
    const float2 gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gp + 64 * r));
    *reinterpret_cast<__nv_bfloat162*>(op + 64 * r) =
        __floats2bfloat162_rn(o0[r] * rr / (1.f + __expf(-gv.x)), o1[r] * rr / (1.f + __expf(-gv.y)));
  }
}

}  // namespace okc_attn
'''


def _build(fast_scales=True):
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}{cap[1]}" + ("a" if cap[0] == 9 else "")
    src = ("" if fast_scales else "#define MOE_FAST_SCALE 0\n") + _CUDA_SRC + GLUE
    name = "okc_prefill_moe_" + hashlib.sha1((src + CPP + arch).encode()).hexdigest()[:16]
    return load_inline(name, cpp_sources=CPP, cuda_sources=src, functions=["qk_prep", "moe_route", "moe_plan", "moe_gather", "moe_gather_plan", "moe_combine", "moe_gemm", "moe_init", "moe_gemm_p", "moe_fused", "boundary", "boundary_rc", "kv_gather", "attn_prep", "attn_fwd", "attn_init", "attn_set_prof"],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-std=c++20",
                                          "--use_fast_math"])


LT_CPP = r'''
#include <torch/extension.h>
int64_t lt_plan(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos);
int64_t lt_plan_sm(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos, int64_t sm_count);
int64_t lt_nalgo(int64_t id);
std::vector<int64_t> lt_info(int64_t id, int64_t algo);
void lt_run(int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws);
'''
LT_GLUE = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <vector>
struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  std::vector<cublasLtMatmulHeuristicResult_t> res;
};
static std::vector<LtPlan*> g_lt_plans;
static cublasLtHandle_t g_lt = nullptr;
int64_t lt_plan_sm(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos, int64_t sm_count) {
  if (!g_lt) TORCH_CHECK(cublasLtCreate(&g_lt) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate");
  auto* p = new LtPlan();
  TORCH_CHECK(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "desc");
  if (sm_count > 0) {  // size the kernel for fewer SMs: a concurrent stream keeps the rest
    const int32_t smc = (int32_t)sm_count;
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &smc, sizeof(smc));
  }
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  // column-major views: A = W^T (K x N, ld K) used transposed, B = X^T (K x M, ld K), C = out^T (N x M, ld N)
  cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, K, N, K);
  cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, K, M, K);
  cublasLtMatrixLayoutCreate(&p->lc, out_f32 ? CUDA_R_32F : CUDA_R_16BF, N, M, N);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  uint64_t ws = (uint64_t)ws_bytes;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p->res.resize(max_algos);
  int n = 0;
  cublasLtMatmulAlgoGetHeuristic(g_lt, p->op, p->la, p->lb, p->lc, p->lc, pref, (int)max_algos, p->res.data(), &n);
  p->res.resize(n);
  cublasLtMatmulPreferenceDestroy(pref);
  g_lt_plans.push_back(p);
  return (int64_t)g_lt_plans.size() - 1;
}
int64_t lt_plan(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos) {
  return lt_plan_sm(M, N, K, out_f32, ws_bytes, max_algos, 0);
}
int64_t lt_nalgo(int64_t id) { return (int64_t)g_lt_plans[id]->res.size(); }
std::vector<int64_t> lt_info(int64_t id, int64_t algo) {
  const auto& r = g_lt_plans[id]->res[algo];
  int tile = 0, splitk = 0, stages = 0, cluster = 0, aid = 0, red = 0, swz = 0, cust = 0, inner = 0;
  size_t sz = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof(cluster), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_ID, &aid, sizeof(aid), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, sizeof(cust), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof(inner), &sz);
  // [0..3] as before; [4] workspace; [5..9] algo id, reduction scheme, CTA swizzling, custom option, inner shape
  return {tile, splitk, stages, cluster, (int64_t)r.workspaceSize, aid, red, swz, cust, inner};
}
void lt_run(int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws) {
  auto* p = g_lt_plans[id];
  const float alpha = 1.f, beta = 0.f;
  auto st = c10::cuda::getCurrentCUDAStream();
  auto s = cublasLtMatmul(g_lt, p->op, &alpha, w.data_ptr(), p->la, x.data_ptr(), p->lb, &beta, out.data_ptr(), p->lc,
                          out.data_ptr(), p->lc, &p->res[algo].algo, ws.data_ptr(), (size_t)ws.numel(), st);
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed: ", (int)s);
}
'''


def _lt_pin_index(lt, pid, pin):
    """LT_PIN entry -> heuristic index: None (torch.mm), an int index (legacy), or a signature tuple (tile id, split-K,
    reduction scheme, custom option) matched against each heuristic result's config read back from cuBLASLt; returns
    -1 when no result matches (the caller then tunes as usual)."""
    if pin is None:
        return None
    n = lt.lt_nalgo(pid)
    if isinstance(pin, int):
        return pin if pin < n else -1
    for a in range(n):
        info = list(lt.lt_info(pid, a))
        if (info[0], info[1], info[6], info[8]) == tuple(pin):
            return a
    return -1


def _build_lt():
    name = "okc_prefill_lt_" + hashlib.sha1((LT_GLUE + LT_CPP).encode()).hexdigest()[:16]
    # host-only code: compiled by the host compiler (no nvcc), linked against cuBLASLt
    return load_inline(name, cpp_sources=[LT_CPP, LT_GLUE], functions=["lt_plan", "lt_plan_sm", "lt_nalgo", "lt_info", "lt_run"],
                       extra_cflags=["-O3", "-std=c++20"], extra_ldflags=["-lcublasLt"], with_cuda=True)


# dense GEMMs: name -> (state weight [N, K], fp32 output)
LT_GEMMS = {"qkv": ("qkv_gate", False), "o": ("o", False), "router": ("router_hilo", True), "pooler": ("pooler", False),
            "unpooler": ("unpooler", False)}
LT_TOKENS = (753, 3012, 16384)  # token counts of the scored shapes; others use torch.mm
# The router chain (GEMM, split-K reduce, top-8) runs on a side stream during the pooler GEMM. At small T the pooler
# holds every SM, so the reduce and the top-8 kernel wait for the pooler's end and the top-8 lands on the critical path.
# Sizing the pooler for (SMs - reserve) leaves those SMs to the router chain.
POOL_SM_RESERVE = {}  # T -> SMs left free for the router chain (0 / absent: torch.mm or the tuned algorithm)


def _tune_lt(st):
    """Time cuBLASLt's heuristic algorithms for each dense GEMM at the scored token counts; keep one only where it
    beats torch.mm (same library, different kernel choice)."""
    lt = st["lt"]
    dev = "cuda"
    choice = {}
    for T in LT_TOKENS:
        for name, (wn, f32) in LT_GEMMS.items():
            w = st[wn]
            N, K = w.shape
            x = (torch.randn((T, K), device=dev) * 0.5).to(torch.bfloat16)
            odt = torch.float32 if f32 else torch.bfloat16
            ref_fn = (lambda: torch.mm(x, w.t(), out_dtype=torch.float32)) if f32 else (lambda: torch.mm(x, w.t()))
            ref = ref_fn().float()
            t_ref = _time(ref_fn, reps=10)
            if name == "router" and not ROUTER_LT:  # (the split-K pick steals the concurrent pooler's SMs)
                del x
                continue
            out = torch.empty((T, N), dtype=odt, device=dev)
            if name == "pooler" and POOL_SM_RESERVE.get(T, 0) > 0:
                pid = lt.lt_plan_sm(T, N, K, f32, st["lt_ws"].numel(), 4, st["nsm"] - POOL_SM_RESERVE[T])
                if lt.lt_nalgo(pid) > 0:
                    lt.lt_run(pid, 0, x, w, out, st["lt_ws"])
                    torch.cuda.synchronize()
                    err = float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
                    if err < 2e-2:
                        choice[(T, name)] = (pid, 0)
                        t = _time(lambda: lt.lt_run(pid, 0, x, w, out, st["lt_ws"]), reps=10)
                        print(f"LT T={T} pooler sized for {st['nsm'] - POOL_SM_RESERVE[T]} SMs: {t:.1f}us alone (torch.mm {t_ref:.1f}us)", flush=True)
                        del x, out
                        continue
            pid = lt.lt_plan(T, N, K, f32, st["lt_ws"].numel(), 32)
            if (T, name) in LT_PIN:  # a fixed pick: no prepare-time race between near-equal algorithms
                a = _lt_pin_index(lt, pid, LT_PIN[(T, name)])
                if a != -1:  # -1: no heuristic result has the pinned signature -> time them as usual
                    if a is not None:
                        choice[(T, name)] = (pid, a)
                    print(f"LT T={T} {name}: pinned {LT_PIN[(T, name)]} -> algo {a}", flush=True)
                    del x, out
                    continue
                print(f"LT T={T} {name}: pinned signature {LT_PIN[(T, name)]} not offered; tuning instead", flush=True)
            best = (t_ref, None)
            for a in range(lt.lt_nalgo(pid)):
                try:
                    lt.lt_run(pid, a, x, w, out, st["lt_ws"])
                    torch.cuda.synchronize()
                    err = float((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6))
                    if not err < 2e-2:
                        continue
                    t = _time(lambda: lt.lt_run(pid, a, x, w, out, st["lt_ws"]), reps=10)
                except Exception:  # noqa: BLE001
                    continue
                if t < best[0] * 0.97:
                    best = (t, a)
            if best[1] is not None:
                choice[(T, name)] = (pid, best[1])
            print(f"LT tune T={T} {name}: torch.mm {t_ref:.1f}us, best {best[0]:.1f}us algo {best[1]} of {lt.lt_nalgo(pid)}", flush=True)
            del x, out
    torch.cuda.empty_cache()
    return choice


def _mm(st, x, name, side=False):
    """x [T, K] @ W[N, K]^T with the tuned cuBLASLt algorithm for this shape, else torch.mm."""
    T = x.shape[0]
    ch = st["lt_choice"].get((T, name)) if st.get("lt_choice") else None
    wn, f32 = LT_GEMMS[name]
    w = st[wn]
    if ch is None:
        return torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
    out = torch.empty((T, w.shape[0]), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    st["lt"].lt_run(ch[0], ch[1], x, w, out, st["lt_ws_side"] if side else st["lt_ws"])
    return out


_SIDE_STREAM = []  # the side stream for the router GEMM + top-8 (created in prepare)


def _codes(packed):
    """uint8 [..., K/2] -> uint8 codes [..., K] (element 2i = low nibble)."""
    lo = packed & 15
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


# nibble slot i (the old layout's 4i..4i+3) -> bit positions of (m0, m1, m2, sign) in the 32-bit word (see convert8)
_CODE_POS = {3: (6, 7, 8, 15), 7: (22, 23, 24, 31), 2: (2, 3, 4, 11), 6: (18, 19, 20, 27),
             1: (28, 29, 30, 5), 5: (12, 13, 14, 21), 0: (0, 1, 26, 9), 4: (16, 17, 10, 25)}  # MOE_DQ7 layout


def _repack(w, sf, gated):
    """w: uint8 [E, N, 2048]; sf: e4m3 [E, N, 256] -> uint8 [E, N/128, KS, WBLK]."""
    E, N, _ = w.shape
    NT = N // 128
    dev = w.device
    if gated:
        nt = torch.arange(NT, device=dev).view(NT, 1, 1)
        q = torch.arange(8, device=dev).view(1, 8, 1)
        lr = torch.arange(16, device=dev).view(1, 1, 16)
        perm = torch.where(lr < 8, 64 * nt + 8 * q + lr, 4096 + 64 * nt + 8 * q + lr - 8).reshape(-1)
    else:
        perm = torch.arange(N, device=dev)
    out = torch.empty((E, NT, KS, WBLK), dtype=torch.uint8, device=dev)
    sfu = sf.view(torch.uint8)
    shifts = torch.tensor([3, 7, 1, 5, 2, 6, 0, 4], dtype=torch.int32, device=dev)  # (half, hk, b) order below
    pos = torch.tensor([_CODE_POS[int(i)] for i in shifts.tolist()], dtype=torch.int64, device=dev)  # [8, 4]
    for e in range(E):
        c = _codes(w[e][perm]).to(torch.int32)  # [N, 4096]
        # rows: [nt, wg, w, half, g]; k: [ks, j, hk, t, b]
        c = c.view(NT, 2, 4, 2, 8, KS, 4, 2, 4, 2)
        # -> [nt, ks, wg, w, g, t, j, half, hk, b]
        c = c.permute(0, 5, 1, 2, 4, 8, 6, 3, 7, 9).reshape(NT, KS, 2, 4, 8, 4, 4, 8)
        # nibble index for (half, hk, b): i = 3 - (half + 2*hk) + 4*b; flattened order (half, hk, b):
        # (0,0,0)->3 (0,0,1)->7 (0,1,0)->1 (0,1,1)->5 (1,0,0)->2 (1,0,1)->6 (1,1,0)->0 (1,1,1)->4
        # bit layout read by convert8 (rotate + mask per register): code bit b of the code in nibble slot i goes to
        # bit _CODE_POS[i][b] (b: m0, m1, m2, sign)
        cb = (c.to(torch.int64).unsqueeze(-1) >> torch.arange(4, device=dev)) & 1  # [..., 8, 4]
        words = (cb << pos).sum((-1, -2)) & 0xFFFFFFFF
        wbytes = torch.stack([(words >> (8 * i)) & 0xFF for i in range(4)], dim=-1).to(torch.uint8)
        out[e, :, :, :4096] = wbytes.reshape(NT, KS, 4096)
        s = sfu[e][perm].view(NT, 2, 4, 2, 8, KS, 4)  # rows [nt, wg, w, half, g]; cols [ks, j]
        s = s.permute(0, 5, 1, 2, 4, 6, 3).reshape(NT, KS, 512)  # [nt, ks, wg, w, g, j, half]
        out[e, :, :, 4096:] = s
    return out


# ----------------------------------------------------------------------------------------------
# pieces
# ----------------------------------------------------------------------------------------------

def _attention_fallback(data, q8, cfg):
    """Local (non-Hopper) attention over the paged FP8 cache, FP32 math. Testing only."""
    B, L = cfg["batch"], cfg["q_len"]
    page, window = cfg["page_size"], cfg["window"]
    ks, vs = data["k_scale"].float(), data["v_scale"].float()
    kf = data["k_cache"].view(-1, HKV, HD)
    vf = data["v_cache"].view(-1, HKV, HD)
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    out = torch.empty((B * L, HQ, HD), dtype=torch.bfloat16, device=q8.device)
    seq_lens = data["seq_lens"].tolist()
    for b in range(B):
        seq = seq_lens[b]
        first = max(0, seq - window + 1) if window else 0
        kp = torch.arange(first, seq + L, device=q8.device)
        rows = data["page_table"][b].long()[kp // page] * page + kp % page
        keys = kf[rows].float() * ks
        vals = vf[rows].float() * vs
        q = q8[b * L:(b + 1) * L].float()
        for c in range(0, L, 1024):
            qp = seq + torch.arange(c, min(c + 1024, L), device=q8.device)
            vis = kp[None, :] <= qp[:, None]
            if window:
                vis &= kp[None, :] > qp[:, None] - window
            qq = q[c:c + 1024].view(-1, HKV, HQ // HKV, HD)
            att = torch.einsum("lkgd,skd->lkgs", qq, keys) * scale
            att = att.masked_fill(~vis[:, None, None, :], float("-inf")).softmax(-1)
            out[b * L + c: b * L + c + qq.shape[0]] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(-1, HQ, HD).to(torch.bfloat16)
    return out


def _attention(data, q8, cfg):
    if _fa3 is None:
        return _attention_fallback(data, q8, cfg)
    window = cfg["window"]
    scaled = cfg["kv_scale"] != 1.0
    descale = {}
    if scaled:
        B = cfg["batch"]
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, HKV),
                       v_descale=data["v_scale"].reshape(1, 1).expand(B, HKV))
    out = _fa3(q=q8.view(-1, HQ, HD), k_cache=data["k_cache"], v_cache=data["v_cache"],
               page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
               cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=cfg["q_len"],
               softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
               window_size=(window - 1, 0) if window else (-1, -1),
               scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _moe_forward(st, p, lid, pos, cnt, T, order=None, zb=None, cit=None):
    """Routed experts on the pooler output p [T, 4096]: returns per-route rows y [rows, 4096] (bf16) and rrow. With cit
    (dict: zc = [max tiles | 4] zeroed counters, tok = [2T + rows] ints, topw, out) the fused kernel also writes the
    combined output (combine in the tail) and the caller skips the combine kernel."""
    dev = p.device
    mod = st["mod"]
    maxrows = T * TOP_K + LOCAL_EXPERTS * 256
    maxtiles = T * TOP_K // 32 + 64
    offsets = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)
    rows = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)
    tiles = torch.empty((maxtiles, 4), dtype=torch.int32, device=dev)
    ntiles = torch.empty((1,), dtype=torch.int32, device=dev)
    large = T >= MOE_LARGE_T
    mode = MOE_MODE_LARGE if large else MOE_MODE_SMALL
    if order is None:
        order = MOE_ORDER_LARGE if large else MOE_ORDER_SMALL
    fine = 1 if (FINE_MIN_T and T >= FINE_MIN_T) else 0  # remainder tiles rounded to 32 rows (plan + kernel agree)
    order = order | (fine << 8)
    xt = torch.empty((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
    rrow = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    if cit is not None and (not GATHER_PLAN or mode != "fused"):
        raise RuntimeError("CIT needs the fused kernel and the gather+plan launch")
    cz = cit["zc"] if cit is not None else st["no_zero"]
    ctok = cit["tok"] if cit is not None else st["no_zero"]
    if GATHER_PLAN:  # the gather kernel also writes the plan (one launch fewer)
        mod.moe_gather_plan(p, st["g_pre_expert_norm"], lid, pos, cnt, xt, rrow, offsets, tiles, ntiles, rows, order, cz, ctok)
    else:
        mod.moe_plan(cnt, offsets, tiles, ntiles, rows, order)
        mod.moe_gather(p, st["g_pre_expert_norm"], lid, pos, offsets, cnt, xt, rrow)
    inter = torch.empty((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
    y = torch.empty((maxrows, D_EXPERT), dtype=torch.bfloat16, device=dev)
    if mode == "fused":
        if zb is None:
            zb = torch.zeros((1 + maxtiles,), dtype=torch.int32, device=dev)  # work counter + per-tile GEMM1 completion
        mod.moe_fused(st["w13r"], st["w2r"], xt, inter, y, tiles, ntiles, st["w13_gs"], st["w2_gs"], st["nsm"] * (2 if SKINNY_TEST else 1), st["smem"],
                      zb[0:1], zb[1:1 + maxtiles], st["moe_dbg"], st["moe_flags"] | (fine << 8), cz, ctok, rrow, lid,
                      cit["topw"] if cit is not None else st["no_zero_f"], st["g_post_expert_norm"],
                      cit["out"] if cit is not None else st["no_zero_b"])
    elif mode == "persist":
        ctr = torch.zeros((2,), dtype=torch.int32, device=dev)
        mod.moe_gemm_p(st["w13r"], xt, tiles, ntiles, st["w13_gs"], inter, 64, st["nsm"], True, st["smem"], ctr[0:1])
        mod.moe_gemm_p(st["w2r"], inter, tiles, ntiles, st["w2_gs"], y, 32, st["nsm"], False, st["smem"], ctr[1:2])
    else:
        mod.moe_gemm(st["w13r"], xt, tiles, ntiles, st["w13_gs"], inter, 64, P_GEMM1, True, st["smem"])
        mod.moe_gemm(st["w2r"], inter, tiles, ntiles, st["w2_gs"], y, 32, P_GEMM2, False, st["smem"])
    return dict(y=y, rrow=rrow, xt=xt, inter=inter, tiles=tiles, ntiles=ntiles, offsets=offsets)


def kernel_fn(data, state):
    st = state
    cfg = data["config"]
    T = cfg["tokens"]
    dev = data["attn_input"].device
    # --- attention ---
    av = _attn_variant(st, cfg)
    pre = None
    split = av is not None and T <= PREP_SPLIT_MAX_T and PREP_SPLIT.get("slide" if cfg["window"] else "full", False)
    if split and not PREP_SPLIT_AFTER:
        pre = _attn_prep_old(st, data, cfg)
    ev0 = None
    if split and PREP_SPLIT_AFTER:
        ev0 = torch.cuda.Event()
        ev0.record()
    qkv = _mm(st, data["attn_input"], "qkv")
    if ev0 is not None:
        pre = _attn_prep_old(st, data, cfg, ev0)
    q8 = torch.empty((T, Q_SIZE), dtype=torch.uint8, device=dev)
    if QK_PREP_CUDA:
        st["mod"].qk_prep(qkv, data["positions"], data["cache_loc"], st["inv_freq"], q8, data["k_cache"].view(torch.uint8),
                          data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], bool(cfg["rope_theta"]),
                          cfg["kv_scale"] != 1.0)
    else:
      _qk_prep_kernel[(T,)](qkv, data["positions"], data["cache_loc"], st["inv_freq"], q8,
                          data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"],
                          ROPE=bool(cfg["rope_theta"]), SCALED=cfg["kv_scale"] != 1.0, num_warps=4,
                          **(dict(PDL=True, launch_pdl=True) if TRITON_PDL else {}))
    if av is not None:
        ns = _attn_splits(cfg, data, av)
        sk = ATTN_SK.get(_attn_class(cfg), 0)
        if sk and av in (8, 43) and cfg["batch"] * ((cfg["q_len"] + 23) // 24) * 2 <= 1536:
            ns = -sk
        og = _attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=av, pre=pre)
    else:
        att = _attention(data, q8.view(FP8), cfg)
        og = torch.empty((T, Q_SIZE), dtype=torch.bfloat16, device=dev)
        _norm_gate_kernel[(T,)](att, qkv, og, num_warps=4)
    a = _mm(st, og, "o")
    h = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    maxtiles = T * TOP_K // 32 + 64
    use_cit = CIT and MOE_MODE_SMALL == "fused" and MOE_MODE_LARGE == "fused" and GATHER_PLAN
    zlen = LOCAL_EXPERTS + 1 + maxtiles + ((maxtiles + 4) if use_cit else 0)  # counts | work | done | (CIT: done2 | head, tail, err, pad)
    rc = T >= BND_CUDA_MIN and BND_RECOMPUTE and T >= BND_RC_MIN_T
    if rc:  # the first boundary keeps only the branch's rms factor; the second rebuilds the fp32 stream from it
        zb = torch.empty((zlen,), dtype=torch.int32, device=dev)
        r1 = torch.empty((T,), dtype=torch.float32, device=dev)
        st["mod"].boundary_rc(a, data["stream"], st["a1"], st["a1"], st["b1"], st["g_pre_ffn_norm"], data["stream"], h, 1, zb,
                              a, r1, st["a1"], st["b1"])
    elif T >= BND_CUDA_MIN:  # the boundary kernel also zeroes zb (counts | work counter | tile done)
        stream1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
        zb = torch.empty((zlen,), dtype=torch.int32, device=dev)
        st["mod"].boundary(a, data["stream"], st["a1"], st["a1"], st["b1"], st["g_pre_ffn_norm"], stream1, h, False, zb)
    else:
        stream1 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
        zb = torch.zeros((zlen,), dtype=torch.int32, device=dev)
        _boundary_kernel[(T,)](a, data["stream"], a, st["a1"], st["b1"], st["g_pre_ffn_norm"], stream1, h,
                               HAS_GPOST=False, H=HIDDEN, num_warps=16)
    # --- router + top-8 on a side stream, concurrent with the pooler GEMM ---
    main = torch.cuda.current_stream()
    side = _SIDE_STREAM[0]
    serial = T <= ROUTER_SERIAL_MAX or T >= ROUTER_SERIAL_MIN
    if serial:  # the router GEMM alone on all SMs first: cheaper than slowing the pooler down by sharing SMs with it
        logits = _mm(st, h, "router")
    side.wait_stream(main)
    with torch.cuda.stream(side):
        if not serial:
            logits = _mm(st, h, "router", side=True)  # [T, 512] fp32: hi | lo partial logits
        ids = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
        topw = torch.empty((T, TOP_K), dtype=torch.float32, device=dev)
        lid = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
        pos = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
        cnt = zb[:LOCAL_EXPERTS]
        cit = None
        if use_cit:  # tok_left (T) | combine queue (T) | row -> token map (rows)
            ctok = torch.empty((2 * T + T * TOP_K + LOCAL_EXPERTS * 256,), dtype=torch.int32, device=dev)
            cit = dict(zc=zb[LOCAL_EXPERTS + 1 + maxtiles:], tok=ctok, topw=topw)
        st["mod"].moe_route(logits, st["router_bias"], data["local_start"], ids, topw, lid, pos, cnt,
                            ctok[:T] if use_cit else st["no_zero"], ctok[T:2 * T] if use_cit else st["no_zero"])  # one warp per token
    # --- experts ---
    if POOL_FP8_MIN_T <= T <= POOL_FP8_MAX_T or (T > POOL_FP8_MAX_T and
                                                  data["page_table"].shape[1] * cfg["page_size"] - cfg["q_len"] >= POOL_FP8_CTX_MIN):
        h8 = torch.empty((T, HIDDEN), dtype=FP8, device=dev)
        hsc = torch.empty((T, 1), dtype=torch.float32, device=dev)
        _rowq8_kernel[(T,)](h, h8, hsc, H=HIDDEN, num_warps=16)
        p = torch._scaled_mm(h8, st["pooler8"].t(), scale_a=hsc, scale_b=st["pooler_ws"], out_dtype=torch.bfloat16)
    else:
        p = _mm(st, h, "pooler")
    main.wait_stream(side)
    moe = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
    if cit is not None:
        cit["out"] = moe
    ex = _moe_forward(st, p, lid, pos, cnt, T, zb=zb[LOCAL_EXPERTS:], cit=cit)
    y, rrow = ex["y"], ex["rrow"]
    u8 = UNPOOL_FP8 and cit is None and T >= UNPOOL_FP8_MIN_T
    if u8:
        moe8 = torch.empty((T, D_EXPERT), dtype=FP8, device=dev)
        usc = torch.empty((T, 1), dtype=torch.float32, device=dev)
        _combine8_kernel[(T,)](y, rrow, lid, topw, st["ones32"], st["g_post_expert_norm"], moe8, usc, H=D_EXPERT,
                               num_warps=COMBINE_NW_SMALL if T <= COMBINE_SMALL_T else 8)
    elif cit is not None:
        pass  # combined in the fused kernel's tail
    elif CUDA_COMBINE:  # one block per token, PDL (the per-expert scale is 1: st["ones32"])
        st["mod"].moe_combine(y, rrow, lid, topw, st["g_post_expert_norm"], moe)
    elif TRITON_PDL:
        _combine_kernel[(T,)](y, rrow, lid, topw, st["ones32"], st["g_post_expert_norm"], moe, H=D_EXPERT, PDL=True, num_warps=8,
                              launch_pdl=True)
    else:
        _combine_kernel[(T,)](y, rrow, lid, topw, st["ones32"], st["g_post_expert_norm"], moe, H=D_EXPERT,
                              num_warps=COMBINE_NW_SMALL if T <= COMBINE_SMALL_T else 8)
    if u8:
        f = torch._scaled_mm(moe8, st["unpooler8"].t(), scale_a=usc, scale_b=st["unpooler_ws"], out_dtype=torch.bfloat16)
    else:
        f = _mm(st, moe, "unpooler")
    stream2 = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
    hidden = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
    # (a recompute-the-stream variant, _boundary2_kernel, ran at only 2.4 TB/s on the H200: 772 us vs 436 us at 16K)
    if rc:
        st["mod"].boundary_rc(f, data["stream"], st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], stream2, hidden, 2,
                              st["no_zero"], a, r1, st["a1"], st["b1"])
    elif T >= BND_CUDA_MIN:
        st["mod"].boundary(f, stream1, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], stream2, hidden, True, st["no_zero"])
    else:
        _boundary_kernel[(T,)](f, stream1, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"],
                               stream2, hidden, HAS_GPOST=True, H=HIDDEN, num_warps=16)
    return hidden, stream2, ids


# ----------------------------------------------------------------------------------------------
# self-test / micro-benchmarks printed from prepare() (check runs only)
# ----------------------------------------------------------------------------------------------

SELFTEST = False

SCEN = {
    "T16384": (16384, [12547, 11910, 2820, 2392, 270, 242, 97, 60, 34, 16, 14, 2, 1, 1, 1]),
    "T3012": (3012, [2277, 2191, 416, 400, 47, 37, 32, 11, 7, 5, 4, 3, 2, 1, 1]),
    "T753": (753, [540, 529, 127, 108, 14, 6, 3, 3, 3, 2, 1, 1]),
}


def _routes_from_counts(T, counts, dev):
    lid = torch.full((T, TOP_K), -1, dtype=torch.int32, device=dev)
    pos = torch.zeros((T, TOP_K), dtype=torch.int32, device=dev)
    r = 0
    for e, c in enumerate(counts):
        idx = torch.arange(r, r + c, device=dev)
        lid.view(-1)[(idx % T) * TOP_K + idx // T] = e
        pos.view(-1)[(idx % T) * TOP_K + idx // T] = torch.arange(c, device=dev, dtype=torch.int32)
        r += c
    cnt = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device=dev)
    cnt[: len(counts)] = torch.tensor(counts, dtype=torch.int32, device=dev)
    return lid, pos, cnt


_FLUSH = []


def _time(fn, reps=5):
    """Median time in us with a cold L2 (a read sweep over 256 MB before each call, like the judge's replays)."""
    if not _FLUSH:
        _FLUSH.append((torch.ones(128 << 20, dtype=torch.float16, device="cuda"), torch.zeros((), dtype=torch.float32, device="cuda")))
    buf, sink = _FLUSH[0]
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        sink.copy_(buf.sum(dtype=torch.float32))
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def _selftest(st, weights):
    from nvfp4 import dequantize
    dev = weights["w13"].device
    mod = st["mod"]
    torch.manual_seed(0)
    # correctness on a small skewed routing
    T = 700
    counts = [300, 200, 100, 60, 33, 17, 14, 5, 1]  # 14 -> N=16, 5 and 1 -> N=8 tiles (below FINE_MIN_T)
    lid, pos, cnt = _routes_from_counts(T, counts, dev)
    p = torch.randn((T, D_EXPERT), device=dev).to(torch.bfloat16)
    gw = st["g_pre_expert_norm"]
    out = _moe_forward(st, p, lid, pos, cnt, T)
    y, rrow = out["y"], out["rrow"]
    pf = p.float()
    xn = (pf * torch.rsqrt(pf.pow(2).mean(-1, keepdim=True) + 1e-5) * (1.0 + gw.float())).to(torch.bfloat16)
    msg = []
    for e in range(len(counts)):
        tok, slot = (lid == e).nonzero(as_tuple=True)
        W13 = dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).to(torch.bfloat16)
        W2 = dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).to(torch.bfloat16)
        hh = xn[tok].float() @ W13.float().T
        inter = (F.silu(hh[:, :D_EXPERT]) * hh[:, D_EXPERT:]).to(torch.bfloat16)
        ref = inter.float() @ W2.float().T
        got = y[rrow[tok, slot].long()].float()
        msg.append(f"e{e}:n{len(tok)}:{float((got - ref).norm() / ref.norm()):.4f}")
    print("SELFTEST moe rel err", " ".join(msg), flush=True)
    # timing
    for name, (T, counts) in SCEN.items():
        lid, pos, cnt = _routes_from_counts(T, counts, dev)
        p = torch.randn((T, D_EXPERT), device=dev).to(torch.bfloat16)
        out = _moe_forward(st, p, lid, pos, cnt, T)
        nt = int(out["ntiles"])
        rows = int(out["tiles"][:nt, 2].sum())
        R = sum(counts)
        res = []
        for P1, P2 in ((8, 16),):
            t1 = _time(lambda: mod.moe_gemm(st["w13r"], out["xt"], out["tiles"], out["ntiles"], st["w13_gs"], out["inter"], 64, P1, True, st["smem"]))
            t2 = _time(lambda: mod.moe_gemm(st["w2r"], out["inter"], out["tiles"], out["ntiles"], st["w2_gs"], out["y"], 32, P2, False, st["smem"]))
            f1 = 2 * R * 8192 * 4096 / t1 / 1e6
            f2 = 2 * R * 4096 * 4096 / t2 / 1e6
            res.append(f"P{P1}/{P2}: g1 {t1:.0f}us ({f1:.0f}TF) g2 {t2:.0f}us ({f2:.0f}TF)")
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        rr = out["rrow"][out["rrow"] >= 0].long()
        y0 = out["y"][rr].clone()
        ctr = torch.zeros((2,), dtype=torch.int32, device=dev)
        rows_t = torch.empty((LOCAL_EXPERTS,), dtype=torch.int32, device=dev)

        def g1p():
            ctr.zero_()
            mod.moe_gemm_p(st["w13r"], out["xt"], out["tiles"], out["ntiles"], st["w13_gs"], out["inter"], 64, nsm, True, st["smem"], ctr[0:1])

        def g2p():
            ctr.zero_()
            mod.moe_gemm_p(st["w2r"], out["inter"], out["tiles"], out["ntiles"], st["w2_gs"], out["y"], 32, nsm, False, st["smem"], ctr[1:2])
        zb = torch.zeros((1 + out["tiles"].shape[0],), dtype=torch.int32, device=dev)

        def gf():
            zb.zero_()
            mod.moe_fused(st["w13r"], st["w2r"], out["xt"], out["inter"], out["y"], out["tiles"], out["ntiles"], st["w13_gs"],
                          st["w2_gs"], nsm, st["smem"], zb[0:1], zb[1:], st["moe_dbg"], st["moe_flags"], st["no_zero"], st["no_zero"],
                          out["rrow"], lid, st["no_zero_f"], st["g_post_expert_norm"], st["no_zero_b"])
        for order in (0, 1, 2):
            mod.moe_plan(cnt, out["offsets"], out["tiles"], out["ntiles"], rows_t, order)
            tf = _time(gf)
            same_f = bool(torch.equal(out["y"][rr], y0))
            res.append(f"fused o{order}: {tf:.0f}us same={same_f}")
            t1 = _time(g1p)
            t2 = _time(g2p)
            f1 = 2 * R * 8192 * 4096 / t1 / 1e6
            f2 = 2 * R * 4096 * 4096 / t2 / 1e6
            same = bool(torch.equal(out["y"][rr], y0))
            res.append(f"persist o{order}: g1 {t1:.0f}us ({f1:.0f}TF) g2 {t2:.0f}us ({f2:.0f}TF) same={same}")
        tg = _time(lambda: mod.moe_gather(p, st["g_pre_expert_norm"], lid, pos, out["offsets"], cnt, out["xt"], out["rrow"]))
        print(f"SELFTEST {name} R={R} padded={rows} tiles={nt} gather {tg:.0f}us |", " | ".join(res), flush=True)
    a = torch.randn((16384, 8192), device=dev).to(torch.bfloat16)
    tmm = _time(lambda: torch.mm(a, st["pooler"].t()))
    print(f"SELFTEST cublas 16384x4096x8192 {tmm:.0f}us {2 * 16384 * 4096 * 8192 / tmm / 1e6:.0f}TF", flush=True)
    s = torch.randn((16384, 8192), device=dev)
    o1 = torch.empty_like(s)
    o2 = torch.empty((16384, 8192), device=dev, dtype=torch.bfloat16)
    for nw in (4, 8, 16):
        tb = _time(lambda: _boundary_kernel[(16384,)](a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], o1, o2, HAS_GPOST=True, H=HIDDEN, num_warps=nw))
        print(f"SELFTEST boundary nw{nw} {tb:.0f}us {16384 * 8192 * 12 / tb / 1e6:.2f}TB/s", flush=True)
    del a, s, o1, o2
    torch.cuda.empty_cache()


ATTN_SCEN = [
    ("turn1_full", 1, 753, 80000, 0),
    ("turn4_full", 4, 753, 80000, 0),
    ("ft_full32k", 1, 16384, 32768, 0),
    ("ft_full64k", 1, 16384, 65536, 0),
    ("ft_slide", 1, 16384, 32768, 2048),
]


def _selftest_attn():
    if _fa3 is None:
        return
    dev = "cuda"
    torch.manual_seed(1)
    for name, B, L, ctx, window in ATTN_SCEN:
        S = ctx + L
        T = B * L
        q = (torch.randn((T, HQ, HD), device=dev) * 0.5).to(FP8)
        kc = torch.randn((B * S, 1, HKV, HD), device=dev).to(FP8)
        vc = torch.randn((B * S, 1, HKV, HD), device=dev).to(FP8)
        perm = torch.randperm(B * S, device=dev).to(torch.int32)
        pt = perm.view(B, S)
        seqlens = torch.full((B,), S, dtype=torch.int32, device=dev)
        cuq = torch.arange(0, T + 1, L, dtype=torch.int32, device=dev)
        kcont = kc.view(B * S, HKV, HD)[perm.long()].view(B, S, HKV, HD)
        vcont = vc.view(B * S, HKV, HD)[perm.long()].view(B, S, HKV, HD)
        ws = (window - 1, 0) if window else (-1, -1)
        scale = 0.125 * 5.4730064863875
        res = []
        for label, kk, vv, ptab, extra in (
            ("paged", kc, vc, pt, {}),
            ("paged_pack", kc, vc, pt, dict(pack_gqa=True)),
            ("contig", kcont, vcont, None, {}),
            ("contig_pack", kcont, vcont, None, dict(pack_gqa=True)),
        ):
            try:
                fn = lambda: _fa3(q=q, k_cache=kk, v_cache=vv, page_table=ptab, cache_seqlens=seqlens, cu_seqlens_q=cuq,
                                  max_seqlen_q=L, softmax_scale=scale, causal=True, window_size=ws, **extra)
                t = _time(fn, reps=3)
                o = fn()
                o = o[0] if isinstance(o, tuple) else o
                res.append(f"{label} {t:.0f}us")
                if label == "paged":
                    ref = o.float()
                else:
                    res[-1] += f" d={float((o.float() - ref).abs().max()):.1e}"
            except Exception as exc:  # noqa: BLE001
                res.append(f"{label} ERR {str(exc)[:80]}")
        tg = _time(lambda: kc.view(B * S, HKV, HD)[perm.long()], reps=3)
        flops = 4.0 * HQ * HD * (B * L * (ctx if not window else min(ctx, window)) + B * L * L / 2)
        print(f"SELFTEST attn {name} ({flops / 1e9:.0f} GF) gather {tg:.0f}us |", " | ".join(res), flush=True)
        del q, kc, vc, kcont, vcont
    torch.cuda.empty_cache()


def _selftest_gemm(st):
    dev = "cuda"
    try:
        import deep_gemm
        names = [n for n in dir(deep_gemm) if not n.startswith("_")]
        print("SELFTEST deep_gemm:", " ".join(names)[:1500], flush=True)
    except Exception as exc:  # noqa: BLE001
        deep_gemm = None
        print("SELFTEST deep_gemm import failed", repr(exc)[:200], flush=True)
    try:
        import quack
        import pkgutil
        mods = [m.name for m in pkgutil.iter_modules(quack.__path__)]
        print("SELFTEST quack:", " ".join(mods)[:800], flush=True)
    except Exception as exc:  # noqa: BLE001
        print("SELFTEST quack import failed", repr(exc)[:200], flush=True)
    shapes = [("qkv", 2304, 8192), ("o", 8192, 1024), ("pooler", 4096, 8192), ("unpooler", 8192, 4096)]
    for T in (753, 3012, 16384):
        line = []
        for name, N, K in shapes:
            a = torch.randn((T, K), device=dev).to(torch.bfloat16)
            w = torch.randn((N, K), device=dev).to(torch.bfloat16)
            t0 = _time(lambda: torch.mm(a, w.t()))
            s = f"{name} cublas {t0:.0f}us ({2 * T * N * K / t0 / 1e6:.0f}TF)"
            if deep_gemm is not None:
                for fname in ("bf16_gemm_nt",):
                    fn = getattr(deep_gemm, fname, None)
                    if fn is None:
                        continue
                    try:
                        out = torch.empty((T, N), device=dev, dtype=torch.bfloat16)
                        fn(a, w, out)
                        err = float((out.float() - (a @ w.t()).float()).abs().max())
                        t1 = _time(lambda: fn(a, w, out))
                        s += f" dg {t1:.0f}us ({2 * T * N * K / t1 / 1e6:.0f}TF, err {err:.1e})"
                    except Exception as exc:  # noqa: BLE001
                        s += f" dg ERR {str(exc)[:60]}"
            line.append(s)
        print(f"SELFTEST gemm T={T}: " + " | ".join(line), flush=True)
    torch.cuda.empty_cache()


import re as _re
_m_ast = _re.search(r"#define ATTN_ASTAGES (\d+)", _CUDA_SRC[:2000])
ATTN_SMEM = 1024 + 1024 + 4096 * 3 + (int(_m_ast.group(1)) if _m_ast else 4) * (8192 + 72 * 128)
# own-kernel variants: (consumer warpgroups, exponentials per thread per tile on the FMA pipe)
ATTN_VARIANTS = {0: (2, 0), 1: (2, 16), 8: (3, 0), 9: (3, 16), 16: (2, 0), 24: (3, 0), 43: (3, 0)}
if "ATTN_NWG1" in _CUDA_SRC[:400]:
    ATTN_VARIANTS[99] = (1, 0)  # timing experiment: a lone warpgroup
if "ATTN_V52" in _CUDA_SRC[:400]:
    ATTN_VARIANTS[52] = (2, 0)  # timing experiment: 2 warpgroups, two S buffers (PIPE2)
if "ATTN_V53" in _CUDA_SRC[:400]:
    ATTN_VARIANTS[53] = (2, 0)  # 2 warpgroups, two S sets, one P set, exp turns (PIPE3)  # 43: pipelined, exp turns  # 16, 24: QK(t+1) overlaps P.V(t); 32, 33: softmax(t) overlaps P.V(t-1)
OWN_ATTN_LOCAL = False  # local (sm_80) runs use the FP32 fallback attention unless set


ROUTER_SERIAL_MAX = 0  # T <= this: router GEMM on the main stream before the pooler
COMBINE_NW_SMALL = 8  # Triton combine warps at T <= COMBINE_SMALL_T (occupancy: one wave of programs at 753)
COMBINE_SMALL_T = 4096
QK_PREP_CUDA = True  # (Oct 26 05:48: 16K -15 us, bit-identical but 3 of 3M q8 bytes, checks PASS) CUDA q/k prep (norm + RoPE + e4m3 + cache rows, PDL) instead of the Triton kernel
CIT = False  # combine in the tail of the fused expert kernel (no combine launch); True also needs #define MOE_CIT 1 (dead: TRICKS 5.16)
ATTN_SK = {"slide": 0, "full_long": 0, "full_short": 0, "full_batch": 0}  # stream-K attention grid per class (0: off)
PREP_SPLIT_MAX_T = 1024  # stage the cache-only attention tiles on the side stream during the qkv GEMM (T=753: 6 idle SMs)
PREP_SPLIT = {"slide": False, "full": False}  # Oct 26 04:56: +14 us at 753 sliding (side branch captured first)
PREP_SPLIT_AFTER = True  # capture the side branch after the qkv GEMM node (forked from an event recorded before it)
LT_PIN = {(753, "router"): (33, 4, 2, 3), (16384, "o"): (201, 1, 0, 2), (16384, "unpooler"): (201, 1, 0, 2),
          (16384, "pooler"): (29, 1, 0, 1)}  # signatures (tile id, split-K, reduction, custom option): 753 router =
# nvjet 128x96 1x2 split-K (v89c's judge draw), 16K o/unpooler = 176x128 1x2, 16K pooler = 128x160 2x1 (Oct 26: indices
# 6/6/6/5 on the pod and on the judge alike; v113 pinned the router to #4 = 64x152 4x1 split-K by mistake)  # (T, gemm) -> cuBLASLt heuristic index (None: torch.mm), instead of the prepare-time race (Oct 26 search r28/r29: 16K unpooler 6 +0.15%, pooler 5 +0.17% vs the tuner)
ROUTER_LT = True  # False: the router GEMM always runs as torch.mm (no tuned split-K algorithm)
ATTN_PF_MAX_T = 0  # tokens <= this: the attention kernel prefetches the o-proj weight into L2 (0: off)
ROUTER_SERIAL_MIN = 1 << 30  # T >= this: the same (at 16K the router's first wave delays the pooler anyway)
FINE_MIN_T = 2048  # from this token count the plan rounds remainder tiles to 32 rows (160/192 widths; fused kernel <1>)
SKINNY_TEST = False  # with #define MOE_SKINNY_TEST 1: the fused MoE kernel at 2 CTAs/SM (small tiles only)
COMBINE_PW_EARLY = tl.constexpr(False)  # Triton combine: norm weight loaded before the PDL wait
CUDA_COMBINE = False  # CUDA combine kernel (PDL, COMB_GROUP routes in flight): slower than Triton at 753/3012 (+1..9 us)
TRITON_PDL = False  # qk_prep and combine (Triton) launched with programmatic dependent launch too
UNPOOL_FP8 = True  # unpooler GEMM in E4M3 (per-token x per-channel scales)
UNPOOL_FP8_MIN_T = 1
POOL_FP8_MIN_T = 512  # pooler GEMM in E4M3 for these token counts
POOL_FP8_MAX_T = 4096
POOL_FP8_CTX_MIN = 4096  # longer blocks: pooler in E4M3 only with this much cached context
GATHER_PLAN = True  # plan folded into the gather kernel (block 0 writes the tile list)
BND_RECOMPUTE = True  # second boundary rebuilds the fp32 stream (alpha1*s + beta1*a*r1) instead of a 32 KB/row round trip
BND_RC_MIN_T = 8192  # ...from this token count (Oct 26 00:05 on v89c: 16K sliding -30 us, 753 +6.6 us, 3012 equal)
BND_CUDA_MIN = 1  # token count from which the CUDA boundary kernel replaces the Triton one (v69: every T, +0.12%)
ATTN_SPLIT_MODE = 1  # 0: fill two waves; 1: minimize waves / splits (wave-quantization aware)
# own attention variant per shape class when it validates (None: time all variants in prepare). 43 (softmax(t) overlaps
# P.V(t-1); the three warpgroups take turns on the exp2 units for the first 40 of their 64 exps) is ~4% faster than 8 on
# long full attention (64K: 6979 vs 7309 us; 4 x 80K: 1485 vs 1552 us) and ~3% slower on sliding windows.
ATTN_FIXED = {"slide": 8, "full_long": 43, "full_short": 43, "full_batch": 43}  # turn-1 full: 418 vs 430 us
_NSM = [132]


def _attn_splits(cfg, data, variant):
    """Key-range splits so that small batches still fill the GPU (each split keeps >= 8 key tiles)."""
    B, L, window = cfg["batch"], cfg["q_len"], cfg["window"]
    qb = 8 * ATTN_VARIANTS[variant][0]
    base = ((L + qb - 1) // qb) * 2 * B
    nsm = _NSM[0]
    if base >= 2 * nsm:
        return 1
    keys = (window + qb) if window else data["page_table"].shape[1] * cfg["page_size"]
    nmax = max(1, min(16, (keys // 128) // 8))
    if ATTN_SPLIT_MODE == 1:
        best = None
        for ns in range(1, nmax + 1):
            cost = -(-base * ns // nsm) / ns + (0.02 if ns > 1 else 0.0)
            if best is None or cost < best[0] - 1e-9:
                best = (cost, ns)
        return best[1]
    ns = min((2 * nsm + base - 1) // base, nmax)
    return max(1, min(ns, 16))


def _synthetic_attn_case(B, L, ctx, window, page, seed):
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    S = ctx + L
    npg = (S + page - 1) // page
    T = B * L
    q8 = (torch.randn((T, HQ, HD), device=dev, generator=g) * 0.7).to(FP8)
    qkv = torch.randn((T, QKV_SIZE), device=dev, generator=g).to(torch.bfloat16)
    kc = (torch.randn((B * npg, page, HKV, HD), device=dev, generator=g) * 0.8 + 0.3).to(FP8)  # [pages, page_size, 2, 64]
    vc = torch.randn((B * npg, page, HKV, HD), device=dev, generator=g).to(FP8)
    pt = torch.randperm(B * npg, device=dev, generator=g).to(torch.int32).view(B, npg)
    data = {"k_cache": kc, "v_cache": vc, "page_table": pt, "seq_lens": torch.full((B,), ctx, dtype=torch.int32, device=dev),
            "cache_seqlens": torch.full((B,), S, dtype=torch.int32, device=dev),
            "cu_seqlens_q": torch.arange(0, T + 1, L, dtype=torch.int32, device=dev), "scheduler_metadata": None,
            "k_scale": torch.tensor(1.0, device=dev), "v_scale": torch.tensor(1.0, device=dev)}
    cfg = {"batch": B, "q_len": L, "window": window, "page_size": page, "sm_scale": 0.125, "qk_scale_factor": 5.4730064863875,
           "kv_scale": 1.0, "tokens": T}
    return data, cfg, q8, qkv


def _attn_ref(data, cfg, q8, qkv):
    T = cfg["tokens"]
    att = _attention(data, q8, cfg)
    og = torch.empty((T, Q_SIZE), dtype=torch.bfloat16, device=q8.device)
    _norm_gate_kernel[(T,)](att, qkv, og, num_warps=4)
    return og


def _validate_attn(st, variants):
    """Check own-kernel variants against the reference attention path on small synthetic cases; returns the
    variants that agree."""
    worst = {v: 0.0 for v in variants}
    for i, (B, L, ctx, window, page) in enumerate(((2, 300, 1000, 0, 1), (1, 400, 3000, 256, 1), (2, 100, 200, 0, 64))):
        data, cfg, q8, qkv = _synthetic_attn_case(B, L, ctx, window, page, 7 + i)
        ref = _attn_ref(data, cfg, q8, qkv).float()
        for v in variants:
            for ns in (1, 3):
                o = _attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=v).float()
                err = float((o - ref).norm() / ref.norm())
                worst[v] = max(worst[v], err if err == err else 1e9)
    ok = [v for v in variants if worst[v] < 0.03]
    print("ATTN own kernel validation (worst rel err):", {v: round(e, 4) for v, e in worst.items()}, "->", ok, flush=True)
    return ok


def _tune_attn(st, variants):
    """Pick, per attention shape class, the fastest of the reference path and the validated own variants."""
    choice = {}
    # the scored shapes themselves (synthetic data): sliding 16K chunk, full 16K chunk at 32K context, turn 1 / turn 4 full
    for kind, (B, L, ctx, window) in (("slide", (1, 16384, 32768, 2048)), ("full_long", (1, 16384, 32768, 0)),
                                      ("full_short", (1, 753, 80000, 0)), ("full_batch", (4, 753, 80000, 0))):
        data, cfg, q8, qkv = _synthetic_attn_case(B, L, ctx, window, 1, 11)
        res = {"ref": _time(lambda: _attn_ref(data, cfg, q8, qkv), reps=5)}
        for v in variants:
            ns = _attn_splits(cfg, data, v)
            res[v] = _time(lambda: _attn_custom(st, data, q8, qkv, cfg, nsplit=ns, variant=v), reps=5)
        best = min(res, key=res.get)
        choice[kind] = None if best == "ref" else best
        print(f"ATTN tune {kind}: " + " ".join(f"{k}={t:.0f}us" for k, t in res.items()) + f" -> {best}", flush=True)
        del data, q8, qkv
    torch.cuda.empty_cache()
    return choice


def _attn_class(cfg):
    if cfg["window"]:
        return "slide"
    if cfg["tokens"] > 4096:
        return "full_long"
    return "full_batch" if cfg["batch"] > 1 else "full_short"


def _attn_variant(st, cfg):
    ch = st.get("attn_choice")
    if not ch:
        return None
    if cfg["window"]:
        return ch["slide"]
    if cfg["tokens"] > 4096:
        return ch["full_long"]
    return ch["full_batch"] if cfg["batch"] > 1 else ch["full_short"]


def _attn_stage_buffers(data, cfg, dev):
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    smax = (window - 1 + L) if window else data["page_table"].shape[1] * page
    ntk = (smax + 127) // 128
    kt = torch.empty((B, 2, ntk, 128 * 64), dtype=torch.uint8, device=dev)
    vt = torch.empty((B, 2, ntk, 72 * 128), dtype=torch.uint8, device=dev)
    return ntk, kt, vt


def _attn_prep_old(st, data, cfg, ev=None):
    """Stage the tiles made only of keys already in the cache on the side stream (it needs nothing from this call's
    qk_prep, so it overlaps the qkv GEMM's idle SMs); _attn_custom stages the rest after joining it. With `ev` (an
    event recorded before the qkv GEMM was enqueued) the side branch is captured after the GEMM node."""
    main, side = torch.cuda.current_stream(), _SIDE_STREAM[0]
    pre = _attn_stage_buffers(data, cfg, data["attn_input"].device)
    if ev is not None:
        side.wait_event(ev)
    else:
        side.wait_stream(main)
    with torch.cuda.stream(side):
        st["mod"].attn_prep(data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                            data["seq_lens"], cfg["page_size"], cfg["q_len"], cfg["window"], pre[0], pre[1], pre[2], 1)
    return pre


def _attn_custom(st, data, q8, qkv, cfg, nsplit=1, variant=0, pre=None):
    """FP8 attention (own kernel) with the output norm * sigmoid(gate) fused: returns og [T, 1024] bf16."""
    B, L, window, page = cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"]
    dev = q8.device
    if pre is not None:  # part 1 was issued on the side stream by _attn_prep_old: join it, then stage the rest
        ntk, kt, vt = pre
        torch.cuda.current_stream().wait_stream(_SIDE_STREAM[0])
    else:
        ntk, kt, vt = _attn_stage_buffers(data, cfg, dev)
    st["mod"].attn_prep(data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), data["page_table"],
                        data["seq_lens"], page, L, window, ntk, kt, vt, 2 if pre is not None else 0)
    out = torch.empty((B * L, Q_SIZE), dtype=torch.bfloat16, device=dev)
    c = cfg["sm_scale"] * cfg["qk_scale_factor"] * cfg["kv_scale"] * 1.4426950408889634
    if nsplit < 0:  # stream-K over -nsplit CTAs: int header (W, pair prefix, first pairs) + npairs + G partial slots
        G, nqb = -nsplit, (L + 23) // 24
        hdr = (2 + B * nqb * 2 + 1 + G + 63) // 64 * 64
        ws = torch.empty((hdr + (B * nqb * 2 + G) * 24 * 8 * 66,), dtype=torch.float32, device=dev)
    else:
        ws = torch.empty((nsplit, B * L, HQ, 66), dtype=torch.float32, device=dev) if nsplit > 1 else out
    pf = st["o"] if B * L <= ATTN_PF_MAX_T else st["no_pf"]  # o-proj weight into L2 during the attention
    st["mod"].attn_fwd(q8.view(torch.uint8), kt, vt, data["seq_lens"], qkv, out, L, window, ntk, c, cfg["kv_scale"],
                       st["attn_smem"], nsplit, ws, variant, pf)
    return out


def _selftest_attn2(st):
    if _fa3 is None:
        return
    dev = "cuda"
    torch.manual_seed(2)
    for name, B, L, ctx, window in ATTN_SCEN:
        S = ctx + L
        T = B * L
        q8 = (torch.randn((T, HQ, HD), device=dev) * 0.7).to(FP8)
        qkv = torch.randn((T, QKV_SIZE), device=dev).to(torch.bfloat16)
        kc = (torch.randn((B * S, 1, HKV, HD), device=dev) * 0.8 + 0.4).to(FP8)
        vc = torch.randn((B * S, 1, HKV, HD), device=dev).to(FP8)
        perm = torch.randperm(B * S, device=dev).to(torch.int32)
        data = {"k_cache": kc, "v_cache": vc, "page_table": perm.view(B, S), "seq_lens": torch.full((B,), ctx, dtype=torch.int32, device=dev),
                "cache_seqlens": torch.full((B,), S, dtype=torch.int32, device=dev),
                "cu_seqlens_q": torch.arange(0, T + 1, L, dtype=torch.int32, device=dev), "scheduler_metadata": None,
                "k_scale": torch.tensor(1.0, device=dev), "v_scale": torch.tensor(1.0, device=dev)}
        cfg = {"batch": B, "q_len": L, "window": window, "page_size": 1, "sm_scale": 0.125, "qk_scale_factor": 5.4730064863875,
               "kv_scale": 1.0, "tokens": T}
        def ref_fn():
            att = _attention(data, q8, cfg)
            og = torch.empty((T, Q_SIZE), dtype=torch.bfloat16, device=dev)
            _norm_gate_kernel[(T,)](att, qkv, og, num_warps=4)
            return og
        flops = 4.0 * HQ * HD * (B * L * (ctx if not window else min(ctx, window)) + B * L * L / 2)
        try:
            r = ref_fn()
            t_ref = _time(ref_fn, reps=3)
            res = [f"fa3+ng {t_ref:.0f}us ({flops / t_ref / 1e6:.0f}TF)"]
        except Exception as exc:  # noqa: BLE001
            print(f"SELFTEST attn2 {name} FA3 ERR {repr(exc)[:300]}", flush=True)
            continue
        for variant, nsplit in [(v, 1) for v in ATTN_VARIANTS] + [(v, _attn_splits(cfg, data, v)) for v in (1, 9)]:
            try:
                o = _attn_custom(st, data, q8, qkv, cfg, nsplit=nsplit, variant=variant)
                torch.cuda.synchronize()
                err = float((o.float() - r.float()).norm() / r.float().norm())
                t_own = _time(lambda: _attn_custom(st, data, q8, qkv, cfg, nsplit=nsplit, variant=variant), reps=3)
                res.append(f"v{variant}s{nsplit} {t_own:.0f}us ({flops / t_own / 1e6:.0f}TF) err {err:.4f}")
            except Exception as exc:  # noqa: BLE001
                res.append(f"v{variant}s{nsplit} ERR {repr(exc)[:120]}")
        print(f"SELFTEST attn2 {name}: " + " | ".join(res), flush=True)
        del q8, qkv, kc, vc
    torch.cuda.empty_cache()


def _selftest_boundary(st):
    dev = "cuda"
    T = 16384
    a = torch.randn((T, HIDDEN), device=dev).to(torch.bfloat16)
    s = torch.randn((T, HIDDEN), device=dev)
    o1 = torch.empty_like(s)
    o2 = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    o3 = torch.empty_like(s)
    o4 = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    _boundary_kernel[(T,)](a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], o1, o2, HAS_GPOST=True, H=HIDDEN, num_warps=16)
    st["mod"].boundary(a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], o3, o4, True, st["no_zero"])
    torch.cuda.synchronize()
    d1 = float((o1 - o3).abs().max())
    d2 = float((o2.float() - o4.float()).abs().max())
    t1 = _time(lambda: _boundary_kernel[(T,)](a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], o1, o2, HAS_GPOST=True, H=HIDDEN, num_warps=16))
    t2 = _time(lambda: st["mod"].boundary(a, s, st["g_post_ffn_norm"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], o3, o4, True, st["no_zero"]))
    print(f"SELFTEST boundary triton {t1:.0f}us cuda {t2:.0f}us (diff {d1:.1e} {d2:.1e})", flush=True)
    del a, s, o1, o2, o3, o4
    torch.cuda.empty_cache()
