# okc: problem=muse_bf16_proj
"""Dense projections: SGLang's cuBLAS path, plus Triton GEMV/small-GEMM kernels, a
transposed-weight cuBLAS layout and a 2-kernel router hi/lo (mm + addmm epilogue add),
chosen per (layer, m, n) by timing CUDA-graph replays against the baseline path in
prepare() (untimed). Unchosen or untuned shapes run exactly the baseline path.
All weights are used at their given precision (BF16; router FP32 or BF16 hi + lo)."""

import time
from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
except Exception:  # pragma: no cover
    def _bf16_gemm_dispatch_impl(x, w, b):
        return F.linear(x, w, b)


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


ROUTER_FP32_MAX_TOKENS = 8
SCORED_M = (1, 8, 16, 32, 128, 256)
TUNE_BUDGET_S = 220.0


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, o_ptr, M, N, K: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # out[m, n] = sum_k x[m, k] * w[n, k], FP32 FMA accumulation; reduced once at the end.
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    mmask = rm < M
    acc = tl.zeros([BM, BN, BK], dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :]).to(tl.float32)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)
        acc += x[:, None, :] * w[None, :, :]
    out = tl.sum(acc, axis=2)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], out.to(o_ptr.dtype.element_ty), mask=mmask[:, None])


@triton.jit
def _gemv_split_kernel(x_ptr, w_ptr, o_ptr, M, N, K: tl.constexpr, KS: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Split-K GEMV into a zeroed FP32 output: each program reduces KS of K and adds atomically.
    pid = tl.program_id(0)
    ks = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    mmask = rm < M
    acc = tl.zeros([BM, BN, BK], dtype=tl.float32)
    for k0 in tl.range(ks * KS, ks * KS + KS, BK):
        rk = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :]).to(tl.float32)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)
        acc += x[:, None, :] * w[None, :, :]
    out = tl.sum(acc, axis=2)
    tl.atomic_add(o_ptr + rm[:, None] * N + rn[None, :], out, mask=mmask[:, None])


@triton.jit
def _router_hilo_kernel(x_ptr, hi_ptr, lo_ptr, o_ptr, M, N, K: tl.constexpr, KS: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # out += x @ hi.T + x @ lo.T over one K split, BF16 tensor cores with FP32 accumulation.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    ks = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in tl.range(ks * KS, ks * KS + KS, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0)
        hi = tl.load(hi_ptr + rn[:, None] * K + rk[None, :])
        lo = tl.load(lo_ptr + rn[:, None] * K + rk[None, :])
        acc = tl.dot(x, tl.trans(hi), acc)
        acc = tl.dot(x, tl.trans(lo), acc)
    tl.atomic_add(o_ptr + rm[:, None] * N + rn[None, :], acc, mask=mmask[:, None])


@triton.jit
def _gemm_small_kernel(x_ptr, w_ptr, o_ptr, M, N, K: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # BF16 tensor-core GEMM for small M: out = x @ w.T, FP32 accumulation.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mmask = rm < M
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc.to(o_ptr.dtype.element_ty), mask=mmask[:, None])


def _next_pow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


def _gemv(x, w, n, out_dtype, cfg):
    bm, bn, bk, warps, stages = cfg
    m, k = x.shape
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    _gemv_kernel[(n // bn,)](x, w, out, m, n, K=k, BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


def _gemv_split(x, w, n, cfg):
    bm, bn, bk, warps, stages, split = cfg
    m, k = x.shape
    out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    _gemv_split_kernel[(n // bn, split)](x, w, out, m, n, K=k, KS=k // split, BM=bm, BN=bn, BK=bk,
                                         num_warps=warps, num_stages=stages)
    return out


def _router_hilo(x, hi, lo, cfg):
    bm, bn, bk, warps, stages, split = cfg
    m, k = x.shape
    n = hi.shape[0]
    out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    _router_hilo_kernel[(n // bn, triton.cdiv(m, bm), split)](x, hi, lo, out, m, n, K=k, KS=k // split,
                                                              BM=bm, BN=bn, BK=bk, num_warps=warps,
                                                              num_stages=stages)
    return out


def _gemm_small(x, w, n, out_dtype, cfg):
    bm, bn, bk, warps, stages = cfg
    m, k = x.shape
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    _gemm_small_kernel[(n // bn, triton.cdiv(m, bm))](x, w, out, m, n, K=k, BM=bm, BN=bn, BK=bk,
                                                       num_warps=warps, num_stages=stages)
    return out


def _baseline(x, layer, n, state):
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _gemv_configs(m, n, k, limit=10):
    bm = _next_pow2(m)
    cfgs = []
    for bn in (1, 2, 4, 8, 16, 32):
        if n % bn or n // bn < 64:
            continue
        for bk in (256, 512, 1024, 2048):
            if bk > k or k % bk:
                continue
            for warps in (2, 4, 8):
                regs = bm * bn * bk // (32 * warps)
                if 8 <= regs <= 64:
                    cfgs.append((bm, bn, bk, warps, 3))
    if len(cfgs) > limit:
        step = len(cfgs) / float(limit)
        cfgs = [cfgs[int(i * step)] for i in range(limit)]
    return cfgs


def _split_configs(m, n, k):
    cfgs = []
    for cfg in _gemv_configs(m, n, k // 8, limit=6):
        for split in (4, 8):
            if (k // split) % cfg[2] == 0 and (n // cfg[1]) * split <= 8192:
                cfgs.append(cfg + (split,))
    return cfgs[:8]


def _hilo_configs(m, n, k):
    bm = 16 if m <= 16 else (32 if m <= 32 else 64)
    cfgs = []
    for bn, bk, split in ((32, 128, 8), (32, 128, 16), (64, 128, 8), (64, 128, 16), (64, 64, 32), (32, 256, 8)):
        if n % bn or (k // split) % bk:
            continue
        cfgs.append((bm, bn, bk, 4, 3, split))
    return cfgs


def _dot_configs(m, n, k):
    bm = max(16, _next_pow2(m))
    cfgs = []
    for bn in (16, 32, 64, 128):
        if n % bn or not (64 <= n // bn <= 1024):
            continue
        for bk in (64, 128, 256):
            if k % bk:
                continue
            smem = (bm + bn) * bk * 2
            stages = 4 if smem * 4 <= 160 * 1024 else 3
            if smem * stages > 200 * 1024:
                continue
            cfgs.append((bm, bn, bk, 4, stages))
    if len(cfgs) > 6:
        step = len(cfgs) / 6.0
        cfgs = [cfgs[int(i * step)] for i in range(6)]
    return cfgs


def _candidates(layer, m, n, k, state):
    cands = []
    if layer == "router":
        w = state["router"]
        if m <= ROUTER_FP32_MAX_TOKENS:
            for cfg in _gemv_configs(m, n, k):
                cands.append((("gemv",) + cfg, lambda x, c=cfg: _gemv(x, w, n, torch.float32, c)))
            for cfg in _split_configs(m, n, k):
                cands.append((("gemv_split",) + cfg, lambda x, c=cfg: _gemv_split(x, w, n, c)))
        hi, lo = state["router_hi"], state["router_lo"]
        for cfg in _hilo_configs(m, n, k):
            cands.append((("hilo_split",) + cfg, lambda x, c=cfg: _router_hilo(x, hi, lo, c)))
        hi_t, lo_t = state["router_hi_t"], state["router_lo_t"]
        if m > ROUTER_FP32_MAX_TOKENS:
            def addmm_path(x):
                acc = torch.mm(x, hi_t, out_dtype=torch.float32)
                return torch.addmm(acc, x, lo_t, out_dtype=torch.float32)
            cands.append((("hilo_addmm",), addmm_path))
        return cands
    w = state[layer][:n]
    wt = state["t_" + layer + ("_slice" if n != state[layer].shape[0] else "")]
    cands.append((("cublas_t",), lambda x: torch.mm(x, wt)))
    if m <= 8:
        for cfg in _gemv_configs(m, n, k):
            cands.append((("gemv",) + cfg, lambda x, c=cfg: _gemv(x, w, n, torch.bfloat16, c)))
    if 2 <= m <= 32:
        for cfg in _dot_configs(m, n, k):
            cands.append((("dot",) + cfg, lambda x, c=cfg: _gemm_small(x, w, n, torch.bfloat16, c)))
    return cands


def _bench(fn, x, flush, calls=20, reps=7):
    for _ in range(2):
        fn(x)
    torch.cuda.synchronize()
    g_warm, g_cold, g_flush = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_warm):
        for _ in range(calls):
            fn(x)
    with torch.cuda.graph(g_cold):
        for _ in range(calls):
            flush.zero_()
            fn(x)
    with torch.cuda.graph(g_flush):
        for _ in range(calls):
            flush.zero_()
    res = {}
    for name, g in (("warm", g_warm), ("cold", g_cold), ("flush", g_flush)):
        g.replay()
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / calls)
        ts.sort()
        res[name] = ts[len(ts) // 2]
    del g_warm, g_cold, g_flush
    return res["warm"], max(res["cold"] - res["flush"], 1e-3)


def _close(out, ref, tol):
    if out.shape != ref.shape or out.dtype != ref.dtype or not torch.isfinite(out).all():
        return False
    d = (out.double() - ref.double()).norm() / ref.double().norm().clamp(min=1e-30)
    return float(d) <= tol


def _tune(state):
    t0 = time.time()
    dev = state["router"].device
    flush = torch.empty(96 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(1234)
    choice = {}
    for layer, wfull in (("qkv_gate", state["qkv_gate"]), ("o", state["o"]), ("pooler", state["pooler"]),
                         ("unpooler", state["unpooler"]), ("router", state["router"])):
        k = wfull.shape[1]
        for m in SCORED_M:
            if time.time() - t0 > TUNE_BUDGET_S:
                print("tune budget exhausted")
                state["choice"] = choice
                return
            n = 1024 if (layer == "unpooler" and m <= 8) else wfull.shape[0]
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            x[0].zero_()
            x[0, 7] = 32.0  # a probe row
            base = lambda xx, l=layer, nn_=n: _baseline(xx, l, nn_, state)
            ref = base(x)
            tol = 2e-5 if layer == "router" else 2e-3
            try:
                bw, bc = _bench(base, x, flush)
            except Exception as e:
                print("baseline bench failed", layer, m, repr(e)[:200])
                continue
            best = None
            for name, fn in _candidates(layer, m, n, k, state):
                try:
                    out = fn(x)
                    torch.cuda.synchronize()
                    if not _close(out, ref, tol) or not _close(out[:1], ref[:1], 1e-3 if layer != "router" else 1e-5):
                        print("reject (accuracy)", layer, m, name)
                        continue
                    cw, cc = _bench(fn, x, flush)
                except Exception as e:
                    print("candidate failed", layer, m, name, repr(e)[:200])
                    continue
                rw, rc = cw / bw, cc / bc
                score = 0.5 * (rw + rc)
                need = 0.97 if layer == "router" else 0.93
                if max(rw, rc) < 1.0 and score < need and (best is None or score < best[0]):
                    best = (score, name, fn)
            if best is not None:
                # Confirm against a fresh baseline measurement.
                bw2, bc2 = _bench(base, x, flush)
                cw2, cc2 = _bench(best[2], x, flush)
                s2 = 0.5 * (cw2 / bw2 + cc2 / bc2)
                if s2 < (0.98 if layer == "router" else 0.95):
                    choice[(layer, m, n)] = best[2]
                    print(f"choose {layer} m={m} n={n}: {best[1]} ratio {best[0]:.3f}/{s2:.3f} base {bw:.2f}/{bc:.2f}us")
                else:
                    print(f"keep baseline {layer} m={m}: {best[1]} not confirmed ({s2:.3f})")
            else:
                print(f"keep baseline {layer} m={m} n={n}: base {bw:.2f}/{bc:.2f}us")
    state["choice"] = choice
    print(f"tuning took {time.time() - t0:.1f}s")


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state["t_" + name] = state[name].t().contiguous()
    state["t_unpooler_slice"] = state["unpooler"][:1024].t().contiguous()
    state["choice"] = {}
    try:
        _tune(state)
    except Exception as e:
        print("tuning failed, using baseline paths:", repr(e)[:300])
        state["choice"] = {}
    torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    fn = state["choice"].get((layer, x.shape[0], n))
    if fn is not None and x.is_contiguous():
        return fn(x)
    return _baseline(x, layer, n, state)
