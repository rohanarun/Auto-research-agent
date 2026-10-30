# okc: problem=muse_bf16_proj
# Ported from team vjeux's Open Kernel Challenge entry: https://github.com/vjeux/okc-2026 (muse_bf16_proj.py),
# licensed under the Apache License, Version 2.0; used and redistributed under its terms. Credit: team vjeux.
# Modifications by team rohanarun: this header, extra autotuner candidates (rq_*), and a per-shape composition block at the end.
"""Dense projections for Muse Spark Open: skinny swap-AB tensor-core GEMV/GEMM with
losslessly compressed BF16 weights.

out[m, n] = x[m, k] @ W[n, k]^T.  For m <= 64 every weight byte is streamed once from HBM
straight into registers, in mma.m16n8k16 A-fragment order (prepared once), and multiplied
against x (staged in shared memory) on BF16 tensor cores with FP32 accumulation.

Lossless 12-bit weights (BF16 projections): each BF16 weight is stored as its low byte
(exponent bit 0 + 7 mantissa bits, raw) and a 4-bit code for its high byte (sign + exponent
bits 7..1, as an offset into an 8-value window chosen per matrix).  The kernel rebuilds the
exact BF16 bits in registers (a few integer ops per weight).  The rare weights whose exponent
falls outside the window (tiny magnitudes) are exact-corrected: the kernel adds
x[:, k] * (w - w_decoded) for each of them (both factors exact in FP32), so every output
equals the product with the original weights up to FP32 accumulation order.

Split-K across warps (shared-memory reduction) and across CTAs (FP32 partials + a
self-resetting semaphore; the last CTA of a row block reduces) keeps ~128 CTAs streaming on
every shape.  The router's FP32 weights run as a BF16 hi + lo pair (concatenated along K)
with FP32 accumulation, as SGLang does above 8 tokens.
"""

from typing import TypedDict

import torch
import torch.nn.functional as F
import sys

from torch.utils.cpp_extension import load_inline
import os as _os
_SK2_DELAY = int(_os.environ.get("SK2_DELAY_ESC", "0"))   # debug only: one random escape warp per CTA stalls this many ns at a random stage; ring waits trap after 2 s


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


CUDA_SRC = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.
// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low
//   bytes of the lane's 32 weights of 4 k-steps, part 2 = their 32 codes.
// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).
// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).
// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).
// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).
@@MBAR@@
@@BODY@@
"""

MBAR_HELPERS = r"""
// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#endif
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
// escape correction from the stored original BF16 weight t: its 12-bit decode keeps the sign and low byte and clamps the
// exponent into [base, base + 7]; the exact difference (asserted in prepare) is formed here in FP32
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
#if __CUDA_ARCH__ >= 900
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
#else
  mbar_wait(bar, parity);
#endif
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
// L2 policies (v132: used by the 2-D TMA boxes of the m >= 33 routes only - gq/gqx/gemm90/wide): weights evict_first,
// x evict_last
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}
#endif

"""

WG_HELPERS = r"""
// ---------------------------------------------------------------- wgmma helpers (sm_90a)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// smem matrix descriptor: K-major, 128B swizzle, SBO = 1024 B (8 rows x 128 B)
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;                 // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 62;                 // 128B swizzle
  return d;
}


"""


def wgmma_defs(bns):
    """Generate wgmma_call overloads for m64nBNk16 bf16 with A from registers."""
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        a0 = nacc
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        pi = a0 + 5
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{pi}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
      "}}\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}""")
    return "\n".join(out)


TMA_SRC = r"""
// Weights: per-warp contiguous layout, uncompressed [nrg][S][32][16B], compressed [nrg][S/4][3][32][16B].
// One stage = KS k-steps (uncompressed) or KS/4 units (compressed) for every consumer warp.
// Warps 0..NW-1 consume, warp NW produces (one elected lane issues bulk copies on sm_90;
// on sm_80 all 32 producer lanes issue 16-byte cp.async and arrive with cp.async.mbarrier.arrive).
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
  // dg (diagnostics only, 0 in production): 1 = skip epilogue, 2 = also skip x / escape-table loads,
  // 3 = also skip all compute (pure weight streaming through the pipeline)
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;                       // consumer threads
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;  // padded red row (bank-conflict-free epilogue reads)
  // bytes per warp per stage (HL: router hi and lo tiles of the same k-step back to back)
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  // layout: [bars: full NSTAGE, empty NSTAGE, xbar XP (pad 256)] [ring NSTAGE*SLOT] [x rows M*XS*2 + zero row] [red]
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
  // WGM: x as 128B-swizzled K-major tiles [Kc/64][TOK rows][128 B] (wgmma B operand), 1024-aligned
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;   // WGM: absolute shared address 1024-aligned
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);       // k-steps per consumer warp
  const int nst = NS / KS;              // stages
  // WS == 1: stream is [rb][stage over all K][warp][chunk] (shared with the wide kernel); else [cta][stage][warp][chunk]
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
#if __CUDA_ARCH__ >= 900
      mbar_init(full0 + 8 * i, 1);
#else
      mbar_init(full0 + 8 * i, 32);
#endif
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // first stages go out right away (no wait for the CTA barrier)
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
#endif
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;  // zero row
  __syncthreads();

  if (warp == NW) {
    // ---------------- producer: this CTA's weight stream is contiguous
#if __CUDA_ARCH__ >= 900
    const int st0 = NSTAGE < nst ? NSTAGE : nst;   // already issued by thread 0
#else
    const int st0 = 0;
#endif
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
#if __CUDA_ARCH__ >= 900
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
#else
      for (int off = lane * 16; off < SLOT; off += 512)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + off), "l"(src + off) : "memory");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(full0 + 8 * slot) : "memory");
#endif
    }
    return;
  }

  // ---------------- consumers
  const int rg = rb * RG + warp % RG;   // warps of a warpgroup: 4 consecutive row groups, same k range
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
    // this CTA's exact-correction entries -> smem (tracked by x part 0's mbarrier below)
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
  // x chunk in XP parts, each completion tracked by its own mbarrier (no CTA-wide wait)
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;  // 16-byte chunks per token row per part
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);   // c: 16-byte chunk index along k
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;  // k per part

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];

  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;   // bytes per k-step pair... per k-step: 32 B
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;  // x part holding this stage's last k
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
#if __CUDA_ARCH__ >= 900
        if constexpr (WGM) fence_async_smem();   // cp.async (generic proxy) writes -> wgmma (async proxy) reads
#endif
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    // pull this warp's stage data into registers, then hand the slot back to the producer at once
    constexpr int NL = CHUNK / 512;  // 16-byte loads per lane per stage
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (dg == 3) {
      if (wr[0].x == 0x9e3779b9u && wr[NL - 1].w == 0x7f4a7c15u) acc[0][0] += 1.f;   // keep the loads
      continue;
    }
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#if __CUDA_ARCH__ >= 900
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;   // local k of this unit's first k-step (multiple of 64)
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();   // the wgmma that used afrag[(j+1)&1] is done -> it can be overwritten next step
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
#else
      // sm_80 check path: same data layout, mma.sync with B fragments read from the swizzled x tile
      const int g8 = lane >> 2, t8 = lane & 3;
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;
        const unsigned char* tile = reinterpret_cast<const unsigned char*>(xs) + (kl >> 6) * TOK * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int jj = 0; jj < MT; ++jj) {
            const int tok = jj * 8 + g8;
            const unsigned char* row = tile + tok * 128;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t8);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t8);
            mma_bf16(wacc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
          }
        }
      }
#endif
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
#if __CUDA_ARCH__ >= 900
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }
#endif

  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  if (dg >= 1) {
    if (acc[0][0] == 1.2345e-30f) out[threadIdx.x] = cvt_out<OutT>(acc[0][1]);
    return;   // (the producer warp has already returned)
  }
  // per-warp partials -> shared memory: red[w][row16][TOK]
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
    // all x parts and the patch table are in smem; apply exact corrections in parallel (smem atomics)
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;   // warp (wsub 0, row group rl/16)
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {   // swizzled K-major tiles [Kc/64][TOK][128 B]
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
  // dist: all `splits` CTAs of this row block are co-resident (grid fits the GPU at once): monotonic-counter
  // barrier, then each reduces its own slice of the row block.  Otherwise the last-arriving CTA reduces all.
  // Counters only ever increase (both modes), so the modes can share them.
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      while (*((volatile int*)(counters + rb)) - target < 0) __nanosleep(20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }  // dist || s_last
}

template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
"""

MAP_SRC = r"""
#include <cudaTypedefs.h>
#include <cstring>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
#endif
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

"""

GEMM_SRC = r"""

@@WGMMA_SS_DEFS@@

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;       // BM rows x 64 bf16
  constexpr int B_BYTES = BN * 128;       // BN rows x 64 bf16
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;           // consumer threads
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;          // full[STAGES], empty[STAGES]
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();

  if (warp >= NWG * 4) {
    // ---------------- producer warp (one lane)
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
        if constexpr (AS)   // A tile pre-swizzled and contiguous: [rb][kb][BM x 128 B]
          bulk_g2s(sbase + slot * ST_BYTES, Wst + ((size_t)rb * nkb_all + kb0 + i) * A_BYTES, A_BYTES, full0 + 8 * slot);
        else
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }

  // ---------------- consumers
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;

  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    // release the previous stage once its wgmmas are done (keep one group in flight)
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    while (true) {
      int v = *((volatile int*)(counters + rb));
      if (v - target >= 0) break;
      __nanosleep(32);
    }
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
"""


SK2_SRC = r"""

#ifndef SK2_DELAY_ESC
#define SK2_DELAY_ESC 0
#endif
#if SK2_DELAY_ESC
// debug builds only: every ring wait of the escape and producer warps gives up after 2 s (globaltimer) with a trap, so
// a wedged ring aborts the launch instead of hanging the GPU
__device__ __forceinline__ void sk2_wait_dbg(uint32_t bar, uint32_t parity) {
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  uint32_t ok = 0;
  for (;;) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(1000u) : "memory");
    if (ok) return;
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t - t0 > 2000000000ull) {
      printf("sk2 ring wedged: block %d warp %d bar 0x%x parity %u\n", (int)blockIdx.x, (int)(threadIdx.x >> 5), bar, parity);
      __trap();
    }
  }
}
#define SK2_WAIT(bar, par) sk2_wait_dbg((bar), (par))
// one (escape warp, stage) per CTA per launch stalls SK2_DELAY_ESC ns at the end of its stage step; the pair comes from a
// hash of the launch's start time (read once per CTA by every escape warp from the same smem word) and the block index
#define SK2_DELAY_HOOK(st_) do { if ((st_) == dly_st && ew == dly_w) { const unsigned long long t0_ = gtime(); \
    while (gtime() - t0_ < (unsigned long long)SK2_DELAY_ESC) __nanosleep(500); } } while (0)
#define SK2_DLY_SEED_DECL __shared__ unsigned long long sk2_dly_seed;
#define SK2_DLY_SEED_SET() (sk2_dly_seed = gtime())
#define SK2_DLY_PICK() \
    unsigned long long h_ = sk2_dly_seed ^ (0x9E3779B97F4A7C15ull * (unsigned long long)(blockIdx.x + 1)); \
    h_ ^= h_ >> 30; h_ *= 0xBF58476D1CE4E5B9ull; h_ ^= h_ >> 27; h_ *= 0x94D049BB133111EBull; h_ ^= h_ >> 31; \
    const int dly_w = (int)(h_ % (unsigned long long)NESC), dly_st = (int)((h_ >> 32) % (unsigned long long)(nst > 0 ? nst : 1));
#else
#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#define SK2_DELAY_HOOK(st_) do { } while (0)
#define SK2_DLY_SEED_DECL
#define SK2_DLY_SEED_SET() ((void)0)
#define SK2_DLY_PICK()
#endif
#ifndef SK2_NESC
#define SK2_NESC 3
#endif
#ifndef SK2_DIAG
#define SK2_DIAG 0
#endif
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg) {
#if __CUDA_ARCH__ >= 900
  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds
  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;
  const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);
  unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;
  const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;
  if (trace && threadIdx.x == 0) tr[0] = t_start;
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  SK2_DLY_SEED_DECL
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                      // stages over all of K
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    SK2_DLY_SEED_SET();
    mbar_init(full0, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, ((dgv & 4) ? 0 : XBYr) + WBY);
    bulk_g2s(sbase + XBY, wsrc + (size_t)gs0 * wstride, WBY, full0);
    if (!(dgv & 4)) tma_load_3d(sbase, &xmap, full0, 0, 0, gs0 * UPS);
    if (COMP) {   // this CTA's escape table
      if constexpr (CL == 1) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
  __syncthreads();
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = 1; st < nst; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
        if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    SK2_DLY_PICK()
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      const uint32_t bq = base4 & 0xFFu;
      int pe = 1;
      for (int st = 0; st < nst; ++st) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        const int slot = st % NSTAGE;
        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          SK2_DELAY_HOOK(st);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
        SK2_DELAY_HOOK(st);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();
    const uint32_t s0 = sbase + slot * SST;
    if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      // 3) tensor-core work: independent accumulators per k-step
      if (dgv & 8) {   // diagnostic: replace the tensor-core instructions by ALU work
        const int nh = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);
          }
        continue;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();
  }
  if (trace && lane == 0) tr[1 + warp] = gtime();   // per consumer warp: main loop done

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if (dgv & 1) {
    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);
    return;
  }
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    if (trace && tid == 0) tr[15] = gtime();
    return;
  }
  named_bar(1, NC);
  if (trace && tid == 0) tr[9] = gtime();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      if (trace && mt == 0 && nt == 0) {
        asm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
        if (tid == 0) tr[12] = gtime();
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  if (trace && tid == 0) tr[11] = gtime();
  named_bar(1, NC);
  if (trace && tid == 0) tr[10] = gtime();
  if (trace && tid == 0) tr[13] = gtime();
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  if (trace && tid == 0) tr[14] = gtime();
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if (COMP && ptab_s[0].y != 0)   // escape-balanced row order (v135): 4-row groups moved whole, so they stay contiguous
      o = out + (size_t)tt * N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
  if (trace && tid == 0) tr[15] = gtime();
  if ((dgv & 32) && dbg != nullptr && tid == 0) {   // every CTA: SM id, start and end time
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();
  }
#endif
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    if constexpr (CL > 1) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[1];
      at_[0].id = cudaLaunchAttributeClusterDimension;
      at_[0].val.clusterDim.x = CL;
      at_[0].val.clusterDim.y = 1;
      at_[0].val.clusterDim.z = 1;
      lc.attrs = at_;
      lc.numAttrs = 1;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""


WIDE_SRC = r"""
// ============================================================================ wide kernel (m 33..128)
// Weights: the per-row-block compressed stream [rb][unit][warp][1536 B] (unit = 16 rows x 64 k per warp),
// one bulk copy per stage; x: one TMA tile [TOK x 64] (128B swizzle) per stage.  Consumers decode the
// weights into registers and run RS-wgmma m64nTOKk16 (sm_90) / mma.sync on the swizzled tile (sm_80 check
// path).  Split-K over units (uneven splits allowed); escapes corrected from a per-CTA table.
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const __grid_constant__ CUtensorMap mapA, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
  // No separate producer warp (17 warps would cap registers at 96 per thread: 5 warps on one SM sub-partition):
  // thread 0 issues the stage copies.
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
  // per warp per 64-k unit: compressed 1536 B, or (HLW, router) BF16 hi tile 2 KB + lo tile 2 KB in fragment order
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;                    // token block: rows [zh*TOK, zh*TOK + Mz) of x / out
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
#if __CUDA_ARCH__ >= 900
      mbar_init(full0 + 8 * i, 1);
#else
      mbar_init(full0 + 8 * i, 32);
#endif
      mbar_init(empty0 + 8 * i, NWG);
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }

  // stage issue (sm_90: thread 0, TMA + one bulk copy; sm_80 check path: warp 0, cp.async)
  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
#if __CUDA_ARCH__ >= 900
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_el(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    if constexpr (HLW) {   // router: BF16 hi and lo tiles [64 rows x 64 k] by TMA (128B swizzle) from [hi | lo]
      tma_load_2d_ef(st + B_BYTES, &mapA, full0 + 8 * slot, (u0 + i) * 64, rb * 64);
      tma_load_2d_ef(st + B_BYTES + 8192, &mapA, full0 + 8 * slot, K + (u0 + i) * 64, rb * 64);
    } else {
      bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
    }
#else
    for (int idx = lane; idx < TOK * 8; idx += 32) {
      const int t = idx >> 3, c = idx & 7;
      const uint32_t dst = st + t * 128 + ((c ^ (t & 7)) << 4);
      const __nv_bfloat16* src = x + (size_t)(t0z + (t < Mz ? t : 0)) * K + (u0 + i) * 64 + c * 8;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < Mz ? 16 : 0) : "memory");
    }
    const char* wa = reinterpret_cast<const char*>(wsrc) + (size_t)i * A_BYTES;
    for (int off = lane * 16; off < A_BYTES; off += 512)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(st + B_BYTES + off), "l"(wa + off) : "memory");
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(full0 + 8 * slot) : "memory");
#endif
  };
#if __CUDA_ARCH__ >= 900
  const bool producer = tid == 0;
#else
  const bool producer = warp == 0;
#endif
#if __CUDA_ARCH__ >= 900
  if (producer) {   // thread 0: barriers are initialized (program order) -> issue before the CTA barrier
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    if constexpr (HLW) asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
#else
  __syncthreads();
  if (producer)
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
#endif
  // escape table + the x values it needs -> shared memory (used after the main loop)
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }

  // ---------------- consumers
  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
#if __CUDA_ARCH__ >= 900
      // router: SS-wgmma on the TMA'd hi and lo tiles (no register-sourced operands: the wgmma pipeline stays full)
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + j * 32), make_desc_sw128(st + j * 32));
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wgmma_ss<TOK>(acc, make_desc_sw128(st + B_BYTES + 8192 + j * 32), make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
#else
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {   // h = 0: hi tile, 1: lo tile (same B) -> FP32 sum of both
          uint32_t* af = afrag[h];
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                       : "r"(wa + h * 2048 + j * 512));
#if __CUDA_ARCH__ >= 900
#pragma unroll
          for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
          wg_commit();
          wg_wait<1>();
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
#else
          const unsigned char* tile = gbase + slot * ST_BYTES;
#pragma unroll
          for (int jj = 0; jj < TOK / 8; ++jj) {
            const int tok = jj * 8 + g;
            const unsigned char* row = tile + tok * 128;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t4);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t4);
            mma_bf16(acc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
          }
#endif
        }
      }
#endif
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // k-step j: low bytes = 8 B at (j/2)*512 + (j%2)*8, codes = 4 B at 1024 + 4j (per-k-step loads: low pressure)
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#if __CUDA_ARCH__ >= 900
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
#else
      const unsigned char* tile = gbase + slot * ST_BYTES;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const int tok = jj * 8 + g;
        const unsigned char* row = tile + tok * 128;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t4);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t4);
        mma_bf16(acc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
      }
#endif
    }
    }  // !HLW
#if __CUDA_ARCH__ >= 900
    // stage i-1's wgmmas are complete (one group in flight) -> its slot can be refilled
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
#else
    named_bar(1, NT);   // check path: every warp is done with this slot -> refill it
    if (producer && i + STAGES < nst) issue(i + STAGES);
#endif
  }
#if __CUDA_ARCH__ >= 900
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
#endif

  // ---------------- exact corrections for escapes (table + x values staged at kernel start)
  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }

  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  named_bar(1, NT);
  if (tid == 0) {
    // release: every thread's partial (ordered before the CTA barrier) is visible before the count; acquire on the
    // spin makes the other CTAs' partials visible to this CTA after the barrier below
    int old;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(counters + zh * nrb + rb) : "memory");
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      int v;
      do {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(counters + zh * nrb + rb) : "memory");
      } while (v - target < 0);
    } else if (s_last) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}

template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;   // token blocks (same weight stream: concurrent CTAs share it in L2)
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB, mA;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  mA = mB;
  if (HLW && major >= 9) mA = make_map_2d(Wst.data_ptr(), (uint64_t)Wst.size(0), (uint64_t)Wst.size(1), 64);   // [hi | lo]
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, mA, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
"""


def wgmma_ss_defs(bns):
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        a, b, p = nacc, nacc + 1, nacc + 2
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{p}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
      "}}\\n"
      : {cons}
      : "l"(da), "l"(db), "r"(1));
}}""")
    return "\n".join(out)


CPP_SRC = """
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void rgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M);
void sgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t N);
void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth);
void gqx(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage);
void rk(torch::Tensor Wc, torch::Tensor x, torch::Tensor out, int64_t S, torch::Tensor zc);
"""

# (layer, mt) -> (splits, RG, WS); grid = (N / (16 RG), splits) ~ 128 CTAs
CONFIGS = {
    ("pooler", 1): (1, 2, 8), ("pooler", 2): (2, 4, 4), ("pooler", 4): (4, 8, 2), ("pooler", 8): (16, 8, 1),
    ("unpooler", 1): (2, 1, 8), ("unpooler", 2): (1, 4, 4), ("unpooler", 4): (2, 8, 2), ("unpooler", 8): (8, 8, 1),
    ("qkv_gate", 1): (8, 9, 2), ("qkv_gate", 2): (8, 9, 2), ("qkv_gate", 4): (8, 9, 1), ("qkv_gate", 8): (16, 9, 1),
    ("o", 1): (1, 4, 4), ("o", 2): (1, 4, 4), ("o", 4): (1, 4, 4), ("o", 8): (2, 8, 1),
    ("router", 1): (8, 1, 8), ("router", 2): (8, 1, 8), ("router", 4): (8, 1, 8), ("router", 8): (16, 1, 4),
}
MAX_SKINNY_M = 32
w_rows = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
COMPRESS = {"qkv_gate", "o", "pooler", "unpooler"}
D_PLAIN = 16
D_COMP = 12
PF = 0

_ext = None


def _dispatch_src():
    return ""  # LDG skinny kernel unused in v6 (TMA path); keep the template uninstantiated
    lines = []
    for (layer, mt), (splits, rg, wsn) in sorted(CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for comp in ((0,) if f32 else (0, 1)):
            d = D_COMP if comp else (D_PLAIN if rg * wsn <= 9 and mt <= 4 else 8)
            lines.append(f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                         f"{{ launch<{mt}, {rg}, {wsn}, {d}, {str(bool(comp)).lower()}, {PF}, {ot}>(Wp, x, out, ws, counters, "
                         f"poff, pk, pr, pd, base4, M, N, Kp, Kx, splits, nrg); return; }}")
    return "\n  ".join(dict.fromkeys(lines))


MID_CONFIGS = {  # (layer, bn) -> (nwg, splits); grid = (N / (64 nwg), splits) <= 132 CTAs (co-resident)
    ("pooler", 128): (2, 4), ("unpooler", 128): (2, 2), ("qkv_gate", 128): (2, 7), ("o", 128): (2, 2),
    ("router", 128): (2, 64),
    ("pooler", 256): (2, 4), ("unpooler", 256): (2, 2), ("qkv_gate", 256): (2, 7), ("o", 256): (2, 2),
    ("router", 256): (2, 64),
}
TMA_CONFIGS = {  # (layer, mt) -> (splits, RG, WS, KS); x chunk M*Kc*2 small, ~128 CTAs; mt 0 = m == 1
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
TMA_KS = 4
USE_TMA = True
USE_WGM = True   # m >= 9 compressed configs with RG % 4 == 0 use RS-wgmma (sm_90)
WGM_SM80_TEST = False  # local check of the wgmma path's data layout (mma.sync emulation on sm_80)


def _tma_nstage(nw, comp, sm90, ks, hl=False):
    chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
    budget = (112 if sm90 else 40) * 1024
    return max(2, min(8, budget // (nw * chunk)))


TMA_TU = {}  # (layer, mtk) -> 0 / 1 (which translation unit instantiates it)


# alternative skinny configs (more splits -> fewer x bytes per CTA); the prepare()-time autotuner times them
# against the default config and cuBLAS on every scored shape
TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def _tma_cfg_items():
    items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(TMA_CONFIGS.items())]
    items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(TMA_ALT.items())]
    return items


def _tma_srcs(sm90):
    lines = {}
    for key, (splits, rg, wsn, ks) in _tma_cfg_items():
        if key[0] == "alt":
            layer, mtk = key[1], _mt_key(key[2])
        else:
            layer, mtk = key
        mt = max(mtk, 1)
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        comp = 0 if f32 else 1
        hl = f32
        ns = _tma_nstage(rg * wsn, comp, sm90, ks, hl)
        if f32 and mt >= 4:
            ns = min(ns, 3)  # x chunk (32 x 1024) + ring must fit in shared memory
        n_cfg = 1024 if (layer == "unpooler" and mtk <= 1) else w_rows[layer]
        wgm = bool((sm90 or WGM_SM80_TEST) and comp and USE_WGM and rg % 4 == 0 and mtk >= 2)
        line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
        lines.setdefault(line, []).append(key)
    keys = list(lines)
    for ln in keys:
        for cfg in lines[ln]:
            TMA_TU[cfg] = 0
    return TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(keys)).replace("@@TMA_NAME@@", "skinny_tma_a")


GEMM_CONFIGS = {  # (layer, bn) -> (nwg, splits, stages); grid = (N / (64 nwg), splits) <= 132 (co-resident)
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


DIAG_LAYERS = ()
GEMM_DIAG = False   # diagnostic: also time gemm90 variants (TMA-2D A vs contiguous pre-swizzled A) on m 65..256


def swizzle_tiles(w, bm):
    """W [N, K] -> [N/bm][K/64][bm rows x 128 B] with the 128B swizzle (16-byte chunk c of row r at c ^ (r % 8))."""
    n, k = w.shape
    t = w.view(n // bm, bm, k // 64, 8, 8).permute(0, 2, 1, 3, 4)  # [rb, kb, r, c, 8]
    r = torch.arange(bm, device=w.device).view(bm, 1)
    p = torch.arange(8, device=w.device).view(1, 8)
    src_c = (p ^ (r % 8)).view(1, 1, bm, 8, 1).expand(t.shape[0], t.shape[1], bm, 8, 8)
    return torch.gather(t, 3, src_c).contiguous()


def _gemm_src(sm90):
    if not sm90:
        return ('void gemm90(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, '
                'int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) { TORCH_CHECK(false, "gemm90 needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, splits, stages) in sorted(GEMM_CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for a_s in ((0, 1) if (not f32 and GEMM_DIAG) else (0,)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                         f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
    bns = sorted({bn for (_, bn) in GEMM_CONFIGS})
    return (GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", wgmma_ss_defs(bns))
            .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


GQ_SRC = r"""
// ============================================================================ gq (m 33..256): BF16 weights, SS-wgmma
// One CTA = BM = NWG*64 weight rows x BN tokens: W tile [BM x 64 k] and x tile [BN x 64 k] arrive by TMA (128B
// swizzle) into a ring fed by a producer warp; warpgroup w multiplies rows [64w, 64w+64) of every stage (all share
// the x tile, so x is read from L2 once per BM rows).  CL > 1: a cluster of CL CTAs splits K; ranks > 0 hand their
// accumulators to rank 0 by st.async (completing rank 0's barrier transaction count); rank 0 adds and stores BF16.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
#if __CUDA_ARCH__ >= 900
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);   // [CL-1][NWG][NACC][128]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);   // x rows actually loaded (rows >= M of the tile only feed outputs that are dropped)

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);   // split epilogue: the peer's half of our rows
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    // the first ring of stages goes out right away (no cluster barrier in front of it: a cluster-scope release
    // issued after a copy is in flight waits for it, and the stages only need this CTA's barriers)
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
  // cluster barrier: arrived here (relaxed: the barrier inits are fenced above), waited on only before remote stores
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();   // the previous stage's group is done: hand its slot back
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  if constexpr (CL <= 2) {
    // CL = 2: rank r finishes rows half r of every warpgroup (warps 2r, 2r+1); the other two warps hand their sums
    // to the peer by st.async (into [wg][warp][lane][NACC + 4]). The kept rows go out through a shared-memory
    // staging tile [wg][token][rows] so that each token's rows are written as contiguous 16-byte chunks.
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;          // kept rows per warpgroup
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;   // first kept warp of the warpgroup
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // the peer's qbar is initialized
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
    // stage BF16 [wg][token][KR rows] in the idle ring (every warpgroup's ring reads are over: wg_wait<0> above;
    // the TMA writes all landed before the last full-barrier wait)
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;   // this warp's first kept row
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;               // 16-byte chunks per token row
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
  if (CL > 1 && crank > 0) {
    // stage the accumulators in this CTA's (now idle) ring, then one bulk shared::cluster copy into rank 0's buffer
    float* mine = reinterpret_cast<float*>(gbase);   // [NWG][NACC][128]
    if constexpr (NWG > 1) named_bar(1, NWG * 128);   // every warpgroup is done reading the ring
#pragma unroll
    for (int q = 0; q < NACC; ++q) mine[(wg * NACC + q) * 128 + t] = acc[q];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    named_bar(1, NWG * 128);
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    if (tid == 0) {
      const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * NWG * NACC * 128);
      uint32_t peer_r, qbar_r;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(peer_r), "r"((uint32_t)__cvta_generic_to_shared(mine)), "r"(NWG * NACC * 128 * 4), "r"(qbar_r)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    return;
  }
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
#pragma unroll
    for (int r = 0; r < CL - 1; ++r)
#pragma unroll
      for (int q = 0; q < NACC; ++q) acc[q] += peer[((r * NWG + wg) * NACC + q) * 128 + t];
  }
  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + wg * 64 + (warp & 3) * 16;
#pragma unroll
  for (int j = 0; j < BN / 8; ++j) {
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = __float2bfloat16_rn(acc[4 * j]);
      out[(size_t)t0 * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = __float2bfloat16_rn(acc[4 * j + 1]);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
#endif
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
    kern<<<N / BM, NWG * 128 + 32, smem, at::cuda::getCurrentCUDAStream()>>>(mW, mX, op, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
"""

GQ_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): BM = 64*nwg rows per CTA, grid = N/BM * cl CTAs (cl-CTA clusters split K)
    # measured on H200 against cuBLAS (graph replay, cold L2): only the pooler shape wins (64 row blocks x 2 K halves)
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 8),
}
USE_GQ = True

GQX_SRC = r"""
// ============================================================================ gqx (m 33..256): BF16 weights, big tiles
// CTA = BM = 64 NWG weight rows x BN tokens over 1/CL of K; a cluster of CL CTAs covers all of K for one row block.
// Ring of TMA'd W [BM x 64] and x [BN x 64] tiles (128B swizzle) fed by a producer warp; warpgroup w runs SS-wgmma
// m64nBN on rows [64w, 64w + 64). Reduction: after every rank's main loop (cluster barrier: every ring is free), each
// rank owns BN/CL tokens; every thread st.async's the other ranks' token blocks of its accumulator fragment into the
// owner's ring space ([src rank][thread][4 JR floats]) and sums the CL-1 incoming copies of its own block.
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gqx_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
           __nv_bfloat16* __restrict__ out, int M, int N, int K) {
#if __CUDA_ARCH__ >= 900
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NTH = NWG * 128;
  constexpr int NB = BN / 8;                 // 8-token blocks of the tile; rank r owns blocks [r NB / CL, (r+1) NB / CL)
  constexpr int JM = (NB + CL - 1) / CL;     // most blocks a rank owns
  static_assert(CL <= NB, "gqx: every rank owns at least one token block");
  static_assert(((NB + CL - 1) / CL) * 8 * BM * 2 <= NSTAGE * ST, "gqx: the BF16 staging tile must fit in the ring");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, rbar = bars + 16 * NSTAGE;
  // CL > 1: this rank's accumulator fragments [NB][NTH] float4, outside the ring (written right after the main loop)
  float4* frag = reinterpret_cast<float4*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);
  float* recv = reinterpret_cast<float*>(gbase);   // [CL][NTH][4 JM] (the ring, once every rank is done with it)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int U = K >> 6;
  const int u0 = ((int)crank * U) / CL, u1 = (((int)crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(rbar, 1);

    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();

  if (warp == NWG * 4) {   // producer: refills; then joins the cluster barrier
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        mbar_wait(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);   // busy wait: a sleeping producer refills late
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) {   // the consumers' two cluster barriers (fragments written, reads done)
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
      asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row0 = rb * BM + wg * 64 + (warp & 3) * 16 + g;
  if constexpr (CL > 1) {
    // 1) fragments out right away (own area: no need to wait for the other ranks' main loops), 2) one cluster
    // barrier, 3) the owner of each token block sums the other ranks' copies over DSMEM, stages BF16 [token][64 rows]
    // in its own (free) ring and 4) writes each token's 64 rows as 128 contiguous bytes; 5) exit barrier
#pragma unroll
    for (int j = 0; j < NB; ++j) frag[j * NTH + tid] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    const uint32_t fl = (uint32_t)__cvta_generic_to_shared(frag);
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);   // [owned tokens][BM rows]
    const int jlo = ((int)crank * NB) / CL, jhi = (((int)crank + 1) * NB) / CL;
    const int rl = wg * 64 + (warp & 3) * 16 + g;   // this thread's first row within the tile
#pragma unroll
    for (int r = 0; r < CL; ++r) {
      if (r != (int)crank) continue;
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        if (j < (r * NB) / CL || j >= ((r + 1) * NB) / CL) continue;
        float4 pv[CL];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          uint32_t ra;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(fl + (uint32_t)((j * NTH + tid) * 16)), "r"(r2));
          asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                       : "=f"(pv[r2].x), "=f"(pv[r2].y), "=f"(pv[r2].z), "=f"(pv[r2].w) : "r"(ra) : "memory");
        }
        float v0 = acc[4 * j], v1 = acc[4 * j + 1], v2 = acc[4 * j + 2], v3 = acc[4 * j + 3];
#pragma unroll
        for (int r2 = 0; r2 < CL; ++r2) {
          if (r2 == r) continue;
          v0 += pv[r2].x; v1 += pv[r2].y; v2 += pv[r2].z; v3 += pv[r2].w;
        }
        const int tl = 8 * (j - (r * NB) / CL) + 2 * t4;   // token within the owned slice
        stg[tl * BM + rl] = __float2bfloat16_rn(v0);
        stg[tl * BM + rl + 8] = __float2bfloat16_rn(v2);
        stg[(tl + 1) * BM + rl] = __float2bfloat16_rn(v1);
        stg[(tl + 1) * BM + rl + 8] = __float2bfloat16_rn(v3);
      }
    }
    named_bar(1, NTH);
    {
      constexpr int CH = BM / 8;                 // 16-byte chunks per token row
      const int nt = 8 * (jhi - jlo);
      for (int idx = tid; idx < nt * CH; idx += NTH) {
        const int tt = idx / CH, c = idx - tt * CH;
        const int tok = 8 * jlo + tt;
        if (tok < M)
          *reinterpret_cast<uint4*>(out + (size_t)tok * N + rb * BM + c * 8) =
              *reinterpret_cast<const uint4*>(stg + tt * BM + c * 8);
      }
    }
    // nobody leaves while another rank may still read its fragments
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row0] = __float2bfloat16_rn(acc[4 * j]);
        out[(size_t)t0 * N + row0 + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row0] = __float2bfloat16_rn(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row0 + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
  }
#endif
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gqx(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 + (CL > 1 ? (size_t)(BN / 8) * NWG * 128 * 16 : 0);
  auto kern = gqx_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (CL > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(N / BM * CL);
  lc.blockDim = dim3(NWG * 128 + 32);
  lc.dynamicSmemBytes = smem;
  lc.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeClusterDimension;
  at_[0].val.clusterDim.x = CL;
  at_[0].val.clusterDim.y = 1;
  at_[0].val.clusterDim.z = 1;
  lc.attrs = at_;
  lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
}

void gqx(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage) {
  @@GQX_DISPATCH@@
  TORCH_CHECK(false, "gqx: no config");
}
"""

GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
    ("qkv_gate", 128): (1, 3, 8),   # m 33..128: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
}
USE_GQX = True
TORCHT_LAYERS = ("unpooler", "o")   # layers that also get a K-major BF16 copy for the "torchT" (mm(x, W^T)) route


def _gqx_src(sm90):
    if not sm90:
        return ('void gqx(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t) { TORCH_CHECK(false, "gqx needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQX_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQX_SRC.replace("@@GQX_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gqx_bn(m):
    return 128 if m <= 128 else 256


def _run_gqx(x, layer, m, n, k, state):
    nwg, cl, ns = GQX_CONFIGS[(layer, _gqx_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().gqx(state["raw"][layer], x, out, m, n, k, _gqx_bn(m), nwg, cl, ns)
    return out



RGEMV_SRC = r"""
// ============================================================================ router GEMV (m = 1): FP32 weights
// read directly (8.4 MB, no hi/lo split), ROWS rows per CTA, 256 threads; thread i owns k = 8i + 2048j (j < 4): every
// weight and x load is issued before the first FMA; warp butterfly then a fixed-order cross-warp sum (deterministic).
template <int ROWS, int MMAX>
__global__ void __launch_bounds__(256) rgemv_kernel(const float* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    float* __restrict__ out, int M, int N) {
  constexpr int K = 8192;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  float4 w[ROWS][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4* p = reinterpret_cast<const float4*>(W + (size_t)(r0 + r) * K + j * 2048 + tid * 8);
      w[r][2 * j] = __ldg(p); w[r][2 * j + 1] = __ldg(p + 1);
    }
  float acc[ROWS][MMAX];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int t = 0; t < MMAX; ++t) {
    if (t >= M) break;
    uint4 xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)t * K + j * 2048 + tid * 8));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
      float xf[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const float4 a = w[r][2 * j], b = w[r][2 * j + 1];
        float v = acc[r][t];
        v = fmaf(a.x, xf[0], v); v = fmaf(a.y, xf[1], v); v = fmaf(a.z, xf[2], v); v = fmaf(a.w, xf[3], v);
        v = fmaf(b.x, xf[4], v); v = fmaf(b.y, xf[5], v); v = fmaf(b.z, xf[6], v); v = fmaf(b.w, xf[7], v);
        acc[r][t] = v;
      }
    }
  }
  __shared__ float red[ROWS][MMAX][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][t][warp] = v;
    }
  __syncthreads();
  if (tid < ROWS * MMAX) {
    const int r = tid / MMAX, t = tid % MMAX;
    if (t < M) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q) v += red[r][t][q];
      out[(size_t)t * N + r0 + r] = v;
    }
  }
}


// ============================================================================ cgemv (m = 1): 12-bit compressed GEMV
// Lossless planes, row-major: L [N][K] low bytes, C [N][K/2] 4-bit codes (sign | exponent offset from base), byte j of
// each 4-byte group g holding codes 8g + j and 8g + j + 4. Exponents outside the 8-value window decode wrong and are
// corrected exactly by a dedicated escape warp from a per-CTA table (entry 0 = count; ((row << 16) | k, delta bits)).
// Every weight and x load is issued before the first FMA; fixed-order reductions (deterministic).
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void cgemv(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor etab, int64_t pmax, at::Tensor out, int64_t base4,
           int64_t rows, int64_t nth) {
  const int N = out.size(1), K = L.size(1);
  TORCH_CHECK(x.size(0) == 1 && N % rows == 0 && N <= L.size(0), "cgemv: m = 1");
  auto st = at::cuda::getCurrentCUDAStream();
  auto a0 = L.data_ptr<uint8_t>(); auto a1 = C.data_ptr<uint8_t>();
  auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto tp = reinterpret_cast<const int2*>(etab.data_ptr<int>());
  auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const uint32_t b4 = (uint32_t)base4;
  const int pm = (int)pmax;
  if (K == 8192 && rows == 2 && nth == 256) cgemv_kernel<2, 256, 8192><<<N / 2, 256 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 1024 && rows == 2 && nth == 64) cgemv_kernel<2, 64, 1024><<<N / 2, 64 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 4096 && rows == 2 && nth == 128) cgemv_kernel<2, 128, 4096><<<N / 2, 128 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else if (K == 8192 && rows == 2 && nth == 512) cgemv_kernel<2, 512, 8192><<<N / 2, 512 + 32, 0, st>>>(a0, a1, xp, tp, pm, op, b4);
  else TORCH_CHECK(false, "cgemv: no instantiation for K=", K, " rows=", rows, " nth=", nth);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// BF16 GEMV (the unpooler slice at m = 1: W[:1024] bf16 [1024, 4096]): 4 rows per CTA, 512 threads, thread i owns
// k = 8i (one 16 B load per row and one of x), fixed-order reductions.
template <int ROWS, int NTH, int K>
__global__ void __launch_bounds__(NTH) sgemv_kernel(const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    __nv_bfloat16* __restrict__ out) {
  constexpr int PER = K / NTH / 8;
  constexpr int NWARP = NTH / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  uint4 w[ROWS][PER], xv[PER];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) w[r][j] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)(r0 + r) * K + j * (NTH * 8) + tid * 8));
#pragma unroll
  for (int j = 0; j < PER; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + j * (NTH * 8) + tid * 8));
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat162* hx = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
    float xf[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(hx[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const __nv_bfloat162* hw = reinterpret_cast<const __nv_bfloat162*>(&w[r][j]);
      float v = acc[r];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 a = __bfloat1622float2(hw[e]); v = fmaf(a.x, xf[2 * e], v); v = fmaf(a.y, xf[2 * e + 1], v); }
      acc[r] = v;
    }
  }
  __shared__ float red[ROWS][NWARP];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}

void sgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t N) {
  TORCH_CHECK(W.size(1) == 4096 && x.size(0) == 1 && N % 4 == 0 && N <= W.size(0), "sgemv: m = 1, K = 4096");
  sgemv_kernel<4, 512, 4096><<<(int)(N / 4), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rgemv(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M) {
  TORCH_CHECK(W.size(1) == 8192 && M == 1, "rgemv: K = 8192, m = 1");
  rgemv_kernel<1, 1><<<(int)W.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), (int)M,
      (int)W.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _gq_src(sm90):
    if not sm90:
        return ('void gq(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t) { TORCH_CHECK(false, "gq needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(GQ_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def _gq_bn(m):
    return 64 if m <= 64 else 128 if m <= 128 else 256


def _run_gq(x, layer, m, n, k, state):
    nwg, cl, ns = GQ_CONFIGS[(layer, _gq_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _get_ext().gq(state["raw"][layer], x, out, m, n, k, _gq_bn(m), nwg, cl, ns)
    return out



WIDE_CONFIGS = {  # (layer, tok, nz) -> (nwg, splits, stages); grid = (N / (64 nwg), splits, nz) <= 132 (co-resident)
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    # router: BF16 hi|lo tiles (plain), 64-row blocks
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


def _wide_key(m):
    return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


def _wide_src(sm90):
    lines = []
    for (layer, tok, nz), (nwg, splits, stages) in sorted(WIDE_CONFIGS.items()):
        hl = layer == "router"
        if not sm90:
            stages = min(stages, 3 if not hl else 2)  # sm_80 check path: 163 KB of shared memory
        ot = "float" if hl else "__nv_bfloat16"
        lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                     f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                     f"M, N, K, splits); return; }}")
    return WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


MID_STAGES = 4
MID_DU = 2


def _mid_src(sm90):
    if not sm90:
        return ('void mid(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, '
                'torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, int64_t) { TORCH_CHECK(false, "mid needs sm_90"); }')
    lines = []
    for (layer, bn), (nwg, splits) in []:
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for comp in ((0,) if f32 else (0, 1)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                         f"{{ launch_mid<{nwg}, {bn}, {MID_STAGES}, {MID_DU}, {str(bool(comp)).lower()}, {ot}>(Wp, x, out, ws, "
                         f"counters, poff, pk, pr, pd, base4, M, N, Kp, Kx, splits, nrg); return; }}")
    bns = sorted({bn for (_, bn) in MID_CONFIGS})
    return (MID_SRC.replace("@@WGMMA_DEFS@@", wgmma_defs(bns))
            .replace("@@MID_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


RK_SRC = r"""
// ============================================================================ rk: router GEMM, m 33..256 (sm_90)
// out[M, 256] (FP32) = x[M, 8192] (BF16) @ (hi + lo)[256, 8192]^T on router_cat = [hi | lo] [256, 16384] (BF16).
// CTA (192 threads) = 1 consumer warpgroup (SS-wgmma m64n64k16 on the hi tile, then the lo tile, of each 64-k block)
// + 1 producer warp (TMA, RK_QS-stage ring of 24 KB: hi 8 KB | lo 8 KB | x 8 KB, 128B swizzle) + 1 helper warp.
// Tile = 64 W rows x 64 tokens; split-K over S CTAs per tile (grid 4 x ntt x S <= 128: one CTA per SM, all resident).
// Each CTA ADDS its partial tile into the FP32 output (smem-transposed, 16 B red.add.v4.f32): the reduction is
// ORDER-DEPENDENT (run-to-run differences ~1e-7 relative, inside the router tolerance; POD item 6). The output is
// cleared inside the kernel: every CTA's helper warp reads the grid counter of this grid size (generation =
// value / grid: earlier launches' counts are complete and this CTA has not counted yet), clears its slice of out,
// counts it (release), waits (acquire, 2 s limit -> trap) until every CTA has counted, and releases the consumers'
// adds through a CTA mbarrier. Monotonic counters (one per token-tile count = per grid size): no host reset.
#define RK_QS 6
__device__ __forceinline__ uint64_t rk_desc(uint32_t addr) {   // K-major tile, 128 B rows, 128B swizzle, SBO 1024
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void rk_mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
__device__ __forceinline__ void rk_mbar_expect(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void rk_mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void rk_mbar_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRKW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKW_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void rk_tma2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void rk_wgmma(float d[4][8], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
               : "l"(da), "l"(db));
}

__global__ void __launch_bounds__(192, 1) rk_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                     float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int BM = 64, BN = 64, NJ = 4, QS = RK_QS;
  constexpr int W_BYTES = 2 * BM * 128, ST = W_BYTES + BN * 128;
  extern __shared__ __align__(1024) unsigned char rk_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(rk_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  const int rt = blockIdx.x & 3, tt = (blockIdx.x >> 2) % ntt, sp = (blockIdx.x >> 2) / ntt;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, 1); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 5) {   // helper: clear this CTA's slice of out, count it, wait for every slice, release the adds
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;   // float4s in out
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        if (v - target >= 0) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) __trap();   // 2 s: a clear that never lands is a bug, never a slow node
        __nanosleep(128);
      }
      rk_mbar_arrive(zbar);
    }
    return;
  }
  if (warp == 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        rk_tma2d(st, &mapW, fb, kb * 64, rt * BM);
        rk_tma2d(st + BM * 128, &mapW, fb, 8192 + kb * 64, rt * BM);
        rk_tma2d(st + W_BYTES, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  // consumer warpgroup
  float d[NJ][8];
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) d[j][e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST, b = a + W_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      rk_wgmma(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
      rk_wgmma(d, rk_desc(a + BM * 128 + 32 * k16), rk_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));   // stage i-1's wgmmas are done
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+f"(d[j][e]) :: "memory");
  // partial tile -> shared memory [token][row] (the ring is idle: every stage was waited for), then 16 B adds
  constexpr int TP = 68;
  float* T = reinterpret_cast<float*>(rk_smem + (sbase - sraw));
  const int rl = (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int cl = j * 16 + h * 8 + (tid & 3) * 2;
      T[cl * TP + rl] = d[j][4 * h]; T[cl * TP + rl + 8] = d[j][4 * h + 2];
      T[(cl + 1) * TP + rl] = d[j][4 * h + 1]; T[(cl + 1) * TP + rl + 8] = d[j][4 * h + 3];
    }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  rk_mbar_wait(zbar, 0);   // every CTA's slice of out is clear (acquired by the helper warp)
  for (int idx = tid; idx < BN * 16; idx += 128) {
    const int tk = idx >> 4, qd = idx & 15;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(out + (size_t)c * 256 + rt * BM + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
#endif
}
static PFN_cuTensorMapEncodeTiled_v12000 rk_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qr; void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr);
    TORCH_CHECK(p != nullptr, "rk: cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static CUtensorMap rk_map(const void* ptr, uint64_t rows, uint64_t cols) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows}; cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, 64}; cuuint32_t estr[2] = {1, 1};
  CUresult r = rk_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "rk: cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
void rk(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t S, at::Tensor zc) {
  const int M = x.size(0);
  TORCH_CHECK(Wc.size(0) == 256 && Wc.size(1) == 16384 && x.size(1) == 8192 && M >= 1 && M <= 256, "rk: shape");
  const int ntt = (M + 63) / 64;
  const int grid = 4 * ntt * (int)S;
  static int nsm = 0;
  if (!nsm) { int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)); }
  TORCH_CHECK(grid <= nsm, "rk: grid ", grid, " exceeds the SM count ", nsm, " (every CTA must be resident: one per SM)");
  const CUtensorMap mw = rk_map(Wc.data_ptr(), 256, 16384), mx = rk_map(x.data_ptr(), M, 8192);
  const int smem = 1024 + RK_QS * 24576 + 8 * (2 * RK_QS + 1) + 64;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(rk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
  rk_kernel<<<grid, 192, smem, at::cuda::getCurrentCUDAStream()>>>(mw, mx, out.data_ptr<float>(), M, (int)S, ntt, zc.data_ptr<int>() + ntt);
  C10_CUDA_CHECK(cudaGetLastError());
}
"""


def _rk_src(sm90):
    if sm90:
        return RK_SRC
    return 'void rk(at::Tensor, at::Tensor, at::Tensor, int64_t, at::Tensor) { TORCH_CHECK(false, "rk needs sm_90"); }'


def _get_ext():
    """Compile everything as one CUDA translation unit with load_inline.  no_implicit_headers: the CUDA file includes
    only the ATen headers it needs (torch/types.h alone costs ~45 s of nvcc time)."""
    global _ext
    if _ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        common = CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace("@@BODY@@", WG_HELPERS if sm90 else "")
        defs = wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
        cuda_src = (common + defs + _tma_srcs(sm90) + MAP_SRC + _gemm_src(sm90) + _wide_src(sm90) + _sk2_src(sm90)
                    + _gq_src(sm90) + _gqx_src(sm90) + RGEMV_SRC + _rk_src(sm90))
        _ext = load_inline(
            name=f"okc_bf16proj_v141_{arch}" + (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return _ext


# ---------------------------------------------------------------------------------------------- CUTLASS route (m 129..256)
# CUTLASS 3.x sm90 TMA warp-specialized PINGPONG GEMM, persistent tile scheduler, swap-AB (D^T = W x^T: A = W [N, K]
# row-major, B = x [M, K] as a K x M column-major matrix, D column-major [N, M] = out [M, N] row-major), BF16 in/out,
# FP32 accumulate, tile 128 (W rows) x 64 (tokens) x 64, cluster 1x1. Headers: CUTLASS bundled with flashinfer (in the
# SGLang image). Sweep (pod/cutsweep.py, 54 configs x 8 shapes): it beats cuBLAS only for pooler m=256 (31.3 vs 34.5 us
# raw); a missing header or a failed build just leaves the route out.
CUTLASS_LAYERS = set()   # rq_lt is 2.66 us faster (27.9 vs 30.56 us); removing CUTLASS saves ~80s prepare time
CUTLASS_BUILD_TIMEOUT = 170   # seconds prepare waits for the CUTLASS build (main build ~25 s; prepare's limit is 300 s)
CUTLASS_SRC = r"""
/***************************************************************************************************
 * The GEMM below instantiates NVIDIA CUTLASS 3.x/4.x templates (header-only, compiled from the CUTLASS headers that
 * ship in this image's FlashInfer package; nothing prebuilt is linked). Its type composition follows CUTLASS's Hopper
 * GEMM examples (examples/48_hopper_warp_specialized_gemm, 49_hopper_gemm_with_collective_builder), so it carries
 * CUTLASS's license notice:
 *
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
using namespace cute;
using EA = cutlass::bfloat16_t; using LA = cutlass::layout::RowMajor;
using EB = cutlass::bfloat16_t; using LB = cutlass::layout::ColumnMajor;
using EC = cutlass::bfloat16_t; using LC = cutlass::layout::ColumnMajor;
using TileShape = Shape<_128, _64, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, EC, LC, 8, EC, LC, 8,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, LA, 8, EB, LB, 8, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
int64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  return (int64_t)Gemm::get_workspace_size(args);
}
void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws) {
  const int M = W.size(0), K = W.size(1), N = x.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA*>(W.data_ptr()), sA, reinterpret_cast<EB*>(x.data_ptr()), sB},
      {{1.f, 0.f}, reinterpret_cast<EC*>(out.data_ptr()), sC, reinterpret_cast<EC*>(out.data_ptr()), sC}};
  Gemm gemm;
  TORCH_CHECK((size_t)ws.numel() >= Gemm::get_workspace_size(args), "cut_gemm: workspace");
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cut_gemm: cannot implement");
  TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), st) == cutlass::Status::kSuccess, "cut_gemm: init");
  TORCH_CHECK(gemm.run(st) == cutlass::Status::kSuccess, "cut_gemm: run");
}
"""
_cut_ext = None
_cut_err = None


def _build_cutlass():
    """Build the CUTLASS extension (sm_90 only); on any failure record it and leave the route out."""
    global _cut_ext, _cut_err
    try:
        import importlib.util as _ilu
        import os as _os
        major, minor = torch.cuda.get_device_capability()
        if major < 9:
            raise RuntimeError("needs sm_90")
        fi = _ilu.find_spec("flashinfer")
        if fi is None or not fi.submodule_search_locations:
            raise RuntimeError("flashinfer (CUTLASS headers) not installed")
        root = _os.path.join(list(fi.submodule_search_locations)[0], "data", "cutlass")
        inc = [_os.path.join(root, "include"), _os.path.join(root, "tools", "util", "include")]
        if not all(_os.path.isdir(d) for d in inc):
            raise RuntimeError("CUTLASS headers missing under " + root)
        _cut_ext = load_inline(
            name=f"okc_bf16proj_v141_cutlass_{major}{minor}a",
            cpp_sources="#include <torch/extension.h>\nint64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out);\n"
                        "void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws);",
            cuda_sources=CUTLASS_SRC, functions=["cut_ws", "cut_gemm"], extra_include_paths=inc,
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a", "-DNDEBUG"],
            verbose=False)
    except Exception as e:   # noqa: BLE001 - any failure only drops this optional route
        _cut_ext, _cut_err = None, f"{type(e).__name__}: {str(e)[:200]}"


def pack_fragments(w: torch.Tensor) -> torch.Tensor:
    """W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order."""
    n, k = w.shape
    t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  # [rg, hr, g, s, hk, t, e]: row = hr*8+g, col = hk*8+2t+e
    t = t.permute(0, 3, 2, 5, 4, 1, 6)  # [rg, s, g, t, hk, hr, e]
    return t.contiguous().view(n // 16, k // 16, 32, 8)


def pack_compressed(w: torch.Tensor):
    """Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it."""
    n, k = w.shape
    if n % 16 or k % 64:
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8                  # sign | exponent bits 7..1
    e71 = hi & 0x7F
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((hi >> 7) << 3) | o
    dec_hi = ((hi >> 7) << 7) | (base + o)
    dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
    esc = (e71 < base) | (e71 > base + 7)
    n_esc = int(esc.sum())
    if n_esc > n * k // 200:
        return None
    rows, cols = esc.nonzero(as_tuple=True)
    wtrue = w[rows, cols].float()
    wdec = dec[rows, cols].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    # fragment order for low bytes and codes: [rg, s, lane, 8]
    lo_f = pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
    cd_f = pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
    nrg, S = n // 16, k // 16
    lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   # [rg, u, ks, lane, 8]
    cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
    # parts 0,1: lane's low bytes for k-steps (0,1) and (2,3) -> [rg, u, 2, lane, 16]
    lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
    # part 2: word i (k-step i): byte b = code[8i + b] | code[8i + 4 + b] << 4
    cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           # [rg, u, ks, lane, 4]
    cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
    packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [u, rg, 3, lane, 16]
    order = torch.argsort(rows * k + cols)
    rows, cols, delta = rows[order], cols[order], delta[order]
    rg_idx = rows // 16
    poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
    poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
    return {
        "packed": packed,
        "base4": base * 0x01010101,
        "poff": poff,
        "pk": cols.to(torch.int32).contiguous(),
        "pr": (rows % 16).to(torch.int32).contiguous(),
        "pd": delta.to(torch.float32).contiguous(),
        "n_esc": n_esc,
    }


def build_patch_table(c, n_cfg, k, splits, rg):
    """Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits)."""
    dev = c["pk"].device
    rows_cta = rg * 16
    nrb = n_cfg // rows_cta
    nunits = k // 64
    poff = c["poff"].long()
    nrg_all = poff.numel() - 1
    rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
    rows = rgi * 16 + c["pr"].long()
    cols = c["pk"].long()
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
    unit = cols // 64
    spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          # split owning this 64-k unit
    k_start = (spl * nunits // splits) * 64
    cta = spl * nrb + rows // rows_cta
    ncta = nrb * splits
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta, stable=True)
    cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
    tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
    return tab.contiguous(), pmax


def build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
    """Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16."""
    dev = src.device
    nw = rg * wsn
    nrb = (n_cfg // 16) // rg
    ar = lambda n: torch.arange(n, device=dev)
    if wsn == 1:
        s_all = (k // 16) // ks
        rb = ar(nrb).view(-1, 1, 1, 1)
        st = ar(s_all).view(1, -1, 1, 1)
        w = ar(nw).view(1, 1, -1, 1)
        rg_idx = rb * rg + w
        if comp:
            u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
            return src[rg_idx, u_idx].contiguous()
        s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
        return src[rg_idx, s_idx].contiguous()
    kc = k // splits
    ns = kc // (16 * wsn)
    nst = ns // ks
    sp = ar(splits).view(-1, 1, 1, 1, 1)
    rb = ar(nrb).view(1, -1, 1, 1, 1)
    st = ar(nst).view(1, 1, -1, 1, 1)
    w = ar(nw).view(1, 1, 1, -1, 1)
    rg_idx = rb * rg + w % rg
    if comp:
        ku = ar(ks // 4).view(1, 1, 1, 1, -1)
        u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
        out = src[rg_idx, u_idx]
    else:
        j = ar(ks).view(1, 1, 1, 1, -1)
        s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
        out = src[rg_idx, s_idx]
    return out.contiguous()


# ---------------------------------------------------------------- sk2: m <= 32, swapped operands, no cross-CTA reduction
SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL); a name ending in _rNNN caps the ring at NNN KB
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
    # okc_tune (14173ac4) retimed winners, Oct 25 12:30 (search-harness ratios vs production in brackets); pair.py both
    # orders: pooler16 -0.43, unpooler16 -0.60, o16 -0.15, qkv8 -0.12, qkv16 -0.08 us (pooler32 r160 / o32 r88: no gain)
    ('pooler16t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),      # tokr16 [0.973 vs 1.0]
    ('unpooler16t_r72', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # tokr16 [0.961 vs 0.969]
    ('qkv8t_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),         # tokr8 [0.969 vs 0.979]
    ('qkv16t_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),        # tokr16 [0.985 vs 0.994]
    ('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1),                  # tokr16 [0.937 vs 0.957]
]
SK2_PICK = {  # (layer, tokr) -> config name
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o16t_r68',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler16t_r192',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8t_r144',
    ('qkv_gate', 16): 'qkv16t_r200',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16t_r72',
    ('unpooler', 32): 'unpooler32_r144',
}
SK2_ROT = False
SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
USE_SK2 = True
WIDE_BF16 = False  # the wide (RS-wgmma) kernel never beats cuBLAS for BF16 m 33..256


def _sk2_name(layer, m):
    return SK2_PICK[(layer, _sk2_tokr(m))]


def _sk2_tokr(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32


def _sk2_geom(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl = layer == "router"
    tu = 2048 if hl else 768
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * tu
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(SK2_K[layer] // (ups * 64)) // CL)   # (the largest rank's stages)
    mt = 2 if tokr == 32 else 1
    toks = 16 * mt
    cp = R + 1
    nesc = 4 if not hl else 1   # (3 when the ring has 3 stages; sized for 4 here)
    rp = R + ((40 - (R % 32)) % 32)
    # everything but the ring: alignment pad, barriers, corrections, peer sums, escape table (<= 64 entries), zero row
    extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
             + 64 * 8 + 128)
    cap = 200
    tail = name.rsplit("_r", 1)
    if len(tail) == 2 and tail[1].isdigit():
        cap = int(tail[1])
    ring = min(cap * 1024, 232448 - extra)
    nstage = max(2, min(16, nst, ring // sst))
    return hl, nstage


def _sk2_src(sm90):
    if not sm90:
        return ('void sk2(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, '
                'int64_t, int64_t, int64_t, at::Tensor) { TORCH_CHECK(false, "sk2 needs sm_90"); }')
    lines = []
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            if SK2_PICK.get((layer, tokr)) != name:
                continue
            hl, nstage = _sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                         f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
    return SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))


def sk2_tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def sk2_compress(w):
    """Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable."""
    n, k = w.shape
    tu = sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = (sign << 3) | o
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)                  # [T, U, 32, 8]
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()                                   # [E, 4]: tile, unit, lane, v
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    wbits = (tu[esc].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).contiguous()   # the original BF16 weights
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "wbits": wbits}


CG_CONFIGS = {"qkv_gate": (2, 256), "o": (2, 64), "unpooler": (2, 128), "pooler": (2, 512)}   # layer -> (rows per CTA, threads); m = 1
# (v137: pooler m=1 cgemv 2x512 measured -0.34 us vs sk2; 2x256 ties sk2, 4/8 rows or 1 row lose)


def cg_compress(w, rows_per_cta):
    """Row-major 12-bit planes for cgemv (lossless; escapes corrected exactly from a per-CTA table)."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)          # row-major order
    dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
    wtrue = w[rows, cols].float()
    delta = (wtrue - dec).contiguous()
    if not bool(((dec + delta) == wtrue).all()):
        return None
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)   # the original BF16 weights; the kernel forms the correction
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101}


def sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    """Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows."""
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


SK2_BALANCE = True   # v135: deal 4-row groups to sk2 row blocks so every block carries ~the same number of escapes
SK2_BALANCE_LAYERS = {"qkv_gate"}   # v136: only where it measured a gain (qkv m=8/16 -0.5 us; o/unpooler m=32 lost)


def sk2_balance_perm(w, n_rows, R):
    """Row order for R-row sk2 blocks over the first n_rows rows of w: rows sorted by escape count and snake-dealt, so
    every block holds ~the same number of escapes (the escape warps' work sets the CTA tail). perm[p] = original row."""
    wl = w[:n_rows]
    bits = wl.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    cnt = ((e71 < base) | (e71 > base + 7)).sum(1)
    # 4-row groups are dealt (keeps 4-column output groups contiguous: vector stores in the epilogue)
    G4 = 4
    tcnt = cnt.view(n_rows // G4, G4).sum(1)
    order = torch.argsort(tcnt, descending=True, stable=True)
    TB = R // G4                                   # groups per block
    nb = n_rows // R
    i = torch.arange(n_rows // G4, device=w.device)
    rnd, pos = i // nb, i % nb
    blk = torch.where(rnd % 2 == 0, pos, nb - 1 - pos)
    tperm = torch.empty(n_rows // G4, dtype=torch.long, device=w.device)
    tperm[blk * TB + rnd] = order
    return (tperm.view(-1, 1) * G4 + torch.arange(G4, device=w.device).view(1, G4)).reshape(-1)


def sk2_table_add_perm(tab, pmax, perm, R):
    """Append each CTA's row map (R int32 output columns) after its escapes; entry 0 .y = the map's int2 offset."""
    ncta = tab.shape[0]
    pm2 = pmax + R // 2
    pm2 += pm2 % 2
    out = torch.zeros(ncta, pm2, 2, dtype=torch.int32, device=tab.device)
    out[:, :pmax] = tab
    out[:, pmax:pmax + R // 2] = perm.to(torch.int32).view(ncta, R // 2, 2)
    out[:, 0, 1] = pmax
    return out.contiguous(), pm2


def sk2_table(comp, n_cfg, R):
    """Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, the escaped weight's
    original BF16 bits); the kernel forms the exact correction (w - decode(w)) itself."""
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["wbits"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())   # per CTA in k order (the kernel walks K once)
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.to(torch.int32)
    return tab.contiguous(), pmax


def sk2_prepare(weights, state):
    state["sk2"] = {}
    if not (USE_SK2 and state["sm90"]):
        return
    dev = weights["o"].device
    shared = {}
    for ci, cfg in enumerate(SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if layer == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            T, U = w.shape[0] // 8, w.shape[1] // 64
            tub = torch.cat([sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                             sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
            stream = sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            perm = None
            if SK2_BALANCE and CL == 1 and layer in SK2_BALANCE_LAYERS:
                pkey = (layer, n_rows, R)
                bal = state.setdefault("sk2_bal", {})
                if pkey not in bal:
                    pr = sk2_balance_perm(w, n_rows, R)
                    bal[pkey] = (pr, sk2_compress(w[:n_rows][pr].contiguous()))
                perm, comp = bal[pkey]
                if comp is None:
                    perm = None
            if perm is None:
                comp = state.setdefault("sk2_comp", {}).get(layer)
                if comp is None:
                    comp = sk2_compress(w.contiguous())
                    state["sk2_comp"][layer] = comp
            if comp is None:
                continue
            lkey = (layer, n_rows, R, NT, NWR, WSK, UPW, perm is not None)   # configs differing only in NACC / ring depth share a stream
            if lkey not in shared:
                tab, pmax = sk2_table(comp, n_rows, R)
                if perm is not None:
                    tab, pmax = sk2_table_add_perm(tab, pmax, perm, R)
                shared[lkey] = (sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tab, pmax)
            stream, tab, pmax = shared[lkey]
            b4 = comp["base4"]
        state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
    state.pop("sk2_comp", None)
    state.pop("sk2_bal", None)


def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    name = _sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _get_ext().sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


def _mt_key(m):
    return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def prepare(weights):
    import threading
    ext = _get_ext()
    # the CUTLASS extension builds in the background (nvcc is a subprocess) while the weights are compressed; started
    # after the main build so the two load_inline calls never overlap
    cut_thread = None
    if CUTLASS_LAYERS and torch.cuda.get_device_capability()[0] >= 9 and _cut_ext is None and _cut_err is None:
        cut_thread = threading.Thread(target=_build_cutlass, daemon=True)
        cut_thread.start()
    dev = weights["qkv_gate"].device
    sm90_ = torch.cuda.get_device_capability()[0] >= 9
    # on sm_90 the sk2 kernels replace the older skinny kernels and the BF16 wide kernel (never faster than cuBLAS)
    old_paths = not (sm90_ and USE_SK2)
    state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
    # split-K partials: kernels run one at a time, so every layer shares one workspace
    ws_floats = 0
    for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
        mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
        ws_floats = max(ws_floats, splits * mmax * w_rows[lname])
    for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
        ws_floats = max(ws_floats, splits * mm * w_rows[lname])
    state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    gemm_floats = 0
    for (lname, bn), (nwg, splits, stages) in GEMM_CONFIGS.items():
        gemm_floats = max(gemm_floats, (w_rows[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
    for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
        gemm_floats = max(gemm_floats, nz * (w_rows[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
    state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
    if GEMM_DIAG:
        state["ws_shared_diag"] = torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=dev)
    for name, w in weights.items():
        if name == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            state["router_hi_t"] = hi.t().contiguous()
            state["router_f32"] = w.float().contiguous()   # the router GEMV (m = 1) reads the FP32 weights as given
            state["router_lo_t"] = lo.t().contiguous()
            state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   # gemm90 fallback (m 33..256), rk
            state["rk_zc"] = torch.zeros(8, dtype=torch.int32, device=w.device)   # rk: grid counters, one per token-tile count
            # hi and lo tiles of the same k-step back to back: [rg, s, 2, 32, 8]
            src = torch.stack([pack_fragments(hi), pack_fragments(lo)], dim=2).contiguous()
            comp, kk = False, w.shape[1]
            c = None
            # wide kernel stream: [rb][unit][warp][hi 4 k-steps x 512 B | lo 4 k-steps x 512 B], 64-row blocks
            nrg_r = w.shape[0] // 16
            u_all = w.shape[1] // 64
            t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   # [rg, u, hl, 4, 32, 8]
            nwg_r = WIDE_CONFIGS[("router", 128, 1)][0]
            t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
            if not sm90_:   # sm_90 runs the router's wide path on router_cat (TMA tiles); the fragment stream is sm_80's
                state["streams"][("router", 0, 4 * nwg_r, 1, 4, w.shape[0])] = t
            del t
            for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
                if lname == "router":
                    state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
        else:
            w = w.contiguous()
            state["raw"][name] = w
            if name in CG_CONFIGS:
                wc = w[:1024] if name == "unpooler" else w   # the unpooler at m = 1 is the 1024-row slice
                cg = cg_compress(wc.contiguous(), CG_CONFIGS[name][0])
                if cg is not None:
                    state.setdefault("cg", {})[name] = cg
            if name in TORCHT_LAYERS and sm90_:
                state.setdefault("rawT", {})[name] = w.t().contiguous()
            c = pack_compressed(w) if name in COMPRESS else None
            if c is not None:
                src, comp = c["packed"].transpose(0, 1).contiguous(), True  # [rg, u, 3, 32, 16]
                state["base4"][name] = c["base4"]
                del c["packed"]
            else:
                src, comp = pack_fragments(w), False
            kk = w.shape[1]
        n = w.shape[0]
        state["comp"][name] = comp
        for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
            if lname != name or not old_paths:
                continue
            n_cfg = 1024 if (name == "unpooler" and mtk <= 1) else n
            key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
            if key not in state["streams"]:
                state["streams"][key] = build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
            ckey = (name, splits, rgc, wsn, ks, n_cfg)
            if ckey not in state["cnt"]:
                state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  # per-config counters
            if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n_cfg)] = build_patch_table(c, n_cfg, kk, splits, rgc)
        if comp and old_paths:
            for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
                if lname != name:
                    continue
                n_cfg = 1024 if (name == "unpooler" and mm <= 8) else n
                key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
                if key not in state["streams"]:
                    state["streams"][key] = build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
                ckey = (name, splits, rgc, wsn, ks, n_cfg)
                if ckey not in state["cnt"]:
                    state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
                if (name, splits, rgc, n_cfg) not in state["tabs"]:
                    state["tabs"][(name, splits, rgc, n_cfg)] = build_patch_table(c, n_cfg, kk, splits, rgc)
        for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
            if lname != name or not comp or not (old_paths or WIDE_BF16):   # unused on sm_90 (cuBLAS / gq / gqx)
                continue
            rgc = 4 * nwg
            key = (name, 0, rgc, 1, 4, n)
            if key not in state["streams"]:
                state["streams"][key] = build_stream(src, comp, n, kk, splits, rgc, 1, 4)
            if (name, splits, rgc, n) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n)] = build_patch_table(c, n, kk, splits, rgc)
            state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        del src
        state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    state["sm90"] = sm90_
    state["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count   # rk: one resident CTA per SM
    state["old_paths"] = old_paths
    sk2_prepare(weights, state)
    if GEMM_DIAG and state["sm90"]:
        state["wtile"], state["diag_cnt"] = {}, {}
        for layer in DIAG_LAYERS:
            state["wtile"][layer] = swizzle_tiles(state["raw"][layer], 128)
            for a_s in (0, 1):
                for sp_ in sorted({GEMM_CONFIGS[(layer, 128)][1], GEMM_CONFIGS[(layer, 256)][1], 1}):
                    state["diag_cnt"][(layer, a_s, sp_)] = torch.zeros(w_rows[layer] // 16, dtype=torch.int32, device=dev)
    if cut_thread is not None:
        cut_thread.join(timeout=CUTLASS_BUILD_TIMEOUT)   # ~70-80 s on an H200 pod; never let it threaten prepare's limit
        if cut_thread.is_alive():
            print(f"[okc] CUTLASS route off: build still running after {CUTLASS_BUILD_TIMEOUT} s", file=sys.stderr)
    if _cut_ext is not None and (cut_thread is None or not cut_thread.is_alive()):
        state["cut_ws"] = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)   # persistent scheduler: a few KB at most
        state["cut_layers"] = set(CUTLASS_LAYERS) & set(state["raw"])
    elif _cut_err is not None:
        print(f"[okc] CUTLASS route off: {_cut_err}", file=sys.stderr)
    _selftest(weights, state)
    rq_prepare(weights, state)
    _autotune(weights, state)
    torch.cuda.empty_cache()
    return state


# Which implementation runs a (layer, m): "skinny" (TMA-streamed tensor-core GEMV, m <= 32),
# "gemm" (TMA + wgmma split-K GEMM, sm_90, m <= 256) or "torch" (cuBLAS, as SGLang).
ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
WIDE_SM80_TEST = False  # local check of the wide kernel's sm_80 path
ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def _route(layer, m, sm90):
    for mmax, r in (ROUTE_ROUTER if layer == "router" else ROUTE_BF16):
        if m <= mmax:
            if r == "gemm" and not sm90:
                return "torch"
            if r == "wide" and not (sm90 or WIDE_SM80_TEST):
                return "torch"
            return r
    return "torch"


USE_RK = True   # v141: router m 33..256 candidate 'rk' (order-dependent FP32 adds; stress compares it by tolerance)


def _route_key(layer, m, route):
    if route == "sk2":
        return (layer, route, _sk2_name(layer, m), _sk2_tokr(m))
    if route == "skinny":
        return (layer, route, _mt_key(m))
    if route == "wide":
        return (layer, route) + _wide_key(m)
    if route == "rk":
        return (layer, route, (m + 63) // 64)
    return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def _selftest(weights, state):
    """Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on."""
    state["disabled"] = set()
    g = torch.Generator(device=weights["o"].device)
    g.manual_seed(1234)
    for layer, w in weights.items():
        n, k = w.shape
        for m in SELFTEST_M:
            nn = 1024 if (layer == "unpooler" and m <= 8) else n
            x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
            ref = x.double() @ w[:nn].double().T
            if layer != "router":
                ref = ref.to(torch.bfloat16).double()
            tol = 2e-5 if layer == "router" else 5e-3
            for ch in _candidates(layer, m, state):
                if ch[0] == "torch":
                    continue
                key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                       else _route_key(layer, m, ch[0]))
                try:
                    out = _run(ch, x, layer, m, nn, k, state).double()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                except RuntimeError as e:  # launch failure -> fall back
                    err = float("inf")
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: {e}", file=sys.stderr)
                if not err < tol:
                    state["disabled"].add(key)
                    print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)


def _run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
    splits, rg, wsn, ks = cfg
    mtk = _mt_key(m)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
    cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
    fn = _get_ext().skinny_tma_a
    if state["comp"][layer]:
        tab, pmax = state["tabs"][(layer, splits, rg, n)]
        fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 1, w_rows[layer] // 16, ks, dg)
    else:
        fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 0, w_rows[layer] // 16, ks, dg)
    return out


def _run_wide(x, layer, m, n, k, state):
    tok, nz = _wide_key(m)
    nwg, splits, _ = WIDE_CONFIGS[(layer, tok, nz)]
    rgc = 4 * nwg
    if layer == "router":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        tab, pmax, b4 = state["dummy"], 0, 0
    else:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
    wst = state["router_cat"] if layer == "router" else state["streams"][(layer, 0, rgc, 1, 4, n)]
    _get_ext().wide(wst, x, out, state["mid_ws_shared"],
                    state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
    return out


def _run_gemm(x, layer, m, n, k, state):
    bn = 64 if m <= 64 else 128 if m <= 128 else 256
    nwg, splits, _ = GEMM_CONFIGS[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    w = state["router_cat"] if layer == "router" else state["raw"][layer]
    _get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                      splits, nwg, bn, 0)
    return out


def _run_torch(x, layer, m, n, k, state):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


def _run(choice, x, layer, m, n, k, state):
    kind = choice[0]
    if kind in ("rq_lt", "rq_tma"):
        return rq_run(choice, x, layer, m, n, k, state)
    if kind == "skinny":
        return _run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
    if kind == "wide":
        return _run_wide(x, layer, m, n, k, state)
    if kind == "sk2":
        return _run_sk2(x, layer, m, n, k, state)
    if kind == "gemm":
        return _run_gemm(x, layer, m, n, k, state)
    if kind == "gq":
        return _run_gq(x, layer, m, n, k, state)
    if kind == "gqx":
        return _run_gqx(x, layer, m, n, k, state)
    if kind == "cgemv":   # m = 1: 12-bit compressed GEMV (qkv, o, unpooler slice, pooler)
        cg = state["cg"][layer]
        rows, nth = CG_CONFIGS[layer]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth)
        return out
    if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _get_ext().sgemv(state["raw"][layer], x, out, n)
        return out
    if kind == "rgemv":   # router, m = 1: FP32 weights straight from memory (one row per CTA)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _get_ext().rgemv(state["router_f32"], x, out, m)
        return out
    if kind == "cutlass":   # m 129..256: CUTLASS pingpong 128x64 (see CUTLASS_SRC)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _cut_ext.cut_gemm(state["raw"][layer], x, out, state["cut_ws"])
        return out
    if kind == "rk":   # router m 33..256: dedicated split-K GEMM with order-dependent FP32 adds (see RK_SRC)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _get_ext().rk(state["router_cat"], x, out, _rk_splits(m, state["nsm"]), state["rk_zc"])
        return out
    if kind == "torchT":   # cuBLAS on a K-major copy of W: a different (faster) kernel choice for unpooler m=128
        return torch.mm(x, state["rawT"][layer])
    return _run_torch(x, layer, m, n, k, state)


RK_SPLITS = {1: 16, 2: 12, 3: 10, 4: 8}   # token tiles of 64 -> K splits (grid 4 x ntt x S <= 128; pod/rk.py sweep)


def _rk_splits(m, nsm=132):
    """K splits for m tokens: the measured table, capped so the grid (4 x ntt x S, one CTA per SM, all resident: the
    in-kernel clear waits for every CTA) fits the GPU's SMs."""
    ntt = (m + 63) // 64
    return max(1, min(RK_SPLITS[ntt], nsm // (4 * ntt)))


def _static_choice(layer, m, state):
    route = _route(layer, m, state["sm90"])
    if not state.get("old_paths", True):
        if route == "skinny":
            route = "sk2" if (_sk2_name(layer, m) in state.get("sk2", {})
                              and _route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
        elif route == "wide" and layer != "router" and not WIDE_BF16:
            route = "torch"
    if route != "torch" and _route_key(layer, m, route) in state.get("disabled", ()):
        # fall back: router -> the wgmma GEMM (validated earlier) when possible, else cuBLAS
        route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                           and _route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
    if route == "skinny":
        mtk = _mt_key(m)
        return ("skinny", TMA_CONFIGS[(layer, mtk)], (layer, mtk))
    return (route,)


def _candidates(layer, m, state):
    """Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS."""
    c = [_static_choice(layer, m, state)]
    if c[0][0] == "skinny" and (layer, m) in TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
        c.append(("skinny", TMA_ALT[(layer, m)], ("alt", layer, m)))
    if (c[0][0] == "wide" and layer == "router" and state["sm90"]
            and _route_key(layer, m, "gemm") not in state["disabled"]):
        c.append(("gemm",))
    if (m <= 32 and c[0][0] != "sk2" and _sk2_name(layer, m) in state.get("sk2", {})
            and _route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
    if (USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gq_bn(m)) in GQ_CONFIGS
            and _route_key(layer, m, "gq") not in state["disabled"]):
        c.insert(0, ("gq",))
    if (USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQX_CONFIGS
            and _route_key(layer, m, "gqx") not in state["disabled"]):
        c.insert(0, ("gqx",))
    if layer == "router" and m == 1 and "router_f32" in state:
        c.insert(0, ("rgemv",))
    if (USE_RK and layer == "router" and 32 < m <= 256 and state["sm90"] and "rk_zc" in state and state.get("nsm", 0) >= 16
            and _route_key(layer, m, "rk") not in state["disabled"]):
        c.insert(0, ("rk",))
    if layer == "unpooler" and m == 1:
        c.insert(0, ("sgemv",))
    if m == 1 and layer in state.get("cg", {}) and _route_key(layer, m, "cgemv") not in state["disabled"]:
        c.insert(0, ("cgemv",))
    if layer in TORCHT_LAYERS and 32 < m <= 256 and layer in state.get("rawT", {}):
        c.append(("torchT",))
    if (128 < m <= 256 and layer in state.get("cut_layers", ()) and _cut_ext is not None
            and _route_key(layer, m, "cutlass") not in state["disabled"]):
        c.append(("cutlass",))
    c.extend(rq_candidates(layer, m, state))
    if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
        c.append(("torch",))
    return c


AUTOTUNE = True
TUNE_M = (1, 16, 32, 8, 128, 256)   # the scored token counts


def _autotune(weights, state):
    """Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m)."""
    state["choice"] = {}
    if not AUTOTUNE:
        return
    dev = weights["o"].device
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)   # 256 MB, read-only sweep
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(4321)
    reps = 24
    for m in TUNE_M:
        for layer, w in weights.items():
            n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
            k = w.shape[1]
            cands = _candidates(layer, m, state)
            if len(cands) < 2:
                continue
            x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
            graphs = []
            for ch in cands:
                _run(ch, x, layer, m, n, k, state)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    _run(ch, x, layer, m, n, k, state)
                graphs.append(gr)
            evs = []
            for r in range(reps):
                order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                for i in order:
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[i].replay()
                    e1.record()
                    evs.append((i, e0, e1))
            torch.cuda.synchronize()
            per = [[] for _ in cands]
            for i, e0, e1 in evs:
                per[i].append(e0.elapsed_time(e1))
            med = [sorted(v)[len(v) // 2] for v in per]
            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
            # CUTLASS is tuned for the scored m only (its partial token tiles are slow: m=129 +14 us, m=200 +4 us vs
            # cuBLAS): other m of the bucket take the best other candidate
            alt = [i for i in range(len(cands)) if cands[i][0] != "cutlass"]
            if alt:
                state.setdefault("choice_alt", {})[(layer, m)] = cands[min(alt, key=lambda i: med[i])]
            state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
            del graphs
    del flush
    torch.cuda.synchronize()
    # one line per shape in the run's log: candidate times (us, raw graph replay incl. ~4 us replay cost) and the pick
    for layer, m, tl in state.get("tune_log", []):
        ch = state["choice"][(layer, m)]
        desc = " ".join(f"{name}{'/' + 'x'.join(map(str, c[1])) if c[0] == 'skinny' else ''}={t}"
                        for (name, t), c in zip(tl, _candidates(layer, m, state)))
        pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")
        print(f"[okc] tune {layer} m={m}: {desc} -> {pick}", file=sys.stderr)
    print(f"[okc] disabled: {sorted(state['disabled'])}", file=sys.stderr)


def _nearby_choice(layer, m, state):
    """An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket."""
    tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
    if not tm:
        return None
    if tm == 256 and m != 256 and layer != "router":
        # an untuned m in 129..255 runs SGLang's own call (F.linear): the m=256 picks (CUTLASS, gqx, torchT) were
        # only timed at m=256 and some lose to cuBLAS at partial tiles (pooler CUTLASS m=129 +14 us, unpooler
        # torchT m=240 +11%)
        return ("torch",)
    ch = state.get("choice", {}).get((layer, tm))
    if ch is None:
        return None
    if ch[0] == "cutlass" and m != tm:
        ch = state.get("choice_alt", {}).get((layer, tm))
        if ch is None:
            return None
    if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
        return None
    return ch


def kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    choice = state.get("choice", {}).get((layer, m)) or _nearby_choice(layer, m, state) or _static_choice(layer, m, state)
    out = _run(choice, x, layer, m, n, k, state)
    if GEMM_DIAG:
        _diag(x, layer, m, n, k, state, choice)
    return out


def _diag(x, layer, m, n, k, state, choice):
    """Extra kernels for the judge's per-kernel profile (outputs discarded)."""
    if choice[0] == "skinny" and m == 1 and layer in ("o", "unpooler") and state["comp"][layer]:
        for dg in (1, 2, 3):   # same kernel without epilogue / x loads / compute
            _run_skinny(x, layer, m, n, k, state, choice[1], choice[2], dg)


# ----------------------------------------------------------------------------------------------------------------
# Team rohanarun additions (our own work): extra tuner candidates for the m >= 120 shapes where our kernels measured
# faster on H200 (pooler m=128 TMA kernel; cuBLASLt plans with tuned tile/schedule for pooler/qkv/o/unpooler).
# They only enter vjeux's per-shape autotuner as candidates; the tuner keeps whichever is fastest on the judge GPU.
import ctypes as _ct
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def rq_deep_tma_pooler(DX, DW, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     EVEN: tl.constexpr = False):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(K // BK):
        a = DX.load([pm * BM, kb * BK])
        b = DW.load([kb * BK, pn * BN])
        acc = tl.dot(a, b, acc)
    mm = pm * BM + tl.arange(0, BM)
    nn = pn * BN + tl.arange(0, BN)
    if EVEN:
        tl.store(OUT + mm[:, None] * N + nn[None, :], acc.to(tl.bfloat16))
    else:
        tl.store(OUT + mm[:, None] * N + nn[None, :], acc.to(tl.bfloat16),
                 (mm[:, None] < M) & (nn[None, :] < N))


class rq_LtAlgo(_ct.Structure):
    _fields_ = [("data", _ct.c_uint64 * 8)]

class rq_LtResult(_ct.Structure):
    _fields_ = [("algo", rq_LtAlgo), ("workspace", _ct.c_size_t),
                ("state", _ct.c_int), ("waves", _ct.c_float),
                ("reserved", _ct.c_int * 4)]

class rq_LtAPI:
    def __init__(self):
        # Let the dynamic loader resolve the installed CUDA library. Avoid
        # ctypes.util.find_library, whose Linux implementation can start processes.
        self.lib = None
        for name in ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"):
            try:
                self.lib = _ct.CDLL(name)
                break
            except OSError:
                pass
        if self.lib is None:
            raise RuntimeError("cuBLASLt unavailable")
        V, I, S = _ct.c_void_p, _ct.c_int, _ct.c_size_t
        signatures = {
            "cublasLtCreate": [_ct.POINTER(V)],
            "cublasLtDestroy": [V],
            "cublasLtMatmulDescCreate": [_ct.POINTER(V), I, I],
            "cublasLtMatmulDescSetAttribute": [V, I, V, S],
            "cublasLtMatrixLayoutCreate": [_ct.POINTER(V), I, _ct.c_uint64, _ct.c_uint64, _ct.c_int64],
            "cublasLtMatmulPreferenceCreate": [_ct.POINTER(V)],
            "cublasLtMatmulPreferenceSetAttribute": [V, I, V, S],
            "cublasLtMatmulAlgoGetHeuristic": [V, V, V, V, V, V, V, I, _ct.POINTER(rq_LtResult), _ct.POINTER(I)],
            "cublasLtMatmulAlgoConfigGetAttribute": [_ct.POINTER(rq_LtAlgo), I, V, S, _ct.POINTER(S)],
            "cublasLtMatmulAlgoConfigSetAttribute": [_ct.POINTER(rq_LtAlgo), I, V, S],
            "cublasLtMatmulAlgoCheck": [V, V, V, V, V, V, _ct.POINTER(rq_LtAlgo), _ct.POINTER(rq_LtResult)],
            "cublasLtMatmul": [V, V, V, V, V, V, V, V, V, V, V, V, _ct.POINTER(rq_LtAlgo), V, S, V],
            "cublasLtMatmulPreferenceDestroy": [V],
            "cublasLtMatmulDescDestroy": [V],
            "cublasLtMatrixLayoutDestroy": [V],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = I
        self.handle = V()
        self.check(self.lib.cublasLtCreate(_ct.byref(self.handle)))
        self.alpha, self.beta = _ct.c_float(1.0), _ct.c_float(0.0)

    def __del__(self):
        try:
            handle = getattr(self, "handle", None)
            if handle is not None and handle.value:
                self.lib.cublasLtDestroy(handle)
        except (AttributeError, TypeError):
            pass

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError("cuBLASLt status " + str(status))

class rq_LtPlan:
    def __init__(self, api, m, n, k, transposed, tile, custom, pick=False):
        self.api = api
        self.transposed = transposed
        V, I = _ct.c_void_p, _ct.c_int
        self.desc, self.a, self.b, self.c, pref = V(), V(), V(), V(), V()
        lib, check = api.lib, api.check
        check(lib.cublasLtMatmulDescCreate(_ct.byref(self.desc), 68, 0))
        trans = I(0 if transposed else 1)
        check(lib.cublasLtMatmulDescSetAttribute(self.desc, 3, _ct.byref(trans), _ct.sizeof(trans)))
        check(lib.cublasLtMatrixLayoutCreate(_ct.byref(self.a), 14, n if transposed else k,
                                             k if transposed else n, n if transposed else k))
        check(lib.cublasLtMatrixLayoutCreate(_ct.byref(self.b), 14, k, m, k))
        check(lib.cublasLtMatrixLayoutCreate(_ct.byref(self.c), 14, n, m, n))
        check(lib.cublasLtMatmulPreferenceCreate(_ct.byref(pref)))
        # Only select algorithms with no scratch workspace and FP32 reduction.
        workspace = _ct.c_uint64(0)
        reduction = I(2)
        try:
            check(lib.cublasLtMatmulPreferenceSetAttribute(pref, 1, _ct.byref(workspace), _ct.sizeof(workspace)))
            check(lib.cublasLtMatmulPreferenceSetAttribute(pref, 3, _ct.byref(reduction), _ct.sizeof(reduction)))
            result, count = rq_LtResult(), I()
            if pick:
                # Keep a heuristic candidate exactly as returned, selected by its tile and option.
                many = (rq_LtResult * 48)()
                check(lib.cublasLtMatmulAlgoGetHeuristic(api.handle, self.desc, self.a, self.b,
                    self.c, self.c, pref, 48, many, _ct.byref(count)))
                for i in range(count.value):
                    got = []
                    for attribute in (0, 1, 5, 2):
                        val, size = I(), _ct.c_size_t()
                        check(lib.cublasLtMatmulAlgoConfigGetAttribute(_ct.byref(many[i].algo), attribute,
                            _ct.byref(val), _ct.sizeof(val), _ct.byref(size)))
                        got.append(val.value)
                    if not many[i].state and not many[i].workspace and got == [66, tile, custom, 1]:
                        self.algo = many[i].algo
                        return
                raise RuntimeError("No compatible library algorithm")
            check(lib.cublasLtMatmulAlgoGetHeuristic(api.handle, self.desc, self.a, self.b,
                self.c, self.c, pref, 1, _ct.byref(result), _ct.byref(count)))
            if count.value != 1 or result.state:
                raise RuntimeError("No compatible library algorithm")
            algorithm_id, size = I(), _ct.c_size_t()
            check(lib.cublasLtMatmulAlgoConfigGetAttribute(_ct.byref(result.algo), 0,
                _ct.byref(algorithm_id), _ct.sizeof(algorithm_id), _ct.byref(size)))
            if algorithm_id.value != 66:
                raise RuntimeError("Unsupported library algorithm")
            for attribute, value in ((1, tile), (6, 35), (2, 1), (3, 0), (5, custom)):
                v = I(value)
                check(lib.cublasLtMatmulAlgoConfigSetAttribute(_ct.byref(result.algo),
                    attribute, _ct.byref(v), _ct.sizeof(v)))
            checked = rq_LtResult()
            check(lib.cublasLtMatmulAlgoCheck(api.handle, self.desc, self.a, self.b,
                self.c, self.c, _ct.byref(result.algo), _ct.byref(checked)))
            check(checked.state)
            if checked.workspace:
                raise RuntimeError("Unexpected workspace requirement")
            self.algo = result.algo
        finally:
            lib.cublasLtMatmulPreferenceDestroy(pref)

    def __del__(self):
        try:
            self.api.lib.cublasLtMatmulDescDestroy(self.desc)
            for layout in (self.a, self.b, self.c):
                self.api.lib.cublasLtMatrixLayoutDestroy(layout)
        except (AttributeError, TypeError):
            pass

    def run(self, x, weight, out):
        api, V = self.api, _ct.c_void_p
        # These pointers are constructed from this invocation, never stored in a plan.
        status = api.lib.cublasLtMatmul(api.handle, self.desc, _ct.byref(api.alpha),
            V(weight.data_ptr()), self.a, V(x.data_ptr()), self.b, _ct.byref(api.beta),
            V(out.data_ptr()), self.c, V(out.data_ptr()), self.c, _ct.byref(self.algo),
            None, 0, V(torch.cuda.current_stream(x.device).cuda_stream))
        if status in (7, 15):
            return False
        api.check(status)
        return True


RQ_LT = [   # (layer, m, weight pre-transposed, tile id, custom option, pick from heuristic list)
    ("pooler", 256, True, 17, 2, False), ("qkv_gate", 256, False, 134, 2, True), ("o", 256, False, 20, 4, False),
    ("o", 128, False, 18, 2, False), ("unpooler", 128, True, 18, 4, False), ("unpooler", 256, True, 20, 4, False),
]
RQ_TMA = {("pooler", 128): (64, 64, 128, 6), ("pooler", 256): (64, 64, 128, 6),
          ("unpooler", 128): (64, 128, 128, 4), ("unpooler", 256): (128, 128, 64, 6)}


def rq_prepare(weights, state):
    rq = {"t": {}, "plans": {}}
    state["rq"] = rq
    try:
        for name in ("pooler", "unpooler", "o", "qkv_gate"):
            rq["t"][name] = weights[name].t().contiguous()
        api = rq_LtAPI()
        rq["api"] = api
        for layer, m, tr, tile, custom, pick in RQ_LT:
            n, k = weights[layer].shape
            try:
                rq["plans"][(layer, m)] = rq_LtPlan(api, m, n, k, tr, tile, custom, pick)
            except Exception:
                pass
    except Exception as e:
        print(f"[okc] rq_prepare: {type(e).__name__}: {e}", file=sys.stderr)


def rq_candidates(layer, m, state):
    rq = state.get("rq", {})
    c = []
    if (layer, m) in RQ_TMA and layer in rq.get("t", {}):
        c.append(("rq_tma",))
    if (layer, m) in rq.get("plans", {}):
        c.append(("rq_lt",))
    return c


def rq_run(choice, x, layer, m, n, k, state):
    rq = state["rq"]
    if choice[0] == "rq_lt":
        plan = rq["plans"].get((layer, m))
        if plan is not None and x.is_contiguous():
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            w = rq["t"][layer] if plan.transposed else state["raw"].get(layer, None)
            if w is None:
                w = rq["t"][layer].t()
            if plan.run(x, w, out):
                return out
        return _run_torch(x, layer, m, n, k, state)
    # rq_tma: TMA + tl.dot pooler/unpooler kernel on the K-major weight copy (any m)
    bm, bn, bk, stg = RQ_TMA.get((layer, m), RQ_TMA.get((layer, 256 if m > 128 else 128)))
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    dx = TensorDescriptor.from_tensor(x, [bm, bk])
    dw = TensorDescriptor.from_tensor(rq["t"][layer], [bk, bn])
    rq_deep_tma_pooler[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        dx, dw, out, m, n, k, bm, bn, bk, EVEN=(m % bm == 0 and n % bn == 0), num_warps=4, num_stages=stg)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Per-shape composition: the shapes in _ALT_ROUTES run the embedded _ALT_SRC module (its own namespace, its own
# prepare state); every other shape, and every m outside the scored set, runs the code above.
# Portions of this file (the embedded _ALT_SRC module) are adapted from software under the MIT License:
#   Copyright (c) 2026 Nathen Fernandes
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright
#   notice and this permission notice shall be included in all copies or substantial portions of the Software.
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import types as _types, zlib as _zl, base64 as _b64
_ALT_SRC = "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"
_ALT_ROUTES = frozenset([('o', 8), ('o', 128), ('o', 256), ('pooler', 8), ('pooler', 16), ('qkv_gate', 128), ('router', 8), ('router', 16), ('router', 32), ('unpooler', 8), ('unpooler', 16)])
_alt = _types.ModuleType("_alt_proj")
_alt.__file__ = __file__
exec(compile(_zl.decompress(_b64.b64decode(_ALT_SRC)).decode(), "_alt_proj", "exec"), _alt.__dict__)
_main_prepare, _main_kernel_fn = prepare, kernel_fn


def prepare(weights):
    state = _main_prepare(weights)
    if _ALT_ROUTES:
        state["_alt"] = _alt.prepare(weights)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    if (cfg["layer"], cfg["m"]) in _ALT_ROUTES and "_alt" in state:
        return _alt.kernel_fn(data, state["_alt"])
    return _main_kernel_fn(data, state)
