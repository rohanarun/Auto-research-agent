"""NVFP4 (E2M1 values, 16-element blocks, E4M3 block scales, FP32 tensor scale).

A value is stored as  x ~= e2m1 * block_scale(fp8 e4m3) * global_scale(fp32).
Packed tensors hold two E2M1 codes per uint8: element 2i in the low nibble,
element 2i+1 in the high nibble, matching torch.float4_e2m1fn_x2.

Depends only on torch.
"""

from __future__ import annotations

import math

import torch

BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
# random_weights' block scales: log2 of the scale is normal with this mean and deviation.
SCALE_LOG2_MEAN = 7.0
SCALE_LOG2_STD = 0.4

# Magnitude of each 3-bit E2M1 code (the sign is bit 3).
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between grid values, for round-to-nearest.
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


_GRIDS: dict[torch.device, torch.Tensor] = {}


def _grid(device: torch.device) -> torch.Tensor:
    # Cached per device, so dequantize is CUDA-graph capturable after its first eager call.
    device = torch.device(device)
    if device not in _GRIDS:
        _GRIDS[device] = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=device)
    return _GRIDS[device]


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> float32 [..., K] of E2M1 values (unscaled)."""
    p = packed.view(torch.uint8)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    codes = torch.stack((lo, hi), dim=-1).flatten(-2)
    mag = _grid(packed.device)[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def pack_e2m1(values: torch.Tensor) -> torch.Tensor:
    """float [..., K] already on the E2M1 grid (after scaling) -> uint8 [..., K/2]."""
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=values.device)
    v = values.float()
    mag_code = torch.bucketize(v.abs().clamp(max=E2M1_MAX), mids).to(torch.uint8)
    sign = ((v < 0) & (mag_code > 0)).to(torch.uint8) << 3
    codes = (sign | mag_code).view(*v.shape[:-1], v.shape[-1] // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4)


def global_scale_for(amax: torch.Tensor | float) -> torch.Tensor:
    """Tensor scale chosen so the largest block scale lands at the top of E4M3."""
    amax = torch.as_tensor(amax, dtype=torch.float32)
    return torch.where(amax > 0, amax / (E2M1_MAX * E4M3_MAX), torch.ones_like(amax))


def quantize(x: torch.Tensor, global_scale: torch.Tensor | float | None = None):
    """Quantize the last dim of x to NVFP4.

    Returns (packed uint8 [..., K/2], block scales fp8 e4m3 [..., K/16], global scale fp32 []).
    If global_scale is None it is computed from x (dynamic); otherwise it is used as given
    (static).
    """
    k = x.shape[-1]
    if k % BLOCK:
        raise ValueError(f"last dim {k} is not a multiple of {BLOCK}")
    xf = x.float()
    gs = global_scale_for(xf.abs().amax()) if global_scale is None else torch.as_tensor(
        global_scale, dtype=torch.float32, device=x.device
    )
    gs = gs.to(x.device)
    blocks = xf.view(*xf.shape[:-1], k // BLOCK, BLOCK)
    block_amax = blocks.abs().amax(dim=-1)
    sf = (block_amax / E2M1_MAX / gs).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    scale = (sf.float() * gs).unsqueeze(-1)
    q = torch.where(scale > 0, blocks / scale, torch.zeros_like(blocks))
    packed = pack_e2m1(q.view_as(xf))
    return packed, sf, gs


def dequantize(packed: torch.Tensor, sf: torch.Tensor, global_scale: torch.Tensor | float) -> torch.Tensor:
    """Inverse of quantize. Returns float32 [..., K]."""
    vals = unpack_e2m1(packed)
    k = vals.shape[-1]
    blocks = vals.view(*vals.shape[:-1], k // BLOCK, BLOCK)
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=packed.device)
    out = blocks * (sf.float() * gs).unsqueeze(-1)
    return out.view_as(vals)


def fake_quantize(x: torch.Tensor, global_scale: torch.Tensor | float | None = None) -> torch.Tensor:
    """Quantize then dequantize, i.e. what an NVFP4 kernel sees as its input."""
    return dequantize(*quantize(x, global_scale))


def random_weights(rows: int, k: int, *, generator: torch.Generator, device: str | torch.device,
                   leading: tuple[int, ...] = (), wide_scales: bool = False):
    """Random NVFP4 weights whose dequantized values have roughly unit-variance dot products.

    Codes are uniform over all 16 bit patterns; block scales are log-normal, log2 ~ N(7, 0.4)
    clamped to [8, 448]; the global scale
    normalizes by sqrt(k). With wide_scales, block scales are log-uniform over 2^-6 to 448
    instead.
    """
    shape = (*leading, rows, k // 2)
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator, device=device)
    scale_shape = (*leading, rows, k // BLOCK)
    if wide_scales:
        u = torch.rand(scale_shape, generator=generator, device=device)
        sf = torch.exp2(-6 + u * (math.log2(E4M3_MAX) + 6)).to(torch.float8_e4m3fn)
        mean_sq = float(sf.float().pow(2).mean())
    else:
        z = torch.randn(scale_shape, generator=generator, device=device)
        sf = torch.exp2(SCALE_LOG2_MEAN + SCALE_LOG2_STD * z).clamp(8.0, E4M3_MAX).to(torch.float8_e4m3fn)
        # E[scale^2] of the log-normal (the clamp moves it by under 0.1%).
        mean_sq = 4.0 ** SCALE_LOG2_MEAN * math.exp(2 * (math.log(2) * SCALE_LOG2_STD) ** 2)
    # E[e2m1^2] over uniform codes is ~8.6.
    gs = torch.tensor(1.0 / (8.6 * mean_sq * k) ** 0.5, dtype=torch.float32, device=device)
    return packed, sf, gs


def outlier_channels(x: torch.Tensor, *, generator: torch.Generator, frac: float = 0.01,
                     scale: float = 30.0) -> torch.Tensor:
    """x with a few channels (last dim) scaled up, like the outlier features of real LLM
    activations."""
    k = x.shape[-1]
    n = max(1, int(k * frac))
    idx = torch.randperm(k, generator=generator, device=x.device)[:n]
    out = x.clone()
    out[..., idx] = (out[..., idx].float() * scale).to(x.dtype)
    return out

