"""Pure-torch stand-in for the reference `inference/kernel.py` (tilelang, which does
not run on Windows). Installed as the `kernel` module before `model.py` is
imported, so the reference model code runs unmodified.

Each function reproduces the tilelang kernel's arithmetic, not just its intent:
power-of-two scales via the same ceil-log2 bit trick, clamps before the fp8/fp4
cast, fp32 accumulation, and the bf16 cast of attention probabilities before
the PV product. Speed is irrelevant here; this is the numerical oracle the Rust
port is tested against.
"""

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
# e2m1 magnitudes by code 0..7; code parity is the mantissa bit (round-half-to-even picks even codes)
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# checkpoint nibble -> value (sign in bit 3), as in reference convert.py
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) for positive fp32 x, exact: the kernel's fast_log2_ceil reads the
    exponent bits and adds one unless the mantissa is zero (x already a power of two)."""
    m, e = torch.frexp(x.float())  # x = m * 2**e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), e)


def e8m0_to_f32(s: torch.Tensor) -> torch.Tensor:
    """float8_e8m0fnu (or its raw uint8 bits) -> fp32 power of two."""
    bits = s.view(torch.uint8).int() if s.dtype != torch.uint8 else s.int()
    return torch.ldexp(torch.ones_like(bits, dtype=torch.float32), bits - 127)


def f32_to_e8m0(s: torch.Tensor) -> torch.Tensor:
    """Exact powers of two -> float8_e8m0fnu."""
    _, e = torch.frexp(s.float())
    return (e - 1 + 127).clamp(0, 254).to(torch.uint8).view(torch.float8_e8m0fnu)


def round_to_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 values (already clamped to +-6) onto the e2m1 grid, half to even."""
    grid = FP4_GRID.to(v.device)
    mag = v.abs()
    hi = torch.bucketize(mag, grid).clamp(max=7)  # first code with grid >= mag
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = mag - grid[lo], grid[hi] - mag
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    code = torch.where(pick_hi, hi, lo)
    return torch.copysign(grid[code], v)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise fp8 e4m3 quantization along the last dim. inplace=True writes the
    quantize-dequantize result back into x (the KV-cache simulation)."""
    shape = x.shape
    n = shape[-1]
    assert n % block_size == 0
    xf = x.float().reshape(-1, n // block_size, block_size)
    amax = xf.abs().amax(-1).clamp_min(1e-4)
    s = pow2_ceil(amax * (1.0 / FP8_MAX)) if scale_fmt is not None else amax * (1.0 / FP8_MAX)
    y = (xf / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s.unsqueeze(-1)).reshape(shape).to(x.dtype))
        return x
    s_out = f32_to_e8m0(s) if scale_dtype == torch.float8_e8m0fnu else s
    return y.reshape(shape), s_out.reshape(*shape[:-1], n // block_size)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise fp4 e2m1: power-of-two (e8m0) scales for the indexer, e4m3 scales in
    groups of 16 for the compressed KV. Only the in-place (simulated) form is used."""
    assert inplace, "model.py only calls fp4_act_quant in place"
    shape = x.shape
    n = shape[-1]
    xf = x.float().reshape(-1, n // block_size, block_size)
    amax = xf.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * 2**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(FP4_MAX * 2**-126)
        s = pow2_ceil(amax * (1.0 / FP4_MAX))
    q = round_to_fp4((xf / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((q * s.unsqueeze(-1)).reshape(shape).to(x.dtype))
    return x


def dequant_fp8_weight(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """fp8 [N, K] with one e8m0 scale per block x block tile -> fp32."""
    n, k = w.shape
    sf = e8m0_to_f32(s)
    sf = sf.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return w.float() * sf


def dequant_fp4_weight(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Packed e2m1 [N, K/2] (low nibble = even element) with one e8m0 scale per 32 along K -> fp32."""
    b = w.view(torch.uint8)
    table = FP4_TABLE.to(b.device)
    vals = torch.stack([table[(b & 0x0F).long()], table[(b >> 4).long()]], dim=-1).flatten(-2)
    return vals * e8m0_to_f32(s).repeat_interleave(block, -1)


def _act_dequant(a, a_s, block):
    k = a.shape[-1]
    af = a.float().reshape(-1, k // block, block)
    s = e8m0_to_f32(a_s) if a_s.dtype == torch.float8_e8m0fnu else a_s.float()
    return (af * s.reshape(-1, k // block, 1)).reshape(-1, k)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C = A @ B^T, fp8 activations and weights with block scales, fp32 accumulate."""
    k = a.shape[-1]
    c = _act_dequant(a, a_s, block_size) @ dequant_fp8_weight(b, b_s, block_size).t()
    return c.reshape(*a.shape[:-1], b.shape[0]).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C = A_fp8 @ B_fp4^T: fp8 activations (per act_block_size), fp4 weights (per 32)."""
    c = _act_dequant(a, a_s, act_block_size) @ dequant_fp4_weight(b, b_s).t()
    return c.reshape(*a.shape[:-1], b.shape[0]).to(torch.get_default_dtype())


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Split the per-token mix projection into pre / post / comb; comb doubly stochastic."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = m[..., 2 * hc:].unflatten(-1, (hc, hc)) * hc_scale[2] + hc_base[2 * hc:].view(hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Each query attends to its own gathered KV positions (-1 = none), plus a per-head
    sink logit in the denominator. Matches the kernel's finite -1e30 running-max start
    (an all-invalid row yields zeros) and its bf16 cast of probabilities before PV."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    gathered = torch.gather(
        kv.unsqueeze(1).expand(b, m, kv.shape[1], d), 2, idx.clamp_min(0).unsqueeze(-1).expand(b, m, idx.shape[-1], d)
    )  # [b, m, k, d]
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    smax = scores.amax(-1).clamp_min(-1e30)  # [b, m, h]
    p = torch.exp(scores - smax.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - smax)
    num = torch.einsum("bmhk,bmkd->bmhd", p.to(torch.bfloat16).float(), gathered.to(torch.bfloat16).float())
    return (num / denom.unsqueeze(-1)).to(q.dtype)
