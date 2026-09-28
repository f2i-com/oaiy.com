"""Minimal `flash_attn` shim backed by PyTorch SDPA (OAIY-companion installed).

Lance hard-imports `flash_attn` in modeling/lance/qwen2_navit.py and calls
`flash_attn_varlen_func` for its packed (NaViT) attention. Real flash-attn has
no Windows wheel matching this torch/CUDA/Python on Blackwell (sm_120), so this
provides a correct — if slower — implementation via
torch.nn.functional.scaled_dot_product_attention. SDPA is exact attention, so
outputs match flash-attn numerically; it just isn't fused.

Installed as `flash_attn.py` in the venv's site-packages, with NO package
metadata, so transformers' `is_flash_attn_2_available()` stays False and the
OTHER Lance model files (vit, qwen2_5_vl) take their built-in SDPA paths — they
only import from flash_attn under that guard. qwen2_navit's unguarded
`from flash_attn import flash_attn_varlen_func` resolves here.

Set OAIY_LANCE_FLASH=1 at install time to instead build/install real flash-attn
and skip this shim.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Intentionally NOT a real version — kept off PyPI-style metadata so
# is_flash_attn_2_available() (which reads importlib.metadata) returns False.
__version__ = "0.0.0+oaiy-sdpa-shim"


def _seg_attention(q, k, v, causal: bool, softmax_scale: float):
    """Attention over one variable-length segment.

    q: (sq, nh, hd)   k,v: (sk, nkv, hd)   ->  (sq, nh, hd)
    """
    sq, nh, hd = q.shape
    sk, nkv, _ = k.shape
    if nkv != nh:  # GQA / MQA: expand kv heads to match query heads
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    # -> (1, nh, s, hd) for SDPA
    qd = q.transpose(0, 1).unsqueeze(0)
    kd = k.transpose(0, 1).unsqueeze(0)
    vd = v.transpose(0, 1).unsqueeze(0)
    if causal and sq != sk:
        # Bottom-right aligned causal (flash-attn >= 2.1 semantics): query i
        # may attend to keys 0..(sk-sq)+i. SDPA's is_causal assumes top-left,
        # which is wrong when sq != sk (e.g. decode with a KV cache).
        qi = torch.arange(sq, device=q.device).unsqueeze(1)
        kj = torch.arange(sk, device=q.device).unsqueeze(0)
        allowed = kj <= (sk - sq) + qi
        mask = torch.zeros(sq, sk, dtype=q.dtype, device=q.device)
        mask.masked_fill_(~allowed, float("-inf"))
        out = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=mask, scale=softmax_scale)
    else:
        out = F.scaled_dot_product_attention(qd, kd, vd, is_causal=causal, scale=softmax_scale)
    return out.squeeze(0).transpose(0, 1).contiguous()  # (sq, nh, hd)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    **kwargs,
):
    """SDPA-backed drop-in for flash_attn.flash_attn_varlen_func.

    q: (total_q, nheads, headdim), k/v: (total_k, nheads_k, headdim), with
    cu_seqlens_* the cumulative segment boundaries. Returns (total_q, nheads,
    headdim). dropout/window/softcap/alibi are not used by Lance and ignored.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    out = torch.empty_like(q)
    for i in range(len(cq) - 1):
        qs, qe = int(cq[i]), int(cq[i + 1])
        ks, ke = int(ck[i]), int(ck[i + 1])
        if qe > qs:
            out[qs:qe] = _seg_attention(q[qs:qe], k[ks:ke], v[ks:ke], bool(causal), softmax_scale)
    return out


# A couple of common aliases Lance/other code might reach for. Harmless if
# unused (Lance only needs flash_attn_varlen_func).
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    # q,k,v: (batch, seqlen, nheads, headdim)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    qd = q.transpose(1, 2)
    kd = k.transpose(1, 2)
    vd = v.transpose(1, 2)
    if kd.shape[1] != qd.shape[1]:
        rep = qd.shape[1] // kd.shape[1]
        kd = kd.repeat_interleave(rep, dim=1)
        vd = vd.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(qd, kd, vd, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()
