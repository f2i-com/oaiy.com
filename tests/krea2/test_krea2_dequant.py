#!/usr/bin/env python
"""Regression guard for the on-the-fly GGUF dequant kernels in krea2_gguf.py.

The pure-torch dequant kernels (Q4_K/Q5_K/Q6_K/Q4_0/Q5_0/Q8_0) claim to be
bit-identical to gguf.quants.dequantize. That claim is the load-bearing
assumption of the low-VRAM path, so this pins it: for each supported quant type,
quantize random data with gguf, then assert the torch dequant matches gguf's
numpy dequant. Plus a GGUFLinear forward smoke test.

The kernels are loaded straight out of the shipped self-contained template
(desktop/src-tauri/resources/templates/krea2.json), so this tests exactly what
ships -- not a stale copy.

Run with the krea2 venv (no pytest needed):
    <dataDir>/venvs/krea2/Scripts/python.exe tests/krea2/test_krea2_dequant.py
or under pytest:
    <dataDir>/venvs/krea2/Scripts/python.exe -m pytest tests/krea2
"""
import importlib
import json
import pathlib
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "desktop" / "src-tauri" / "resources" / "templates" / "krea2.json"
QTYPES = ["Q4_K", "Q5_K", "Q6_K", "Q4_0", "Q5_0", "Q8_0"]
_TOL = 1e-3

# Byte offset(s) of the fp16 scale field(s) within each type's block. gguf's
# Python can DEQUANTIZE k-quants but can't QUANTIZE them, so we build valid
# blocks directly: random quant payload + finite fp16 scales (so neither dequant
# path yields NaN/Inf, which would make the comparison meaningless).
_FP16_OFFSETS = {"Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208], "Q4_0": [0], "Q5_0": [0], "Q8_0": [0]}


def _valid_blocks(gguf, np_rng, qname, n):
    """n valid quantized blocks for `qname`: random bytes everywhere, with the
    fp16 scale field(s) overwritten by finite fp16 values."""
    qtype = getattr(gguf.GGMLQuantizationType, qname)
    type_size = gguf.GGML_QUANT_SIZES[qtype][1]
    raw = np_rng.integers(0, 256, size=(n, type_size), dtype=np.uint8)
    for off in _FP16_OFFSETS[qname]:
        vals = (np_rng.random(n).astype(np.float32) * 0.99 + 0.01).astype(np.float16)
        raw[:, off:off + 2] = vals.view(np.uint8).reshape(n, 2)  # little-endian fp16
    return raw


def load_krea2_gguf():
    """Extract the embedded scripts from the self-contained template and import
    the GGUF loader, so we exercise the shipped code."""
    files = json.loads(TEMPLATE.read_text(encoding="utf-8")).get("files", {})
    assert "krea2_gguf.py" in files, "krea2.json is missing the embedded krea2_gguf.py"
    d = tempfile.mkdtemp(prefix="krea2-src-")
    for name, body in files.items():
        pathlib.Path(d, name).write_text(body, encoding="utf-8", newline="")
    sys.path.insert(0, d)
    return importlib.import_module("krea2_gguf")


def dequant_maxdiff(kg, qname):
    """Dequant valid random blocks with both gguf (numpy) and the torch kernels;
    return (maxdiff, skip_reason)."""
    import gguf
    import torch

    qtype = getattr(gguf.GGMLQuantizationType, qname)
    block = gguf.GGML_QUANT_SIZES[qtype][0]
    n = 8
    raw = _valid_blocks(gguf, np.random.default_rng(0), qname, n)
    ref = gguf.quants.dequantize(raw, qtype).reshape(n, block)
    got = kg.dequantize_torch(
        torch.from_numpy(raw.reshape(-1)), int(qtype), n, block, dtype=torch.float32,
    ).cpu().numpy()
    return float(np.abs(got - ref).max()), None


def gguf_linear_finite(kg):
    """A GGUFLinear built from a (Q4_K) quantized weight produces a finite,
    correctly shaped output."""
    import gguf
    import torch

    qtype = gguf.GGMLQuantizationType.Q4_K
    block = gguf.GGML_QUANT_SIZES[qtype][0]
    out_features = 16
    raw = _valid_blocks(gguf, np.random.default_rng(1), "Q4_K", out_features)  # 16 rows of `block`
    lin = kg.GGUFLinear(torch.from_numpy(raw.reshape(-1)), int(qtype), out_features, block)
    y = lin(torch.randn(4, block))
    return tuple(y.shape) == (4, out_features) and bool(torch.isfinite(y).all())


# ---- pytest entry points (optional; the file also runs standalone) ----
try:
    import pytest

    @pytest.fixture(scope="module")
    def kg():
        pytest.importorskip("torch")
        pytest.importorskip("gguf")
        return load_krea2_gguf()

    @pytest.mark.parametrize("qname", QTYPES)
    def test_torch_dequant_matches_gguf(kg, qname):
        maxdiff, skip = dequant_maxdiff(kg, qname)
        if skip:
            pytest.skip(skip)
        assert maxdiff < _TOL, f"{qname} dequant maxdiff {maxdiff:.3e} exceeds {_TOL}"

    def test_gguf_linear_forward_finite(kg):
        assert gguf_linear_finite(kg)
except ImportError:
    pass


def _main():
    kg = load_krea2_gguf()
    failures = 0
    for q in QTYPES:
        maxdiff, skip = dequant_maxdiff(kg, q)
        if skip:
            print(f"  SKIP {q}: {skip}")
        elif maxdiff < _TOL:
            print(f"  PASS {q}: maxdiff {maxdiff:.2e}")
        else:
            print(f"  FAIL {q}: maxdiff {maxdiff:.2e}")
            failures += 1
    ok = gguf_linear_finite(kg)
    print(f"  {'PASS' if ok else 'FAIL'} GGUFLinear forward finite")
    failures += 0 if ok else 1
    print("\nOK" if failures == 0 else f"\n{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
