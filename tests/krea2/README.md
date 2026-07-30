# krea2 tests

Regression guard for the **GGUF dequant kernels** in the krea2 service
(`krea2_gguf.py`), which ship inline in the self-contained
`desktop/src-tauri/resources/templates/krea2.json`.

`test_krea2_dequant.py` extracts the kernels straight from that template (so it
tests exactly what ships) and asserts the pure-torch dequant is **bit-identical**
to `gguf.quants.dequantize` for every supported quant type
(Q4_K / Q5_K / Q6_K / Q4_0 / Q5_0 / Q8_0), plus a `GGUFLinear` forward smoke
test. It builds valid quantized blocks directly (random payload + finite fp16
scales), so it needs **no multi-GB GGUF** — just `torch` + `gguf` + `numpy`,
which the krea2 venv already has.

## Run

Use the krea2 service venv (it has torch + gguf):

```bash
# standalone (no pytest needed)
"$APPDATA/com.oaiy/venvs/krea2/Scripts/python.exe" tests/krea2/test_krea2_dequant.py

# or under pytest, if installed in the venv
"$APPDATA/com.oaiy/venvs/krea2/Scripts/python.exe" -m pytest tests/krea2
```

On Linux the venv python is `~/.local/share/com.oaiy/venvs/krea2/bin/python` (or
wherever `OAIY_DATA_DIR` points).

Editing the kernels: extract the embedded source, edit, and re-embed with the
package tool — see `tools/service-package.mjs`.
