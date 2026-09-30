"""CPU-only tiny transformer oracle. Never used by the native runtime.

Usage: python tools/klein/reference.py path/to/BFL/src/flux2/model.py
Uses the published Apache-2.0 implementation, with an 8-wide test model.
Writes small deterministic fixtures; never opens production model weights.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import torch

source = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("bfl_klein_reference", source)
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
torch.set_num_threads(1)
torch.manual_seed(20260930)
model = ref.Flux2(ref.Flux2Params(in_channels=8, context_in_dim=6,
    hidden_size=8, num_heads=1, depth=1, depth_single_blocks=1,
    axes_dim=[2, 2, 2, 2], use_guidance_embed=False)).cpu().float().eval()
# Nontrivial modulation and normalization, still bounded and reproducible.
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith(".scale"):
            p.copy_(torch.linspace(0.75, 1.25, p.numel()).reshape(p.shape))
        else:
            p.copy_(torch.randn_like(p) * 0.12)
x = torch.linspace(-1, 1, 32).reshape(1, 4, 8)
ctx = torch.linspace(0.7, -0.3, 18).reshape(1, 3, 6)
x_ids = torch.tensor([[[0, y, z, 0] for y in range(2) for z in range(2)]], dtype=torch.float32)
ctx_ids = torch.tensor([[[0, 0, 0, i] for i in range(3)]], dtype=torch.float32)
with torch.no_grad():
    y = model(x, x_ids, torch.tensor([0.625]), ctx, ctx_ids, None)
folder = Path(__file__).resolve().parents[2] / "crates/oaiy-media/tests/klein"
folder.mkdir(parents=True, exist_ok=True)
header, chunks, offset = {}, [], 0
for name, t in model.state_dict().items():
    raw = t.contiguous().numpy().tobytes()
    header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
    chunks.append(raw)
    offset += len(raw)
h = json.dumps(header, separators=(",", ":")).encode()
h += b" " * (-len(h) % 8)
(folder / "tiny.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))
(folder / "expected.json").write_text(json.dumps({"source": "https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "torch": torch.__version__,
    "input": x.flatten().tolist(), "context": ctx.flatten().tolist(), "output": y.flatten().tolist()}, indent=2) + "\n")
print("CPU reference fixtures written:", folder)

# Independent Hugging Face Qwen3 decoder-layer oracle, including GQA and padding.
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
qc = Qwen3Config(hidden_size=8, intermediate_size=12, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, head_dim=4, rope_theta=1000000,
    attention_bias=False, use_sliding_window=False)
qc._attn_implementation = "eager"
layer = Qwen3DecoderLayer(qc, 0).cpu().float().eval()
with torch.no_grad():
    for name, p in layer.named_parameters():
        p.copy_(torch.linspace(0.8, 1.2, p.numel()).reshape(p.shape) if "norm.weight" in name else torch.randn_like(p) * 0.15)
tx = torch.linspace(-0.6, 0.9, 32).reshape(1, 4, 8)
pos = torch.arange(4).unsqueeze(0)
pe = Qwen3RotaryEmbedding(qc)(tx, pos)
mask = torch.tensor([[[[0, float("-inf"), float("-inf"), float("-inf")],
    [0, 0, float("-inf"), float("-inf")], [0, 0, float("-inf"), float("-inf")],
    [0, 0, float("-inf"), float("-inf")]]]])
with torch.no_grad():
    ty = layer(tx, attention_mask=mask, position_ids=pos, position_embeddings=pe)
header, chunks, offset = {}, [], 0
for name, t in layer.state_dict().items():
    raw = t.contiguous().numpy().tobytes()
    header["model.layers.0." + name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
    chunks.append(raw)
    offset += len(raw)
h = json.dumps(header, separators=(",", ":")).encode()
h += b" " * (-len(h) % 8)
(folder / "qwen-tiny.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))
(folder / "qwen-expected.json").write_text(json.dumps({"source": "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py",
    "input": tx.flatten().tolist(), "output": ty.flatten().tolist()}, indent=2) + "\n")
print("CPU Qwen3 reference fixtures written")
