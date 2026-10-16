"""CPU-only tiny MOSS-Image transformer oracle. Never used by the native runtime.

Usage: python tools/moss_image/reference.py path/to/MOSS-Image-SFT/pipeline_moss_image.py
Runs OpenMOSS's published Apache-2.0 `MOSSImageTransformer` with its sizes made small (64 wide, two heads of 32, three
blocks, MLP 96, text states 48 wide) on random weights, rounded to f16 as OAIY holds them, in f32. Writes one
safetensors file: the weights, a step's inputs (a 3x4 picture's latent, five text tokens, the step's time) and the
velocity the transformer gives them, with every block and with block 1 skipped (the negative branch's skip-layer
guidance). Never opens production weights.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import torch

source = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("moss_image_pipeline", source)
moss = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = moss
spec.loader.exec_module(moss)
torch.set_num_threads(1)
torch.manual_seed(20261017)
# the published sizes are 3072 wide, 24 heads of 128, 42 blocks, MLP 8192, text 4096 wide
model = moss.MOSSImageTransformer(dim=64, ffn_dim=96, num_heads=2, num_layers=3, text_dim=48).float().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight") or "final_norm" in name:
            p.copy_(torch.linspace(0.75, 1.25, p.numel()).reshape(p.shape))
        elif name.endswith(".bias"):
            p.copy_(torch.randn_like(p) * 0.1)
        else:
            p.copy_(torch.randn_like(p) * (1.5 / p.shape[-1] ** 0.5))
        # as OAIY holds a checkpoint's matrices: f16
        p.copy_(p.half().float())

H, W = 3, 4
latent = torch.randn(1, 128, H, W)
text = torch.randn(1, 5, 48) * 3
text[:, :, 7] *= 300  # a few channels far larger than the rest: the text's projection reads its states as f32
time = torch.tensor([700.5])


def velocity(skip):
    with torch.no_grad():
        out = model(latent, time, text, skip_layers=skip).sample
    return out[0].flatten(1).transpose(0, 1).contiguous()  # [h w, 128], a token a row


state = model.state_dict()
tensors = {name: (t.float() if name == "rope.bands" else t.detach().half()) for name, t in state.items()}
tensors.update({
    "fixture.latent": latent[0].flatten(1).transpose(0, 1).contiguous(),
    "fixture.size": torch.tensor([H, W], dtype=torch.float32),
    "fixture.text": text[0].contiguous(),
    "fixture.time": time,
    "fixture.velocity": velocity(None),
    "fixture.velocity_skipping_1": velocity([1]),
})
dtypes = {torch.float16: "F16", torch.float32: "F32"}
header, chunks, offset = {}, [], 0
for name, tensor in tensors.items():
    raw = tensor.contiguous().numpy().tobytes()
    header[name] = {"dtype": dtypes[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
    chunks.append(raw)
    offset += len(raw)
header["__metadata__"] = {"source": "https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/pipeline_moss_image.py",
                          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "torch": torch.__version__}
h = json.dumps(header, separators=(",", ":")).encode()
h += b" " * (-len(h) % 8)
folder = Path(__file__).resolve().parents[2] / "crates/oaiy-media/tests/moss_image"
folder.mkdir(parents=True, exist_ok=True)
(folder / "tiny.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))
print("CPU reference fixture written:", folder / "tiny.safetensors", f"({(folder / 'tiny.safetensors').stat().st_size} bytes)")
