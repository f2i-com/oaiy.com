"""CPU-only tiny FLUX 3 Image Klein transformer oracle. Never used by the native runtime.

Usage: python tools/flux3/reference.py path/to/flux-3-image-klein/src/flux3/dit.py
Runs BFL's published Apache-2.0 DiT with its sizes made small (32 wide, two heads of 16, one block a kind, two
shared) on random weights, rounded to f16 as OAIY holds them, in f32. Writes one safetensors file: the weights, a
step's inputs (a 3x4 picture's latent, two references of 4 and 3 tokens, 5 text tokens with a few channels far past
f16's range) and the velocity the DiT gives them, with and without the references. Never opens production weights.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import torch

source = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("bfl_flux3_dit", source)
dit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dit
spec.loader.exec_module(dit)
# the published sizes are 3072 wide, 24 heads of 128, 5 blocks a kind, 28 shared, MLP 3x, text 20480 wide
dit.HIDDEN, dit.HEADS, dit.AXES_DIM = 32, 2, (4, 4, 4, 4)
dit.DEPTH, dit.DEPTH_SINGLE, dit.MLP_RATIO = 1, 2, 1.5
dit.IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 48}
torch.set_num_threads(1)
torch.manual_seed(20261016)
model = dit.DiT(flow_map=True).float().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith(".scale"):
            p.copy_(torch.linspace(0.75, 1.25, p.numel()).reshape(p.shape))
        else:
            # (this large, the references move the velocity by a quarter: what leaving them out would miss)
            p.copy_(torch.randn_like(p) * (1.5 / p.shape[-1] ** 0.5))
        # as OAIY holds a checkpoint's matrices: f16
        p.copy_(p.half().float())


def ids(rows, cols, l=0):
    """A latent's positions as the pipeline gives them: (0, row, col, l), rows and columns times 0.5."""
    return torch.tensor([[0, r * 0.5, c * 0.5, l] for r in range(rows) for c in range(cols)], dtype=torch.float32)


image = torch.randn(12, 128)
image_ids = ids(3, 4)
cond = torch.randn(7, 128)
cond_ids = torch.cat([ids(2, 2, -1), ids(1, 3, -2)])
text = torch.randn(5, 48) * 3
text[:, 7] *= 400  # an encoder's states: a few channels far larger than the rest
text[2, 30] = 90000.0  # one past f16's range outright: the text's input must read its states as f32
text_ids = torch.tensor([[0, 0, 0, p] for p in range(5)], dtype=torch.float32)
# times as the pipeline holds them: bfloat16
t = torch.tensor(0.7).bfloat16().float().item()
target = torch.tensor(0.45).bfloat16().float().item()


def velocity(with_cond):
    nc = 7 if with_cond else 0
    packed = dit.Packed(
        data={"image": image, "image_cond": cond[:nc], "text": text},
        ids={"image": image_ids, "image_cond": cond_ids[:nc], "text": text_ids},
        timesteps={"image": torch.full((12,), t), "image_cond": torch.zeros(nc), "text": torch.zeros(5)},
        mode_indices={"image": [(0, 12)], "image_cond": [(0, nc)], "text": [(0, 5)]},
    )
    with torch.no_grad():
        return model(packed, timesteps_ref=torch.tensor([target]))


tensors = {name: p.detach().half() for name, p in model.state_dict().items()}
tensors.update({
    "fixture.image": image, "fixture.image_ids": image_ids, "fixture.cond": cond, "fixture.cond_ids": cond_ids,
    "fixture.text": text, "fixture.text_ids": text_ids, "fixture.times": torch.tensor([t, target]),
    "fixture.velocity": velocity(True), "fixture.velocity_without_references": velocity(False),
})
dtypes = {torch.float16: "F16", torch.float32: "F32"}
header, chunks, offset = {}, [], 0
for name, tensor in tensors.items():
    raw = tensor.contiguous().numpy().tobytes()
    header[name] = {"dtype": dtypes[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
    chunks.append(raw)
    offset += len(raw)
header["__metadata__"] = {"source": "https://github.com/black-forest-labs/flux-3-image-klein/blob/main/src/flux3/dit.py",
                          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "torch": torch.__version__}
h = json.dumps(header, separators=(",", ":")).encode()
h += b" " * (-len(h) % 8)
folder = Path(__file__).resolve().parents[2] / "crates/oaiy-media/tests/flux3"
folder.mkdir(parents=True, exist_ok=True)
(folder / "tiny.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))
print("CPU reference fixture written:", folder / "tiny.safetensors", f"({(folder / 'tiny.safetensors').stat().st_size} bytes)")
