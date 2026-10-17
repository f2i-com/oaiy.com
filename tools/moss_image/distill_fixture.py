"""CPU-only oracle for a distillation adapter on the tiny MOSS-Image transformer. Never used by the native runtime.

Usage: python tools/moss_image/distill_fixture.py path/to/MOSS-Image-SFT/pipeline_moss_image.py
Reads `crates/oaiy-media/tests/moss_image/tiny.safetensors` (`reference.py`'s transformer) and makes a random adapter
for it in the layout a data-space DMD distillation of MOSS-Image ships in one file: each block's six projections'
low-rank factors (`layers.{i}.proj.{q,k,v,o,ffn_up_gate,ffn_down}.lora_A` `[rank, in]` and `.lora_B` `[out, rank]`),
the time embedding's two layers and the two modulations' biases in place of the transformer's, the latent mix
`latent_mix.Q` `[128, 128]` and the step codes. Rounded to f16 as OAIY holds them. Runs OpenMOSS's published
`MOSSImageTransformer` with the factors merged (`W + B A`) and the replacements in, in f32: a step's velocity at two
of the codes, and the sampler's three passes from a fixed noise (`x = z + Q x0`, `x0 = z - v`). Writes
`crates/oaiy-media/tests/moss_image/tiny_distill.safetensors`. Never opens production weights.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import torch
from safetensors.torch import load_file

source = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("moss_image_pipeline", source)
moss = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = moss
spec.loader.exec_module(moss)
torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.manual_seed(20261018)
folder = Path(__file__).resolve().parents[2] / "crates/oaiy-media/tests/moss_image"
tiny = load_file(folder / "tiny.safetensors")
model = moss.MOSSImageTransformer(dim=64, ffn_dim=96, num_heads=2, num_layers=3, text_dim=48).float().eval()
model.load_state_dict({k: v.float() for k, v in tiny.items() if not k.startswith("fixture.")}, strict=True)
f16 = lambda t: t.half().float()

RANK = 4
adapter = {}
state = model.state_dict()
for i in range(3):
    for m in ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]:
        weight = state[f"layers.{i}.proj.{m}.weight"]
        out_dim, in_dim = weight.shape
        a = f16(torch.randn(RANK, in_dim) / in_dim ** 0.5)
        # (a change of some tenth of the weight: large enough that a step without it is told apart)
        b = f16(torch.randn(out_dim, RANK) * 0.1 * weight.abs().mean() * in_dim ** 0.5 / RANK ** 0.5)
        adapter[f"layers.{i}.proj.{m}.lora_A"] = a
        adapter[f"layers.{i}.proj.{m}.lora_B"] = b
for name in ["time_embedding.0.weight", "time_embedding.0.bias", "time_embedding.2.weight", "time_embedding.2.bias",
             "shared_block_modulation.bias", "video_head_modulation.bias"]:
    adapter[name] = f16(state[name] + torch.randn_like(state[name]) * state[name].std())
adapter["latent_mix.Q"] = torch.randn(128, 128) * 0.01
adapter["step_codes"] = torch.tensor([1000.0, 791.0, 24.0])

# the adapted transformer: the factors merged in f32, the replacements in
merged = dict(state)
for name, value in adapter.items():
    if name.endswith(".lora_A"):
        base = name[: -len(".lora_A")]
        merged[f"{base}.weight"] = state[f"{base}.weight"] + adapter[f"{base}.lora_B"] @ value
    elif name in state:
        merged[name] = value
model.load_state_dict(merged, strict=True)

latent, text = tiny["fixture.latent"], tiny["fixture.text"][None].float()
H, W = 3, 4
grid = lambda rows: rows.transpose(0, 1).reshape(1, 128, H, W)  # [h w, 128] -> [1, 128, h, w]
rows = lambda x: x[0].flatten(1).transpose(0, 1).contiguous()


def velocity(x, code):
    return model(x, torch.tensor([code]), text).sample.float()


z = grid(torch.randn(H * W, 128))
x0 = torch.zeros_like(z)
for code in adapter["step_codes"].tolist():
    x = z + torch.einsum("oc,bchw->bohw", adapter["latent_mix.Q"], x0)
    x0 = z - velocity(x, code)

tensors = dict(adapter)
tensors.update({
    "fixture.velocity_1000": rows(velocity(grid(latent), 1000.0)),
    "fixture.velocity_791": rows(velocity(grid(latent), 791.0)),
    "fixture.noise": rows(z),
    "fixture.sampled": rows(x0),
})
dtypes = {torch.float16: "F16", torch.float32: "F32"}
# (the factors and replacements as f16 values in f32, as the adapters ship f32)
header, chunks, offset = {}, [], 0
for name, tensor in tensors.items():
    raw = tensor.float().contiguous().numpy().tobytes()
    header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
    chunks.append(raw)
    offset += len(raw)
header["__metadata__"] = {"source": "https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/pipeline_moss_image.py",
                          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "torch": torch.__version__}
h = json.dumps(header, separators=(",", ":")).encode()
h += b" " * (-len(h) % 8)
out = folder / "tiny_distill.safetensors"
out.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))
print("CPU reference fixture written:", out, f"({out.stat().st_size} bytes)")
