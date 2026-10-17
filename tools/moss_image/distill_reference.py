"""CPU-only oracle for a distillation of MOSS-Image's real transformer. Never used by the native runtime.

Usage: python tools/moss_image/distill_reference.py pipeline_moss_image.py TRANSFORMER_DIR DISTILLATION TEXT_STATES.safetensors OUT.safetensors [PASSES]
Runs OpenMOSS's published `MOSSImageTransformer` (its own modules, from `pipeline_moss_image.py`) on the real weights in
float32 with a data-space DMD distillation's factors merged into each block's projections (`W + scale B A`, in
float32), its time embedding and the two modulations' biases in place of the transformer's, the 42 blocks made and run
one at a time from the checkpoint. Its sampler on a 16 x 16 latent from seeded noise under a prompt's text states
(`tools/moss_image/text_reference.py`'s `prompt0.final`): each pass `x = z + Q x0`, `v` the transformer at the pass's
step code, `x0 = z - v`. Writes the noise, each pass's input and velocity and the last estimate (`[256, 128]`, a token
a row), for `moss_image::webgpu`'s ignored test `a_distilled_step_is_the_references`. DISTILLATION is its one file
(`layers.{i}.proj.{module}.lora_A`/`lora_B`, the replacements, `latent_mix.Q`, `step_codes`).
"""
import gc
import importlib.util
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

source, weights, adapter_file, text_file, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
passes = int(sys.argv[6]) if len(sys.argv) > 6 else 8
spec = importlib.util.spec_from_file_location("moss_image_pipeline", source)
moss = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = moss
spec.loader.exec_module(moss)
torch.set_grad_enabled(False)
config = json.loads((weights / "config.json").read_text())
where = {}
for f in sorted(weights.glob("*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            where[k] = f
adapter = safe_open(adapter_file, "pt")
names = set(adapter.keys())


def tensors(prefix):
    got = {}
    for k, f in where.items():
        if k.startswith(prefix):
            with safe_open(f, "pt") as h:
                got[k[len(prefix):]] = h.get_tensor(k).float()
    return got


def replaced(prefix):
    # the transformer's, the distillation's in place where it has them
    got = tensors(prefix)
    for k in got:
        if prefix + k in names:
            got[k] = adapter.get_tensor(prefix + k).float()
    return got


dim, heads, ffn = config["dim"], config["num_heads"], config["ffn_dim"]
with torch.device("meta"):
    model = moss.MOSSImageTransformer(**{k: v for k, v in config.items() if not k.startswith("_")})
for name in ["video_embedder", "text_embedder", "time_embedding", "shared_block_modulation", "final_norm_video", "final_linear_video", "video_head_modulation"]:
    part = getattr(model, name).to_empty(device="cpu").float()
    part.load_state_dict(replaced(f"{name}."), strict=True)
model.rope = moss._RoPE(dim // heads, config["rope_base"])
text = load_file(text_file)["prompt0.final"][None].float()
Q, codes = adapter.get_tensor("latent_mix.Q").float(), adapter.get_tensor("step_codes").float()
torch.manual_seed(20261018)
H = W = 16
z = torch.randn(1, 128, H, W)


def block_weights(i):
    w = tensors(f"layers.{i}.")
    for m in ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]:
        a, b = f"layers.{i}.proj.{m}.lora_A", f"layers.{i}.proj.{m}.lora_B"
        if a in names:
            w[f"proj.{m}.weight"] = w[f"proj.{m}.weight"] + adapter.get_tensor(b).float() @ adapter.get_tensor(a).float()
    return w


def forward(latent, time):
    # MOSSImageTransformer.forward, its blocks streamed
    half = config["freq_dim"] // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = time.reshape(-1).double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    t = model.time_embedding(embedding)
    mod = model.shared_block_modulation(t).view(1, 6, -1)
    head = model.video_head_modulation(t).view(1, 2, -1)
    video = model.video_embedder(latent.flatten(2).transpose(1, 2))
    rope = model.rope(H, W, text.shape[1], "cpu")
    x = torch.cat((video, model.text_embedder(text, rope[:, H * W:])), 1)
    for i in range(config["num_layers"]):
        with torch.device("meta"):
            block = moss._Block(dim, ffn, heads)
        block = block.to_empty(device="cpu").float()
        block.load_state_dict(block_weights(i), strict=True)
        x = block(x, rope, mod)
        del block
        gc.collect()
    x = model.final_norm_video(x[:, :H * W])
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    v = model.final_linear_video(x)  # [1, h w, 128]
    return v.transpose(1, 2).reshape(1, 128, H, W)


rows = lambda x: x[0].flatten(1).transpose(0, 1).contiguous()
saved = {"noise": rows(z), "text": text[0].contiguous(), "codes": codes}
x0 = torch.zeros_like(z)
for r in range(passes):
    x = z + torch.einsum("oc,bchw->bohw", Q, x0)
    v = forward(x, codes[r:r + 1])
    x0 = z - v
    saved[f"pass{r}.input"], saved[f"pass{r}.velocity"] = rows(x), rows(v)
    print(f"pass {r} (time {codes[r].item():.0f}): velocity RMS {v.pow(2).mean().sqrt():.4f}", flush=True)
saved["sampled"] = rows(x0)
save_file(saved, out, metadata={"size": f"{H}x{W}", "passes": str(passes), "torch": torch.__version__})
print(f"estimate RMS {x0.pow(2).mean().sqrt():.4f} -> {out}")
