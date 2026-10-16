"""CPU-only oracle for one step of MOSS-Image's real transformer. Never used by the native runtime.

Usage: python tools/moss_image/velocity_reference.py pipeline_moss_image.py TRANSFORMER_DIR TEXT_STATES.safetensors OUT.safetensors
Runs OpenMOSS's published `MOSSImageTransformer` (its own modules, from `pipeline_moss_image.py`) on the real weights
in float32, its 42 blocks made and run one at a time from the checkpoint (the whole would want 20 GB): a 16 x 16
latent's step (random, seeded) at time 700 under a prompt's text states (`tools/moss_image/text_reference.py`'s
`prompt0.final`), with every block and with block 22 skipped. Writes the inputs and both velocities (`[256, 128]`, a
token a row), for `moss_image::webgpu`'s ignored test `a_step_is_openmosss`.
"""
import gc
import importlib.util
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

source, weights, text_file, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
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


def tensors(prefix):
    got = {}
    for k, f in where.items():
        if k.startswith(prefix):
            with safe_open(f, "pt") as h:
                got[k[len(prefix):]] = h.get_tensor(k).float()
    return got


dim, heads, ffn = config["dim"], config["num_heads"], config["ffn_dim"]
with torch.device("meta"):
    model = moss.MOSSImageTransformer(**{k: v for k, v in config.items() if not k.startswith("_")})
# the small parts whole, in float32; the blocks one at a time below
for name in ["video_embedder", "text_embedder", "time_embedding", "shared_block_modulation", "final_norm_video", "final_linear_video", "video_head_modulation"]:
    part = getattr(model, name).to_empty(device="cpu").float()
    part.load_state_dict(tensors(f"{name}."), strict=True)
model.rope = moss._RoPE(dim // heads, config["rope_base"])
text = load_file(text_file)["prompt0.final"][None].float()
torch.manual_seed(20261017)
H = W = 16
latent = torch.randn(1, 128, H, W)
time = torch.tensor([700.0])


def forward(skip):
    # MOSSImageTransformer.forward, its blocks streamed
    times = time.reshape(-1)
    half = config["freq_dim"] // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    t = model.time_embedding(embedding)
    mod = model.shared_block_modulation(t).view(1, 6, -1)
    head = model.video_head_modulation(t).view(1, 2, -1)
    video = model.video_embedder(latent.flatten(2).transpose(1, 2))
    rope = model.rope(H, W, text.shape[1], "cpu")
    x = torch.cat((video, model.text_embedder(text, rope[:, H * W:])), 1)
    for i in range(config["num_layers"]):
        if i in skip:
            continue
        with torch.device("meta"):
            block = moss._Block(dim, ffn, heads)
        block = block.to_empty(device="cpu").float()
        block.load_state_dict(tensors(f"layers.{i}."), strict=True)
        x = block(x, rope, mod)
        del block
        gc.collect()
    x = model.final_norm_video(x[:, :H * W])
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    return model.final_linear_video(x)[0].contiguous()


velocity = forward(set())
skipped = forward({22})
save_file({"latent": latent[0].flatten(1).transpose(0, 1).contiguous(), "text": text[0].contiguous(), "time": time,
           "velocity": velocity, "velocity_skipping_22": skipped}, out,
          metadata={"size": f"{H}x{W}", "torch": torch.__version__})
print(f"velocity RMS {velocity.pow(2).mean().sqrt():.4f}, skipping 22 moves it {(velocity - skipped).pow(2).mean().sqrt():.4f} -> {out}")
