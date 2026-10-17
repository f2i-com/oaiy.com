"""Qwen3-VL-4B's vision tower, as transformers runs it (float32, its weights alone), on one picture at 768 x 768:
the picture as the tower takes it ([3, 768, 768], normalised), its 576 soft tokens and its 3 DeepStack features
([576, 2560] each), saved for the Rust side to compare with.

python vision_reference.py CHECKPOINT_DIR PICTURE OUT.safetensors"""
import json, sys
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

ckpt, picture, out = sys.argv[1:4]
cfg = json.load(open(f"{ckpt}/config.json"))
vcfg = Qwen3VLVisionConfig(**cfg["vision_config"])
vcfg._attn_implementation = "eager"
model = Qwen3VLVisionModel(vcfg).eval()
index = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
state = {}
for shard in sorted(set(f for k, f in index.items() if k.startswith("model.visual."))):
    with safe_open(f"{ckpt}/{shard}", "pt") as f:
        for k in f.keys():
            if k.startswith("model.visual."):
                state[k[len("model.visual."):]] = f.get_tensor(k).float()
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing", missing, "unexpected", unexpected)
model = model.float()

# the picture at 768 x 768 (bicubic), normalised as the processor does (mean 0.5, std 0.5)
img = Image.open(picture).convert("RGB").resize((768, 768), Image.BICUBIC)
arr = (np.asarray(img).astype(np.float32) / 255.0 - 0.5) / 0.5          # [768, 768, 3]
chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()               # [3, 768, 768]

# the processor's patches: [grid_t * grid_h * grid_w, C * T * P * P], each 2 x 2 window's four together
P, M, T = 16, 2, 2
gh = gw = 768 // P
x = chw.unsqueeze(0).repeat(T, 1, 1, 1)                                  # [T, C, H, W] (a still: the frame twice)
x = x.reshape(1, T, 3, gh // M, M, P, gw // M, M, P)
x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)                                 # [t, bh, bw, mh, mw, C, T, P, P]
patches = x.reshape(gh * gw, 3 * T * P * P)
grid = torch.tensor([[1, gh, gw]])
with torch.no_grad():
    o = model(patches, grid_thw=grid)
tokens = o.pooler_output
ds = torch.stack(o.deepstack_features)
print("tokens", tuple(tokens.shape), "deepstack", tuple(ds.shape), "rms", tokens.pow(2).mean().sqrt().item())
save_file({"image": chw.contiguous(), "tokens": tokens.contiguous(), "deepstack": ds.contiguous()}, out)
