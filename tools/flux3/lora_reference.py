"""CPU-only reference for FLUX 3 Image Klein's LoRAs: diffusers' own. Never used by the native runtime.

Usage: python tools/flux3/lora_reference.py crates/oaiy-media/tests/flux3/tiny.safetensors OUT_DIR
Needs diffusers with FLUX 3 (0.41, `Flux3Transformer2DModel`) and peft. Builds diffusers' transformer at the tiny
fixture's sizes (tools/flux3/reference.py) from the fixture's weights, converted as diffusers converts BFL's checkpoint,
and checks it gives the fixture's velocity (BFL's DiT's). Then two LoRAs, as trainers save them:

- diffusers.safetensors: made with PEFT and saved by diffusers' `Flux3LoraLoaderMixin` (diffusers' names, its scale in
  the file's metadata: `lora_alpha` over the rank, the `to_k` projections' own alpha), adapting every linear layer, the
  final modulation (whose halves diffusers keeps the other way round from BFL) among them;
- bfl.safetensors: BFL's names as ai-toolkit saves them (`diffusion_model.` prefix, lora_A/lora_B, no alphas), the
  blocks' projections, loaded through diffusers' own conversion of such a file.

Writes them to OUT_DIR, and expected.safetensors: the velocity diffusers gives the fixture's step (its references
included) with both applied, at strengths 0.7 and 1.3. Never opens production weights.
"""
import json
from pathlib import Path
import sys

import torch
from diffusers import Flux3Transformer2DModel
from diffusers.loaders.lora_pipeline import Flux3LoraLoaderMixin
from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import load_file, save_file

fixture, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(20261016)
tensors = load_file(fixture)
x = {k.removeprefix("fixture."): v for k, v in tensors.items() if k.startswith("fixture.")}
weights = {k: v.float() for k, v in tensors.items() if not k.startswith("fixture.")}
model = Flux3Transformer2DModel(
    in_channels=128, cond_in_channels=128, num_layers=1, num_single_layers=2, attention_head_dim=16,
    num_attention_heads=2, joint_attention_dim=48, timestep_channels=256, mlp_ratio=1.5, axes_dims_rope=(4, 4, 4, 4),
    rope_theta=10000, eps=1e-6, flow_map=True,
).float().eval()
model.load_state_dict(convert_flux3_transformer_checkpoint_to_diffusers(dict(weights)), strict=True)
t, target = x["times"].tolist()


def velocity():
    with torch.no_grad():
        return model(
            hidden_states=x["image"][None], encoder_hidden_states=x["text"][None], timestep=torch.tensor([t]),
            img_ids=x["image_ids"], txt_ids=x["text_ids"], image_cond_hidden_states=x["cond"][None],
            image_cond_ids=x["cond_ids"], target_timestep=torch.tensor([target]),
        ).sample[0]


def compare(a, b):
    cosine = torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()
    return f"cosine {cosine:.7f}, the worst difference {(a - b).abs().max().item():.2e} of an RMS {b.pow(2).mean().sqrt().item():.3f}"


plain = velocity()
print("diffusers against BFL's DiT, without LoRAs:", compare(plain, x["velocity"]))

# a LoRA made with PEFT, saved by diffusers: every linear layer, `to_k`'s own alpha
config = LoraConfig(
    r=4, lora_alpha=8, alpha_pattern={"to_k": 2}, init_lora_weights="gaussian",
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "linear_in", "linear_out", "linear", "linear_1", "linear_2",
                    "x_embedders.image", "x_embedders.image_cond", "context_embedder", "proj_out"],
)
model.add_adapter(config, adapter_name="diffusers")
with torch.no_grad():
    for name, p in model.named_parameters():
        if ".lora_B." in name:
            # (gaussian's B starts at zero: this large, the two adapters turn the velocity some 60 degrees)
            p.copy_(torch.randn_like(p) * 0.04)
adapted = sorted({n.split(".lora_")[0] for n, _ in model.named_parameters() if ".lora_A." in n})
assert "norm_out.linear" in adapted and "context_embedder" in adapted, adapted
Flux3LoraLoaderMixin.save_lora_weights(
    out, transformer_lora_layers=get_peft_model_state_dict(model, adapter_name="diffusers"),
    transformer_lora_adapter_metadata=model.peft_config["diffusers"].to_dict(), weight_name="diffusers.safetensors",
)

# a LoRA in BFL's names as ai-toolkit saves them, loaded as diffusers loads such a file
bfl = {}
for module, (rows, cols) in {
    "content_mode_blocks.image.0.q_proj": (32, 32), "content_mode_blocks.image_cond.0.mlp_in": (96, 32),
    "txt_mode_blocks.0.attn_out": (32, 32), "single_blocks.0.k_proj": (32, 32), "single_blocks.1.mlp_out": (32, 48),
    "single_blocks.1.v_proj": (32, 32),
}.items():
    bfl[f"diffusion_model.{module}.lora_A.weight"] = torch.randn(4, cols) / cols ** 0.5
    bfl[f"diffusion_model.{module}.lora_B.weight"] = torch.randn(rows, 4) * 0.1
save_file(bfl, out / "bfl.safetensors", metadata={"format": "pt"})
model.load_lora_adapter(Flux3LoraLoaderMixin.lora_state_dict(str(out / "bfl.safetensors")), prefix="transformer", adapter_name="bfl")
model.set_adapters(["diffusers", "bfl"], weights=[0.7, 1.3])
both = velocity()
print("with both LoRAs, against without:", compare(plain, both))
save_file({"velocity": both.contiguous()}, out / "expected.safetensors", metadata={
    "strengths": json.dumps({"diffusers.safetensors": 0.7, "bfl.safetensors": 1.3}),
    "source": "diffusers Flux3Transformer2DModel with peft LoRA adapters (tools/flux3/lora_reference.py)",
})
