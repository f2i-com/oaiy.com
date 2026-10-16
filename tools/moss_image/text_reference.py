"""CPU-only oracle for MOSS-Image's text conditioning. Never used by the native runtime.

Usage: python tools/moss_image/text_reference.py QWEN3.5-9B_DIR TOKENIZER_DIR CONFIG_JSON OUT.safetensors
Encodes two prompts (one of them the empty negative prompt) as `MOSSImagePipeline.encode_prompt` does: the chat
template of one user turn, the assistant's turn opened with thinking on, then Qwen3.5-9B's last hidden state (after
its final norm), through Hugging Face transformers in float32. The 9B's 37 GB of float32 never sit in memory at
once: its layers are made and run one at a time from the checkpoint's own tensors (Hugging Face's
`model.language_model.` names, or `language_model.` as MOSS-Image's `text_encoder` keeps them). Writes each prompt's
tokens, its final states (`[n, 4096]`) and its states after four layers, normed by the final norm (what the
encoder cut to four layers gives), for the ignored tests in `crates/oaiy-media/src/moss_image/webgpu/text.rs`.
"""
import gc
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding

weights, tokenizer_dir, config_json, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
PROMPTS = [
    "A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator "
    "door is handwritten in Chinese:\n\"记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\"",
    "",
]
EARLY = 4

config = Qwen3_5TextConfig(**json.loads(Path(config_json).read_text())["text_config"])
config._attn_implementation = "sdpa"
files = sorted(weights.glob("*.safetensors"))
where = {}
for f in files:
    with safe_open(f, "pt") as h:
        for k in h.keys():
            where[k] = f
prefix = next(p for p in ("model.language_model.", "language_model.") if f"{p}embed_tokens.weight" in where)


def tensors(start):
    out = {}
    for k, f in where.items():
        if k.startswith(start):
            with safe_open(f, "pt") as h:
                out[k[len(start):]] = h.get_tensor(k).float()
    return out


tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
ids = []
for p in PROMPTS:
    text = tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids.append(tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"][0])
with safe_open(where[f"{prefix}embed_tokens.weight"], "pt") as h:
    embed = h.get_tensor(f"{prefix}embed_tokens.weight")
states = [embed[i].float()[None] for i in ids]
del embed
norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
norm.load_state_dict(tensors(f"{prefix}norm."))
rotary = Qwen3_5TextRotaryEmbedding(config=config)
early = []
with torch.inference_mode():
    for i in range(config.num_hidden_layers):
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(config, i)
        layer = layer.to_empty(device="cpu").float()
        layer.load_state_dict(tensors(f"{prefix}layers.{i}."), strict=True)
        layer.eval()
        for j, x in enumerate(states):
            n = x.shape[1]
            positions = torch.arange(n).view(1, 1, -1).expand(3, 1, -1)
            states[j] = layer(x, position_embeddings=rotary(x, positions), attention_mask=None, position_ids=torch.arange(n)[None])
        if i + 1 == EARLY:
            early = [norm(x)[0].clone() for x in states]
        print(f"layer {i} ({config.layer_types[i]})", flush=True)
        del layer
        gc.collect()
    final = [norm(x)[0] for x in states]
saved = {}
for j, (t, e, f) in enumerate(zip(ids, early, final)):
    saved[f"prompt{j}.tokens"] = t.float()
    saved[f"prompt{j}.early"] = e.contiguous()
    saved[f"prompt{j}.final"] = f.contiguous()
save_file(saved, out, metadata={"prompts": json.dumps(PROMPTS, ensure_ascii=False), "early_layers": str(EARLY),
                                "transformers": __import__("transformers").__version__, "torch": torch.__version__})
for j, f in enumerate(final):
    print(f"prompt {j}: {len(ids[j])} tokens, final RMS {f.pow(2).mean().sqrt():.3f}, largest {f.abs().max():.1f}")
print("->", out)
