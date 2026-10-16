"""CPU-only reference for MiniCPM-V 4.7: transformers' own model and processor. Never used by the native runtime.

Usage: python tools/minicpmv/reference.py MODEL_DIR IMAGE OUT_DIR [16x|4x] [PROMPT]
MODEL_DIR is the original release (openbmb/MiniCPM-V-4.7-1B: config, tokenizer, model.safetensors); needs transformers
with MiniCPM-V 4.7 (5.18 or later). Runs it in float32 on one picture and writes OUT_DIR/reference.safetensors:

- `tile.K`: each of the picture's tiles as the processor gives the vision tower its pixels ([3, H, W], normalised),
  the overview first, then the slices row by row; `target_sizes` their patch grids;
- `features.K`: each tile's embeddings out of the merger, as the language model receives them;
- `input_ids`, `mm_token_type_ids` and `position_ids` ([3, S]: the canvas M-RoPE's T, H and W of every token);
- `logits`: the next token's logits after the prompt; `text_logits` likewise after a text-only prompt;

and OUT_DIR/reference.json: the prompt as text, the grid, and the greedy reply's ids and text.
"""
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, MiniCPMV4_7ForConditionalGeneration

model_dir, image_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "16x"
prompt = sys.argv[5] if len(sys.argv) > 5 else "Describe this picture in two sentences."
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
processor = AutoProcessor.from_pretrained(model_dir)
model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
image = Image.open(image_path).convert("RGB")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
inputs = processor(images=[image], text=[text], downsample_mode=mode, return_tensors="pt")
ids = inputs["input_ids"]
sizes = inputs["target_sizes"]
tensors = {
    "input_ids": ids[0].to(torch.int64),
    "mm_token_type_ids": inputs["mm_token_type_ids"][0].to(torch.int64),
    "target_sizes": sizes.to(torch.int64),
}

# the tiles' pixels, unpacked from the NaViT packing (`reshape_by_patch`: [3, 14, n * 14] a tile, n its patches)
packed = inputs["pixel_values"][0]
at = 0
for k, (h, w) in enumerate(sizes.tolist()):
    n = h * w
    seg = packed[:, :, at * 14:(at + n) * 14]
    at += n
    tile = seg.reshape(3, 14, n, 14).permute(0, 2, 1, 3).reshape(3, h, w, 14, 14).permute(0, 1, 3, 2, 4).reshape(3, h * 14, w * 14)
    tensors[f"tile.{k}"] = tile.contiguous().float()

with torch.no_grad():
    feats = model.get_image_features(inputs["pixel_values"], sizes, downsample_mode=mode).pooler_output
    for k, f in enumerate(feats):
        tensors[f"features.{k}"] = f.contiguous().float()
    position_ids, _ = model.model.get_rope_index(ids, mm_token_type_ids=inputs["mm_token_type_ids"], target_sizes=sizes, downsample_mode=mode)
    tensors["position_ids"] = position_ids[:, 0, :].contiguous().to(torch.int64)
    logits = model(**inputs, downsample_mode=mode).logits[0, -1]
    tensors["logits"] = logits.contiguous().float()
    generated = model.generate(**inputs, downsample_mode=mode, max_new_tokens=48, do_sample=False)
    reply = generated[0, ids.shape[1]:]
    text_prompt = processor.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer in one word."}],
                                                add_generation_prompt=True, enable_thinking=False, tokenize=False)
    text_ids = processor.tokenizer(text_prompt, return_tensors="pt")["input_ids"]
    tensors["text_input_ids"] = text_ids[0].to(torch.int64)
    tensors["text_logits"] = model(input_ids=text_ids).logits[0, -1].contiguous().float()

save_file(tensors, out / "reference.safetensors", metadata={"source": "transformers MiniCPMV4_7ForConditionalGeneration (tools/minicpmv/reference.py)", "mode": mode})
json.dump({
    "prompt": processor.tokenizer.decode(ids[0]),
    "mode": mode,
    "image": str(image_path),
    "image_size": list(image.size),
    "grid": inputs.get("grids", None),
    "reply_ids": reply.tolist(),
    "reply": processor.tokenizer.decode(reply, skip_special_tokens=True),
}, open(out / "reference.json", "w"), indent=1, default=str)
print("tiles", sizes.tolist(), "tokens", ids.shape[1], "features", [tuple(f.shape) for f in feats])
print("reply:", processor.tokenizer.decode(reply, skip_special_tokens=True))
