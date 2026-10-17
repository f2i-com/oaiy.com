"""Qwen3-VL-4B-Instruct as transformers runs it (bf16, CPU), for oaiy-llm-server to be compared with: two prompts
(text alone; a picture at 768 x 768, 576 tokens, and a question), each one's first next-token top 10 and its greedy reply.
The prompt is made as Qwen3-VL's chat template makes it (the processor needs torchvision; this does not), and the
picture's patches as `vision_reference.py` makes them.

python text_reference.py CHECKPOINT_DIR PICTURE OUT.json"""
import json, sys
import torch
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

ckpt, picture, out = sys.argv[1:4]
torch.set_num_threads(12)
model = Qwen3VLForConditionalGeneration.from_pretrained(ckpt, dtype=torch.bfloat16).eval()
tok = AutoTokenizer.from_pretrained(ckpt)
# the picture at 768 x 768, normalised, as the processor's patches (each 2 x 2 window's four together)
img = Image.open(picture).convert("RGB").resize((768, 768), Image.BICUBIC)
chw = torch.from_numpy((np.asarray(img).astype(np.float32) / 255.0 - 0.5) / 0.5).permute(2, 0, 1)
P, M, T, g = 16, 2, 2, 48
x = chw.unsqueeze(0).repeat(T, 1, 1, 1).reshape(1, T, 3, g // M, M, P, g // M, M, P).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
pixel_values = x.reshape(g * g, 3 * T * P * P).to(torch.bfloat16)
grid = torch.tensor([[1, g, g]])
IMAGE = "<|vision_start|>" + "<|image_pad|>" * 576 + "<|vision_end|>"
cases = {
    "text": "<|im_start|>user\nName three primary colours, then say which one the sky usually is.<|im_end|>\n<|im_start|>assistant\n",
    "picture": "<|im_start|>user\n" + IMAGE + "What words are printed on the mug, and what colour is the mug? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n",
}
import os
results = json.load(open(out)) if os.path.exists(out) else {}
for name, text in cases.items():
    if name in results:
        continue
    ids = tok(text, add_special_tokens=False)["input_ids"]
    inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
    if "<|image_pad|>" in text:
        inputs.update(pixel_values=pixel_values, image_grid_thw=grid, mm_token_type_ids=torch.tensor([[1 if t == 151655 else 0 for t in ids]]))
    print(name, "prompt tokens", len(ids), "image tokens", ids.count(151655))
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
        top = torch.topk(logits, 10)
        gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    reply_ids = gen[0, len(ids):].tolist()
    results[name] = {
        "prompt": tok.decode(ids),
        "prompt_ids": ids,
        "top_ids": top.indices.tolist(),
        "top_logits": top.values.tolist(),
        "top_tokens": [tok.decode([i]) for i in top.indices.tolist()],
        "reply_ids": reply_ids,
        "reply": tok.decode(reply_ids, skip_special_tokens=True),
    }
    print(name, "top", results[name]["top_tokens"][:5], "reply:", results[name]["reply"])
    json.dump(results, open(out, "w"), indent=1)
