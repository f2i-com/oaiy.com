"""CPU-only oracle for FLUX 3 Image Klein's text conditioning. Never used by the native runtime.

Usage: python tools/flux3/text_reference.py TEXT_ENCODER_DIR OUT.safetensors [LAYERS]
Encodes one prompt as BFL's `flux3/text_encoder.py` does (Qwen3-VL-4B-Instruct through Hugging Face transformers:
its chat template, right-padded to the next multiple of 80 tokens, the padding's positions all 1), in float32, with
the model cut to the taps up to LAYERS (8: the taps 4 and 8; every later layer is the same code, and float32's whole
32 would want some 15 GB) and one layer more: the last of the hidden states transformers gives is after the final
norm, which no tap of the whole model is. Writes the prompt, its token count and the tapped states side by side (`[L, 2560 x
taps]`), for `flux3_text`'s ignored test in `crates/oaiy-media/src/klein_text_wgpu.rs`.
"""
import math
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

path, out = sys.argv[1], sys.argv[2]
layers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
taps = [k for k in (4, 8, 12, 16, 20, 24, 28, 32) if k <= layers]
PROMPT = ("A cosy mountain cabin at dusk beside a frozen lake, a hand-painted sign above the door that reads "
          "\"WebGPU Lodge\", warm light in the windows, pine trees dusted with snow, a thin column of smoke rising "
          "from the chimney, cinematic wide shot.")
PAD_MULTIPLE, MAX_LENGTH = 80, 4096

config = AutoConfig.from_pretrained(path)
config.text_config.num_hidden_layers = layers + 1
model = Qwen3VLForConditionalGeneration.from_pretrained(path, config=config, dtype=torch.float32).eval()
processor = AutoProcessor.from_pretrained(path)
processor.tokenizer.padding_side = "right"
# flux3/text_encoder.py's encode, in float32 and over the layers kept
text = processor.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
tokenizer = processor.tokenizer
real = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"].shape[1]
length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
mask = inputs["attention_mask"]
positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
with torch.inference_mode():
    output = model(input_ids=inputs["input_ids"], attention_mask=mask, position_ids=positions[None].expand(4, -1, -1),
                   output_hidden_states=True, use_cache=False)
stacked = torch.stack([output.hidden_states[k] for k in taps], dim=1)
tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).float().contiguous()
save_file({"tokens": tokens, "real": torch.tensor([real], dtype=torch.float32), "taps": torch.tensor(taps, dtype=torch.float32)}, out,
          metadata={"prompt": PROMPT, "transformers": __import__("transformers").__version__, "torch": torch.__version__})
print(f"{real} tokens padded to {length}, taps {taps}: {tuple(tokens.shape)} -> {out}")
