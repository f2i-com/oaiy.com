"""Encode a chat prompt to DeepSeek-V4.1 token ids with the reference
encoding (encoding/encoding.py) and the checkpoint's tokenizer — a stopgap
until OAIY has a Rust encoder (docs/DEEPSEEK_V41.md, Phase E).

Usage: python encode.py "Write a haiku about rivers." [--thinking] > prompt.ids
Prints comma-separated ids on one line.
"""

import argparse
import os
import sys

REF = os.environ.get("DSV41_REF", r"E:\deepseek\reference")
sys.path.insert(0, os.path.join(REF, "encoding"))
from encoding import encode_messages  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--model", default=r"E:\deepseek\model")
ap.add_argument("--thinking", action="store_true", help="thinking mode (default: chat, no reasoning)")
a = ap.parse_args()
tok = PreTrainedTokenizerFast.from_pretrained(a.model)
text = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="thinking" if a.thinking else "chat")
print(",".join(str(i) for i in tok.encode(text)))
