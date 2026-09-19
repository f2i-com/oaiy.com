"""Write golden token ids for the Rust DeepSeek-V4.1 tokenizer
(crates/dsv41/src/tokenizer.rs), from the reference Hugging Face tokenizer.

The corpus mixes this repo's own sources and docs (code-heavy, like a coding
harness's prompts), the reference encoding's test prompts, hand-written edge
cases, and random Unicode strings.

Usage: python tools/dsv41/tokenizer_golden.py [--out E:/deepseek/golden/tokenizer_cases.json]
"""

import argparse
import glob
import json
import os
import random

from transformers import PreTrainedTokenizerFast

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"E:\deepseek\model")
ap.add_argument("--ref", default=r"E:\deepseek\reference")
ap.add_argument("--out", default=r"E:\deepseek\golden\tokenizer_cases.json")
a = ap.parse_args()

tok = PreTrainedTokenizerFast.from_pretrained(a.model)
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

texts = []

# this repo: code and docs, cut into chunks of a few KB
for pattern in ["crates/**/*.rs", "crates/**/*.cu", "docs/*.md", "*.md", "tools/**/*.py", "**/*.toml"]:
    for path in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
        if os.sep + "target" + os.sep in path:
            continue
        with open(path, encoding="utf-8") as f:
            s = f.read()
        for i in range(0, len(s), 4000):
            texts.append(s[i:i + 4000])

# the reference encoding's rendered prompts (special tokens, DSML, CJK)
for path in sorted(glob.glob(os.path.join(a.ref, "encoding", "tests", "test_output_*.txt"))):
    with open(path, encoding="utf-8") as f:
        texts.append(f.read())

texts += [
    "", " ", "  ", "\n", "\r\n", "\t", "a", "Hello world", " Hello  world ", "hello   123",
    "1", "12", "123", "1234567", "3.14159", "x=1e-5; y=-42", "0xFF 0b1010 1_000_000",
    "  \n\n  x", "\n\n\n", "a\n\n\nb", "a \n b", "a  \n  b", "foo\r\nbar\r\n", " \t \n",
    "...!!!\n\n", "!!", " !", "(){}[]", "foo_bar.baz()", "self.x += 1", "#include <stdio.h>",
    "It's I'm we're they've you'll he'd", "don't", "'s", "\"quoted\"", "`code`",
    "https://example.com/a/b?c=d&e=f#g", "user@example.com", "C:\\Users\\x\\file.txt",
    '{"a": [1, 2.5, "s"], "b": null}', "<div class=\"x\">hi</div>", "a->b::c<T>",
    "中文测试，你好世界！", "日本語のテキスト、カタカナ、ひらがな。", "한국어 텍스트", "混合English中文123",
    "Ελληνικά", "Русский текст", "العربية", "हिन्दी", "ภาษาไทย", "עברית",
    "e\u0301", "\u0301abc", "a\u200db", "👍🏽 emoji 😀🎉", "👨‍👩‍👧", "\u3000全角空格\u3000",
    "\u00a0nbsp\u00a0", "\x00\x07ctrl", "tab\tsep\tvalues", "trailing   ", "   leading",
    "ℕ ∑ ∫ √ ≤ ≥ ≠ ∞", "½ ¼ ² ³ ① Ⅳ", "٣٤٥ ১২৩", "$100 €50 ¥30", "a1b2c3", "A.B.C",
    "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>ok<｜end▁of▁sentence｜>",
    "<think>reasoning</think>answer", "<｜DSML｜ calls>", "<｜Use", "<|EOT|>", "<｜User｜><｜User｜>",
    "text<｜deepseek_image｜>text", "<tool_result>{}</tool_result>", "<dsml:x></dsml:x>",
]

# random strings over a pool that exercises every pre-tokenizer class
pool = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    + list(" \t\n\r  ")
    + list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    + list("中文日本語カタナひらが한국어ÄéñüßØ\u0301\u0308\u200d😀👍€£©®™°±×÷→⇒∀∃")
    + list("٠١٢३४५①½²\u3000\u00a0\u2028\u0085")
    + ["<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "｜DSML｜", "<｜end▁of▁sentence｜>"]
)
rng = random.Random(1234)
for _ in range(3000):
    n = rng.randint(1, 60)
    texts.append("".join(rng.choice(pool) for _ in range(n)))

cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in texts]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False)
print(f"{len(cases)} cases, {sum(len(c['ids']) for c in cases)} tokens -> {a.out}")
