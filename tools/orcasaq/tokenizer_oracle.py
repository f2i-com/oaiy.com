"""Generate deterministic token IDs with Hugging Face tokenizers for native QA."""
import json
import pathlib
import random
from tokenizers import Tokenizer

root = pathlib.Path(__file__).resolve().parents[2]
tokenizer = Tokenizer.from_file(str(root / "models/OrcaSAQ-2-27B/tokenizer.json"))
samples = [
    "The capital of France is Paris.", "1234567890", "  hello   world\t\t!",
    "cafe\u0301 = café", "किताब", "中文测试。日本語", "👩‍💻 emoji 🚀",
    "<|im_start|>assistant\n<think>\n", "<tool_call>\n<function=add>\n<parameter=a>\n84\n</parameter>\n</function>\n</tool_call>",
    "\r\n\t x\n\n", "def foo(x):\n    return x + 12345\n", "I'm WE'RE isn't we've",
]
rng = random.Random(3827)
parts = ["hello", "cafe\u0301", "नमस्ते", "汉字", "\u0301", "\n", "\r\n", " ", "   ", "\t", "123456", "42", "!!", "<>/", "😀", "'s", "<|im_end|>"]
samples += ["".join(rng.choices(parts, k=12)) for _ in range(128)]
records = [{"text": s, "ids": tokenizer.encode(s, add_special_tokens=False).ids} for s in samples]
output = pathlib.Path(__file__).with_name("tokenizer-cases.json")
output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8")
print(f"Wrote {len(records)} tokenizer reference cases to {output}")
