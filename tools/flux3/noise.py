"""The starting noise of BFL's FLUX 3 Image Klein examples, as their pipeline draws it. Never used by the runtime.

Usage: python tools/flux3/noise.py path/to/flux-3-image-klein OUT_DIR [examples/distilled]
For each example record (its seed, its references), the picture's size as `flux3/generate.py` gives it (the first
reference prepared, or 1024 x 1024) and its noise as `Pipeline.generate` draws it: torch.randn of (h w, 128) on the
CPU from the seed, held as bfloat16. Writes OUT_DIR/<example>.f32 (little-endian f32, h w 128 of them), for
`flux3`'s ignored test `the_examples_are_bfls` in `crates/oaiy-media/src/flux3/mod.rs`.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
folder = repo / (sys.argv[3] if len(sys.argv) > 3 else "examples/distilled")
sys.path.insert(0, str(repo / "src"))
from flux3.pipeline import default_size, prepare_multi_reference, prepare_reference  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
for record in sorted(folder.glob("*.json")):
    j = json.loads(record.read_text())
    images = [Image.open(repo / r) for r in j.get("references") or []]
    references = [prepare_multi_reference(i) for i in images] if len(images) > 1 else [prepare_reference(i)[0] for i in images]
    width, height = default_size(references)
    noise = torch.randn((height // 16 * (width // 16), 128), generator=torch.Generator().manual_seed(j["seed"]))
    noise.to(torch.bfloat16).float().numpy().astype("<f4").tofile(out / f"{record.stem}.f32")
    print(f"{record.stem}: seed {j['seed']}, {width}x{height}, {len(images)} references")
