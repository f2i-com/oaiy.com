"""The starting noise of OpenMOSS's MOSS-Image examples, as their Gradio app draws it. Never used by the runtime.

Usage: python tools/moss_image/noise.py path/to/MOSS-Image/assets/images OUT_DIR
For each example (`NNN/parameters.json`: its seed and size), its noise as `MOSSImagePipeline.__call__` draws it with
the app's `torch.Generator("cpu").manual_seed(seed)`: diffusers' `randn_tensor` of (1, 128, height / 16, width / 16) in
float32 on the CPU. Writes OUT_DIR/NNN.f32 (little-endian f32, channel-major: 128 planes of the latent's rows), for
`moss_image`'s ignored test `the_examples_are_openmosss` in `crates/oaiy-media/src/moss_image/mod.rs`.
"""
import json
from pathlib import Path
import sys

import torch
from diffusers.utils.torch_utils import randn_tensor

examples, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for record in sorted(examples.glob("*/parameters.json")):
    p = json.loads(record.read_text())["parameters"]
    shape = (1, 128, p["height"] // 16, p["width"] // 16)
    noise = randn_tensor(shape, generator=torch.Generator("cpu").manual_seed(p["seed"]), device=torch.device("cpu"), dtype=torch.float32)
    noise[0].numpy().astype("<f4").tofile(out / f"{record.parent.name}.f32")
    print(f"{record.parent.name}: seed {p['seed']}, {p['width']}x{p['height']}")
