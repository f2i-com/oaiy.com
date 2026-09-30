"""Read native PNGs and verify a same-seed zero-strength control.

Usage: python tools/klein/compare.py path/to/task-storage
Requires Pillow and NumPy only for image analysis; never performs inference.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

root = Path(sys.argv[1]).resolve()
results = {name: json.loads((root / f"klein-{name}.result.json").read_text(encoding="utf-8-sig"))
    for name in ["baseline", "style", "zero"]}
images = {}
records = {}
for name, r in results.items():
    record = r["data"][0]
    image_path = Path(record["path"])
    with Image.open(image_path) as im:
        images[name] = np.asarray(im.convert("RGB"), dtype=np.int16)
    records[name] = {**record, "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "total_seconds": r["seconds"], "residency": r["residency"]}
    assert images[name].shape == (512, 512, 3)
    for key in ["prompt", "seed", "width", "height", "steps", "cfg", "transformer", "variant"]:
        assert record[key] == results["baseline"]["data"][0][key], f"mismatched {key}"
assert np.array_equal(images["baseline"], images["zero"]), "zero-strength control differs from clean baseline"
delta = np.abs(images["baseline"] - images["style"])
assert delta.max() > 0, "LoRA has no effect on pixels"
report = {"runs": records, "baseline_equals_zero_strength_pixels": True,
    "baseline_equals_zero_strength_png": records["baseline"]["sha256"] == records["zero"]["sha256"],
    "style_changed_pixels": int(np.any(delta > 0, axis=2).sum()),
    "style_changed_pixel_fraction": float(np.any(delta > 0, axis=2).mean()),
    "mean_absolute_channel_difference": float(delta.mean()),
    "max_channel_difference": int(delta.max())}
report_path = root / "klein-comparison.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k:v for k,v in report.items() if k != "runs"}, indent=2))
print("Evidence:", report_path)
