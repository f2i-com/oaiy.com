"""Generate an independent HF vision oracle; testing only, never a runtime dependency."""
import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-deps', type=pathlib.Path)
parser.add_argument('--model', type=pathlib.Path, default=pathlib.Path('models/OrcaSAQ-2-27B/vision'))
parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('target/orca-vision-research'))
args = parser.parse_args()
if args.reference_deps:
    sys.path.insert(0, str(args.reference_deps.resolve()))
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

args.output.mkdir(parents=True, exist_ok=True)
img = Image.new('RGB', (768, 768), 'white')
draw = ImageDraw.Draw(img)
draw.rectangle((60, 90, 330, 350), fill='red')
draw.ellipse((430, 110, 680, 360), fill='blue')
draw.polygon([(120, 650), (300, 400), (470, 650)], fill='green')
try:
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 60)
except OSError:
    font = ImageFont.load_default(size=60)
draw.text((400, 550), 'ORCA 42', fill='black', font=font)
img.save(args.output / 'shapes.png')
pixels = torch.from_numpy(np.asarray(img).copy()).float().permute(2, 0, 1) / 127.5 - 1.0
# Official processor patch ordering: spatial 2x2 groups, duplicated temporal slots.
patches = pixels.unsqueeze(0).repeat(2, 1, 1, 1)
patches = patches.reshape(1, 2, 3, 24, 2, 16, 24, 2, 16)
patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(2304, -1)
config = Qwen3_5VisionConfig(**json.loads((args.model / 'config.json').read_text())['vision_config'])
config._attn_implementation = 'sdpa'
model = Qwen3_5VisionModel(config)
weights = {}
with safe_open(str(args.model / 'model-00001-of-00018.safetensors'), framework='pt') as f:
    for key in f.keys():
        if key.startswith('model.visual.'):
            weights[key.removeprefix('model.visual.')] = f.get_tensor(key).float()
model.load_state_dict(weights, strict=True, assign=True)
del weights
model = model.eval().to('cuda:1')
torch.backends.cuda.matmul.allow_tf32 = False
with torch.inference_mode():
    result = model(patches.to('cuda:1'), grid_thw=torch.tensor([[1, 48, 48]], device='cuda:1'))
    embeddings = result.pooler_output.cpu()
save_file({'pixels': pixels.contiguous(), 'embeddings': embeddings.contiguous()}, str(args.output / 'oracle.safetensors'))
print('HF oracle:', tuple(embeddings.shape), 'finite:', bool(embeddings.isfinite().all()), flush=True)
