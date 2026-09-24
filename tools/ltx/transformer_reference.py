"""Compare the complete native video transformer with the official LTX model."""
import argparse
import json
import pathlib
import sys

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--reference-deps')
parser.add_argument('--output', default='target/ltx-golden')
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
sys.path.insert(0, str(pathlib.Path(args.source) / 'packages/ltx-core/src'))
if args.reference_deps:
    sys.path.insert(0, args.reference_deps)
from ltx_core.model.transformer.model_configurator import LTXVideoOnlyModelConfigurator
from ltx_core.model.transformer.modality import Modality

path = pathlib.Path(args.checkpoint)
if path.suffix != '.safetensors':
    raise ValueError('Use a complete safetensors checkpoint')
weights = safe_open(str(path), framework='pt', device=args.device)
metadata = weights.metadata()
metadata['config'] = json.loads(metadata['config'])
with torch.device('meta'):
    model = LTXVideoOnlyModelConfigurator.from_metadata(metadata)
state = {key: weights.get_tensor('model.diffusion_model.' + key)
         for key in model.state_dict()}
model.load_state_dict(state, strict=True, assign=True)
print('Loaded official video-only transformer', flush=True)
x = torch.sin(torch.arange(16 * 128, device=args.device) * .013).reshape(1, 16, 128).to(torch.bfloat16)
context = torch.sin(torch.arange(8 * 4096, device=args.device) * .021).reshape(1, 8, 4096).to(torch.bfloat16)
positions = torch.tensor([[[0., 1. / 24], [y * 32, (y + 1) * 32], [x * 32, (x + 1) * 32]]
                          for y in range(4) for x in range(4)],
                         device=args.device, dtype=torch.float32).permute(1, 0, 2).unsqueeze(0)
video = Modality(latent=x, sigma=torch.tensor([.725], device=args.device),
                 timesteps=torch.full((1, 16), .725, device=args.device),
                 positions=positions, context=context)
with torch.inference_mode():
    result, _ = model(video, None, None)
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
for name, value in [('input', x), ('context', context), ('output', result)]:
    value.float().cpu().numpy().tofile(out / ('transformer-' + name + '.f32'))
print('Official transformer RMS', result.float().square().mean().sqrt().item(), flush=True)
