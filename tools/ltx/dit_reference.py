"""Offline reference from the official Lightricks video block; never used at runtime."""
import argparse
import json
import math
import pathlib
import sys

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--source', required=True, help='Official LTX-2 repository checkout')
parser.add_argument('--reference-deps', help='Optional isolated Python dependency directory')
parser.add_argument('--output', default='target/ltx-golden')
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
sys.path.insert(0, str(pathlib.Path(args.source) / 'packages/ltx-core/src'))
if args.reference_deps:
    sys.path.insert(0, args.reference_deps)

from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.model.transformer.rope import precompute_freqs_cis, generate_freq_grid_np, generate_freq_grid_pytorch

path = pathlib.Path(args.checkpoint)
if path.suffix != '.safetensors':
    raise ValueError('Use a complete safetensors checkpoint')
weights = safe_open(str(path), framework='pt', device=args.device)
prefix = 'model.diffusion_model.transformer_blocks.0.'
with torch.device('meta'):
    model = BasicAVTransformerBlock(video=TransformerConfig(
        dim=4096, heads=32, d_head=128, context_dim=4096,
        apply_gated_attention=True, cross_attention_adaln=True,
        ff_bias=prefix + 'ff.net.2.bias' in weights.keys()))
state = {k: weights.get_tensor(prefix + k) for k in model.state_dict()}
model.load_state_dict(state, strict=True, assign=True)

def values(shape, frequency):
    return torch.sin(torch.arange(math.prod(shape), device=args.device,
                                  dtype=torch.float32) * frequency).reshape(shape).to(torch.bfloat16)

x = values((1, 16, 4096), .013)
context = values((1, 8, 4096), .021)
modulation = values((1, 9, 4096), .007) * .1
prompt = values((1, 2, 4096), .023) * .1
positions = torch.tensor([[.5 / 24, (y + .5) * 32, (x + .5) * 32]
                          for y in range(4) for x in range(4)],
                         device=args.device, dtype=torch.float32).T.unsqueeze(0)
config = json.loads(weights.metadata()['config'])['transformer']
frequency_grid = generate_freq_grid_np if config.get('frequencies_precision') == 'float64' else generate_freq_grid_pytorch
rope = precompute_freqs_cis(positions, 4096, torch.bfloat16, freq_grid_generator=frequency_grid)
video = TransformerArgs(x=x, context=context, context_mask=None,
                        timesteps=modulation.reshape(1, 1, -1), embedded_timestep=None,
                        positional_embeddings=rope, cross_positional_embeddings=None,
                        cross_scale_shift_timestep=None, cross_gate_timestep=None,
                        enabled=True, prompt_timestep=prompt.reshape(1, 1, -1))
with torch.inference_mode():
    result, _ = model(video, None)
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
for name, tensor in [('input', x), ('context', context), ('modulation', modulation),
                     ('prompt', prompt), ('output', result.x)]:
    tensor.float().cpu().numpy().tofile(out / ('dit-' + name + '.f32'))
print('Official video block RMS', result.x.float().square().mean().sqrt().item())
