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
parser.add_argument('--conditioned-tokens', type=int, default=0)
parser.add_argument('--end-frame', action='store_true')
args = parser.parse_args()
if not 0 <= args.conditioned_tokens < 16:
    parser.error('--conditioned-tokens must be in 0..15')
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
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
timesteps = torch.full((1, 16), .725, device=args.device)
timesteps[:, :args.conditioned_tokens] = 0
prefix = 'i2v-transformer' if args.conditioned_tokens else 'transformer'
hidden_name = 'i2v-hidden.f32'
marker = torch.ones((1, 16, 1), device=args.device)
if args.end_frame:
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
    tools = VideoLatentTools(VideoLatentPatchifier(1), VideoLatentShape(1, 128, 2, 4, 4), 24.)
    pixels = (torch.sin(torch.arange(32 * 128, device=args.device) * .013)
              .reshape(1, 2, 4, 4, 128).permute(0, 4, 1, 2, 3).to(torch.bfloat16))
    state = tools.create_initial_state(args.device, torch.bfloat16, pixels)
    state.denoise_mask[:, :16] = 0
    end = (torch.cos(torch.arange(16 * 128, device=args.device) * .019)
           .reshape(1, 1, 4, 4, 128).permute(0, 4, 1, 2, 3).to(torch.bfloat16))
    state = VideoConditionByKeyframeIndex(end, frame_idx=8, strength=1.).apply_to(state, tools)
    x = state.latent * state.denoise_mask + state.clean_latent * (1 - state.denoise_mask)
    x = x.to(torch.bfloat16)
    positions = state.positions
    timesteps = state.denoise_mask.squeeze(-1) * .725
    marker = state.keyframes_mask
    prefix = 'end-transformer'
    hidden_name = 'end-hidden.f32'
video = Modality(latent=x, sigma=torch.tensor([.725], device=args.device),
                 timesteps=timesteps, positions=positions, context=context, keyframes_mask=marker)
if args.end_frame:
    model.transformer_blocks[0].register_forward_hook(
        lambda m, a, o: o[0].x.float().cpu().numpy().tofile(out / 'end-first-block.f32'))
if args.conditioned_tokens or args.end_frame:
    model.transformer_blocks[-1].register_forward_hook(
        lambda m, a, o: o[0].x.float().cpu().numpy().tofile(out / hidden_name))
with torch.inference_mode():
    result, _ = model(video, None, None)
for name, value in [('input', x), ('context', context), ('output', result)]:
    value.float().cpu().numpy().tofile(out / (prefix + '-' + name + '.f32'))
print('Official transformer RMS', result.float().square().mean().sqrt().item(), flush=True)
