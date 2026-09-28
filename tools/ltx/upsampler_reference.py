"""Reference activations for OAIY's LTX latent upsampler.

The official LatentUpsampler (ltx_core.model.upsampler: 3D, x2 spatial) on a
fixed random latent, in F32 on the CPU. Writes `input` and `output` to a
safetensors file for `ltx::upsampler::tests::upsampler_matches_reference`.

    python tools/ltx/upsampler_reference.py --weights <spatial-upscaler.safetensors> --out upsampler_ref.safetensors
"""
import argparse
import json

import torch
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()


# The official modules (ltx_core/model/upsampler), for dims=3 spatial x2.
class ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(32, channels)
        self.conv2 = torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class PixelShuffle2D(torch.nn.Module):
    def forward(self, x):
        return rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2)


class LatentUpsampler(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, blocks):
        super().__init__()
        self.initial_conv = torch.nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, mid_channels)
        self.initial_activation = torch.nn.SiLU()
        self.res_blocks = torch.nn.ModuleList([ResBlock(mid_channels) for _ in range(blocks)])
        self.upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
            PixelShuffle2D(),
        )
        self.post_upsample_res_blocks = torch.nn.ModuleList([ResBlock(mid_channels) for _ in range(blocks)])
        self.final_conv = torch.nn.Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent):
        b, _, f, _, _ = latent.shape
        x = self.initial_activation(self.initial_norm(self.initial_conv(latent)))
        for block in self.res_blocks:
            x = block(x)
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.upsampler(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        for block in self.post_upsample_res_blocks:
            x = block(x)
        return self.final_conv(x)


with safe_open(args.weights, 'pt') as f:
    config = json.loads(f.metadata()['config'])
    state = {k: f.get_tensor(k).float() for k in f.keys()}
assert config['dims'] == 3 and config['spatial_upsample'] and not config['temporal_upsample']
model = LatentUpsampler(config['in_channels'], config['mid_channels'], config['num_blocks_per_stage'])
model.load_state_dict(state)
model.eval()

torch.manual_seed(0)
latent = torch.randn(1, 128, 3, 4, 5)
with torch.no_grad():
    out = model(latent)
save_file({'input': latent.contiguous(), 'output': out.contiguous()}, args.out)
print('output', tuple(out.shape), 'rms', out.pow(2).mean().sqrt().item())
