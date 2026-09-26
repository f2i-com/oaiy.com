"""Official LTX-2 audio reference activations for offline Rust verification.

Parts:
  decode       audio VAE decoder, vocoder and bandwidth extension on a fixed latent
               (float32, as the worker runs them).
  transformer  one audio-video block and the complete AV transformer on a tiny
               clip, plus the audio text connector (needs --checkpoint).

Outputs raw little-endian float32 files under --output.
"""
import argparse
import json
import pathlib
import sys

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--part', choices=['decode', 'transformer'], required=True)
parser.add_argument('--audio-vae', help='Audio VAE + vocoder safetensors (decode)')
parser.add_argument('--checkpoint', help='LTX 2.5 transformer safetensors (transformer)')
parser.add_argument('--source', required=True, help='Official LTX-2 repository checkout')
parser.add_argument('--reference-deps', help='Optional isolated Python dependency directory')
parser.add_argument('--output', default='target/ltx-golden')
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
sys.path.insert(0, str(pathlib.Path(args.source) / 'packages/ltx-core/src'))
if args.reference_deps:
    sys.path.insert(0, args.reference_deps)
# Strict float32: TF32 convolutions compound across the vocoder's ~100 convs.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)


def dump(name, t):
    t.detach().float().contiguous().cpu().numpy().tofile(out / name)
    print(f'{name}: {tuple(t.shape)} RMS {t.float().square().mean().sqrt().item():.6f}', flush=True)


def pattern(shape, step=0.031, device=None):
    n = 1
    for d in shape:
        n *= d
    return torch.sin(torch.arange(n, device=device or args.device, dtype=torch.float32) * step).reshape(shape)


def decode():
    from ltx_core.model.audio_vae.model_configurator import AudioDecoderConfigurator, VocoderConfigurator
    f = safe_open(args.audio_vae, framework='pt', device=args.device)
    metadata = f.metadata()
    metadata['config'] = json.loads(metadata['config'])
    decoder = AudioDecoderConfigurator.from_metadata(metadata)
    vocoder = VocoderConfigurator.from_metadata(metadata)
    dec_state, voc_state = {}, {}
    for k in f.keys():
        t = f.get_tensor(k).float()
        if k.startswith('audio_vae.decoder.'):
            dec_state[k.removeprefix('audio_vae.decoder.')] = t
        elif k.startswith('audio_vae.per_channel_statistics.'):
            dec_state['per_channel_statistics.' + k.removeprefix('audio_vae.per_channel_statistics.')] = t
        elif k.startswith('vocoder.'):
            voc_state[k.removeprefix('vocoder.')] = t
    print('decoder', decoder.load_state_dict(dec_state, strict=True), flush=True)
    # The BWE resampler filter is computed, not stored.
    missing = vocoder.load_state_dict(voc_state, strict=False)
    print('vocoder', missing, flush=True)
    assert all('resampler' in k for k in missing.missing_keys) and not missing.unexpected_keys
    decoder = decoder.to(args.device).eval()
    vocoder = vocoder.to(args.device).eval()
    latent = pattern((1, 8, 26, 16), 0.017) * 1.3
    with torch.inference_mode():
        mel = decoder(latent)
        low = vocoder.vocoder(mel.float())
        wave = vocoder(mel)
    dump('audio-latent.f32', latent)
    dump('audio-mel.f32', mel)
    dump('audio-vocoder.f32', low)
    dump('audio-wave.f32', wave)


if args.part == 'decode':
    decode()
