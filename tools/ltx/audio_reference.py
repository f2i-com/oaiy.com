"""Official LTX-2 audio reference activations for offline Rust verification.

Parts:
  decode       audio VAE decoder, vocoder and bandwidth extension on a fixed latent
               (float32, as the worker runs them).
  encode       audio VAE encoder on a fixed 24 kHz stereo waveform: the resampler,
               the log-mel spectrogram and the latent (float32).
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
parser.add_argument('--part', choices=['decode', 'encode', 'transformer'], required=True)
parser.add_argument('--audio-vae', help='Audio VAE + vocoder safetensors (decode, encode)')
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


def encode():
    from ltx_core.model.audio_vae.model_configurator import AudioEncoderConfigurator
    from ltx_core.model.audio_vae.ops import AudioProcessor
    from ltx_core.types import Audio
    f = safe_open(args.audio_vae, framework='pt', device=args.device)
    metadata = f.metadata()
    metadata['config'] = json.loads(metadata['config'])
    encoder = AudioEncoderConfigurator.from_metadata(metadata)
    state = {}
    for k in f.keys():
        if k.startswith('audio_vae.encoder.'):
            state[k.removeprefix('audio_vae.encoder.')] = f.get_tensor(k).float()
        elif k.startswith('audio_vae.per_channel_statistics.'):
            state['per_channel_statistics.' + k.removeprefix('audio_vae.per_channel_statistics.')] = f.get_tensor(k).float()
    print('encoder', encoder.load_state_dict(state, strict=True), flush=True)
    encoder = encoder.to(args.device).eval()
    # 2.3 s of a chirp with a tremolo (left) and a detuned copy (right) at 24 kHz.
    n = 55_200
    t = torch.arange(n, device=args.device, dtype=torch.float64) / 24_000
    left = torch.sin(2 * torch.pi * (220 * t + 400 * t * t)) * (0.6 + 0.3 * torch.sin(2 * torch.pi * 3 * t))
    right = 0.5 * torch.sin(2 * torch.pi * (330 * t + 250 * t * t))
    wave = torch.stack([left, right]).float().unsqueeze(0)
    processor = AudioProcessor(target_sample_rate=encoder.sample_rate, mel_bins=encoder.mel_bins,
                               mel_hop_length=encoder.mel_hop_length, n_fft=encoder.n_fft).to(args.device)
    with torch.inference_mode():
        resampled = processor.resample_audio(Audio(waveform=wave, sampling_rate=24_000)).waveform
        mel = processor.waveform_to_mel(Audio(waveform=wave, sampling_rate=24_000))
        latent = encoder(mel)
    dump('audio-encode-wave.f32', wave)
    dump('audio-encode-resampled.f32', resampled)
    dump('audio-encode-mel.f32', mel)
    dump('audio-encode-latent.f32', latent)


def transformer():
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import AudioEmbeddings1DConnectorConfigurator
    w = safe_open(args.checkpoint, framework='pt', device=args.device)
    keys = set(w.keys())

    def get(k):
        full = 'model.diffusion_model.' + k
        return w.get_tensor(full if full in keys else k)

    metadata = w.metadata()
    metadata['config'] = json.loads(metadata['config'])
    # The first two blocks exercise every audio path and fit on one GPU.
    small = json.loads(json.dumps(metadata))
    small['config']['transformer']['num_layers'] = 2
    with torch.device('meta'):
        model = LTXModelConfigurator.from_metadata(small)
    model.load_state_dict({k: get(k) for k in model.state_dict()}, strict=True, assign=True)
    print('Loaded a two-block official audio-video transformer', flush=True)

    d = args.device
    tv, conditioned, ta = 16, 4, 6
    x = torch.sin(torch.arange(tv * 128, device=d) * .013).reshape(1, tv, 128).to(torch.bfloat16)
    context = torch.sin(torch.arange(8 * 4096, device=d) * .021).reshape(1, 8, 4096).to(torch.bfloat16)
    positions = torch.tensor([[[0., 1. / 24], [y * 32, (y + 1) * 32], [x_ * 32, (x_ + 1) * 32]]
                              for y in range(4) for x_ in range(4)],
                             device=d, dtype=torch.float32).permute(1, 0, 2).unsqueeze(0)
    timesteps = torch.full((1, tv), .725, device=d)
    timesteps[:, :conditioned] = 0
    ax = torch.cos(torch.arange(ta * 128, device=d) * .017).reshape(1, ta, 128).to(torch.bfloat16)
    actx = torch.sin(torch.arange(8 * 2048, device=d) * .023).reshape(1, 8, 2048).to(torch.bfloat16)
    # Latent frame i covers mel frames [max(4i-3, 0), 4i+1) at 100 frames per second.
    apos = torch.tensor([[max(4 * i - 3, 0) * .01, (4 * i + 1) * .01] for i in range(ta)],
                        device=d, dtype=torch.float32).reshape(1, 1, ta, 2)
    sigma = torch.tensor([.725], device=d)
    video = Modality(latent=x, sigma=sigma, timesteps=timesteps, positions=positions, context=context,
                     keyframes_mask=torch.ones((1, tv, 1), device=d))
    audio = Modality(latent=ax, sigma=sigma, timesteps=torch.full((1, ta), .725, device=d),
                     positions=apos, context=actx)
    def after_block0(module, inputs, outputs):
        dump('av-block0-video.f32', outputs[0].x)
        dump('av-block0-audio.f32', outputs[1].x)

    hook = model.transformer_blocks[0].register_forward_hook(after_block0)
    with torch.inference_mode():
        vout, aout = model(video, audio, None)
        hook.remove()
        # Frozen conditioning audio (the official a2vid pipeline): its sigma and
        # every per-token timestep are 0 while the video is denoised.
        frozen = Modality(latent=ax, sigma=torch.zeros_like(sigma), timesteps=torch.zeros((1, ta), device=d),
                          positions=apos, context=actx)
        fvout, faout = model(video, frozen, None)
    dump('av-frozen-video-output.f32', fvout)
    dump('av-frozen-audio-output.f32', faout)
    for name, value in [('av-video-input.f32', x), ('av-video-context.f32', context), ('av-audio-input.f32', ax),
                        ('av-audio-context.f32', actx), ('av-video-output.f32', vout), ('av-audio-output.f32', aout)]:
        dump(name, value)
    del model

    connector = AudioEmbeddings1DConnectorConfigurator.from_metadata(metadata)
    prefix = 'audio_embeddings_connector.'
    connector.load_state_dict({k: get(prefix + k) for k in connector.state_dict()}, strict=True)
    connector = connector.to(d, torch.bfloat16).eval()
    n = 12
    feats = torch.sin(torch.arange(n * 2048, device=d) * .019).reshape(1, n, 2048).to(torch.bfloat16)
    padded = torch.cat([feats, torch.zeros(1, 1024 - n, 2048, device=d, dtype=torch.bfloat16)], 1)
    mask = torch.zeros(1, 1, 1, 1024, device=d, dtype=torch.bfloat16)
    mask[..., n:] = -torch.finfo(torch.bfloat16).max
    with torch.inference_mode():
        out_, _ = connector(padded, mask)
    dump('audio-connector-input.f32', feats)
    dump('audio-connector-output.f32', out_)


if args.part == 'decode':
    decode()
elif args.part == 'encode':
    encode()
else:
    transformer()
