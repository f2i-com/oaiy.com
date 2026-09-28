"""NeMo reference fixtures for oaiy-voice's parity tests (src/stt/tests.rs).

For each model and each clip in ./clips/*.wav (see make_clips.py), NeMo runs
on the CPU in f32 and writes to ./fixtures/<model>/:
  <clip>.mel.f32   the preprocessor's features, valid frames, (n_mels, frames)
  <clip>.enc.f32   the encoder's output, (frames / 8, d_model)
  index.json       shapes, NeMo's greedy tokens and text
Raw little-endian f32. Then:
    OAIY_VOICE_FIXTURES=<this folder>/fixtures cargo test --release -p oaiy-voice parity -- --nocapture

Usage: python nemo_fixtures.py v2 unified v3 ultra
Weights are read from $OAIY_VOICE_MODELS (default E:\\models):
  parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo
  parakeet-unified-en-0.6b/parakeet-unified-en-0.6b.nemo
  parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo
  parakeet-ultra/model.safetensors (loaded into the v3 NeMo model)

Environment: NeMo 2.7 with PyTorch (CPU is enough). NeMo imports numba,
which refuses NumPy 2.4; a venv with the system packages and an older NumPy
works:
    python -m venv --system-site-packages venv
    venv/Scripts/python -m pip install "numpy<2.4"

parakeet-unified-en-0.6b's config names encoder options newer than NeMo 2.7
(att_chunk_context_size, att_context_style chunked_limited_with_rc,
conv_context_style dcc). They only shape its streaming mode; offline it is
full-context attention with ordinary convolutions, so they are removed
before the model is built. onnx_unified.py checks that against an export
made with a NeMo that loads the model as published.
"""
import glob, json, logging, os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import soundfile as sf
import torch

torch.set_grad_enabled(False)
from nemo.utils import logging as nemo_logging

nemo_logging.setLevel(logging.ERROR)
import nemo.collections.asr as nemo_asr

MODELS = os.environ.get("OAIY_VOICE_MODELS", r"E:\models")
OUT = "fixtures"
CLIPS = sorted(glob.glob(os.path.join("clips", "*.wav")))


def hf_to_nemo(k):
    """A transformers ParakeetForTDT parameter name as NeMo's."""
    reps = [
        ("encoder.subsampling.layers.", "encoder.pre_encode.conv."), ("encoder.subsampling.linear.", "encoder.pre_encode.out."),
        (".self_attn.q_proj.", ".self_attn.linear_q."), (".self_attn.k_proj.", ".self_attn.linear_k."), (".self_attn.v_proj.", ".self_attn.linear_v."),
        (".self_attn.o_proj.", ".self_attn.linear_out."), (".self_attn.relative_k_proj.", ".self_attn.linear_pos."),
        (".self_attn.bias_u", ".self_attn.pos_bias_u"), (".self_attn.bias_v", ".self_attn.pos_bias_v"), (".conv.norm.", ".conv.batch_norm."),
        ("decoder.embedding.", "decoder.prediction.embed."), ("decoder.lstm.", "decoder.prediction.dec_rnn.lstm."),
        ("decoder.decoder_projector.", "joint.pred."), ("encoder_projector.", "joint.enc."), ("joint.head.", "joint.joint_net.2."),
    ]
    for a, b in reps:
        k = k.replace(a, b)
    return k


def load(nemo_path, safetensors=None):
    cfg = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu", return_config=True)
    enc = cfg.encoder
    if "att_chunk_context_size" in enc:
        from omegaconf import open_dict

        with open_dict(cfg):
            del enc["att_chunk_context_size"]
            enc.pop("conv_context_style", None)
            enc["att_context_style"] = "regular"
            enc["att_context_size"] = [-1, -1]
    model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu", override_config_path=cfg)
    model.eval()
    if safetensors:
        from safetensors.torch import load_file

        sd = {hf_to_nemo(k): v.float() for k, v in load_file(safetensors).items() if not k.startswith("vad_head.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("preprocessor.")]
        assert not unexpected and not missing, (missing, unexpected)
    return model


def run(name, nemo_path, safetensors=None):
    model = load(nemo_path, safetensors)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    index = {}
    for clip in CLIPS:
        y, sr = sf.read(clip, dtype="float32")
        assert sr == 16000, clip
        x = torch.tensor(y)[None]
        feats, flen = model.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]]))
        enc, elen = model.encoder(audio_signal=feats, length=flen)
        hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=elen, return_hypotheses=True)
        h = hyps[0] if not isinstance(hyps, tuple) else hyps[0][0]
        if isinstance(h, list):
            h = h[0]
        seq = h.y_sequence.tolist() if torch.is_tensor(h.y_sequence) else h.y_sequence
        base = os.path.splitext(os.path.basename(clip))[0]
        fl, el = int(flen[0]), int(elen[0])
        mel = feats[0, :, :fl].contiguous().numpy().astype("<f4")
        e = enc[0, :, :el].T.contiguous().numpy().astype("<f4")
        mel.tofile(os.path.join(d, base + ".mel.f32"))
        e.tofile(os.path.join(d, base + ".enc.f32"))
        index[base] = {"mel": [int(mel.shape[0]), fl], "enc": [el, int(e.shape[1])], "tokens": [int(t) for t in seq], "text": h.text}
        print(name, base, fl, el, repr(h.text), flush=True)
    json.dump(index, open(os.path.join(d, "index.json"), "w"), indent=1)


CHECKPOINTS = {
    "v2": (os.path.join(MODELS, "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2.nemo"), None),
    "unified": (os.path.join(MODELS, "parakeet-unified-en-0.6b", "parakeet-unified-en-0.6b.nemo"), None),
    "v3": (os.path.join(MODELS, "parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3.nemo"), None),
    "ultra": (os.path.join(MODELS, "parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3.nemo"), os.path.join(MODELS, "parakeet-ultra", "model.safetensors")),
}

if __name__ == "__main__":
    for k in sys.argv[1:]:
        run(k, *CHECKPOINTS[k])
