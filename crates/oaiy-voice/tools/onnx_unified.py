"""Cross-check parakeet-unified-en-0.6b's NeMo 2.7 fixtures (made with its
streaming-only config keys removed, see nemo_fixtures.py) against
eschmidbauer/parakeet-unified-en-0.6b-onnx's fp32 export, which was made with
a NeMo that loads the model as published: the export's encoder on the fixture
features against the fixture encoder output, greedy RNN-T through its
decoder_joint, and the same from Aokie's features (no pre-emphasis, periodic
Hann, reflect padding, biased std) for comparison.

Usage (beside clips/ and fixtures/unified/):
    python onnx_unified.py [folder holding onnx_fp32/]
Needs onnxruntime, sentencepiece, soundfile, torch and torchaudio.
"""
import os, json, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, onnxruntime as ort, sentencepiece as spm, soundfile as sf, torch, torchaudio
D = os.path.join(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("OAIY_VOICE_MODELS", r"E:\models"), "parakeet-unified-en-0.6b-onnx"), "onnx_fp32")
so = ort.SessionOptions(); so.intra_op_num_threads = 8
enc = ort.InferenceSession(os.path.join(D, "encoder.onnx"), so, providers=["CPUExecutionProvider"])
dj = ort.InferenceSession(os.path.join(D, "decoder_joint.onnx"), so, providers=["CPUExecutionProvider"])
sp = spm.SentencePieceProcessor(model_file=os.path.join(D, "tokenizer.model"))
BLANK = 1024
def greedy(e):  # e: (1, 1024, T)
    T = e.shape[2]
    names = [i.name for i in dj.get_inputs()]
    s1 = np.zeros((2, 1, 640), np.float32); s2 = np.zeros((2, 1, 640), np.float32)
    last = BLANK; out = []
    for t in range(T):
        for _ in range(10):
            feed = {"encoder_outputs": e[:, :, t:t+1], "targets": np.array([[last]], np.int32), "target_length": np.array([1], np.int32), "input_states_1": s1, "input_states_2": s2}
            logits, _, n1, n2 = dj.run(None, feed)
            k = int(np.argmax(logits.reshape(-1)[:BLANK + 1]))
            if k == BLANK: break
            out.append(k); last = k; s1, s2 = n1, n2
    return out
def aokie_mel(y):
    x = torch.tensor(y)[None]
    m = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=128, window_fn=torch.hann_window, power=2.0, norm="slaney", mel_scale="slaney", center=True)(x)
    m = torch.log(m + 2**-24)
    m = (m - m.mean(dim=-1, keepdim=True)) / (m.std(dim=-1, keepdim=True, unbiased=False) + 1e-5)
    return m.numpy().astype(np.float32)
FX = "fixtures/unified"
index = json.load(open(os.path.join(FX, "index.json")))
for clip, info in index.items():
    n_mels, frames = info["mel"]
    mel = np.fromfile(os.path.join(FX, clip + ".mel.f32"), dtype="<f4").reshape(1, n_mels, frames)
    ours_nemo = np.fromfile(os.path.join(FX, clip + ".enc.f32"), dtype="<f4").reshape(info["enc"])
    e, el = enc.run(None, {"audio_signal": mel, "length": np.array([frames], np.int64)})
    e_t = e[0, :, :int(el[0])].T
    diff = np.abs(e_t - ours_nemo).max() if e_t.shape == ours_nemo.shape else f"shape {e_t.shape} vs {ours_nemo.shape}"
    toks = greedy(e[:, :, :int(el[0])])
    y, _ = sf.read(os.path.join("clips", clip + ".wav"), dtype="float32")
    am = aokie_mel(y)
    e2, el2 = enc.run(None, {"audio_signal": am, "length": np.array([am.shape[2]], np.int64)})
    toks2 = greedy(e2[:, :, :int(el2[0])])
    print(f"{clip}: encoder vs NeMo 2.7 max|diff| {diff}; tokens {'same' if toks == info['tokens'] else 'DIFFER'}")
    print(f"   onnx(nemo mel):  {sp.decode(toks)!r}")
    print(f"   onnx(aokie mel): {sp.decode(toks2)!r}")
    print(f"   nemo 2.7:        {info['text']!r}")
