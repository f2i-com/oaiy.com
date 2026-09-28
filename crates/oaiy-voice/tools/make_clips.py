"""Test clips for oaiy-voice's parity tests, written to ./clips/ with their
reference text in ./clips/reference.json.

- tts0..5: sentences spoken by a running voice server's /v1/audio/speech
  (24 kHz PCM16, resampled to 16 kHz), default http://127.0.0.1:8782.
- libri0..4: five utterances of different lengths (1.6 s to 29.4 s) from
  hf-internal-testing/librispeech_asr_dummy.

Usage (from the folder that will hold clips/ and fixtures/):
    python make_clips.py [--tts http://127.0.0.1:8782]
Needs numpy, scipy, soundfile, pyarrow and huggingface_hub.
"""
import argparse, io, json, os, urllib.request

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TEXTS = [
    "Hello, thanks for calling. How can I help you today?",
    "I'd like to book an appointment for next Tuesday at three thirty in the afternoon.",
    "The quick brown fox jumps over the lazy dog.",
    "My phone number is zero four one two, three four five, six seven eight.",
    "Could you please send the invoice to accounts at example dot com?",
    "Yes.",
]


def tts_clips(base, out, ref):
    for i, text in enumerate(TEXTS):
        body = json.dumps({"input": text, "response_format": "pcm"}).encode()
        req = urllib.request.Request(base + "/v1/audio/speech", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            rate = int(r.headers.get("X-Sample-Rate", "24000"))
            pcm = np.frombuffer(r.read(), dtype="<i2").astype(np.float64) / 32768.0
        g = np.gcd(16000, rate)
        y = resample_poly(pcm, 16000 // g, rate // g)
        y = np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16)
        name = f"tts{i}.wav"
        sf.write(os.path.join(out, name), y, 16000, subtype="PCM_16")
        ref[name] = text
        print(name, len(y) / 16000, "s")


def libri_clips(out, ref):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(p).to_pylist()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["audio"]["bytes"]))
    pick = [order[0], order[len(order) // 4], order[len(order) // 2], order[3 * len(order) // 4], order[-1]]
    for k, i in enumerate(pick):
        y, sr = sf.read(io.BytesIO(rows[i]["audio"]["bytes"]), dtype="int16")
        assert sr == 16000, sr
        name = f"libri{k}.wav"
        sf.write(os.path.join(out, name), y, 16000, subtype="PCM_16")
        ref[name] = rows[i]["text"]
        print(name, len(y) / 16000, "s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", default="http://127.0.0.1:8782", help="a voice server with /v1/audio/speech; empty to skip")
    args = ap.parse_args()
    os.makedirs("clips", exist_ok=True)
    ref = {}
    if args.tts:
        tts_clips(args.tts, "clips", ref)
    libri_clips("clips", ref)
    json.dump(ref, open(os.path.join("clips", "reference.json"), "w"), indent=1)
