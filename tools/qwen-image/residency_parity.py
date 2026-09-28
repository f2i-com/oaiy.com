"""Residency must not change pixels: run one request through oaiy-media in
several memory modes and compare the PNG hashes.

    python tools/qwen-image/residency_parity.py qwen  [gpu ram:64 ssd auto:64:4]
    python tools/qwen-image/residency_parity.py sdxl  [gpu ram:32 ssd]

Modes are memory[:ram_gb[:vram_gb]]. Paths below are this machine's; edit them
(or set OAIY_WORKER) for yours. Outputs go to the system temp folder.
"""
import hashlib, json, os, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.environ.get("OAIY_WORKER", os.path.join(ROOT, "target", "release", "oaiy-media.exe" if os.name == "nt" else "oaiy-media"))
REQUESTS = {
    "qwen": {
        "base": "D:/Qwen-Image-2.1",
        "transformer": os.path.join(ROOT, "models/qwen-image-2.1/qwen-image-2.1-Q4_K_M.gguf"),
        "adapter": os.path.join(ROOT, "models/qwen-image-2.1/viggle-v0.2.1-r128.safetensors"),
        "prompt": "a red fox sitting in fresh snow, photograph",
        "n": 1, "width": 512, "height": 512, "steps": 4, "seed": 7, "device": 1,
    },
    "sdxl": {
        "architecture": "sdxl", "checkpoint": "E:/stuff/sdxlCheckpoint_v90.safetensors",
        "tokenizer": "E:/models/sdxl/clip-tokenizer/tokenizer.json", "prompt": "anime illustration, red fox beside a shrine",
        "negative_prompt": "blurry", "n": 2, "width": 512, "height": 512, "steps": 6, "seed": 3, "device": 1,
    },
}

def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    modes = sys.argv[2:] or (["gpu", "ram:64", "ssd", "auto:64:4"] if kind == "qwen" else ["gpu", "ram:32", "ssd"])
    out_root = tempfile.mkdtemp(prefix=f"oaiy-parity-{kind}-")
    hashes = {}
    for mode in modes:
        parts = mode.split(":")
        req = dict(REQUESTS[kind], memory=parts[0], output_dir=os.path.join(out_root, mode.replace(":", "-")))
        if len(parts) > 1: req["ram_gb"] = int(parts[1])
        if len(parts) > 2: req["vram_gb"] = int(parts[2])
        t = time.time()
        p = subprocess.run([WORKER, "--stdin"], input=json.dumps(req).encode(), capture_output=True)
        if p.returncode:
            print(mode, "FAILED", p.stderr.decode()[-2000:]); continue
        out = json.loads(p.stdout)
        hashes[mode] = tuple(hashlib.sha256(open(d["path"], "rb").read()).hexdigest()[:16] for d in out["data"])
        print(f"{mode}: {time.time() - t:.1f}s {hashes[mode]} {json.dumps(out.get('residency'))}", flush=True)
    same = len(set(hashes.values())) == 1 and len(hashes) == len(modes)
    print("identical" if same else "DIFFERENT")
    sys.exit(0 if same else 1)

if __name__ == "__main__":
    main()
