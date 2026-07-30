"""ltx2_server.py -- thin JSON API over the official LTX-2.3 DistilledPipeline.

The Lightricks/LTX-2 repo ships pipeline *classes*, not a server. This wraps
`DistilledPipeline` (text/image -> video; internally 2-stage: half-res gen ->
2x spatial upscale -> refine) in FastAPI so oaiy-web can drive it over HTTP:

  GET  /health            -> {ok, loaded}
  GET  /resolve           -> the weight paths it found (debug aid)
  POST /generate          -> {ok, videoUrl, videoPath, elapsedSeconds}
  GET  /file?path=<abs>   -> the generated mp4 (guarded to the output dir)

The pipeline is built ONCE (lazily, on the first /generate) and kept warm --
loading the 22B + Gemma + VAEs takes a while, so we don't pay it per request.
One generation runs at a time (they'd contend on the GPU).

Weights are located across one or more model folders (the companion's
multi-folder feature). Resolution order, per role: an explicit env override
wins; otherwise we glob the registered model dirs for the known LTX-2.3
filenames. Set the matching env var if a heuristic ever misses.

Env (set by the companion service template):
  LTX2_PORT          listen port (default 17890)
  LTX2_MODEL_DIRS    os.pathsep-joined model roots to search (e.g. E:\\models;E:\\ckpts)
  OAIY_MODEL_DIRS     fallback for the above (companion exposes both)
  LTX2_CKPT          explicit distilled .safetensors (overrides search)
  LTX2_GEMMA         explicit Gemma 3 text-encoder dir
  LTX2_UPSAMPLER     explicit spatial upscaler .safetensors
  LTX2_QUANT         quantization kind: fp8-cast | fp8-scaled-mm | none (default fp8-cast)
  LTX2_OFFLOAD       OffloadMode: none | cpu | disk (default none)
  LTX2_OUT_DIR       where mp4s are written (default <cwd>/ltx2_out)

Run with the repo's uv .venv python (ltx_pipelines/ltx_core install editable
there, so `import ltx_pipelines` works with no PYTHONPATH). On consumer
Blackwell there's no xformers/flash wheel, so LTX-core auto-falls back to torch
SDPA -- unfused, correct.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Optional
from urllib.parse import quote


def _autoselect_gpu() -> None:
    """Pin to the GPU with the most free VRAM, unless the caller already chose
    one. LTX defaults to cuda:0; on a multi-GPU box where card 0 is busy (e.g.
    driving the desktop) that OOMs the 22B. Must run BEFORE torch initialises
    CUDA, so it lives at module import. nvidia-smi is the probe (no torch yet).
    Honours an explicit CUDA_VISIBLE_DEVICES / LTX2_GPU; no-ops if either is set
    or nvidia-smi is unavailable."""
    # "in" not truthiness: an explicit CUDA_VISIBLE_DEVICES="" (the documented
    # way to hide ALL GPUs) is honoured rather than overwritten.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    forced = os.environ.get("LTX2_GPU", "").strip()
    if forced:
        os.environ["CUDA_VISIBLE_DEVICES"] = forced
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        best_idx, best_free = None, -1
        for line in out.splitlines():
            # Skip rows nvidia-smi can't give a number for ([N/A]/MIG/vGPU,
            # or a stray non-CSV line) instead of abandoning the whole pick.
            try:
                idx_s, free_s = (x.strip() for x in line.split(","))
                free = int(free_s)
            except (ValueError, IndexError):
                continue
            if free > best_free:
                best_idx, best_free = idx_s, free
        if best_idx is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = best_idx
            print(f"[ltx2_server] auto-selected GPU {best_idx} ({best_free} MiB free)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[ltx2_server] GPU auto-select skipped: {e}", flush=True)


# Pick the GPU BEFORE anything imports torch. (We deliberately do NOT set
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments — it's unsupported on Windows and
# setting it there wrecks the allocator: the 22B fp8-cast stage OOMs with a
# bogus "124 GiB allocated" where the plain allocator fits fine in 32 GB.)
_autoselect_gpu()

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


QUANT = _env("LTX2_QUANT", "fp8-cast")
OFFLOAD = _env("LTX2_OFFLOAD", "none")
OUT_DIR = _env("LTX2_OUT_DIR") or str(Path.cwd() / "ltx2_out")

BASE_URL = ""  # set in main() from the bind args, so /file URLs are browser-usable

# Built lazily on first /generate, then reused. One gen at a time.
_pipeline = None
_pipeline_lock = threading.Lock()
_gen_lock = threading.Lock()


# --------------------------------------------------------------------------
# Weight resolution across the registered model folders.
# --------------------------------------------------------------------------
def _model_dirs() -> list[str]:
    raw = _env("LTX2_MODEL_DIRS") or _env("OAIY_MODEL_DIRS") or _env("OAIY_MODELS_DIR")
    dirs, seen = [], set()
    for d in raw.split(os.pathsep):
        d = d.strip().strip('"')
        if d and d not in seen and Path(d).is_dir():
            seen.add(d)
            dirs.append(d)
    return dirs


def _safetensors_in(root: str) -> list[str]:
    """Top-level + one-level-deep .safetensors under root (cheap, no deep walk)."""
    hits = []
    try:
        for entry in os.scandir(root):
            if entry.is_file() and entry.name.lower().endswith(".safetensors"):
                hits.append(entry.path)
            elif entry.is_dir():
                try:
                    for sub in os.scandir(entry.path):
                        if sub.is_file() and sub.name.lower().endswith(".safetensors"):
                            hits.append(sub.path)
                except OSError:
                    pass
    except OSError:
        pass
    return hits


def _pick_file(pred, candidates: list[str]) -> Optional[str]:
    matches = [p for p in candidates if pred(os.path.basename(p).lower())]
    if not matches:
        return None
    # Prefer the largest (the real transformer, not a config/lora shard).
    return max(matches, key=lambda p: os.path.getsize(p))


def _find_gemma_dir(roots: list[str]) -> Optional[str]:
    """A Gemma text-encoder directory: name mentions gemma, has a config +
    at least one weight file (.safetensors). Search each root + its subdirs."""
    def looks_like_gemma(d: str) -> bool:
        name = os.path.basename(d).lower()
        if "gemma" not in name:
            return False
        try:
            files = {f.name.lower() for f in os.scandir(d) if f.is_file()}
        except OSError:
            return False
        has_cfg = any(f.endswith(".json") for f in files)
        has_w = any(f.endswith((".safetensors", ".bin")) for f in files)
        return has_cfg and has_w
    for root in roots:
        if looks_like_gemma(root):
            return root
        try:
            for entry in os.scandir(root):
                if entry.is_dir() and looks_like_gemma(entry.path):
                    return entry.path
        except OSError:
            pass
    return None


def resolve_paths() -> dict:
    """Map the three LTX weight roles -> files on disk. Env overrides win;
    otherwise glob the registered model dirs. Values may be None if missing."""
    ckpt = _env("LTX2_CKPT") or None
    gemma = _env("LTX2_GEMMA") or None
    upsampler = _env("LTX2_UPSAMPLER") or None

    roots = _model_dirs()
    all_st: list[str] = []
    for r in roots:
        all_st.extend(_safetensors_in(r))

    if not ckpt:
        # The distilled transformer: name has "distilled"; else largest 22b file.
        ckpt = _pick_file(lambda n: "distilled" in n and n.endswith(".safetensors"), all_st)
        if not ckpt:
            ckpt = _pick_file(lambda n: "ltx" in n and "22b" in n, all_st)
    if not upsampler:
        upsampler = _pick_file(
            lambda n: "spatial" in n and ("upscal" in n or "upsampl" in n), all_st
        )
    if not gemma:
        gemma = _find_gemma_dir(roots)

    return {
        "modelDirs": roots,
        "checkpoint": ckpt,
        "gemmaRoot": gemma,
        "upsampler": upsampler,
    }


class GenerateRequest(BaseModel):
    prompt: str = ""
    seed: int = 42
    # LTX-2 trained at high res; defaults to a moderate ~720p clip. Two-stage
    # requires height & width divisible by 64 (stage 1 runs at half). num_frames
    # should be 8*K + 1. Bump from oaiy-web for full quality (e.g. 1088x1920).
    height: int = 704
    width: int = 1280
    num_frames: int = 97
    frame_rate: float = 24.0
    enhance_prompt: bool = False


app = FastAPI(title="LTX-2 JSON API (OAIY)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _build_pipeline():
    """Construct DistilledPipeline once (mirrors ltx_pipelines.distilled.main)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        from ltx_pipelines.utils.types import OffloadMode

        paths = resolve_paths()
        missing = [k for k in ("checkpoint", "gemmaRoot", "upsampler") if not paths.get(k)]
        if missing:
            raise RuntimeError(
                f"could not locate LTX-2.3 weights for {missing} in model dirs "
                f"{paths['modelDirs']}. Set LTX2_CKPT / LTX2_GEMMA / LTX2_UPSAMPLER "
                f"to point at the files explicitly."
            )

        quant = None
        if QUANT and QUANT.lower() not in ("", "none"):
            quant = QuantizationKind(QUANT).to_policy(checkpoint_path=paths["checkpoint"])
        offload = OffloadMode(OFFLOAD) if OFFLOAD else OffloadMode.NONE

        print(
            f"[ltx2_server] building DistilledPipeline (quant={QUANT}, offload={offload.value})\n"
            f"  checkpoint={paths['checkpoint']}\n"
            f"  gemmaRoot ={paths['gemmaRoot']}\n"
            f"  upsampler ={paths['upsampler']}",
            flush=True,
        )
        t0 = time.perf_counter()
        _pipeline = DistilledPipeline(
            distilled_checkpoint_path=paths["checkpoint"],
            gemma_root=paths["gemmaRoot"],
            spatial_upsampler_path=paths["upsampler"],
            loras=(),
            quantization=quant,
            offload_mode=offload,
        )
        print(f"[ltx2_server] pipeline ready in {time.perf_counter() - t0:.1f}s", flush=True)
        return _pipeline


def _file_url(p) -> Optional[str]:
    if not p:
        return None
    return f"{BASE_URL}/file?path={quote(str(p))}"


@app.get("/")
def index():
    return {"service": "ltx2", "generate": "POST /generate", "health": "/health"}


@app.get("/health")
def health():
    return {"ok": True, "loaded": _pipeline is not None}


@app.get("/resolve")
def resolve():
    paths = resolve_paths()
    paths["ready"] = all(paths.get(k) for k in ("checkpoint", "gemmaRoot", "upsampler"))
    return paths


@app.get("/file")
def get_file(path: str = Query(...)):
    p = Path(path).resolve()
    out = Path(OUT_DIR).resolve()
    if not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    # Confine to OUT_DIR on a PATH-COMPONENT boundary, not a raw string prefix:
    # a bare startswith() would also admit siblings like `<out>_evil/...`.
    if not (p == out or str(p).startswith(str(out) + os.sep)):
        raise HTTPException(status_code=403, detail="path not allowed")
    return FileResponse(str(p))


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video

        pipe = _build_pipeline()
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(OUT_DIR) / f"ltx2_{int(req.seed)}_{req.width}x{req.height}_{req.num_frames}f.mp4")
        tiling = TilingConfig.default()
        # inference_mode is ESSENTIAL: the LTX CLI's main() is decorated with it.
        # Without it, autograd retains every intermediate across the 22B's 30+
        # blocks — the fp8-cast per-layer upcasts pile up to a bogus "124 GiB
        # allocated" OOM that disappears the moment grad tracking is off.
        with _gen_lock, torch.inference_mode():
            t0 = time.perf_counter()
            video, audio = pipe(
                prompt=req.prompt,
                seed=int(req.seed),
                height=int(req.height),
                width=int(req.width),
                num_frames=int(req.num_frames),
                frame_rate=float(req.frame_rate),
                images=[],
                tiling_config=tiling,
                enhance_prompt=bool(req.enhance_prompt),
            )
            encode_video(
                video=video,
                fps=float(req.frame_rate),
                audio=audio,
                output_path=out_path,
                video_chunks_number=get_video_chunks_number(int(req.num_frames), tiling),
            )
            elapsed = time.perf_counter() - t0
        ok = Path(out_path).is_file()
        return {
            "ok": ok,
            "status": "" if ok else "no output produced",
            "videoUrl": _file_url(out_path) if ok else None,
            "videoPath": out_path if ok else None,
            "elapsedSeconds": round(elapsed, 2),
        }
    except Exception:
        trace = traceback.format_exc()
        print(trace, flush=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": trace})


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2 JSON API server")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=int(_env("LTX2_PORT", "17890")))
    args = parser.parse_args()

    global BASE_URL
    host = args.server_name if args.server_name not in ("0.0.0.0", "::", "") else "127.0.0.1"
    BASE_URL = f"http://{host}:{int(args.server_port)}"

    print("[ltx2_server] resolved at boot: " + str(resolve_paths()), flush=True)
    print(
        f"[ltx2_server] local-only on {BASE_URL}; model loads lazily on first /generate.",
        flush=True,
    )
    uvicorn.run(app, host=args.server_name, port=int(args.server_port), log_level="info")


if __name__ == "__main__":
    main()
