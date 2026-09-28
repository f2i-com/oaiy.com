"""lance_server.py -- thin JSON API over Lance's run_task, for OAIY / oaiy-web.

Lance ships a Gradio app (lance_gradio.py). oaiy-web is a browser app and wants a
clean HTTP API, not Gradio's fn_index protocol, so this wraps run_task() in
FastAPI:

  GET  /health            -> {ok, loaded, variant}
  POST /generate          -> {ok, status, videoUrl, imageUrl, text, ...}
  GET  /file?path=<abs>   -> the generated media bytes (guarded to output dirs)
  GET  /                  -> small JSON index

It also mounts the original Gradio UI at /ui, sharing the SAME model pool
(run_task uses a process-global PipelinePool in lance_gradio), so there is no
double-VRAM cost between the API and the UI.

Run by the OAIY Companion's "Lance (Image+Video)" service (cwd = the Lance repo):
  python lance_server.py --server-name 127.0.0.1 --server-port 17900

Importing lance_gradio is cheap: its Gradio launch is guarded under
`if __name__ == "__main__"`, and the model loads lazily on the first generate.
"""
from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import gradio as gr

# Lance's own settings + entrypoints. The star-free imports keep it explicit.
from common.gradio_utils.settings import (
    DEFAULT_TASK,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_RESOLUTION,
    DEFAULT_TIMESTEPS,
    DEFAULT_TIMESTEP_SHIFT,
    DEFAULT_CFG_TEXT_SCALE,
    DEFAULT_BASIC_SEED,
    DEFAULT_VIDEO_DURATION_SECONDS,
    REPO_ROOT,
    RESULTS_ROOT,
    GRADIO_TMP_ROOT,
    TASK_LABEL_VIDEO_GENERATION,
    TASK_LABEL_IMAGE_GENERATION,
    TASK_LABEL_VIDEO_EDIT,
    TASK_LABEL_IMAGE_EDIT,
    TASK_LABEL_VIDEO_UNDERSTANDING,
    TASK_LABEL_IMAGE_UNDERSTANDING,
)

# Import the MODULE (not just run_task) so /health can read the live
# ACTIVE_PIPELINE_POOL global, which is None until the first generate.
import lance_gradio
from lance_gradio import run_task, build_demo

# normalize_task() only accepts the Gradio display LABELS, so map clean API task
# names (and the labels themselves) onto them.
API_TASK_TO_LABEL = {
    "t2v": TASK_LABEL_VIDEO_GENERATION,
    "video": TASK_LABEL_VIDEO_GENERATION,
    "text-to-video": TASK_LABEL_VIDEO_GENERATION,
    "t2i": TASK_LABEL_IMAGE_GENERATION,
    "image": TASK_LABEL_IMAGE_GENERATION,
    "text-to-image": TASK_LABEL_IMAGE_GENERATION,
    "video_edit": TASK_LABEL_VIDEO_EDIT,
    "image_edit": TASK_LABEL_IMAGE_EDIT,
    "v2t": TASK_LABEL_VIDEO_UNDERSTANDING,
    "x2t_video": TASK_LABEL_VIDEO_UNDERSTANDING,
    "x2t_image": TASK_LABEL_IMAGE_UNDERSTANDING,
}

# Files we will serve over /file -- everything Lance writes lands under one of
# these roots. resolve() so symlink/relative tricks can't escape.
ALLOWED_ROOTS = [
    p.resolve()
    for p in {GRADIO_TMP_ROOT, RESULTS_ROOT, REPO_ROOT}
]


class GenerateRequest(BaseModel):
    task: str = DEFAULT_TASK
    prompt: str = ""
    systemPrompt: Optional[str] = None
    inputVideo: Optional[str] = None
    inputImage: Optional[str] = None
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    # For t2v: clip length in seconds (run_task converts to frames). For other
    # tasks: an explicit frame count (numFrames).
    seconds: Optional[int] = None
    numFrames: Optional[int] = None
    seed: int = DEFAULT_BASIC_SEED
    resolution: str = DEFAULT_RESOLUTION
    steps: int = DEFAULT_TIMESTEPS
    timestepShift: float = DEFAULT_TIMESTEP_SHIFT
    cfg: float = DEFAULT_CFG_TEXT_SCALE


app = FastAPI(title="Lance JSON API (OAIY)")
# oaiy-web runs in a browser on a different origin -> permissive CORS (localhost).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Absolute base URL for /file links, set in main() from the bind args so the
# returned videoUrl/imageUrl work directly as a browser <video>/<img> src
# (a relative path would resolve against the oaiy-web origin, not this server).
BASE_URL = ""


def _file_url(p) -> Optional[str]:
    if not p:
        return None
    return f"{BASE_URL}/file?path={quote(str(p))}"


@app.get("/")
def index():
    return {
        "service": "lance",
        "ui": "/ui",
        "generate": "POST /generate",
        "health": "/health",
    }


@app.get("/health")
def health():
    pool = getattr(lance_gradio, "ACTIVE_PIPELINE_POOL", None)
    loaded = bool(pool is not None and pool.is_initialized)
    return {
        "ok": True,
        "loaded": loaded,
        "variant": (pool.model_variant if pool is not None else None),
    }


@app.get("/file")
def get_file(path: str = Query(...)):
    p = Path(path).resolve()
    if not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    # Path-component containment, not a string prefix (a sibling like
    # `<root>-evil/...` shares the prefix but is outside the allowed root).
    if not any(p == root or str(p).startswith(str(root) + os.sep) for root in ALLOWED_ROOTS):
        raise HTTPException(status_code=403, detail="path not allowed")
    return FileResponse(str(p))


@app.post("/generate")
def generate(req: GenerateRequest):
    key = (req.task or DEFAULT_TASK).strip().lower()
    label = API_TASK_TO_LABEL.get(key, req.task)
    is_video_gen = label == TASK_LABEL_VIDEO_GENERATION

    if is_video_gen:
        # run_task expects SECONDS here and converts internally.
        nf = int(req.seconds if req.seconds is not None else DEFAULT_VIDEO_DURATION_SECONDS)
    else:
        nf = int(req.numFrames if req.numFrames is not None else 1)

    try:
        video_path, image_path, text, status = run_task(
            task=label,
            prompt=req.prompt,
            system_prompt=req.systemPrompt,
            input_video=req.inputVideo,
            input_image=req.inputImage,
            height=int(req.height),
            width=int(req.width),
            num_frames=nf,
            seed=int(req.seed),
            resolution=req.resolution,
            validation_num_timesteps=int(req.steps),
            validation_timestep_shift=float(req.timestepShift),
            cfg_text_scale=float(req.cfg),
        )
    except Exception:
        trace = traceback.format_exc()
        print(trace, flush=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": trace})

    ok = bool(video_path or image_path or (text or "").strip())
    return {
        "ok": ok,
        # run_task returns "" on success, or a human-readable message otherwise.
        "status": status or "",
        "videoUrl": _file_url(video_path),
        "imageUrl": _file_url(image_path),
        "videoPath": str(video_path) if video_path else None,
        "imagePath": str(image_path) if image_path else None,
        "text": text or "",
        "task": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance JSON API server")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=17900)
    # Only override LANCE_GPUS if explicitly given; otherwise honor the env the
    # companion already set on the service (e.g. "0,1").
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--queue-size", type=int, default=32)
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["LANCE_GPUS"] = args.gpus

    # 0.0.0.0/:: bind -> advertise 127.0.0.1 in returned URLs (browser-usable).
    global BASE_URL
    host = args.server_name if args.server_name not in ("0.0.0.0", "::", "") else "127.0.0.1"
    BASE_URL = f"http://{host}:{int(args.server_port)}"

    print(
        "[lance_server] Local-only mode. JSON API at /generate, Gradio UI at /ui. "
        "Model weights load lazily on the first request.",
        flush=True,
    )

    # Mount the original Gradio UI under /ui, sharing run_task's global pool.
    demo = build_demo(run_task)
    demo.queue(max_size=args.queue_size, default_concurrency_limit=1)
    mounted = gr.mount_gradio_app(
        app,
        demo,
        path="/ui",
        allowed_paths=[str(REPO_ROOT.resolve()), str(GRADIO_TMP_ROOT.resolve())],
    )

    uvicorn.run(mounted, host=args.server_name, port=int(args.server_port), log_level="info")


if __name__ == "__main__":
    main()
