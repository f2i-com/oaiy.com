@echo off
setlocal enabledelayedexpansion
rem ===================================================================
rem install-ltx2.bat -- one-click install of the LTX-2.3 video service.
rem
rem Run by the OAIY Companion (which sets OAIY_DATA_DIR / OAIY_MODELS_DIR /
rem OAIY_MODEL_DIRS / OAIY_SCRIPTS_DIR). ASCII-only on purpose (cmd + PS 5.1
rem both mangle non-ASCII).
rem
rem VERIFIED RECIPE (2026-05-29, RTX 5090 / Windows 11):
rem   1. bootstrap uv (manages its own Python -- no bundled interpreter needed)
rem   2. fetch the Lightricks/LTX-2 repo zip (curl + tar, no git)
rem   3. uv sync --frozen  (NO xformers extra -- it's Linux-only; LTX-core
rem      auto-falls back to torch SDPA on consumer Blackwell)
rem   4. uv pip install --reinstall torch/torchaudio from the cu128 index
rem      (PyPI's default torch is +cpu on Windows -- MUST reinstall)
rem   5. uv pip install fastapi + uvicorn (for ltx2_server.py)
rem
rem Weights are NOT downloaded here -- point the service at the folder that
rem already holds them via the companion's model folders (e.g. E:\ckpts).
rem Set OAIY_LTX_DOWNLOAD=1 to pull them from HuggingFace into the models dir.
rem ===================================================================

echo [install-ltx2] starting LTX-2.3 video service install

if "%OAIY_DATA_DIR%"=="" (
  echo [install-ltx2] ERROR: OAIY_DATA_DIR is not set. Run Install from the OAIY app.
  exit /b 1
)

set "DATA=%OAIY_DATA_DIR%"
set "MODELS=%OAIY_MODELS_DIR%"
if "%MODELS%"=="" set "MODELS=%DATA%\models"
set "SVC=%DATA%\services\ltx2"
set "REPO=%SVC%\LTX-2-main"
set "VPY=%REPO%\.venv\Scripts\python.exe"
set "TORCH_INDEX=%OAIY_TORCH_INDEX%"
if "%TORCH_INDEX%"=="" set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"

rem --- 1. uv ---------------------------------------------------------
set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV%" (
  where uv >nul 2>nul && set "UV=uv"
)
if not exist "%UV%" if not "%UV%"=="uv" (
  echo [install-ltx2] bootstrapping uv (standalone installer)
  powershell -NoProfile -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "UV=%USERPROFILE%\.local\bin\uv.exe"
)
if not exist "%UV%" if not "%UV%"=="uv" (
  echo [install-ltx2] ERROR: could not find or install uv. Install it manually from https://astral.sh/uv
  exit /b 1
)
echo [install-ltx2] using uv: %UV%

rem --- 2. fetch the repo --------------------------------------------
if not exist "%SVC%" mkdir "%SVC%"
if exist "%REPO%\pyproject.toml" (
  echo [install-ltx2] reusing existing LTX-2 checkout at %REPO%
) else (
  echo [install-ltx2] downloading the Lightricks/LTX-2 repo
  curl.exe -L -o "%SVC%\ltx2.zip" "https://github.com/Lightricks/LTX-2/archive/refs/heads/main.zip"
  if errorlevel 1 goto :fail
  echo [install-ltx2] extracting
  tar -xf "%SVC%\ltx2.zip" -C "%SVC%"
  if errorlevel 1 goto :fail
  del "%SVC%\ltx2.zip" >nul 2>nul
)
if not exist "%REPO%\pyproject.toml" (
  echo [install-ltx2] ERROR: repo not where expected (%REPO%). The zip layout may have changed.
  exit /b 1
)

rem --- 3. uv sync (no xformers) -------------------------------------
echo [install-ltx2] uv sync --frozen (this creates .venv and downloads a Python; no xformers)
pushd "%REPO%"
"%UV%" sync --frozen
if errorlevel 1 (
  echo [install-ltx2] uv sync --frozen failed; retrying without --frozen
  "%UV%" sync
  if errorlevel 1 ( popd & goto :fail )
)

rem --- 4. CUDA torch (PyPI default is +cpu on Windows) --------------
echo [install-ltx2] installing CUDA torch/torchaudio from %TORCH_INDEX%
"%UV%" pip install --python "%VPY%" --reinstall torch==2.9.1 torchaudio==2.9.1 --index-url %TORCH_INDEX%
if errorlevel 1 ( popd & goto :fail )

rem --- 5. server deps ----------------------------------------------
echo [install-ltx2] installing fastapi + uvicorn (for the JSON API server)
"%UV%" pip install --python "%VPY%" fastapi "uvicorn[standard]"
if errorlevel 1 ( popd & goto :fail )
popd

rem --- 6. weights (opt-in) -----------------------------------------
if "%OAIY_LTX_DOWNLOAD%"=="1" (
  echo [install-ltx2] downloading LTX-2.3 weights into %MODELS% (LARGE; resumes if interrupted)
  "%UV%" pip install --python "%VPY%" huggingface_hub
  "%VPY%" -c "import os;from huggingface_hub import hf_hub_download as d;r='Lightricks/LTX-2';o=os.environ['MODELS'];[d(repo_id=r,filename=f,local_dir=o) for f in ['ltx-2.3-22b-distilled-1.1.safetensors','ltx-2.3-spatial-upscaler-x2-1.0.safetensors']]"
  if errorlevel 1 goto :fail
  echo [install-ltx2] NOTE: the Gemma 3 text encoder is gated -- download it separately and place it in a model folder.
) else (
  echo [install-ltx2] skipping weight download. Point the service at the folder holding the
  echo [install-ltx2]   ltx-2.3-*-distilled / spatial-upscaler .safetensors + a gemma-3-* dir
  echo [install-ltx2]   via Settings -^> Model Folders (e.g. E:\ckpts).
)

echo [install-ltx2] DONE. Start "LTX-2.3 Video" in the app, then POST to
echo [install-ltx2]   http://127.0.0.1:17890/generate  with {"prompt":"..."}.
exit /b 0

:fail
echo [install-ltx2] FAILED (exit code %errorlevel%). Scroll up for the cause.
exit /b 1
