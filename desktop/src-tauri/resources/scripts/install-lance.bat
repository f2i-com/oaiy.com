@echo off
setlocal
rem ===================================================================
rem install-lance.bat -- one-click install of ByteDance "Lance", a 3B
rem unified image+video generate/edit model. Runs its own Gradio server.
rem
rem Run by the OAIY Companion (sets OAIY_DATA_DIR / OAIY_MODELS_DIR /
rem OAIY_SCRIPTS_DIR). ASCII-only on purpose.
rem
rem Steps: standalone Python 3.11 -> venv -> pinned CUDA PyTorch 2.8 ->
rem fetch the bytedance/Lance repo (no git) -> requirements.txt ->
rem (optional) flash-attn -> download weights into their OWN folder under the
rem models dir (e.g. E:\models\lance). Lance finds them via the
rem LANCE_MODEL_BASE_DIR env set on the service -- no junction, weights off C:.
rem
rem Why 3.11 (not the companion's bundled 3.12): Lance pins old deps
rem (numpy 1.23.5, opencv 4.7.0.72, ...) that have NO Python 3.12 wheels, so
rem 3.12 fails at `pip install -r requirements.txt`. Lance's own recipe is
rem python=3.11; we fetch a standalone 3.11 just for this venv.
rem
rem flash-attn is SKIPPED by default: building it on Windows is slow and
rem fragile, and Lance runs with PyTorch SDPA attention without it. If
rem Lance later complains "No module named flash_attn", set
rem OAIY_LANCE_FLASH=1 and re-run to attempt the build. Needs a 40GB-class
rem GPU per the model card (A100); expect tight fit / offloading below that.
rem ===================================================================

echo [install-lance] starting Lance (image+video) service install

if "%OAIY_DATA_DIR%"=="" (
  echo [install-lance] ERROR: OAIY_DATA_DIR is not set. Run Install from the OAIY app.
  exit /b 1
)

set "DATA=%OAIY_DATA_DIR%"
set "PY=%DATA%\python\python.exe"
set "SCRIPTS=%OAIY_SCRIPTS_DIR%"
if "%SCRIPTS%"=="" set "SCRIPTS=%DATA%\scripts"
set "VENV=%DATA%\venvs\lance"
set "VPY=%VENV%\Scripts\python.exe"
set "SVC=%DATA%\services\lance"
set "REPO=%SVC%\Lance-main"
rem Weights live in their OWN folder under the models dir (E:\models\lance),
rem NOT under AppData. lance_gradio.py reads a relative "downloads" dir, so we
rem junction <repo>\downloads -> %LANCEW% below.
set "MODELS=%OAIY_MODELS_DIR%"
if "%MODELS%"=="" set "MODELS=%DATA%\models"
set "LANCEW=%MODELS%\lance"
set "REPODL=%REPO%\downloads"
set "TORCH_INDEX=%OAIY_TORCH_INDEX%"
if "%TORCH_INDEX%"=="" set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
rem Standalone Python 3.11 just for the Lance venv (its deps lack 3.12 wheels).
set "PY311DIR=%SVC%\python311"
set "PY311=%PY311DIR%\python\python.exe"
set "PBS_REL=20260510"
set "PBS_PY311=3.11.15"

if not exist "%PY%" (
  echo [install-lance] ERROR: bundled Python not found at %PY%
  echo [install-lance] Install Python first in the OAIY app's Python tab, then re-run.
  exit /b 1
)

if not exist "%SVC%" mkdir "%SVC%"

if not exist "%PY311%" (
  echo [install-lance] fetching standalone Python 3.11 -- Lance needs 3.11, not 3.12
  "%PY%" "%SCRIPTS%\fetch_zip.py" "https://github.com/astral-sh/python-build-standalone/releases/download/%PBS_REL%/cpython-%PBS_PY311%+%PBS_REL%-x86_64-pc-windows-msvc-install_only.tar.gz" "%PY311DIR%"
  if errorlevel 1 goto :fail
  echo [install-lance] bootstrapping pip in the 3.11 runtime
  "%PY311%" -m ensurepip --upgrade
)

rem A venv left over from an earlier 3.12 attempt would be reused and fail
rem again at requirements; detect a non-3.11 venv and rebuild it.
set "VENV_OK="
if exist "%VPY%" (
  "%VPY%" -c "import sys;exit(0 if sys.version_info[:2]==(3,11) else 1)" 2>nul && set "VENV_OK=1"
)
if not defined VENV_OK if exist "%VENV%" (
  echo [install-lance] existing lance venv isn't Python 3.11; recreating it
  rmdir /s /q "%VENV%"
)

if not exist "%VPY%" (
  echo [install-lance] creating venv at %VENV% -- Python 3.11
  "%PY311%" -m venv "%VENV%"
  if errorlevel 1 goto :fail
) else (
  echo [install-lance] reusing existing Python 3.11 venv at %VENV%
)

echo [install-lance] upgrading pip
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [install-lance] installing CUDA PyTorch 2.8 from %TORCH_INDEX%
"%VPY%" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url %TORCH_INDEX%
if errorlevel 1 goto :fail

echo [install-lance] fetching the bytedance/Lance repo (no git required)
if not exist "%SVC%" mkdir "%SVC%"
"%VPY%" "%SCRIPTS%\fetch_zip.py" "https://github.com/bytedance/Lance/archive/refs/heads/main.zip" "%SVC%"
if errorlevel 1 goto :fail

echo [install-lance] installing requirements.txt
"%VPY%" -m pip install -r "%REPO%\requirements.txt"
if errorlevel 1 goto :fail

echo [install-lance] applying optional multi-GPU sharding patch (inert unless OAIY_LANCE_SHARD=1)
"%VPY%" "%SCRIPTS%\patch_lance_sharding.py" "%REPO%\lance_gradio.py"

echo [install-lance] patching Lance for off-repo weights + SDPA vision tower (no flash-attn)
"%VPY%" "%SCRIPTS%\patch_lance_paths.py" "%REPO%"

rem flash-attn: Lance HARD-imports it (modeling/lance/qwen2_navit.py imports
rem flash_attn_varlen_func at module load), so it is NOT optional -- without it
rem the Gradio server crashes on launch with "No module named flash_attn".
rem There is no Windows wheel matching torch2.8/cu128/cp311 on Blackwell, so by
rem default we install a small SDPA-backed `flash_attn` shim: exact attention via
rem torch.nn.functional.scaled_dot_product_attention (numerically identical to
rem flash-attn, just not fused), which runs on any GPU. Set OAIY_LANCE_FLASH=1 to
rem instead build the real flash-attn 2.8.3 (slow, often fails on Windows).
rem Single-line ifs only -- cmd mis-parses a ')' in echo text inside a (...) block.
set "SITEPK=%VENV%\Lib\site-packages"
if not "%OAIY_LANCE_FLASH%"=="1" echo [install-lance] installing SDPA flash_attn shim (exact attention; set OAIY_LANCE_FLASH=1 to build real flash-attn instead)
if not "%OAIY_LANCE_FLASH%"=="1" if exist "%SITEPK%\flash_attn.py" del /q "%SITEPK%\flash_attn.py"
if not "%OAIY_LANCE_FLASH%"=="1" copy /y "%SCRIPTS%\flash_attn_shim.py" "%SITEPK%\flash_attn.py" >nul
if not "%OAIY_LANCE_FLASH%"=="1" if not exist "%SITEPK%\flash_attn.py" goto :fail
if "%OAIY_LANCE_FLASH%"=="1" echo [install-lance] building real flash-attn 2.8.3 -- this can take a while
if "%OAIY_LANCE_FLASH%"=="1" if exist "%SITEPK%\flash_attn.py" del /q "%SITEPK%\flash_attn.py"
if "%OAIY_LANCE_FLASH%"=="1" "%VPY%" -m pip install flash-attn==2.8.3 --no-build-isolation
if "%OAIY_LANCE_FLASH%"=="1" if errorlevel 1 goto :fail

echo [install-lance] downloading Lance weights into %LANCEW%
echo [install-lance] LARGE download; resumes if interrupted
if not exist "%LANCEW%" mkdir "%LANCEW%"
rem Lance finds weights via LANCE_MODEL_BASE_DIR (set on the service), so they
rem live in the models folder, OUT of the repo + off C:. Relocate any partial
rem an earlier in-repo run left behind so we resume rather than restart.
if exist "%REPODL%" robocopy "%REPODL%" "%LANCEW%" /E /MOVE >nul
if exist "%REPODL%" rmdir /s /q "%REPODL%" 2>nul
set "LANCEDIR=%LANCEW%"
"%VPY%" -c "import os;from huggingface_hub import snapshot_download;snapshot_download(repo_id='bytedance-research/Lance', local_dir=os.environ['LANCEDIR'], allow_patterns=['*.json','*.safetensors','*.bin','*.pth','*.txt','*.model'])"
if errorlevel 1 goto :fail

echo [install-lance] DONE. Start the "Lance (Image+Video)" service in the app,
echo [install-lance] then open its Gradio UI at http://127.0.0.1:17900/.
exit /b 0

:fail
echo [install-lance] FAILED (exit code %errorlevel%). Scroll up for the cause.
exit /b 1
