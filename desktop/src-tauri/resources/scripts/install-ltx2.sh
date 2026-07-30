#!/usr/bin/env sh
# install-ltx2.sh
#
# The LTX-2 text-to-video service is GPU-heavy (CUDA torch + the Lightricks
# repo + model weights). The automated Linux installer isn't provided yet —
# the Windows .bat hasn't been ported + validated on a Linux GPU box.
#
# Manual setup on a CUDA Linux host:
#   1. oaiy venv create ltx2 --req "torch --index-url https://download.pytorch.org/whl/cu128" --req fastapi --req uvicorn
#   2. Fetch the LTX-2 repo into <dataDir>/services/ltx2 and `uv sync` it.
#   3. Point the ltx2-video template's run.command at <dataDir>/venvs/ltx2/bin/python
#      and run ltx2_server.py.
# See desktop/README.md.

echo "[install-ltx2] Automated Linux install for the LTX-2 GPU video service is not implemented yet." >&2
echo "[install-ltx2] Set it up manually on a CUDA host — see the header of this script / desktop/README.md." >&2
exit 1
