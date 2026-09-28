#!/usr/bin/env sh
# install-lance.sh
#
# The Lance (image+video) service is GPU-heavy (Python 3.11 CUDA venv, PyTorch,
# the ByteDance Lance repo, weight downloads, and several patches). The
# automated Linux installer isn't provided yet — the Windows .bat hasn't been
# ported + validated on a Linux GPU box.
#
# Manual setup on a CUDA Linux host:
#   1. Create a Python 3.11 venv (oaiy venv create lance --req "torch==2.8 --index-url https://download.pytorch.org/whl/cu128").
#   2. Fetch the Lance repo into <dataDir>/services/lance, pip install its requirements.
#   3. Download weights (HuggingFace hub) into the models dir.
#   4. Point the lance template's run.command at <dataDir>/venvs/lance/bin/python and run lance_server.py.
# See desktop/README.md.

echo "[install-lance] Automated Linux install for the Lance GPU image+video service is not implemented yet." >&2
echo "[install-lance] Set it up manually on a CUDA host — see the header of this script / desktop/README.md." >&2
exit 1
