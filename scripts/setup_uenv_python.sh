#!/bin/bash

set -euo pipefail

UENV_IMAGE="${UENV_IMAGE:-prgenv-gnu/24.11:v1}"
UENV_VIEW="${UENV_VIEW:-default}"
VENV_DIR="${VENV_DIR:-.venv-uenv}"

uenv image pull "${UENV_IMAGE}"
uenv run "${UENV_IMAGE}" --view="${UENV_VIEW}" -- bash -lc "
set -euo pipefail
python3 -m venv '${VENV_DIR}'
source '${VENV_DIR}/bin/activate'
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall src scripts
python --version
"
