#!/usr/bin/env bash

set -euo pipefail

IMAGE="${UENV_IMAGE:-prgenv-gnu/24.11:v1}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${REPO_DIR}/.venv-uenv"
ENV_FILE="${REPO_DIR}/.env"
REQUIREMENTS_FILE="${REPO_DIR}/requirements.txt"
REQUIREMENTS_STAMP="${VENV_PATH}/.requirements.txt.sha256"

if ! command -v uenv >/dev/null 2>&1; then
	echo "uenv is not available in PATH." >&2
	exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
	cat >&2 <<EOF
Missing virtual environment at ${VENV_PATH}.

Create it with:
./scripts/setup_uenv_python.sh
EOF
	exit 1
fi

if [[ $# -eq 0 ]]; then
	cat >&2 <<EOF
Usage: ./run_uenv.sh <command> [args...]

Examples:
./run_uenv.sh python scripts/prefetch_hf_assets.py --skip-model --skip-datasets --hf-parquet-repo fffoivos/glossapi-greek-nanochat-pretraining-dataset --parquet-file 1000_prwta_xronia_ellhnikhs.parquet
./run_uenv.sh bash
EOF
	exit 1
fi

escaped_args=()
for arg in "$@"; do
	escaped_args+=("$(printf '%q' "$arg")")
done

requirements_hash=""
installed_requirements_hash=""
if [[ -f "${REQUIREMENTS_FILE}" ]]; then
	requirements_hash="$(sha256sum "${REQUIREMENTS_FILE}" | awk '{print $1}')"
	if [[ -f "${REQUIREMENTS_STAMP}" ]]; then
		installed_requirements_hash="$(<"${REQUIREMENTS_STAMP}")"
	fi
fi

cache_root='${SCRATCH:-"${HOME}/scratch"}'

command_string="cd $(printf '%q' "${REPO_DIR}")"
command_string+=" && source .venv-uenv/bin/activate"
command_string+=" && export PYTHONPATH=\"${REPO_DIR}/src\${PYTHONPATH:+:\${PYTHONPATH}}\""
command_string+=" && export HF_HOME=\"\${HF_HOME:-${cache_root}/hf}\""
command_string+=" && export HF_DATASETS_CACHE=\"\${HF_DATASETS_CACHE:-${cache_root}/hf_datasets}\""
command_string+=" && export MPLCONFIGDIR=\"\${MPLCONFIGDIR:-${cache_root}/mplconfig}\""
command_string+=" && mkdir -p \"\${HF_HOME}\" \"\${HF_DATASETS_CACHE}\" \"\${MPLCONFIGDIR}\""

if [[ -f "${ENV_FILE}" ]]; then
	command_string+=" && set -a && source $(printf '%q' "${ENV_FILE}") && set +a"
fi

if [[ -n "${requirements_hash}" && "${requirements_hash}" != "${installed_requirements_hash}" ]]; then
	command_string+=" && python -m pip install -r requirements.txt"
	command_string+=" && printf '%s\\n' '${requirements_hash}' > .venv-uenv/.requirements.txt.sha256"
fi

command_string+=" && ${escaped_args[*]}"

exec uenv run "${IMAGE}" --view=default -- bash -lc "${command_string}"