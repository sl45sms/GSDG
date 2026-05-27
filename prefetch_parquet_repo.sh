#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Load HF_TOKEN from the local .env file (not committed).
# Avoid printing the token.
if [[ -f .env ]]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

: "${HF_TOKEN:?HF_TOKEN is not set. Put it in .env as HF_TOKEN=...}"
: "${HF_PARQUET_REPO:?Set HF_PARQUET_REPO=owner/dataset-repo}"
: "${PARQUET_FILES:?Set PARQUET_FILES=file.parquet,other*.parquet}"

PARQUET_PREFETCH_DIR="${PARQUET_PREFETCH_DIR:-${SCRATCH:-${ROOT_DIR}}/gsdg_parquet_prefetch/${HF_PARQUET_REPO//\//__}}"

cmd=(
	./run_uenv.sh
	python
	scripts/prefetch_hf_assets.py
	--skip-model
	--skip-datasets
	--hf-parquet-repo "$HF_PARQUET_REPO"
	--parquet-out-dir "$PARQUET_PREFETCH_DIR"
)

local_patterns=()
IFS=',' read -r -a parquet_patterns <<< "$PARQUET_FILES"
for parquet_pattern in "${parquet_patterns[@]}"; do
	trimmed_pattern="${parquet_pattern// /}"
	if [[ -z "$trimmed_pattern" ]]; then
		continue
	fi
	cmd+=(--parquet-file "$trimmed_pattern")
	if [[ "$trimmed_pattern" == */* ]]; then
		local_patterns+=("${PARQUET_PREFETCH_DIR}/${trimmed_pattern}")
	else
		local_patterns+=("${PARQUET_PREFETCH_DIR}/data/${trimmed_pattern}")
	fi
done

if [[ ${#local_patterns[@]} -eq 0 ]]; then
	echo "PARQUET_FILES did not contain any usable patterns." >&2
	exit 2
fi

echo "Prefetching parquet files from ${HF_PARQUET_REPO} into ${PARQUET_PREFETCH_DIR}" >&2
"${cmd[@]}"

local_patterns_csv="$(IFS=,; echo "${local_patterns[*]}")"

echo "Prefetched parquet files into ${PARQUET_PREFETCH_DIR}" >&2
echo "Use the local parquet copy with:" >&2
echo "  unset HF_PARQUET_REPO" >&2
echo "  export PARQUET_FILES='${local_patterns_csv}'" >&2