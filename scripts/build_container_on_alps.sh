#!/bin/bash

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-gsdg-qwen3:latest}"
SQSH_PATH="${SQSH_PATH:-${SCRATCH}/images/gsdg-qwen3_latest.sqsh}"
STORAGE_CONF_DIR="${HOME}/.config/containers"
STORAGE_CONF_PATH="${STORAGE_CONF_DIR}/storage.conf"

if [[ ! -f "${STORAGE_CONF_PATH}" ]]; then
	mkdir -p "${STORAGE_CONF_DIR}"
	cat > "${STORAGE_CONF_PATH}" <<EOF
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
EOF
	printf 'Created %s for Alps-safe Podman storage\n' "${STORAGE_CONF_PATH}"
fi

mkdir -p "$(dirname "${SQSH_PATH}")"
podman build -t "${IMAGE_TAG}" .
if [[ -f "${SQSH_PATH}" ]]; then
	rm -f "${SQSH_PATH}"
fi
if ! enroot import -x mount -o "${SQSH_PATH}" "podman://${IMAGE_TAG}"; then
	if [[ -s "${SQSH_PATH}" ]]; then
		printf 'enroot import returned non-zero, but %s was created successfully\n' "${SQSH_PATH}" >&2
	else
		exit 1
	fi
fi
