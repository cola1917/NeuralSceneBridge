#!/usr/bin/env bash
set -euo pipefail

CUDA_TEST_IMAGE="${CUDA_TEST_IMAGE:-nvidia/cuda:12.8.0-base-ubuntu22.04}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || fail "docker is not installed"
docker info >/dev/null 2>&1 || fail "docker daemon is unavailable or the current user lacks permission"

echo "Host GPU:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "  WARN: host nvidia-smi is not on PATH"
fi

echo "Checking Docker GPU access with ${CUDA_TEST_IMAGE}..."
if ! DOCKER_GPU_OUTPUT="$(docker run --rm --gpus all "${CUDA_TEST_IMAGE}" nvidia-smi 2>&1)"; then
  printf '%s\n' "${DOCKER_GPU_OUTPUT}" >&2
  if grep -qiE 'operation not permitted|failed to mount|failed to unmount' <<< "${DOCKER_GPU_OUTPUT}"; then
    fail "Docker cannot create container mount namespaces. If this host is itself a container, relaunch it as privileged or use a VM/bare-metal host with Docker and NVIDIA Container Toolkit installed on the host."
  fi
  fail "Docker cannot access the NVIDIA GPU"
fi
echo "  OK: Docker can access the NVIDIA GPU"

FREE_DISK_GB="$(df -Pk "${REPO_ROOT}" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
echo "Free disk at repository: ${FREE_DISK_GB} GB"
if (( FREE_DISK_GB < MIN_FREE_DISK_GB )); then
  echo "  WARN: less than ${MIN_FREE_DISK_GB} GB free; conversion and reconstruction outputs may exhaust disk"
fi

if [[ -n "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY: set"
else
  echo "NGC_API_KEY: not set (required for NVIDIA NRE images)"
fi

for image in \
  "nsb/ncore-converter:2026-07-10" \
  "nvcr.io/nvidia/nre/nre-tools-ga:26.04" \
  "nvcr.io/nvidia/nre/nre-ga:26.04"; do
  if docker image inspect "${image}" >/dev/null 2>&1; then
    echo "Image present: ${image}"
  else
    echo "Image missing: ${image}"
  fi
done

echo "SERVER PREFLIGHT OK"
