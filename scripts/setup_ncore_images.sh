#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-nsb/ncore-conda-base:2026-07-10}"
CONVERTER_IMAGE="${CONVERTER_IMAGE:-nsb/ncore-converter:2026-07-17-dense-v1}"
BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-1}"
NO_CACHE="${NO_CACHE:-0}"
UPSTREAM_BASE_IMAGE="${UPSTREAM_BASE_IMAGE:-nvcr.io/nvidia/cuda:12.8.0-base-ubuntu22.04}"
MINICONDA_BASE_URL="${MINICONDA_BASE_URL:-https://repo.anaconda.com/miniconda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_ARGS=(build --pull)
if [[ "${NO_CACHE}" == "1" ]]; then
  BUILD_ARGS+=(--no-cache)
fi

if [[ "${BUILD_BASE_IMAGE}" == "1" ]]; then
  echo "Building ${BASE_IMAGE}"
  docker "${BUILD_ARGS[@]}" \
    --build-arg "UPSTREAM_BASE_IMAGE=${UPSTREAM_BASE_IMAGE}" \
    --build-arg "MINICONDA_BASE_URL=${MINICONDA_BASE_URL}" \
    -f docker/ncore-conda-base.Dockerfile -t "${BASE_IMAGE}" .
else
  docker image inspect "${BASE_IMAGE}" >/dev/null
  echo "Reusing existing base image: ${BASE_IMAGE}"
fi

echo "Building ${CONVERTER_IMAGE}"
docker build \
  -f docker/ncore-converter.Dockerfile \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${CONVERTER_IMAGE}" \
  .

echo "Validating ${CONVERTER_IMAGE}"
docker run --rm "${CONVERTER_IMAGE}"

echo "NCore converter images are ready on this Docker host:"
echo "  ${BASE_IMAGE}"
echo "  ${CONVERTER_IMAGE}"
