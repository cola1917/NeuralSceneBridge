#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-nsb/ncore-conda-base:2026-07-10}"
CONVERTER_IMAGE="${CONVERTER_IMAGE:-nsb/ncore-converter:2026-07-10}"
NO_CACHE="${NO_CACHE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_ARGS=(build --pull)
if [[ "${NO_CACHE}" == "1" ]]; then
  BUILD_ARGS+=(--no-cache)
fi

echo "Building ${BASE_IMAGE}"
docker "${BUILD_ARGS[@]}" -f docker/ncore-conda-base.Dockerfile -t "${BASE_IMAGE}" .

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
