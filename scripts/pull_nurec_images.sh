#!/usr/bin/env bash
set -euo pipefail

NUREC_IMAGE="${NUREC_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}"
AUX_IMAGE="${AUX_IMAGE:-nvcr.io/nvidia/nre/nre-tools-ga:26.04}"

echo "Checking Docker GPU runtime..."
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is not set. Export it before pulling/running NuRec containers." >&2
  echo "Example: export NGC_API_KEY=<your-ngc-api-key>" >&2
  exit 1
fi

echo "Logging in to nvcr.io with NGC API key..."
printf '%s\n' "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo "Pulling NuRec training image: ${NUREC_IMAGE}"
docker pull "${NUREC_IMAGE}"

echo "Pulling NuRec auxiliary data image: ${AUX_IMAGE}"
docker pull "${AUX_IMAGE}"

echo "NuRec images are ready:"
echo "  ${NUREC_IMAGE}"
echo "  ${AUX_IMAGE}"
