#!/usr/bin/env bash
set -euo pipefail

AUX_IMAGE="${AUX_IMAGE:-nvcr.io/nvidia/nre/nre-tools-ga:26.04}"
DATASET_DIR="${DATASET_DIR:-outputs/ncore}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}}"
DATASET_PATH="${DATASET_PATH:-}"
SHARD_FILE_PATTERN="${SHARD_FILE_PATTERN:-}"
CAMERA_IDS="${CAMERA_IDS:-}"
NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-auto}"

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is not set. Export it before running the auxiliary data container." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${DATASET_PATH}" && -z "${SHARD_FILE_PATTERN}" ]]; then
  echo "Set DATASET_PATH or SHARD_FILE_PATTERN, relative to DATASET_DIR or absolute inside the container." >&2
  echo "Example:" >&2
  echo "  DATASET_PATH=scene-0061.json bash scripts/run_nurec_aux_data.sh" >&2
  echo "  SHARD_FILE_PATTERN=scene-0061.zarr.itar bash scripts/run_nurec_aux_data.sh" >&2
  exit 1
fi

INPUT_ARGS=()
if [[ -n "${DATASET_PATH}" ]]; then
  if [[ "${DATASET_PATH}" = /* ]]; then
    CONTAINER_DATASET_PATH="${DATASET_PATH}"
  else
    CONTAINER_DATASET_PATH="/workdir/dataset/${DATASET_PATH}"
  fi
  INPUT_ARGS+=(--dataset-path="${CONTAINER_DATASET_PATH}")
fi

if [[ -n "${SHARD_FILE_PATTERN}" ]]; then
  if [[ "${SHARD_FILE_PATTERN}" = /* ]]; then
    CONTAINER_SHARD_FILE_PATTERN="${SHARD_FILE_PATTERN}"
  else
    CONTAINER_SHARD_FILE_PATTERN="/workdir/dataset/${SHARD_FILE_PATTERN}"
  fi
  INPUT_ARGS+=(--shard-file-pattern="${CONTAINER_SHARD_FILE_PATTERN}")
fi

CAMERA_ARGS=()
if [[ -n "${CAMERA_IDS}" ]]; then
  read -r -a CAMERA_ID_ARRAY <<< "${CAMERA_IDS}"
  for camera_id in "${CAMERA_ID_ARRAY[@]}"; do
    CAMERA_ARGS+=(--camera-id="${camera_id}")
  done
fi

docker run --shm-size=2g --rm --gpus all \
  -e "NGC_API_KEY=${NGC_API_KEY}" \
  --volume "${REPO_ROOT}/${DATASET_DIR}:/workdir/dataset" \
  --volume "${REPO_ROOT}/${OUTPUT_DIR}:/workdir/output" \
  "${AUX_IMAGE}" \
  "${INPUT_ARGS[@]}" \
  --output-dir=/workdir/output \
  "${CAMERA_ARGS[@]}" \
  --store-meta \
  --no-seg-logits \
  --lidar-seg-camvis \
  --numba-num-threads="${NUMBA_NUM_THREADS}"
