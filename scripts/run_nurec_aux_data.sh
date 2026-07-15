#!/usr/bin/env bash
set -euo pipefail

AUX_IMAGE="${AUX_IMAGE:-nvcr.io/nvidia/nre/nre-tools-ga:26.04}"
DATASET_DIR="${DATASET_DIR:-outputs/ncore}"
DATASET_PATH="${DATASET_PATH:-}"
SHARD_FILE_PATTERN="${SHARD_FILE_PATTERN:-}"
CAMERA_IDS="${CAMERA_IDS:-}"
NUM_THREADS="${NUM_THREADS:-${NUMBA_NUM_THREADS:-auto}}"

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is not set. Export it before running the auxiliary data container." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${DATASET_PATH}" && -z "${SHARD_FILE_PATTERN}" ]]; then
  echo "Set DATASET_PATH or SHARD_FILE_PATTERN, relative to DATASET_DIR or absolute inside the container." >&2
  echo "Example:" >&2
  echo "  DATASET_PATH=scene-0061.json bash scripts/run_nurec_aux_data.sh" >&2
  echo "  SHARD_FILE_PATTERN=scene-0061.zarr.itar bash scripts/run_nurec_aux_data.sh" >&2
  exit 1
fi

# NuRec discovers auxiliary stores beside the NCore manifest/shard. When the
# input is nested below DATASET_DIR, preserve that relative directory in the
# default output location. An explicitly supplied OUTPUT_DIR always wins.
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="${DATASET_DIR}"
  OUTPUT_INPUT="${DATASET_PATH:-${SHARD_FILE_PATTERN}}"

  if [[ "${OUTPUT_INPUT}" == /workdir/dataset/* ]]; then
    OUTPUT_INPUT="${OUTPUT_INPUT#/workdir/dataset/}"
  fi

  if [[ "${OUTPUT_INPUT}" != /* ]]; then
    OUTPUT_PARENT="$(dirname -- "${OUTPUT_INPUT}")"
    if [[ "${OUTPUT_PARENT}" != "." ]]; then
      OUTPUT_DIR="${DATASET_DIR}/${OUTPUT_PARENT}"
    fi
  fi
fi

if [[ "${DATASET_DIR}" = /* ]]; then
  DATASET_HOST_DIR="${DATASET_DIR}"
else
  DATASET_HOST_DIR="${REPO_ROOT}/${DATASET_DIR}"
fi

if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_HOST_DIR="${OUTPUT_DIR}"
else
  OUTPUT_HOST_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_HOST_DIR}"

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
  NORMALIZED_CAMERA_IDS="${CAMERA_IDS//,/ }"
  read -r -a CAMERA_ID_ARRAY <<< "${NORMALIZED_CAMERA_IDS}"
  for camera_id in "${CAMERA_ID_ARRAY[@]}"; do
    CAMERA_ARGS+=(--camera-id="${camera_id}")
  done
fi

docker run --shm-size=2g --rm --gpus all \
  --env NGC_API_KEY \
  --volume "${DATASET_HOST_DIR}:/workdir/dataset" \
  --volume "${OUTPUT_HOST_DIR}:/workdir/output" \
  "${AUX_IMAGE}" \
  "${INPUT_ARGS[@]}" \
  --output-dir=/workdir/output \
  "${CAMERA_ARGS[@]}" \
  --store-meta \
  --no-seg-logits \
  --lidar-seg-camvis \
  --num-threads="${NUM_THREADS}"
