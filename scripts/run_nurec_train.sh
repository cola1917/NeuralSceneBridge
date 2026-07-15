#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/config/nurec-smoke.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

NUREC_IMAGE="${NUREC_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}"
DATASET_DIR="${DATASET_DIR:-outputs/ncore}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/nurec_smoke}"
CAMERA_IDS="${CAMERA_IDS:-camera_front,camera_front_left,camera_front_right}"
LIDAR_IDS="${LIDAR_IDS:-lidar_top}"
CONFIG_NAME="${CONFIG_NAME:-configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"
SHM_SIZE="${SHM_SIZE:-32g}"
GPUS="${GPUS:-all}"

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is not set." >&2
  exit 1
fi

if [[ -z "${DATASET_PATH}" ]]; then
  echo "DATASET_PATH is required. Set it in config/nurec-smoke.env after NCore conversion." >&2
  exit 1
fi

if [[ "${DATASET_PATH}" = /* ]]; then
  echo "DATASET_PATH must be relative to DATASET_DIR." >&2
  exit 1
fi

if [[ "${SHM_SIZE}" =~ ^([0-9]+)[gG]$ ]]; then
  REQUESTED_SHM_MIB="$((BASH_REMATCH[1] * 1024))"
  HOST_RAM_MIB="$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)"
  if (( REQUESTED_SHM_MIB > HOST_RAM_MIB * 80 / 100 )); then
    echo "SHM_SIZE=${SHM_SIZE} exceeds 80% of host RAM (${HOST_RAM_MIB} MiB)." >&2
    echo "Choose a smaller value such as 32g for a 64 GB-class host." >&2
    exit 1
  fi
fi

DATASET_ABS="${REPO_ROOT}/${DATASET_DIR}"
MANIFEST_ABS="${DATASET_ABS}/${DATASET_PATH}"
OUTPUT_ABS="${REPO_ROOT}/${OUTPUT_DIR}"

if [[ ! -f "${MANIFEST_ABS}" ]]; then
  echo "NCore manifest not found: ${MANIFEST_ABS}" >&2
  exit 1
fi

shopt -s nullglob
AUX_FILES=("${DATASET_ABS}"/*.aux.*.zarr "${DATASET_ABS}"/*.aux.*.zarr.itar)
shopt -u nullglob
if (( ${#AUX_FILES[@]} == 0 )); then
  echo "No NuRec auxiliary stores found beside the NCore manifest in ${DATASET_ABS}." >&2
  echo "Run scripts/run_nurec_aux_data.sh first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ABS}"

DOCKER_ENV=(-e "NGC_API_KEY=${NGC_API_KEY}")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DOCKER_ENV+=(-e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

echo "Starting NuRec smoke training:"
echo "  manifest: ${MANIFEST_ABS}"
echo "  cameras: ${CAMERA_IDS}"
echo "  lidar: ${LIDAR_IDS}"
echo "  epochs: ${MAX_EPOCHS}"
if [[ -n "${SAMPLES_PER_EPOCH}" ]]; then
  echo "  samples per epoch: ${SAMPLES_PER_EPOCH}"
fi
echo "  output: ${OUTPUT_ABS}"

TRAINER_ARGS=("trainer.max_epochs=${MAX_EPOCHS}")
DATASET_ARGS=()
if [[ -n "${SAMPLES_PER_EPOCH}" ]]; then
  DATASET_ARGS+=("dataset.n_samples_per_epoch=${SAMPLES_PER_EPOCH}")
fi

docker run --shm-size="${SHM_SIZE}" --rm --gpus "${GPUS}" \
  "${DOCKER_ENV[@]}" \
  --volume "${DATASET_ABS}:/workdir/dataset" \
  --volume "${OUTPUT_ABS}:/workdir/output" \
  "${NUREC_IMAGE}" \
  mode=train \
  out_dir=/workdir/output \
  --config-name="${CONFIG_NAME}" \
  "dataset.path=/workdir/dataset/${DATASET_PATH}" \
  "dataset.camera_ids=[${CAMERA_IDS}]" \
  "dataset.lidar_ids=[${LIDAR_IDS}]" \
  dataset.aux_data=True \
  "${DATASET_ARGS[@]}" \
  "${TRAINER_ARGS[@]}"
