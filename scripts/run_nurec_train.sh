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
MODE="${MODE:-train}"
DATASET_DIR="${DATASET_DIR:-outputs/ncore}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/nurec_smoke}"
CACHE_DIR="${CACHE_DIR:-.cache/nurec}"
CAMERA_IDS="${CAMERA_IDS:-camera_front,camera_front_left,camera_front_right}"
LIDAR_IDS="${LIDAR_IDS:-lidar_top}"
VAL_CAMERA_IDS="${VAL_CAMERA_IDS:-}"
VAL_LIDAR_IDS="${VAL_LIDAR_IDS:-}"
VAL_LIDAR="${VAL_LIDAR:-0}"
CONFIG_NAME="${CONFIG_NAME:-configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"
SHM_SIZE="${SHM_SIZE:-32g}"
GPUS="${GPUS:-all}"
TRACK_LABEL_SOURCES="${TRACK_LABEL_SOURCES:-AUTOLABEL}"
REQUIRE_DYNAMIC_TRACKS="${REQUIRE_DYNAMIC_TRACKS:-0}"
NCORE_VALIDATION_IMAGE="${NCORE_VALIDATION_IMAGE:-}"
REQUIRE_LIDAR_SUPERVISION="${REQUIRE_LIDAR_SUPERVISION:-0}"
N_TRAIN_SAMPLE_LIDAR_RAYS="${N_TRAIN_SAMPLE_LIDAR_RAYS:-}"
RATIO_LIDAR_SAMPLES="${RATIO_LIDAR_SAMPLES:-}"
LIDAR_LOSS_WEIGHT="${LIDAR_LOSS_WEIGHT:-}"

case "${MODE}" in
  train|trainval) ;;
  *)
    echo "MODE must be train or trainval, got: ${MODE}" >&2
    exit 1
    ;;
esac

for variable in VAL_LIDAR REQUIRE_LIDAR_SUPERVISION; do
  value="${!variable}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${variable} must be 0 or 1, got: ${value}" >&2
    exit 1
  fi
done
if [[ "${VAL_LIDAR}" == "1" && "${MODE}" != "trainval" ]]; then
  echo "VAL_LIDAR=1 requires MODE=trainval." >&2
  exit 1
fi
if [[ "${REQUIRE_LIDAR_SUPERVISION}" == "1" && -z "${LIDAR_IDS//,/}" ]]; then
  echo "LIDAR_IDS must not be empty when REQUIRE_LIDAR_SUPERVISION=1." >&2
  exit 1
fi
if [[ "${REQUIRE_LIDAR_SUPERVISION}" == "1" ]]; then
  for variable in N_TRAIN_SAMPLE_LIDAR_RAYS RATIO_LIDAR_SAMPLES LIDAR_LOSS_WEIGHT; do
    if [[ -z "${!variable}" ]]; then
      echo "${variable} is required when REQUIRE_LIDAR_SUPERVISION=1." >&2
      exit 1
    fi
  done
fi
for variable in N_TRAIN_SAMPLE_LIDAR_RAYS RATIO_LIDAR_SAMPLES LIDAR_LOSS_WEIGHT; do
  value="${!variable}"
  if [[ -n "${value}" ]] && ! awk -v value="${value}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }'; then
    echo "${variable} must be a positive number, got: ${value}" >&2
    exit 1
  fi
done

if [[ -z "${NGC_API_KEY:-}" ]]; then
  if ! docker image inspect "${NUREC_IMAGE}" >/dev/null 2>&1; then
    echo "NGC_API_KEY is not set and the NuRec image is not available locally." >&2
    exit 1
  fi
  echo "NGC_API_KEY is not set; using the already-pulled local NuRec image." >&2
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

if [[ "${CACHE_DIR}" = /* ]]; then
  CACHE_ABS="${CACHE_DIR}"
else
  CACHE_ABS="${REPO_ROOT}/${CACHE_DIR}"
fi

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

mkdir -p "${OUTPUT_ABS}" "${CACHE_ABS}"

if [[ "${REQUIRE_DYNAMIC_TRACKS}" == "1" ]]; then
  if [[ -z "${NCORE_VALIDATION_IMAGE}" ]]; then
    echo "NCORE_VALIDATION_IMAGE is required when REQUIRE_DYNAMIC_TRACKS=1." >&2
    exit 1
  fi
  if ! docker image inspect "${NCORE_VALIDATION_IMAGE}" >/dev/null 2>&1; then
    echo "NCore validation image is not available: ${NCORE_VALIDATION_IMAGE}" >&2
    exit 1
  fi
  mkdir -p "${OUTPUT_ABS}/launcher"
  docker run --rm \
    --volume "${DATASET_ABS}:/ncore-dataset:ro" \
    --volume "${SCRIPT_DIR}/validate_ncore_dynamic_tracks.py:/validate_ncore_dynamic_tracks.py:ro" \
    --volume "${OUTPUT_ABS}/launcher:/validation-output" \
    --entrypoint python \
    "${NCORE_VALIDATION_IMAGE}" \
    /validate_ncore_dynamic_tracks.py \
    "/ncore-dataset/${DATASET_PATH}" \
    --accepted-sources "${TRACK_LABEL_SOURCES}" \
    --vehicle-classes automobile \
    --pedestrian-classes pedestrian \
    --output /validation-output/ncore_dynamic_tracks.json
fi

DOCKER_ENV=()
if [[ -n "${NGC_API_KEY:-}" ]]; then
  DOCKER_ENV+=(--env NGC_API_KEY)
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DOCKER_ENV+=(--env CUDA_VISIBLE_DEVICES)
fi

echo "Starting NuRec ${MODE}:"
echo "  manifest: ${MANIFEST_ABS}"
echo "  cameras: ${CAMERA_IDS}"
echo "  lidar: ${LIDAR_IDS}"
if [[ -n "${VAL_CAMERA_IDS}" ]]; then
  echo "  validation cameras: ${VAL_CAMERA_IDS}"
fi
if [[ -n "${VAL_LIDAR_IDS}" ]]; then
  echo "  validation lidar: ${VAL_LIDAR_IDS}"
fi
echo "  validate lidar: ${VAL_LIDAR}"
echo "  require lidar supervision: ${REQUIRE_LIDAR_SUPERVISION}"
echo "  epochs: ${MAX_EPOCHS}"
if [[ -n "${SAMPLES_PER_EPOCH}" ]]; then
  echo "  samples per epoch: ${SAMPLES_PER_EPOCH}"
  if [[ "${MAX_EPOCHS}" =~ ^[0-9]+$ && "${SAMPLES_PER_EPOCH}" =~ ^[0-9]+$ ]]; then
    echo "  configured training steps: $((MAX_EPOCHS * SAMPLES_PER_EPOCH))"
  fi
fi
echo "  output: ${OUTPUT_ABS}"
echo "  persistent cache: ${CACHE_ABS}"

TRAINER_ARGS=("trainer.max_epochs=${MAX_EPOCHS}")
DATASET_ARGS=()
if [[ -n "${TRACK_LABEL_SOURCES}" ]]; then
  DATASET_ARGS+=("dataset.cuboid_tracks_params.track_label_sources=[${TRACK_LABEL_SOURCES}]")
fi
if [[ -n "${SAMPLES_PER_EPOCH}" ]]; then
  DATASET_ARGS+=("dataset.n_samples_per_epoch=${SAMPLES_PER_EPOCH}")
fi
if [[ -n "${VAL_CAMERA_IDS}" ]]; then
  DATASET_ARGS+=("dataset.val_camera_ids=[${VAL_CAMERA_IDS}]")
fi
if [[ -n "${VAL_LIDAR_IDS}" ]]; then
  DATASET_ARGS+=("dataset.val_lidar_ids=[${VAL_LIDAR_IDS}]")
fi
if [[ "${VAL_LIDAR}" == "1" ]]; then
  DATASET_ARGS+=("dataset.val_lidar=true")
else
  DATASET_ARGS+=("dataset.val_lidar=false")
fi
if [[ -n "${N_TRAIN_SAMPLE_LIDAR_RAYS}" ]]; then
  DATASET_ARGS+=("dataset.n_train_sample_lidar_rays=${N_TRAIN_SAMPLE_LIDAR_RAYS}")
fi
if [[ -n "${RATIO_LIDAR_SAMPLES}" ]]; then
  DATASET_ARGS+=("dataset.samplers.batch_sampler.ratio_lidar_samples=${RATIO_LIDAR_SAMPLES}")
fi
LOSS_ARGS=()
if [[ -n "${LIDAR_LOSS_WEIGHT}" ]]; then
  LOSS_ARGS+=("loss.lidar.lambda_=${LIDAR_LOSS_WEIGHT}")
fi

docker run --shm-size="${SHM_SIZE}" --rm --gpus "${GPUS}" \
  "${DOCKER_ENV[@]}" \
  --volume "${DATASET_ABS}:/workdir/dataset" \
  --volume "${OUTPUT_ABS}:/workdir/output" \
  --volume "${CACHE_ABS}:/home/.cache" \
  "${NUREC_IMAGE}" \
  "mode=${MODE}" \
  out_dir=/workdir/output \
  --config-name="${CONFIG_NAME}" \
  "dataset.path=/workdir/dataset/${DATASET_PATH}" \
  "dataset.camera_ids=[${CAMERA_IDS}]" \
  "dataset.lidar_ids=[${LIDAR_IDS}]" \
  dataset.aux_data=True \
  "${DATASET_ARGS[@]}" \
  "${LOSS_ARGS[@]}" \
  "${TRAINER_ARGS[@]}"
