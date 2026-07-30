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
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
CHECKPOINT_FRIENDLY_BACKWARD="${CHECKPOINT_FRIENDLY_BACKWARD:-0}"
TRACK_LABEL_SOURCES="${TRACK_LABEL_SOURCES:-AUTOLABEL}"
TRACK_VEHICLE_CLASSES="${TRACK_VEHICLE_CLASSES:-automobile}"
TRACK_PEDESTRIAN_CLASSES="${TRACK_PEDESTRIAN_CLASSES:-pedestrian}"
REQUIRE_DYNAMIC_TRACKS="${REQUIRE_DYNAMIC_TRACKS:-0}"
NCORE_VALIDATION_IMAGE="${NCORE_VALIDATION_IMAGE:-}"
REQUIRE_LIDAR_SUPERVISION="${REQUIRE_LIDAR_SUPERVISION:-0}"
REQUIRE_RENDERABLE_LIDAR="${REQUIRE_RENDERABLE_LIDAR:-0}"
N_TRAIN_SAMPLE_LIDAR_RAYS="${N_TRAIN_SAMPLE_LIDAR_RAYS:-}"
RATIO_LIDAR_SAMPLES="${RATIO_LIDAR_SAMPLES:-}"
LIDAR_LOSS_WEIGHT="${LIDAR_LOSS_WEIGHT:-}"
LIDAR_INTENSITY_LOSS_WEIGHT="${LIDAR_INTENSITY_LOSS_WEIGHT:-}"
LIDAR_RAYDROP_LOSS_WEIGHT="${LIDAR_RAYDROP_LOSS_WEIGHT:-}"
DYNAMIC_TRACK_POINTS_PER_TRACK="${DYNAMIC_TRACK_POINTS_PER_TRACK:-}"
DYNAMIC_TRACK_POINTS_PER_LAYER="${DYNAMIC_TRACK_POINTS_PER_LAYER:-}"
DYNAMIC_TRACK_KEEP_ALL_POSES="${DYNAMIC_TRACK_KEEP_ALL_POSES:-}"
DYNAMIC_TRACK_INIT_STEP_FRAME="${DYNAMIC_TRACK_INIT_STEP_FRAME:-1}"
TRACK_MIN_DISTANCE_M="${TRACK_MIN_DISTANCE_M:-}"
TRACK_MIN_DISPLACEMENT_M="${TRACK_MIN_DISPLACEMENT_M:-}"
TRACK_MIN_SPEED_MS="${TRACK_MIN_SPEED_MS:-}"
TRACK_USE_DISPLACEMENT_AND_DISTANCE="${TRACK_USE_DISPLACEMENT_AND_DISTANCE:-}"
NCORE_MIN_TRACK_DISPLACEMENT_M="${NCORE_MIN_TRACK_DISPLACEMENT_M:-${TRACK_MIN_DISPLACEMENT_M:-1.0}}"
NCORE_MIN_TRACK_SPEED_MS="${NCORE_MIN_TRACK_SPEED_MS:-${TRACK_MIN_SPEED_MS:-0.1}}"
DYNAMIC_TRACK_IDS="${DYNAMIC_TRACK_IDS:-}"
DYNAMIC_RIGID_TRACK_IDS="${DYNAMIC_RIGID_TRACK_IDS:-}"
NCORE_SELECTED_TRACK_IDS="${NCORE_SELECTED_TRACK_IDS:-}"

case "${MODE}" in
  train|trainval) ;;
  *)
    echo "MODE must be train or trainval, got: ${MODE}" >&2
    exit 1
    ;;
esac

for variable in VAL_LIDAR REQUIRE_LIDAR_SUPERVISION REQUIRE_RENDERABLE_LIDAR; do
  value="${!variable}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${variable} must be 0 or 1, got: ${value}" >&2
    exit 1
  fi
done
if ! [[ "${DYNAMIC_TRACK_INIT_STEP_FRAME}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DYNAMIC_TRACK_INIT_STEP_FRAME must be a positive integer, got: ${DYNAMIC_TRACK_INIT_STEP_FRAME}" >&2
  exit 1
fi
if [[ "${REQUIRE_RENDERABLE_LIDAR}" == "1" && "${REQUIRE_LIDAR_SUPERVISION}" != "1" ]]; then
  echo "REQUIRE_RENDERABLE_LIDAR=1 requires REQUIRE_LIDAR_SUPERVISION=1." >&2
  exit 1
fi
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
if [[ "${REQUIRE_RENDERABLE_LIDAR}" == "1" ]]; then
  for variable in LIDAR_INTENSITY_LOSS_WEIGHT LIDAR_RAYDROP_LOSS_WEIGHT; do
    if [[ -z "${!variable}" ]]; then
      echo "${variable} is required when REQUIRE_RENDERABLE_LIDAR=1." >&2
      exit 1
    fi
  done
fi
for variable in N_TRAIN_SAMPLE_LIDAR_RAYS RATIO_LIDAR_SAMPLES LIDAR_LOSS_WEIGHT LIDAR_INTENSITY_LOSS_WEIGHT LIDAR_RAYDROP_LOSS_WEIGHT; do
  value="${!variable}"
  if [[ -n "${value}" ]] && ! awk -v value="${value}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }'; then
    echo "${variable} must be a positive number, got: ${value}" >&2
    exit 1
  fi
done
for variable in DYNAMIC_TRACK_POINTS_PER_TRACK DYNAMIC_TRACK_POINTS_PER_LAYER; do
  value="${!variable}"
  if [[ -n "${value}" && ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${variable} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
done
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF}" && ! "${PYTORCH_CUDA_ALLOC_CONF}" =~ ^[A-Za-z0-9_:,=.-]+$ ]]; then
  echo "PYTORCH_CUDA_ALLOC_CONF contains unsupported characters." >&2
  exit 1
fi
if [[ "${CHECKPOINT_FRIENDLY_BACKWARD}" != "0" && "${CHECKPOINT_FRIENDLY_BACKWARD}" != "1" ]]; then
  echo "CHECKPOINT_FRIENDLY_BACKWARD must be 0 or 1, got: ${CHECKPOINT_FRIENDLY_BACKWARD}" >&2
  exit 1
fi
if [[ -n "${DYNAMIC_TRACK_KEEP_ALL_POSES}" && "${DYNAMIC_TRACK_KEEP_ALL_POSES}" != "0" && "${DYNAMIC_TRACK_KEEP_ALL_POSES}" != "1" ]]; then
  echo "DYNAMIC_TRACK_KEEP_ALL_POSES must be 0 or 1, got: ${DYNAMIC_TRACK_KEEP_ALL_POSES}" >&2
  exit 1
fi
for variable in TRACK_MIN_DISTANCE_M TRACK_MIN_DISPLACEMENT_M TRACK_MIN_SPEED_MS NCORE_MIN_TRACK_DISPLACEMENT_M NCORE_MIN_TRACK_SPEED_MS; do
  value="${!variable}"
  if [[ -n "${value}" ]] && ! awk -v value="${value}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0) }'; then
    echo "${variable} must be a non-negative number, got: ${value}" >&2
    exit 1
  fi
done
if [[ -n "${TRACK_USE_DISPLACEMENT_AND_DISTANCE}" && "${TRACK_USE_DISPLACEMENT_AND_DISTANCE}" != "0" && "${TRACK_USE_DISPLACEMENT_AND_DISTANCE}" != "1" ]]; then
  echo "TRACK_USE_DISPLACEMENT_AND_DISTANCE must be 0 or 1, got: ${TRACK_USE_DISPLACEMENT_AND_DISTANCE}" >&2
  exit 1
fi
if [[ -n "${DYNAMIC_TRACK_IDS}" ]]; then
  IFS=',' read -r -a DYNAMIC_TRACK_ID_LIST <<< "${DYNAMIC_TRACK_IDS}"
  if (( ${#DYNAMIC_TRACK_ID_LIST[@]} == 0 )); then
    echo "DYNAMIC_TRACK_IDS must contain at least one ID." >&2
    exit 1
  fi
  for track_id in "${DYNAMIC_TRACK_ID_LIST[@]}"; do
    if [[ ! "${track_id}" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
      echo "DYNAMIC_TRACK_IDS contains an invalid ID: ${track_id}" >&2
      exit 1
    fi
  done
fi
if [[ -n "${DYNAMIC_RIGID_TRACK_IDS}" ]]; then
  IFS=',' read -r -a DYNAMIC_RIGID_TRACK_ID_LIST <<< "${DYNAMIC_RIGID_TRACK_IDS}"
  if (( ${#DYNAMIC_RIGID_TRACK_ID_LIST[@]} == 0 )); then
    echo "DYNAMIC_RIGID_TRACK_IDS must contain at least one ID." >&2
    exit 1
  fi
  for track_id in "${DYNAMIC_RIGID_TRACK_ID_LIST[@]}"; do
    if [[ ! "${track_id}" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
      echo "DYNAMIC_RIGID_TRACK_IDS contains an invalid ID: ${track_id}" >&2
      exit 1
    fi
  done
fi
if [[ -n "${NCORE_SELECTED_TRACK_IDS}" ]]; then
  IFS=',' read -r -a NCORE_SELECTED_TRACK_ID_LIST <<< "${NCORE_SELECTED_TRACK_IDS}"
  for track_id in "${NCORE_SELECTED_TRACK_ID_LIST[@]}"; do
    if [[ ! "${track_id}" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
      echo "NCORE_SELECTED_TRACK_IDS contains an invalid ID: ${track_id}" >&2
      exit 1
    fi
  done
fi
if [[ -n "${DYNAMIC_TRACK_IDS}" && -n "${DYNAMIC_RIGID_TRACK_IDS}" ]]; then
  duplicate_dynamic_track_id="$(tr ',' '\n' <<< "${DYNAMIC_TRACK_IDS},${DYNAMIC_RIGID_TRACK_IDS}" | sort | uniq -d | head -n 1)"
  if [[ -n "${duplicate_dynamic_track_id}" ]]; then
    echo "dynamic track ID occurs in both layers: ${duplicate_dynamic_track_id}" >&2
    exit 1
  fi
fi
if [[ -z "${NCORE_SELECTED_TRACK_IDS}" ]]; then
  NCORE_SELECTED_TRACK_IDS="${DYNAMIC_RIGID_TRACK_IDS}"
  if [[ -n "${DYNAMIC_TRACK_IDS}" ]]; then
    NCORE_SELECTED_TRACK_IDS="${NCORE_SELECTED_TRACK_IDS:+${NCORE_SELECTED_TRACK_IDS},}${DYNAMIC_TRACK_IDS}"
  fi
fi

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
  NCORE_SELECTION_ARGS=()
  if [[ -n "${NCORE_SELECTED_TRACK_IDS}" ]]; then
    NCORE_SELECTION_ARGS+=(--selected-track-ids "${NCORE_SELECTED_TRACK_IDS}")
  fi
  docker run --rm \
    --volume "${DATASET_ABS}:/ncore-dataset:ro" \
    --volume "${SCRIPT_DIR}/validate_ncore_dynamic_tracks.py:/validate_ncore_dynamic_tracks.py:ro" \
    --volume "${OUTPUT_ABS}/launcher:/validation-output" \
    --entrypoint python \
    "${NCORE_VALIDATION_IMAGE}" \
    /validate_ncore_dynamic_tracks.py \
    "/ncore-dataset/${DATASET_PATH}" \
    --accepted-sources "${TRACK_LABEL_SOURCES}" \
    --vehicle-classes "${TRACK_VEHICLE_CLASSES}" \
    --pedestrian-classes "${TRACK_PEDESTRIAN_CLASSES}" \
    --min-displacement-m "${NCORE_MIN_TRACK_DISPLACEMENT_M}" \
    --min-median-speed-ms "${NCORE_MIN_TRACK_SPEED_MS}" \
    "${NCORE_SELECTION_ARGS[@]}" \
    --output /validation-output/ncore_dynamic_tracks.json
fi

DOCKER_ENV=()
if [[ -n "${NGC_API_KEY:-}" ]]; then
  DOCKER_ENV+=(--env NGC_API_KEY)
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DOCKER_ENV+=(--env CUDA_VISIBLE_DEVICES)
fi
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF}" ]]; then
  DOCKER_ENV+=(--env PYTORCH_CUDA_ALLOC_CONF)
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
echo "  require renderable lidar: ${REQUIRE_RENDERABLE_LIDAR}"
if [[ "${REQUIRE_DYNAMIC_TRACKS}" == "1" ]]; then
  echo "  NCore accepted classes: vehicles=${TRACK_VEHICLE_CLASSES} pedestrians=${TRACK_PEDESTRIAN_CLASSES}"
  echo "  NCore track audit thresholds: displacement=${NCORE_MIN_TRACK_DISPLACEMENT_M}m speed=${NCORE_MIN_TRACK_SPEED_MS}m/s"
fi
echo "  epochs: ${MAX_EPOCHS}"
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF}" ]]; then
  echo "  PyTorch CUDA allocator: ${PYTORCH_CUDA_ALLOC_CONF}"
fi
echo "  checkpoint-friendly backward: ${CHECKPOINT_FRIENDLY_BACKWARD}"
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
if [[ -n "${TRACK_MIN_DISTANCE_M}" ]]; then
  DATASET_ARGS+=("dataset.cuboid_tracks_params.track_min_distance_m=${TRACK_MIN_DISTANCE_M}")
fi
if [[ -n "${TRACK_MIN_DISPLACEMENT_M}" ]]; then
  DATASET_ARGS+=("dataset.cuboid_tracks_params.track_min_displacement_m=${TRACK_MIN_DISPLACEMENT_M}")
fi
if [[ -n "${TRACK_MIN_SPEED_MS}" ]]; then
  DATASET_ARGS+=("dataset.cuboid_tracks_params.track_min_speed_ms=${TRACK_MIN_SPEED_MS}")
fi
if [[ "${TRACK_USE_DISPLACEMENT_AND_DISTANCE}" == "1" ]]; then
  DATASET_ARGS+=("dataset.cuboid_tracks_params.use_displacement_and_distance=true")
elif [[ "${TRACK_USE_DISPLACEMENT_AND_DISTANCE}" == "0" ]]; then
  DATASET_ARGS+=("dataset.cuboid_tracks_params.use_displacement_and_distance=false")
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
TRACK_MODEL_ARGS=()
RENDERER_ARGS=()
if [[ "${CHECKPOINT_FRIENDLY_BACKWARD}" == "1" ]]; then
  RENDERER_ARGS+=("model.renderer.checkpoint_friendly_backward=true")
fi
if [[ -n "${DYNAMIC_TRACK_POINTS_PER_TRACK}" ]]; then
  for layer in dynamic_rigids dynamic_deformables; do
    TRACK_MODEL_ARGS+=(
      "model.layers.${layer}.initialization.num_point_cloud_points_per_track=${DYNAMIC_TRACK_POINTS_PER_TRACK}"
    )
  done
fi
if [[ -n "${DYNAMIC_TRACK_POINTS_PER_LAYER}" ]]; then
  for layer in dynamic_rigids dynamic_deformables; do
    TRACK_MODEL_ARGS+=(
      "model.layers.${layer}.initialization.num_point_cloud_points_in_layer=${DYNAMIC_TRACK_POINTS_PER_LAYER}"
    )
  done
fi
if [[ "${DYNAMIC_TRACK_KEEP_ALL_POSES}" == "1" ]]; then
  for layer in dynamic_rigids dynamic_deformables; do
    TRACK_MODEL_ARGS+=("model.layers.${layer}.initialization.keep_all_track_poses=true")
  done
fi
for layer in dynamic_rigids dynamic_deformables; do
  TRACK_MODEL_ARGS+=("+model.layers.${layer}.initialization.step_frame=${DYNAMIC_TRACK_INIT_STEP_FRAME}")
done
if [[ -n "${DYNAMIC_TRACK_IDS}" ]]; then
  TRACK_MODEL_ARGS+=("+model.layers.dynamic_deformables.tracks.ids=[${DYNAMIC_TRACK_IDS}]")
fi
if [[ -n "${DYNAMIC_RIGID_TRACK_IDS}" ]]; then
  TRACK_MODEL_ARGS+=("+model.layers.dynamic_rigids.tracks.ids=[${DYNAMIC_RIGID_TRACK_IDS}]")
fi
EXTRA_SIGNAL_ARGS=()
if [[ "${REQUIRE_RENDERABLE_LIDAR}" == "1" ]]; then
  # The production six-camera recipe enables camera semantic logits only.
  # NuRec gRPC applies the learned raydrop output before returning LiDAR
  # points, so geometry supervision alone can produce a valid-looking USDZ
  # whose render_lidar RPC is always empty. Compose the official 26.04 signal
  # configs into every rendered layer, including dynamic actors.
  for layer in background road dynamic_rigids dynamic_deformables; do
    EXTRA_SIGNAL_ARGS+=(
      "model/gaussians/extra_signal@model.layers.${layer}.extra_signal=[semantic_logits,intensity,raydrop]"
    )
  done
  LOSS_ARGS+=(
    "loss.intensity.lambda_=${LIDAR_INTENSITY_LOSS_WEIGHT}"
    "loss.raydrop.lambda_=${LIDAR_RAYDROP_LOSS_WEIGHT}"
  )
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
  "${EXTRA_SIGNAL_ARGS[@]}" \
  "${TRACK_MODEL_ARGS[@]}" \
  "${RENDERER_ARGS[@]}" \
  "${LOSS_ARGS[@]}" \
  "${TRAINER_ARGS[@]}"
