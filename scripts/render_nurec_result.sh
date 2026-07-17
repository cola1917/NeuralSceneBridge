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
NUREC_OUTPUT_DIR="${NUREC_OUTPUT_DIR:-outputs/nurec_1000step}"
ARTIFACT_PATH="${ARTIFACT_PATH:-}"
RENDER_DIR="${RENDER_DIR:-outputs/nurec_1000step_preview}"
CACHE_DIR="${CACHE_DIR:-.cache/nurec}"
CAMERA_IDS="${RENDER_CAMERA_IDS:-camera_front}"
FRAME_STEP="${FRAME_STEP:-10}"
IMAGE_SCALE="${IMAGE_SCALE:-0.5}"
IMAGE_FORMAT="${IMAGE_FORMAT:-png}"
SHM_SIZE="${SHM_SIZE:-32g}"
GPUS="${GPUS:-all}"
REPLICATE_TRAINING_VIEWS="${REPLICATE_TRAINING_VIEWS:-true}"
ALLOW_NONEMPTY_OUTPUT="${ALLOW_NONEMPTY_OUTPUT:-0}"
DRY_RUN="${DRY_RUN:-0}"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

absolute_from_repo() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${REPO_ROOT}/${value}"
  fi
}

[[ "${FRAME_STEP}" =~ ^[1-9][0-9]*$ ]] || fail "FRAME_STEP must be a positive integer."
[[ "${IMAGE_SCALE}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || fail "IMAGE_SCALE must be in (0, 1]."
[[ "${IMAGE_SCALE}" != "0" && "${IMAGE_SCALE}" != "0.0" ]] || fail "IMAGE_SCALE must be greater than zero."
[[ "${IMAGE_FORMAT}" =~ ^(png|jpg|jpeg)$ ]] || fail "IMAGE_FORMAT must be png, jpg, or jpeg."

NUREC_OUTPUT_ABS="$(absolute_from_repo "${NUREC_OUTPUT_DIR}")"
RENDER_ABS="$(absolute_from_repo "${RENDER_DIR}")"
CACHE_ABS="$(absolute_from_repo "${CACHE_DIR}")"

if [[ -z "${ARTIFACT_PATH}" ]]; then
  [[ -d "${NUREC_OUTPUT_ABS}" ]] || fail "NuRec output directory not found: ${NUREC_OUTPUT_ABS}"
  mapfile -t ARTIFACTS < <(find "${NUREC_OUTPUT_ABS}" -mindepth 3 -maxdepth 3 -type f \
    -path '*/artifacts/last.usdz' -print | sort)
  (( ${#ARTIFACTS[@]} == 1 )) || fail "Expected exactly one artifacts/last.usdz below ${NUREC_OUTPUT_ABS}; found ${#ARTIFACTS[@]}. Set ARTIFACT_PATH explicitly."
  ARTIFACT_ABS="${ARTIFACTS[0]}"
else
  ARTIFACT_ABS="$(absolute_from_repo "${ARTIFACT_PATH}")"
fi

[[ -s "${ARTIFACT_ABS}" ]] || fail "NuRec artifact is missing or empty: ${ARTIFACT_ABS}"

if [[ -d "${RENDER_ABS}" && -n "$(find "${RENDER_ABS}" -mindepth 1 -maxdepth 1 -print -quit)" && "${ALLOW_NONEMPTY_OUTPUT}" != "1" ]]; then
  fail "Render output is not empty: ${RENDER_ABS}. Choose another RENDER_DIR or set ALLOW_NONEMPTY_OUTPUT=1."
fi

command -v docker >/dev/null 2>&1 || fail "docker is not available."
docker image inspect "${NUREC_IMAGE}" >/dev/null 2>&1 || \
  fail "NuRec image is not available locally: ${NUREC_IMAGE}. Run scripts/pull_nurec_images.sh first."

mkdir -p "${RENDER_ABS}" "${CACHE_ABS}"
ARTIFACT_DIR="$(dirname "${ARTIFACT_ABS}")"
ARTIFACT_NAME="$(basename "${ARTIFACT_ABS}")"

IFS=',' read -r -a CAMERA_LIST <<< "${CAMERA_IDS}"
CAMERA_ARGS=()
for camera in "${CAMERA_LIST[@]}"; do
  [[ -n "${camera}" ]] || fail "RENDER_CAMERA_IDS contains an empty camera id."
  CAMERA_ARGS+=(--camera-id "${camera}")
done
DOCKER_ENV=()
if [[ -n "${NGC_API_KEY:-}" ]]; then
  DOCKER_ENV+=(--env NGC_API_KEY)
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DOCKER_ENV+=(--env CUDA_VISIBLE_DEVICES)
fi

RENDER_ARGS=(
  -- render
  --artifact-path "/artifact/${ARTIFACT_NAME}"
  --output-dir /workdir/output
  --frame-step "${FRAME_STEP}"
  --image-scale "${IMAGE_SCALE}"
  "${CAMERA_ARGS[@]}"
  --image-format "${IMAGE_FORMAT}"
  --frame-naming frame-end-timestamp
)
if [[ "${REPLICATE_TRAINING_VIEWS}" == "true" ]]; then
  RENDER_ARGS+=(--replicate-training-views)
elif [[ "${REPLICATE_TRAINING_VIEWS}" == "false" ]]; then
  RENDER_ARGS+=(--no-replicate-training-views)
else
  fail "REPLICATE_TRAINING_VIEWS must be true or false."
fi

echo "Starting NuRec render:"
echo "  artifact: ${ARTIFACT_ABS}"
echo "  cameras: ${CAMERA_IDS}"
echo "  frame step: ${FRAME_STEP}"
echo "  image scale: ${IMAGE_SCALE}"
echo "  output: ${RENDER_ABS}"

COMMAND=(
  docker run --shm-size="${SHM_SIZE}" --rm --gpus "${GPUS}"
  --network host
  "${DOCKER_ENV[@]}"
  --volume "${ARTIFACT_DIR}:/artifact:ro"
  --volume "${RENDER_ABS}:/workdir/output"
  --volume "${CACHE_ABS}:/home/.cache"
  "${NUREC_IMAGE}"
  "${RENDER_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '  %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

"${COMMAND[@]}"

RENDERED_COUNT="$(find "${RENDER_ABS}" -type f \
  \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)"
(( RENDERED_COUNT > 0 )) || fail "NuRec render completed without producing image files."
echo "NuRec render completed: ${RENDERED_COUNT} image(s) in ${RENDER_ABS}"
