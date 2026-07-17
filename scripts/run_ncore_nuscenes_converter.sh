#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-nsb/ncore-converter:2026-07-10}"
ROOT_DIR="${ROOT_DIR:-data/nuscenes-mini-scene-0061}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ncore}"
VERSION="${VERSION:-v1.0-mini}"
SCENE_NAME="${SCENE_NAME:-}"
SCENE_TOKEN="${SCENE_TOKEN:-}"
STORE_TYPE="${STORE_TYPE:-itar}"
PROFILE="${PROFILE:-separate-sensors}"
CUBOID_SAMPLING="${CUBOID_SAMPLING:-lidar-sweeps}"
LIDAR_MODEL_RESOLUTION="${LIDAR_MODEL_RESOLUTION:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${REPO_ROOT}/${OUTPUT_DIR}"

if [[ -n "${SCENE_NAME}" && -n "${SCENE_TOKEN}" ]]; then
  echo "Set only one of SCENE_NAME or SCENE_TOKEN." >&2
  exit 2
fi
if [[ -z "${SCENE_NAME}" && -z "${SCENE_TOKEN}" ]]; then
  SCENE_TOKEN="cc8c0bf57f984915a77078b10eb33198"
fi

if [[ -n "${SCENE_TOKEN}" ]]; then
  SCENE_SELECTOR=(--scene-token "${SCENE_TOKEN}")
else
  SCENE_SELECTOR=(--scene-name "${SCENE_NAME}")
fi

docker run --rm \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace/third_party/ncore_converter \
  "${IMAGE}" \
  python -m tools.data_converter.nuscenes.main \
  --root-dir "/workspace/${ROOT_DIR}" \
  --output-dir "/workspace/${OUTPUT_DIR}" \
  nuscenes-v4 \
  --version "${VERSION}" \
  "${SCENE_SELECTOR[@]}" \
  --store-type "${STORE_TYPE}" \
  --profile "${PROFILE}" \
  --cuboid-sampling "${CUBOID_SAMPLING}" \
  --lidar-model-resolution "${LIDAR_MODEL_RESOLUTION}" \
  --sequence-meta
