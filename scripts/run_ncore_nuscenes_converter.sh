#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-nsb/ncore-converter:2026-07-10}"
ROOT_DIR="${ROOT_DIR:-data/nuscenes-mini-scene-0061}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ncore}"
VERSION="${VERSION:-v1.0-mini}"
SCENE_NAME="${SCENE_NAME:-scene-0061}"
STORE_TYPE="${STORE_TYPE:-itar}"
PROFILE="${PROFILE:-separate-sensors}"
LIDAR_MODEL_RESOLUTION="${LIDAR_MODEL_RESOLUTION:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${REPO_ROOT}/${OUTPUT_DIR}"

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
  --scene-name "${SCENE_NAME}" \
  --store-type "${STORE_TYPE}" \
  --profile "${PROFILE}" \
  --lidar-model-resolution "${LIDAR_MODEL_RESOLUTION}" \
  --sequence-meta
