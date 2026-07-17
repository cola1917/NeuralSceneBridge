#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ncore_dense_lidar_sweeps_v1}"
SCENE_TOKEN="${SCENE_TOKEN:-cc8c0bf57f984915a77078b10eb33198}"

if [[ "${OUTPUT_DIR}" = /* ]]; then
  echo "OUTPUT_DIR must be relative to the repository root." >&2
  exit 2
fi

OUTPUT_ABS="${REPO_ROOT}/${OUTPUT_DIR}"
MANIFEST="${OUTPUT_ABS}/scene-0061/scene-0061.json"
STATE_DIR="${OUTPUT_ABS}/provenance"
LOG_FILE="${STATE_DIR}/conversion.log"
STATUS_FILE="${STATE_DIR}/exit.status"

if [[ -e "${MANIFEST}" ]]; then
  echo "Dense NCore manifest already exists; refusing to mix attempts: ${MANIFEST}" >&2
  exit 1
fi
mkdir -p "${STATE_DIR}"
rm -f "${STATUS_FILE}"

echo "Dense NCore conversion started at $(date --iso-8601=seconds)" | tee -a "${LOG_FILE}"
set +e
OUTPUT_DIR="${OUTPUT_DIR}" \
SCENE_TOKEN="${SCENE_TOKEN}" \
CUBOID_SAMPLING=lidar-sweeps \
  bash "${SCRIPT_DIR}/run_ncore_nuscenes_converter.sh" 2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "${STATUS}" > "${STATUS_FILE}"
echo "Dense NCore conversion finished at $(date --iso-8601=seconds), exit=${STATUS}" \
  | tee -a "${LOG_FILE}"
exit "${STATUS}"
