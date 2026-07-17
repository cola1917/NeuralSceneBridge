#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/config/nurec-formal.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Formal NuRec environment file not found: ${ENV_FILE}" >&2
  echo "Copy config/nurec-formal.env.example to config/nurec-formal.env first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

OUTPUT_DIR="${OUTPUT_DIR:-outputs/nurec_formal_scene0061_dense_lidar_sweeps_v1_6cam_40k_attempt_001}"
if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_ABS="${OUTPUT_DIR}"
else
  OUTPUT_ABS="${REPO_ROOT}/${OUTPUT_DIR}"
fi

STATE_DIR="${OUTPUT_ABS}/launcher"
LOG_FILE="${STATE_DIR}/train.log"
STATUS_FILE="${STATE_DIR}/exit.status"
RESOURCE_LOG="${STATE_DIR}/resources.csv"
RESOURCE_SUMMARY="${STATE_DIR}/resources.summary.json"
mkdir -p "${STATE_DIR}"
rm -f "${STATUS_FILE}"

MONITOR_PID=""
stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  MONITOR_PID=""
}
trap stop_monitor EXIT INT TERM

OUTPUT_FILE="${RESOURCE_LOG}" \
DISK_PATH="${OUTPUT_ABS}" \
INTERVAL_SECONDS="${RESOURCE_MONITOR_INTERVAL_SECONDS:-1}" \
  bash "${SCRIPT_DIR}/monitor_nurec_resources.sh" &
MONITOR_PID="$!"

echo "NuRec formal worker started at $(date --iso-8601=seconds)" | tee -a "${LOG_FILE}"
set +e
ENV_FILE="${ENV_FILE}" bash "${SCRIPT_DIR}/run_nurec_train.sh" 2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e
stop_monitor
trap - EXIT INT TERM
python3 "${SCRIPT_DIR}/summarize_nurec_resources.py" \
  "${RESOURCE_LOG}" --output "${RESOURCE_SUMMARY}" || true
printf '%s\n' "${STATUS}" > "${STATUS_FILE}"
echo "NuRec formal worker finished at $(date --iso-8601=seconds), exit=${STATUS}" | tee -a "${LOG_FILE}"
exit "${STATUS}"
