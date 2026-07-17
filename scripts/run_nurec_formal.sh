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

OUTPUT_DIR="${OUTPUT_DIR:-outputs/nurec_formal_scene0061_6cam_40k}"
if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_ABS="${OUTPUT_DIR}"
else
  OUTPUT_ABS="${REPO_ROOT}/${OUTPUT_DIR}"
fi

STATE_DIR="${OUTPUT_ABS}/launcher"
LOG_FILE="${STATE_DIR}/train.log"
STATUS_FILE="${STATE_DIR}/exit.status"
mkdir -p "${STATE_DIR}"
rm -f "${STATUS_FILE}"

echo "NuRec formal worker started at $(date --iso-8601=seconds)" | tee -a "${LOG_FILE}"
set +e
ENV_FILE="${ENV_FILE}" bash "${SCRIPT_DIR}/run_nurec_train.sh" 2>&1 | tee -a "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "${STATUS}" > "${STATUS_FILE}"
echo "NuRec formal worker finished at $(date --iso-8601=seconds), exit=${STATUS}" | tee -a "${LOG_FILE}"
exit "${STATUS}"
