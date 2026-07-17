#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/config/nurec-formal.env}"
SESSION_NAME="${SESSION_NAME:-nurec-scene0061-formal}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for a training job that survives SSH disconnects." >&2
  exit 1
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Formal NuRec environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${REPO_ROOT}' && exec env ENV_FILE='${ENV_FILE}' bash '${SCRIPT_DIR}/run_nurec_formal.sh'"

echo "Started persistent NuRec session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Worker log is under the OUTPUT_DIR launcher/train.log configured in ${ENV_FILE}."
