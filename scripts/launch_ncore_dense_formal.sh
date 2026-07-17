#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-ncore-scene0061-dense-v1}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for a conversion that survives SSH disconnects." >&2
  exit 1
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${REPO_ROOT}' && exec env OUTPUT_DIR='${OUTPUT_DIR:-outputs/ncore_dense_lidar_sweeps_v1}' bash '${SCRIPT_DIR}/run_ncore_dense_formal.sh'"

echo "Started persistent NCore conversion session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Log: ${OUTPUT_DIR:-outputs/ncore_dense_lidar_sweeps_v1}/provenance/conversion.log"
