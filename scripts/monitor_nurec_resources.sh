#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE="${OUTPUT_FILE:?OUTPUT_FILE is required}"
DISK_PATH="${DISK_PATH:-$(dirname "${OUTPUT_FILE}")}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1}"

if [[ ! "${INTERVAL_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "INTERVAL_SECONDS must be a positive number, got: ${INTERVAL_SECONDS}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"
printf '%s\n' \
  'timestamp_utc,gpu_memory_used_mib,gpu_memory_total_mib,gpu_utilization_percent,power_draw_w,temperature_c,host_memory_available_kib,disk_available_kib' \
  > "${OUTPUT_FILE}"

while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  gpu_row="$(
    nvidia-smi \
      --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
  )"
  if [[ -z "${gpu_row}" ]]; then
    gpu_row=',,,,'
  fi
  host_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  disk_available_kib="$(df -Pk "${DISK_PATH}" | awk 'NR == 2 {print $4}')"
  printf '%s,%s,%s,%s\n' \
    "${timestamp}" "${gpu_row}" "${host_available_kib:-0}" "${disk_available_kib:-0}" \
    >> "${OUTPUT_FILE}"
  sleep "${INTERVAL_SECONDS}"
done
