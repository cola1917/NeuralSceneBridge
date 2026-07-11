#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/nurec_smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ABS="${REPO_ROOT}/${OUTPUT_DIR}"

if [[ ! -d "${OUTPUT_ABS}" ]]; then
  echo "NuRec output directory not found: ${OUTPUT_ABS}" >&2
  exit 1
fi

COMPLETE_RUNS=0
for run_dir in "${OUTPUT_ABS}"/*; do
  [[ -d "${run_dir}" ]] || continue

  usdz="${run_dir}/usd-out/last.usdz"
  config="${run_dir}/config/parsed.yaml"
  checkpoint="${run_dir}/checkpoints/last.ckpt"

  if [[ -f "${usdz}" && -f "${config}" && -f "${checkpoint}" ]]; then
    echo "Complete NuRec run: ${run_dir}"
    echo "  USDZ: ${usdz}"
    echo "  config: ${config}"
    echo "  checkpoint: ${checkpoint}"
    COMPLETE_RUNS=$((COMPLETE_RUNS + 1))
  else
    echo "Incomplete NuRec run: ${run_dir}"
    [[ -f "${usdz}" ]] || echo "  missing: usd-out/last.usdz"
    [[ -f "${config}" ]] || echo "  missing: config/parsed.yaml"
    [[ -f "${checkpoint}" ]] || echo "  missing: checkpoints/last.ckpt"
  fi
done

if (( COMPLETE_RUNS == 0 )); then
  echo "No complete NuRec run artifacts found." >&2
  exit 1
fi

echo "NUREC ARTIFACT VALIDATION OK (${COMPLETE_RUNS} complete run(s))"
