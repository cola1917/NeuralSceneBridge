#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

for script in scripts/*.sh; do
  bash -n "${script}"
done

python3 scripts/validate_ncore_env.py \
  --converter-root third_party/ncore_converter \
  --skip-imports \
  --skip-cli-help \
  --nuscenes-root data/nuscenes-mini-scene-0061 \
  --scene-name scene-0061

echo "LOCAL CHECKS OK"
