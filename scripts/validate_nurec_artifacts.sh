#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/config/nurec-smoke.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/nurec_smoke}"
if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_ABS="${OUTPUT_DIR}"
else
  OUTPUT_ABS="${REPO_ROOT}/${OUTPUT_DIR}"
fi
EXPECTED_CAMERA_IDS="${EXPECTED_CAMERA_IDS:-camera_front,camera_front_left,camera_front_right}"
EXPECTED_MAX_STEPS="${EXPECTED_MAX_STEPS:-1000}"
EXPECTED_MAX_EPOCHS="${EXPECTED_MAX_EPOCHS:--1}"
VALIDATION_BACKEND="${NUREC_VALIDATION_BACKEND:-auto}"
VALIDATION_IMAGE="${NUREC_VALIDATION_IMAGE:-${NUREC_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}}"

if [[ ! -d "${OUTPUT_ABS}" ]]; then
  echo "NuRec output directory not found: ${OUTPUT_ABS}" >&2
  exit 1
fi

if [[ ! "${EXPECTED_MAX_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "EXPECTED_MAX_STEPS must be a non-negative integer, got: ${EXPECTED_MAX_STEPS}" >&2
  exit 1
fi
if [[ ! "${EXPECTED_MAX_EPOCHS}" =~ ^-?[0-9]+$ ]]; then
  echo "EXPECTED_MAX_EPOCHS must be an integer, got: ${EXPECTED_MAX_EPOCHS}" >&2
  exit 1
fi

PYTHON_VALIDATOR='import sys
from pathlib import Path

import torch
import yaml

config_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
expected_cameras = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
expected_steps = int(sys.argv[4])
expected_epochs = int(sys.argv[5])

def find_values(node, key, path=""):
    found = []
    if isinstance(node, dict):
        for child_key, child in node.items():
            child_path = f"{path}.{child_key}" if path else str(child_key)
            if child_key == key:
                found.append((child_path, child))
            found.extend(find_values(child, key, child_path))
    elif isinstance(node, (list, tuple)):
        for index, child in enumerate(node):
            found.extend(find_values(child, key, f"{path}[{index}]"))
    return found

def as_cameras(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value]
    if isinstance(value, str):
        return [item.strip().strip("\"\x27") for item in value.strip("[]").split(",") if item.strip()]
    return None

def as_int(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if not isinstance(config, dict):
    raise SystemExit(f"parsed config is not a mapping: {config_path}")

checks = (
    ("camera_ids", expected_cameras, as_cameras),
    ("max_steps", expected_steps, as_int),
    ("max_epochs", expected_epochs, as_int),
)
for key, expected, normalize in checks:
    candidates = find_values(config, key)
    matches = [(path, value) for path, value in candidates if normalize(value) == expected]
    if not matches:
        rendered = ", ".join(f"{path}={value!r}" for path, value in candidates) or "<not found>"
        raise SystemExit(f"config gate failed: expected {key}={expected!r}; found {rendered}")
    print(f"  config {key}: {matches[0][0]}={expected!r}")

try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
if not isinstance(checkpoint, dict):
    raise SystemExit(f"checkpoint gate failed: root object is {type(checkpoint).__name__}, expected mapping")
raw_step = checkpoint.get("global_step")
actual_step = as_int(raw_step)
if actual_step != expected_steps:
    raise SystemExit(
        f"checkpoint gate failed: expected global_step={expected_steps}, "
        f"found {raw_step!r}"
    )
print(f"  checkpoint global_step: {actual_step}")'

VALIDATOR_KIND=""
if [[ "${VALIDATION_BACKEND}" == "auto" || "${VALIDATION_BACKEND}" == "host" ]]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import torch, yaml' >/dev/null 2>&1; then
    VALIDATOR_KIND="host"
  elif [[ "${VALIDATION_BACKEND}" == "host" ]]; then
    echo "Host validation requested, but python3 with torch and PyYAML is unavailable." >&2
    exit 1
  fi
fi

if [[ -z "${VALIDATOR_KIND}" && ( "${VALIDATION_BACKEND}" == "auto" || "${VALIDATION_BACKEND}" == "docker" ) ]]; then
  if command -v docker >/dev/null 2>&1 && docker image inspect "${VALIDATION_IMAGE}" >/dev/null 2>&1; then
    VALIDATOR_KIND="docker"
  elif [[ "${VALIDATION_BACKEND}" == "docker" ]]; then
    echo "Docker validation requested, but image is unavailable: ${VALIDATION_IMAGE}" >&2
    echo "Pull it first; artifact validation never pulls a multi-GB image implicitly." >&2
    exit 1
  fi
fi

if [[ -z "${VALIDATOR_KIND}" ]]; then
  echo "Cannot inspect parsed.yaml and last.ckpt without torch and PyYAML." >&2
  echo "Install them for host python3, or pull ${VALIDATION_IMAGE} and set NUREC_VALIDATION_BACKEND=docker." >&2
  echo "The gate fails closed so an unchecked checkpoint is never reported as valid." >&2
  exit 1
fi

validate_metadata() {
  local run_dir="$1"
  local config="$2"
  local checkpoint="$3"
  if [[ "${VALIDATOR_KIND}" == "host" ]]; then
    python3 -c "${PYTHON_VALIDATOR}" \
      "${config}" "${checkpoint}" "${EXPECTED_CAMERA_IDS}" \
      "${EXPECTED_MAX_STEPS}" "${EXPECTED_MAX_EPOCHS}"
  else
    docker run --rm --entrypoint python \
      --volume "${run_dir}:/nurec-run:ro" \
      "${VALIDATION_IMAGE}" -c "${PYTHON_VALIDATOR}" \
      "/nurec-run/config/parsed.yaml" "/nurec-run/checkpoints/last.ckpt" \
      "${EXPECTED_CAMERA_IDS}" "${EXPECTED_MAX_STEPS}" "${EXPECTED_MAX_EPOCHS}"
  fi
}

echo "NuRec acceptance gate:"
echo "  cameras: ${EXPECTED_CAMERA_IDS}"
echo "  max steps / checkpoint global_step: ${EXPECTED_MAX_STEPS}"
echo "  max epochs: ${EXPECTED_MAX_EPOCHS}"
echo "  metadata backend: ${VALIDATOR_KIND}"

CANDIDATE_RUNS=0
VALID_RUNS=0
for run_dir in "${OUTPUT_ABS}"/*; do
  [[ -d "${run_dir}" ]] || continue
  CANDIDATE_RUNS=$((CANDIDATE_RUNS + 1))

  usdz="${run_dir}/artifacts/last.usdz"
  if [[ ! -e "${usdz}" ]]; then
    usdz="${run_dir}/usd-out/last.usdz"
  fi
  config="${run_dir}/config/parsed.yaml"
  checkpoint="${run_dir}/checkpoints/last.ckpt"

  echo "Checking NuRec run: ${run_dir}"
  missing=0
  for label_and_path in "USDZ|${usdz}" "config|${config}" "checkpoint|${checkpoint}"; do
    label="${label_and_path%%|*}"
    path="${label_and_path#*|}"
    if [[ ! -f "${path}" ]]; then
      echo "  missing ${label}: ${path}"
      missing=1
    elif [[ ! -s "${path}" ]]; then
      echo "  empty ${label}: ${path}"
      missing=1
    else
      echo "  ${label}: ${path} ($(wc -c < "${path}") bytes)"
    fi
  done
  if (( missing != 0 )); then
    echo "  result: FAIL (missing or empty artifacts)"
    continue
  fi

  validation_output=""
  if validation_output="$(validate_metadata "${run_dir}" "${config}" "${checkpoint}" 2>&1)"; then
    printf '%s\n' "${validation_output}"
    echo "  result: PASS"
    VALID_RUNS=$((VALID_RUNS + 1))
  else
    printf '%s\n' "${validation_output}"
    echo "  result: FAIL (configuration or checkpoint gate)"
  fi
done

if (( CANDIDATE_RUNS == 0 )); then
  echo "No NuRec run directories found below ${OUTPUT_ABS}." >&2
  exit 1
fi
if (( VALID_RUNS == 0 )); then
  echo "No NuRec run passed the strict artifact gate (${CANDIDATE_RUNS} checked)." >&2
  exit 1
fi

echo "NUREC ARTIFACT VALIDATION OK (${VALID_RUNS}/${CANDIDATE_RUNS} run(s) passed)"
