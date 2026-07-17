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
EXPECTED_CAMERA_IDS="${EXPECTED_CAMERA_IDS:-${CAMERA_IDS:-camera_front,camera_front_left,camera_front_right}}"
EXPECTED_GLOBAL_STEP="${EXPECTED_GLOBAL_STEP:-1000}"
EXPECTED_SAMPLES_PER_EPOCH="${EXPECTED_SAMPLES_PER_EPOCH:-1000}"
EXPECTED_MAX_EPOCHS="${EXPECTED_MAX_EPOCHS:-1}"
REQUIRE_SINGLE_RUN="${REQUIRE_SINGLE_RUN:-0}"
VALIDATION_BACKEND="${NUREC_VALIDATION_BACKEND:-auto}"
VALIDATION_IMAGE="${NUREC_VALIDATION_IMAGE:-${NUREC_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}}"

if [[ ! -d "${OUTPUT_ABS}" ]]; then
  echo "NuRec output directory not found: ${OUTPUT_ABS}" >&2
  exit 1
fi

if [[ ! "${EXPECTED_GLOBAL_STEP}" =~ ^[0-9]+$ ]]; then
  echo "EXPECTED_GLOBAL_STEP must be a non-negative integer, got: ${EXPECTED_GLOBAL_STEP}" >&2
  exit 1
fi
if [[ ! "${EXPECTED_SAMPLES_PER_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "EXPECTED_SAMPLES_PER_EPOCH must be a non-negative integer, got: ${EXPECTED_SAMPLES_PER_EPOCH}" >&2
  exit 1
fi
if [[ ! "${EXPECTED_MAX_EPOCHS}" =~ ^-?[0-9]+$ ]]; then
  echo "EXPECTED_MAX_EPOCHS must be an integer, got: ${EXPECTED_MAX_EPOCHS}" >&2
  exit 1
fi
if [[ "${REQUIRE_SINGLE_RUN}" != "0" && "${REQUIRE_SINGLE_RUN}" != "1" ]]; then
  echo "REQUIRE_SINGLE_RUN must be 0 or 1, got: ${REQUIRE_SINGLE_RUN}" >&2
  exit 1
fi

PYTHON_VALIDATOR='import pickletools
import sys
import zipfile
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
expected_cameras = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
expected_steps = int(sys.argv[4])
expected_samples = int(sys.argv[5])
expected_epochs = int(sys.argv[6])

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
    ("n_samples_per_epoch", expected_samples, as_int),
    ("max_epochs", expected_epochs, as_int),
)
for key, expected, normalize in checks:
    candidates = find_values(config, key)
    matches = [(path, value) for path, value in candidates if normalize(value) == expected]
    if not matches:
        rendered = ", ".join(f"{path}={value!r}" for path, value in candidates) or "<not found>"
        raise SystemExit(f"config gate failed: expected {key}={expected!r}; found {rendered}")
    print(f"  config {key}: {matches[0][0]}={expected!r}")

def read_global_step(path):
    # PyTorch checkpoints are ZIP archives containing a pickle instruction
    # stream plus tensor blobs. Inspecting opcodes reads the scalar metadata
    # without importing project classes or executing pickle payloads.
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("/data.pkl") or name == "data.pkl"]
        if len(candidates) != 1:
            raise SystemExit(f"checkpoint gate failed: expected one data.pkl, found {candidates!r}")
        payload = archive.read(candidates[0])

    expecting_value = False
    integer_ops = {"BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4"}
    memo_ops = {"BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT"}
    string_ops = {"BINUNICODE", "BINUNICODE8", "SHORT_BINUNICODE", "UNICODE"}
    for opcode, argument, _ in pickletools.genops(payload):
        if expecting_value:
            if opcode.name in memo_ops:
                continue
            if opcode.name in integer_ops:
                return int(argument)
            raise SystemExit(
                f"checkpoint gate failed: global_step has unsupported opcode {opcode.name}"
            )
        if opcode.name in string_ops and argument == "global_step":
            expecting_value = True
    raise SystemExit("checkpoint gate failed: global_step not found")

actual_step = read_global_step(checkpoint_path)
if actual_step != expected_steps:
    raise SystemExit(
        f"checkpoint gate failed: expected global_step={expected_steps}, "
        f"found {actual_step!r}"
    )
print(f"  checkpoint global_step: {actual_step}")'

VALIDATOR_KIND=""
# In auto mode prefer the same official image that produced the checkpoint.
# A host can import torch/yaml yet still lack Python classes embedded in the
# checkpoint (for example OmegaConf), producing a false-negative validation.
if [[ "${VALIDATION_BACKEND}" == "auto" || "${VALIDATION_BACKEND}" == "docker" ]]; then
  if command -v docker >/dev/null 2>&1 && docker image inspect "${VALIDATION_IMAGE}" >/dev/null 2>&1; then
    VALIDATOR_KIND="docker"
  elif [[ "${VALIDATION_BACKEND}" == "docker" ]]; then
    echo "Docker validation requested, but image is unavailable: ${VALIDATION_IMAGE}" >&2
    echo "Pull it first; artifact validation never pulls a multi-GB image implicitly." >&2
    exit 1
  fi
fi

if [[ -z "${VALIDATOR_KIND}" && ( "${VALIDATION_BACKEND}" == "auto" || "${VALIDATION_BACKEND}" == "host" ) ]]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import yaml' >/dev/null 2>&1; then
    VALIDATOR_KIND="host"
  elif [[ "${VALIDATION_BACKEND}" == "host" ]]; then
    echo "Host validation requested, but python3 with torch and PyYAML is unavailable." >&2
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
      "${EXPECTED_GLOBAL_STEP}" "${EXPECTED_SAMPLES_PER_EPOCH}" "${EXPECTED_MAX_EPOCHS}"
  else
    docker run --rm --entrypoint /bin/bash \
      --volume "${run_dir}:/nurec-run:ro" \
      "${VALIDATION_IMAGE}" -lc \
      'export RUNFILES_DIR=/app/run.runfiles; source <(sed "/# Call obfuscated target/,\$d" /app/run); exec python3 -c "$@"' \
      bash "${PYTHON_VALIDATOR}" \
      "/nurec-run/config/parsed.yaml" "/nurec-run/checkpoints/last.ckpt" \
      "${EXPECTED_CAMERA_IDS}" "${EXPECTED_GLOBAL_STEP}" \
      "${EXPECTED_SAMPLES_PER_EPOCH}" "${EXPECTED_MAX_EPOCHS}"
  fi
}

echo "NuRec acceptance gate:"
echo "  cameras: ${EXPECTED_CAMERA_IDS}"
echo "  samples per epoch: ${EXPECTED_SAMPLES_PER_EPOCH}"
echo "  max epochs: ${EXPECTED_MAX_EPOCHS}"
echo "  checkpoint global_step: ${EXPECTED_GLOBAL_STEP}"
echo "  metadata backend: ${VALIDATOR_KIND}"

CANDIDATE_RUNS=0
VALID_RUNS=0
for run_dir in "${OUTPUT_ABS}"/*; do
  [[ -d "${run_dir}" ]] || continue

  usdz="${run_dir}/artifacts/last.usdz"
  if [[ ! -e "${usdz}" ]]; then
    usdz="${run_dir}/usd-out/last.usdz"
  fi
  config="${run_dir}/config/parsed.yaml"
  checkpoint="${run_dir}/checkpoints/last.ckpt"

  # Output roots may also contain operational directories such as launcher/
  # and logs/. Only directories with at least one NuRec artifact marker are
  # candidate runs; unrelated state must not dilute the acceptance result.
  if [[ ! -e "${usdz}" && ! -e "${config}" && ! -e "${checkpoint}" ]]; then
    continue
  fi
  CANDIDATE_RUNS=$((CANDIDATE_RUNS + 1))

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
if [[ "${REQUIRE_SINGLE_RUN}" == "1" ]] && (( CANDIDATE_RUNS != 1 )); then
  echo "Expected exactly one NuRec run, found ${CANDIDATE_RUNS}." >&2
  exit 1
fi
if (( VALID_RUNS == 0 )); then
  echo "No NuRec run passed the strict artifact gate (${CANDIDATE_RUNS} checked)." >&2
  exit 1
fi

echo "NUREC ARTIFACT VALIDATION OK (${VALID_RUNS}/${CANDIDATE_RUNS} run(s) passed)"
