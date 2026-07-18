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
REQUIRE_DYNAMIC_TRACKS="${REQUIRE_DYNAMIC_TRACKS:-0}"
EXPECTED_MIN_USDZ_TRACKS="${EXPECTED_MIN_USDZ_TRACKS:-1}"
EXPECTED_MIN_USDZ_VEHICLES="${EXPECTED_MIN_USDZ_VEHICLES:-1}"
EXPECTED_MIN_USDZ_PEDESTRIANS="${EXPECTED_MIN_USDZ_PEDESTRIANS:-1}"
REQUIRE_LIDAR_SUPERVISION="${REQUIRE_LIDAR_SUPERVISION:-0}"
EXPECTED_LIDAR_IDS="${EXPECTED_LIDAR_IDS:-${LIDAR_IDS:-lidar_top}}"
EXPECTED_MIN_LIDAR_RAYS="${EXPECTED_MIN_LIDAR_RAYS:-1}"
EXPECTED_MIN_LIDAR_SAMPLE_RATIO="${EXPECTED_MIN_LIDAR_SAMPLE_RATIO:-0.000001}"
EXPECTED_MIN_LIDAR_LOSS_WEIGHT="${EXPECTED_MIN_LIDAR_LOSS_WEIGHT:-0.000001}"
EXPECTED_VAL_LIDAR="${EXPECTED_VAL_LIDAR:-${VAL_LIDAR:-0}}"
REQUIRE_LIDAR_VALIDATION_EVIDENCE="${REQUIRE_LIDAR_VALIDATION_EVIDENCE:-0}"
EXPECTED_REQUIRED_LIDAR_METRICS="${EXPECTED_REQUIRED_LIDAR_METRICS:-test/chamfer_distance,test/raydrop_accuracy}"
EXPECTED_MIN_LIDAR_VALIDATION_FRAMES="${EXPECTED_MIN_LIDAR_VALIDATION_FRAMES:-1}"
EXPECTED_MIN_LIDAR_PLY_PAIRS="${EXPECTED_MIN_LIDAR_PLY_PAIRS:-1}"
ALLOW_NRE_2604_LIDAR_GROUPING_BUG="${ALLOW_NRE_2604_LIDAR_GROUPING_BUG:-0}"
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
if [[ "${REQUIRE_DYNAMIC_TRACKS}" != "0" && "${REQUIRE_DYNAMIC_TRACKS}" != "1" ]]; then
  echo "REQUIRE_DYNAMIC_TRACKS must be 0 or 1, got: ${REQUIRE_DYNAMIC_TRACKS}" >&2
  exit 1
fi
for variable in REQUIRE_LIDAR_SUPERVISION EXPECTED_VAL_LIDAR REQUIRE_LIDAR_VALIDATION_EVIDENCE ALLOW_NRE_2604_LIDAR_GROUPING_BUG; do
  value="${!variable}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${variable} must be 0 or 1, got: ${value}" >&2
    exit 1
  fi
done
for variable in EXPECTED_MIN_LIDAR_VALIDATION_FRAMES EXPECTED_MIN_LIDAR_PLY_PAIRS; do
  value="${!variable}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${variable} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
done
for variable in EXPECTED_MIN_USDZ_TRACKS EXPECTED_MIN_USDZ_VEHICLES EXPECTED_MIN_USDZ_PEDESTRIANS; do
  value="${!variable}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${variable} must be a non-negative integer, got: ${value}" >&2
    exit 1
  fi
done
if [[ ! "${EXPECTED_MIN_LIDAR_RAYS}" =~ ^[0-9]+$ ]]; then
  echo "EXPECTED_MIN_LIDAR_RAYS must be a non-negative integer, got: ${EXPECTED_MIN_LIDAR_RAYS}" >&2
  exit 1
fi
for variable in EXPECTED_MIN_LIDAR_SAMPLE_RATIO EXPECTED_MIN_LIDAR_LOSS_WEIGHT; do
  value="${!variable}"
  if ! awk -v value="${value}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0) }'; then
    echo "${variable} must be a non-negative number, got: ${value}" >&2
    exit 1
  fi
done

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
require_lidar = sys.argv[7] == "1"
expected_lidars = [item.strip() for item in sys.argv[8].split(",") if item.strip()]
min_lidar_rays = int(sys.argv[9])
min_lidar_ratio = float(sys.argv[10])
min_lidar_loss = float(sys.argv[11])
expected_val_lidar = sys.argv[12] == "1"

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

def as_float(value):
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def at_path(node, path):
    value = node
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise SystemExit(f"config gate failed: required path {path} not found")
        value = value[key]
    return value

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

if require_lidar:
    for path in ("dataset.lidar_ids", "dataset.train_lidar_ids"):
        actual = as_cameras(at_path(config, path))
        if actual != expected_lidars:
            raise SystemExit(
                f"config lidar gate failed: expected {path}={expected_lidars!r}, found {actual!r}"
            )
        print(f"  config {path}: {actual!r}")

    lidar_thresholds = (
        ("dataset.n_train_sample_lidar_rays", float(min_lidar_rays)),
        ("dataset.samplers.batch_sampler.ratio_lidar_samples", min_lidar_ratio),
        ("loss.lidar.lambda_", min_lidar_loss),
    )
    for path, minimum in lidar_thresholds:
        actual = as_float(at_path(config, path))
        if actual is None or actual < minimum:
            raise SystemExit(
                f"config lidar gate failed: expected {path}>={minimum}, found {actual!r}"
            )
        print(f"  config {path}: {actual} (minimum {minimum})")

    actual_val_lidar = at_path(config, "dataset.val_lidar")
    if not isinstance(actual_val_lidar, bool) or actual_val_lidar is not expected_val_lidar:
        raise SystemExit(
            "config lidar gate failed: expected "
            f"dataset.val_lidar={expected_val_lidar!r}, found {actual_val_lidar!r}"
        )
    print(f"  config dataset.val_lidar: {actual_val_lidar!r}")

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
      "${EXPECTED_GLOBAL_STEP}" "${EXPECTED_SAMPLES_PER_EPOCH}" "${EXPECTED_MAX_EPOCHS}" \
      "${REQUIRE_LIDAR_SUPERVISION}" "${EXPECTED_LIDAR_IDS}" \
      "${EXPECTED_MIN_LIDAR_RAYS}" "${EXPECTED_MIN_LIDAR_SAMPLE_RATIO}" \
      "${EXPECTED_MIN_LIDAR_LOSS_WEIGHT}" "${EXPECTED_VAL_LIDAR}"
  else
    docker run --rm --entrypoint /bin/bash \
      --volume "${run_dir}:/nurec-run:ro" \
      "${VALIDATION_IMAGE}" -lc \
      'export RUNFILES_DIR=/app/run.runfiles; source <(sed "/# Call obfuscated target/,\$d" /app/run); exec python3 -c "$@"' \
      bash "${PYTHON_VALIDATOR}" \
      "/nurec-run/config/parsed.yaml" "/nurec-run/checkpoints/last.ckpt" \
      "${EXPECTED_CAMERA_IDS}" "${EXPECTED_GLOBAL_STEP}" \
      "${EXPECTED_SAMPLES_PER_EPOCH}" "${EXPECTED_MAX_EPOCHS}" \
      "${REQUIRE_LIDAR_SUPERVISION}" "${EXPECTED_LIDAR_IDS}" \
      "${EXPECTED_MIN_LIDAR_RAYS}" "${EXPECTED_MIN_LIDAR_SAMPLE_RATIO}" \
      "${EXPECTED_MIN_LIDAR_LOSS_WEIGHT}" "${EXPECTED_VAL_LIDAR}"
  fi
}

validate_dynamic_tracks() {
  local usdz="$1"
  python3 "${SCRIPT_DIR}/validate_nurec_usdz_tracks.py" "${usdz}" \
    --min-total "${EXPECTED_MIN_USDZ_TRACKS}" \
    --min-vehicles "${EXPECTED_MIN_USDZ_VEHICLES}" \
    --min-pedestrians "${EXPECTED_MIN_USDZ_PEDESTRIANS}"
}

validate_lidar_evidence() {
  local run_dir="$1"
  local metrics="${run_dir}/val/metrics.yaml"
  local -a bug_flag=()
  if [[ "${ALLOW_NRE_2604_LIDAR_GROUPING_BUG}" == "1" ]]; then
    bug_flag+=(--allow-nre-2604-lidar-grouping-bug)
  fi

  if [[ ! -s "${metrics}" ]]; then
    echo "LiDAR evidence gate failed: missing or empty ${metrics}" >&2
    return 1
  fi

  if [[ "${VALIDATOR_KIND}" == "host" ]]; then
    python3 "${SCRIPT_DIR}/validate_nurec_lidar_metrics.py" "${metrics}" \
      --required-metrics "${EXPECTED_REQUIRED_LIDAR_METRICS}" \
      --min-frame-samples "${EXPECTED_MIN_LIDAR_VALIDATION_FRAMES}" \
      --min-ply-pairs "${EXPECTED_MIN_LIDAR_PLY_PAIRS}" \
      "${bug_flag[@]}"
  else
    docker run --rm --entrypoint /bin/bash \
      --volume "${run_dir}:/nurec-run:ro" \
      --volume "${SCRIPT_DIR}:/validator:ro" \
      "${VALIDATION_IMAGE}" -lc \
      'export RUNFILES_DIR=/app/run.runfiles; source <(sed "/# Call obfuscated target/,\$d" /app/run); exec python3 /validator/validate_nurec_lidar_metrics.py "$@"' \
      bash "/nurec-run/val/metrics.yaml" \
      --required-metrics "${EXPECTED_REQUIRED_LIDAR_METRICS}" \
      --min-frame-samples "${EXPECTED_MIN_LIDAR_VALIDATION_FRAMES}" \
      --min-ply-pairs "${EXPECTED_MIN_LIDAR_PLY_PAIRS}" \
      "${bug_flag[@]}"
  fi
}

echo "NuRec acceptance gate:"
echo "  cameras: ${EXPECTED_CAMERA_IDS}"
echo "  samples per epoch: ${EXPECTED_SAMPLES_PER_EPOCH}"
echo "  max epochs: ${EXPECTED_MAX_EPOCHS}"
echo "  checkpoint global_step: ${EXPECTED_GLOBAL_STEP}"
echo "  metadata backend: ${VALIDATOR_KIND}"
echo "  require dynamic USDZ tracks: ${REQUIRE_DYNAMIC_TRACKS}"
echo "  require lidar supervision: ${REQUIRE_LIDAR_SUPERVISION}"
echo "  require lidar validation evidence: ${REQUIRE_LIDAR_VALIDATION_EVIDENCE}"
if [[ "${REQUIRE_LIDAR_SUPERVISION}" == "1" ]]; then
  echo "  lidar ids: ${EXPECTED_LIDAR_IDS}"
  echo "  minimum lidar rays/ratio/loss: ${EXPECTED_MIN_LIDAR_RAYS}/${EXPECTED_MIN_LIDAR_SAMPLE_RATIO}/${EXPECTED_MIN_LIDAR_LOSS_WEIGHT}"
  echo "  validate lidar: ${EXPECTED_VAL_LIDAR}"
fi
if [[ "${REQUIRE_LIDAR_VALIDATION_EVIDENCE}" == "1" ]]; then
  echo "  required lidar metrics: ${EXPECTED_REQUIRED_LIDAR_METRICS}"
  echo "  minimum lidar validation frames/PLY pairs: ${EXPECTED_MIN_LIDAR_VALIDATION_FRAMES}/${EXPECTED_MIN_LIDAR_PLY_PAIRS}"
  echo "  allow audited NRE 26.04 lidar grouping bug: ${ALLOW_NRE_2604_LIDAR_GROUPING_BUG}"
fi
if [[ "${REQUIRE_DYNAMIC_TRACKS}" == "1" ]]; then
  echo "  minimum USDZ tracks/vehicles/pedestrians: ${EXPECTED_MIN_USDZ_TRACKS}/${EXPECTED_MIN_USDZ_VEHICLES}/${EXPECTED_MIN_USDZ_PEDESTRIANS}"
fi

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
  if ! validation_output="$(validate_metadata "${run_dir}" "${config}" "${checkpoint}" 2>&1)"; then
    printf '%s\n' "${validation_output}"
    echo "  result: FAIL (configuration or checkpoint gate)"
    continue
  fi
  printf '%s\n' "${validation_output}"

  if [[ "${REQUIRE_DYNAMIC_TRACKS}" == "1" ]]; then
    if ! validation_output="$(validate_dynamic_tracks "${usdz}" 2>&1)"; then
      printf '%s\n' "${validation_output}"
      echo "  result: FAIL (dynamic-track gate)"
      continue
    fi
    printf '%s\n' "${validation_output}"
  fi

  if [[ "${REQUIRE_LIDAR_VALIDATION_EVIDENCE}" == "1" ]]; then
    if ! validation_output="$(validate_lidar_evidence "${run_dir}" 2>&1)"; then
      printf '%s\n' "${validation_output}"
      echo "  result: FAIL (LiDAR validation evidence gate)"
      continue
    fi
    printf '%s\n' "${validation_output}"
  fi

  echo "  result: PASS"
  VALID_RUNS=$((VALID_RUNS + 1))
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
