#!/usr/bin/env python3
"""Build a fail-closed quality report from NuRec capture evidence.

The report is deliberately a *summary of evidence*, not a renderer.  A report
can be written while capture is incomplete, but it is only marked ``passed``
when the immutable USDZ identity and all three scene-0061 case captures are
verified.  Missing files, stale evidence, dropped frames, and placeholder
reports therefore remain visible as ``pending_capture`` or ``failed``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "demo" / "scene0061" / "manifest.json"
REQUIRED_CASES = ("V01", "V02", "V03")
VIDEO_NAMES = {
    "V01": "original_replay.mp4",
    "V02": "lead_vehicle_edit.mp4",
    "V03": "camera_pose_sweep.mp4",
}
EVIDENCE_SCHEMA = "nsb.nurec-counterfactual-run.v1"
REPORT_SCHEMA = "nsb.nurec-quality-report.v1"
METRIC_NAMES = ("test/chamfer_distance", "test/raydrop_accuracy")


class QualityReportError(ValueError):
    """Raised when a report input is structurally invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_required(value: object, *, base: Path = REPO_ROOT, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise QualityReportError(f"{label} is required")
    return _resolve(str(value), base=base)


def _resolve_manifest_uri(value: object, manifest_path: Path, *, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise QualityReportError(f"{label} is required")
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    local = (manifest_path.parent / raw).resolve()
    if local.is_file():
        return local
    return _resolve(str(raw), base=REPO_ROOT)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityReportError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityReportError(f"JSON root must be an object: {path}")
    return value


def _sha_ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "size_bytes": None}
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _case_files(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Path]:
    files = manifest.get("case_files")
    if not isinstance(files, list):
        raise QualityReportError("manifest.case_files must be a list")
    result: dict[str, Path] = {}
    for raw in files:
        raw_path = Path(str(raw)).expanduser()
        candidates = [
            raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path,
            raw_path if raw_path.is_absolute() else manifest_path.parent.parent.parent / raw_path,
            _resolve(str(raw_path)),
        ]
        path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())
        case = _load_json(path)
        case_id = str(case.get("case_id") or "")
        if case_id in result:
            raise QualityReportError(f"duplicate manifest case_id: {case_id}")
        result[case_id] = path
    missing = sorted(set(REQUIRED_CASES) - set(result))
    if missing:
        raise QualityReportError(f"manifest is missing required cases: {', '.join(missing)}")
    return result


def _parse_evidence_argument(raw: str, *, base: Path) -> tuple[str, Path]:
    if "=" not in raw:
        raise QualityReportError(f"evidence must be CASE=PATH, got {raw!r}")
    case_id, value = raw.split("=", 1)
    case_id = case_id.strip()
    if case_id not in REQUIRED_CASES or not value.strip():
        raise QualityReportError(f"invalid evidence selector: {raw!r}")
    path = _resolve(value.strip(), base=base)
    if path.is_dir():
        path = path / "evidence.json"
    return case_id, path


def discover_evidence(root: Path) -> dict[str, Path]:
    """Discover one evidence.json per required case under ``root``."""

    if not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("evidence.json")):
        try:
            payload = _load_json(path)
        except QualityReportError:
            continue
        case_id = str((payload.get("case") or {}).get("case_id") or "")
        if case_id in REQUIRED_CASES:
            if case_id in found:
                raise QualityReportError(f"multiple evidence.json files found for {case_id} under {root}")
            found[case_id] = path
    return found


def _load_metrics(path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, []
    if not path.is_file():
        return None, [f"metrics file is unavailable: {path}"]
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ImportError, ValueError) as exc:
        return None, [f"metrics file could not be parsed: {exc}"]
    except Exception as exc:  # PyYAML exposes scanner/parser errors as custom types.
        return None, [f"metrics file could not be parsed: {exc}"]
    if not isinstance(value, Mapping):
        return None, ["metrics file root is not an object"]
    aggregate = value.get("aggregated_metrics")
    if not isinstance(aggregate, Mapping):
        return None, ["metrics file has no aggregated_metrics object"]
    selected: dict[str, float] = {}
    problems: list[str] = []
    for name in METRIC_NAMES:
        entry = aggregate.get(name)
        metric = entry.get("value") if isinstance(entry, Mapping) else None
        if not _finite(metric):
            problems.append(f"metric is unavailable or non-finite: {name}")
        else:
            selected[name] = float(metric)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "program_version": value.get("program_version"),
        "run_info": value.get("run_info"),
        "aggregated": selected,
    }, problems


def _infer_metrics_path(artifact: Path) -> Path | None:
    run_dir = artifact.parent.parent
    candidate = run_dir / "val" / "metrics.yaml"
    return candidate if candidate.is_file() else None


def _infer_config_path(artifact: Path) -> Path | None:
    candidate = artifact.parent.parent / "config" / "parsed.yaml"
    return candidate if candidate.is_file() else None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _quality_metric(values: list[float], *, reason: str) -> dict[str, Any]:
    if not values or not all(_finite(value) for value in values):
        return {"available": False, "value": None, "reason": reason}
    return {
        "available": True,
        "value": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "sample_count": len(values),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered)) - 1)))
    return float(ordered[index])


def _case_output_resolution(case: Mapping[str, Any]) -> tuple[int, int]:
    explicit = case.get("video_resolution") or case.get("output_resolution")
    if isinstance(explicit, Mapping):
        return int(explicit.get("width", 0)), int(explicit.get("height", 0))
    source = case.get("resolution") or {}
    width = int(source.get("width", 0))
    height = int(source.get("height", 0))
    grid = case.get("camera_grid") or {}
    if isinstance(grid, Mapping):
        width *= int(grid.get("columns", 1))
        height *= int(grid.get("rows", 1))
    return width, height


def _case_camera_ids(case: Mapping[str, Any]) -> list[str]:
    raw = case.get("camera_ids")
    if not isinstance(raw, list):
        raw = [case.get("camera_id", "camera_front")]
    return [str(value).split("@", 1)[0] for value in raw]


def _camera_quality_summary(
    frame_rows: list[Mapping[str, Any]], camera_ids: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera_id in camera_ids:
        metrics: dict[str, Any] = {}
        for name in ("invalid_pixel_ratio", "dark_pixel_ratio", "laplacian_sharpness"):
            values = [
                float((row.get("camera_metrics") or {}).get(camera_id, {}).get(name))
                for row in frame_rows
                if _finite((row.get("camera_metrics") or {}).get(camera_id, {}).get(name))
            ]
            metrics[name] = _quality_metric(
                values,
                reason=f"frame metadata has no finite camera metric {camera_id}/{name}",
            )
        result[camera_id] = metrics
    return result


def _frame_quality(
    frame_rows: list[Mapping[str, Any]],
    metadata_path: Path,
    case: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Summarize renderer metrics and compute temporal flicker from JPEGs."""

    reasons: list[str] = []
    quality: dict[str, Any] = {}
    for name in ("invalid_pixel_ratio", "dark_pixel_ratio", "laplacian_sharpness"):
        values = [
            float(row[name])
            for row in frame_rows
            if _finite(row.get(name))
        ]
        quality[name] = _quality_metric(
            values,
            reason=f"frame metadata has no finite {name} measurements",
        )
        if not quality[name]["available"]:
            reasons.append(f"quality metric unavailable: {name}")

    image_arrays: list[Any] = []
    try:
        from PIL import Image
        import numpy as np

        expected_size = _case_output_resolution(case)
        for row in frame_rows:
            frame_value = row.get("frame_path")
            if not isinstance(frame_value, str) or not frame_value:
                reasons.append("frame metadata has no frame_path")
                continue
            frame_path = _resolve(frame_value, base=metadata_path.parent)
            if not frame_path.is_file():
                reasons.append(f"captured frame is unavailable: {frame_path}")
                continue
            with Image.open(frame_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if expected_size != (0, 0) and (rgb.shape[1], rgb.shape[0]) != expected_size:
                reasons.append(
                    f"captured frame dimensions {(rgb.shape[1], rgb.shape[0])} "
                    f"!= case resolution {expected_size}"
                )
            decoded_size = (row.get("decoded_width"), row.get("decoded_height"))
            if decoded_size != (None, None) and decoded_size != (rgb.shape[1], rgb.shape[0]):
                reasons.append("frame metadata decoded dimensions do not match the image")
            image_arrays.append(rgb)
    except (ImportError, OSError, ValueError) as exc:
        reasons.append(f"captured frame pixels could not be measured: {exc}")

    flicker_values: list[float] = []
    if len(image_arrays) >= 2:
        import numpy as np

        for previous, current in zip(image_arrays, image_arrays[1:]):
            if previous.shape != current.shape:
                reasons.append("adjacent frame dimensions differ")
                continue
            flicker_values.append(
                float(np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean() / 255.0)
            )
    quality["temporal_flicker"] = _quality_metric(
        flicker_values,
        reason="at least two decodable frames are required for temporal_flicker",
    )
    if not quality["temporal_flicker"]["available"]:
        reasons.append("quality metric unavailable: temporal_flicker")

    dark_limit = (case.get("quality") or {}).get("max_dark_pixel_ratio")
    dark_metric = quality["dark_pixel_ratio"]
    if _finite(dark_limit) and dark_metric.get("available") and dark_metric["max"] > float(dark_limit):
        reasons.append(
            f"dark pixel ratio {dark_metric['max']:.6f} exceeds configured limit {float(dark_limit):.6f}"
        )
    return quality, reasons


def _case_motion_summary(
    case_id: str,
    frame_rows: list[Mapping[str, Any]],
    probe: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    target_deltas = [
        dict(row["target_pose_delta_m"])
        for row in frame_rows
        if isinstance(row.get("target_pose_delta_m"), Mapping)
    ]
    target_delta = target_deltas[0] if target_deltas else None
    if target_deltas and any(delta != target_delta for delta in target_deltas[1:]):
        reasons.append("target pose delta changes between captured frames")

    translations: list[float] = []
    yaws: list[float] = []
    for row in frame_rows:
        sweep = row.get("camera_sweep_offset")
        if not isinstance(sweep, Mapping):
            continue
        translation = sweep.get("translation_m")
        if isinstance(translation, Mapping):
            translations.append(
                math.sqrt(sum(float(translation.get(axis, 0.0)) ** 2 for axis in ("x", "y", "z")))
            )
        if _finite(sweep.get("yaw_deg")):
            yaws.append(abs(float(sweep["yaw_deg"])))
    motion = {
        "target_pose_delta_m": target_delta,
        "camera_pose_sweep": {
            "max_translation_m": max(translations) if translations else 0.0,
            "max_rotation_deg": max(yaws) if yaws else 0.0,
            "translation_sample_count": len(translations),
            "rotation_sample_count": len(yaws),
        },
    }
    if case_id == "V02" and not target_deltas:
        reasons.append("V02 frame metadata has no target_pose_delta_m")
    if case_id == "V03" and not translations and not yaws:
        reasons.append("V03 frame metadata has no camera_sweep_offset")
    if isinstance(probe, Mapping):
        motion["probe_summary"] = {
            key: probe.get(key)
            for key in (
                "status",
                "aa_request_digest_equal",
                "aa_rgb_repeatable",
                "rgb_response_changed",
                "target_pose_changed",
                "non_target_actors_unchanged",
            )
            if key in probe
        }
    else:
        motion["probe_summary"] = None
    return motion, reasons


def _validate_case_evidence(
    case_id: str,
    evidence_path: Path,
    case_path: Path,
    manifest_path: Path,
    expected_artifact_sha: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    case = _load_json(case_path)
    expected_case_sha = sha256_file(case_path)
    case_entry = _sha_ref(case_path)
    if not evidence_path.is_file():
        reasons.append(f"evidence.json is unavailable: {evidence_path}")
        return {
            "case_id": case_id,
            "kind": case.get("kind"),
            "status": "pending_capture",
            "pass": False,
            "evidence": _sha_ref(evidence_path),
            "case": case_entry,
            "reasons": reasons,
        }
    evidence = _load_json(evidence_path)
    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        reasons.append(f"unsupported evidence schema: {evidence.get('schema_version')!r}")
    if str((evidence.get("case") or {}).get("case_id") or "") != case_id:
        reasons.append("evidence case_id does not match selector")
    if (evidence.get("case") or {}).get("sha256") != expected_case_sha:
        reasons.append("evidence case SHA-256 does not match the checked-in case")
    if (evidence.get("manifest") or {}).get("sha256") != sha256_file(manifest_path):
        reasons.append("evidence manifest SHA-256 does not match the manifest")
    artifact = evidence.get("artifact") or {}
    if artifact.get("sha256") != expected_artifact_sha:
        reasons.append("evidence artifact SHA-256 does not match canonical USDZ")
    if evidence.get("status") != "passed":
        reasons.append(f"capture status is {evidence.get('status')!r}, not 'passed'")
    output = evidence.get("output") or {}
    requested = output.get("frame_count")
    captured = output.get("frame_count")
    requested = (evidence.get("frames") or {}).get("requested_count", requested)
    captured = (evidence.get("frames") or {}).get("captured_count", captured)
    dropped = output.get("dropped_frame_count", 0)
    if not isinstance(requested, int) or requested <= 0:
        reasons.append("requested frame count is missing or empty")
    if not isinstance(captured, int) or captured <= 0:
        reasons.append("captured frame count is missing or empty")
    if requested != captured:
        reasons.append(f"requested/captured frame counts differ: {requested}/{captured}")
    if dropped != 0:
        reasons.append(f"capture dropped {dropped} frame(s)")
    video_value = output.get("video")
    video_path = _resolve(video_value, base=evidence_path.parent) if video_value else None
    if video_path is None or not video_path.is_file() or video_path.stat().st_size <= 0:
        reasons.append("encoded video is unavailable or empty")
        video_ref = _sha_ref(video_path or evidence_path.parent / VIDEO_NAMES[case_id])
    else:
        video_ref = _sha_ref(video_path)
        if video_path.name != VIDEO_NAMES[case_id]:
            reasons.append(f"video name is {video_path.name!r}; expected {VIDEO_NAMES[case_id]!r}")
    probe = evidence.get("probe")
    if case_id in {"V02", "V03"} and (not isinstance(probe, Mapping) or probe.get("status") != "passed"):
        reasons.append(f"{case_id} probe evidence did not pass")
    frame_metadata = output.get("metadata")
    metadata_path = (
        _resolve(frame_metadata, base=evidence_path.parent)
        if frame_metadata
        else evidence_path.parent / "frames.jsonl"
    )
    frame_rows: list[dict[str, Any]] = []
    if metadata_path.is_file():
        try:
            frame_rows = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not all(isinstance(row, Mapping) for row in frame_rows):
                raise ValueError("metadata rows must be objects")
            if isinstance(requested, int) and len(frame_rows) != requested:
                reasons.append(
                    f"frame metadata row count differs from requested count: {len(frame_rows)}/{requested}"
                )
            indices = [row.get("frame_index") for row in frame_rows]
            if indices != list(range(len(indices))):
                reasons.append("frame metadata indices are not contiguous")
            if any(row.get("dropped") for row in frame_rows):
                reasons.append("frame metadata contains dropped rows")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            reasons.append(f"frame metadata is invalid: {exc}")
    else:
        reasons.append(f"frame metadata is unavailable: {metadata_path}")
    quality, quality_reasons = _frame_quality(frame_rows, metadata_path, case)
    reasons.extend(quality_reasons)
    motion, motion_reasons = _case_motion_summary(
        case_id,
        frame_rows,
        probe if isinstance(probe, Mapping) else None,
    )
    reasons.extend(motion_reasons)
    latency_values = [
        float(row.get("frame_rpc_latency_ms", row.get("rpc_latency_ms")))
        for row in frame_rows
        if _finite(row.get("frame_rpc_latency_ms", row.get("rpc_latency_ms")))
    ]
    if not latency_values:
        reasons.append("RPC latency measurements are unavailable")
    expected_camera_ids = _case_camera_ids(case)
    declared_camera_ids = output.get("camera_ids")
    camera_ids = list(declared_camera_ids) if isinstance(declared_camera_ids, list) else expected_camera_ids
    observed_camera_ids: list[str] = []
    for row in frame_rows:
        values = row.get("camera_ids")
        values = values if isinstance(values, list) else [row.get("camera_id")]
        for value in values:
            value = str(value) if value else ""
            if value and value not in observed_camera_ids:
                observed_camera_ids.append(value)
    if observed_camera_ids and observed_camera_ids != camera_ids:
        reasons.append(
            f"camera order differs between evidence and metadata: {camera_ids}/{observed_camera_ids}"
        )
    if camera_ids != expected_camera_ids:
        reasons.append(f"camera IDs differ from case: {camera_ids}/{expected_camera_ids}")
    expected_output_size = _case_output_resolution(case)
    reported_output_size = output.get("video_resolution")
    if isinstance(reported_output_size, Mapping):
        observed_output_size = (
            int(reported_output_size.get("width", 0)),
            int(reported_output_size.get("height", 0)),
        )
        if observed_output_size != expected_output_size:
            reasons.append(
                f"video resolution differs from case: {observed_output_size}/{expected_output_size}"
            )
    camera_quality = _camera_quality_summary(frame_rows, camera_ids)
    video_fps = output.get("video_fps")
    duration_s = (
        float(captured) / float(video_fps)
        if _finite(video_fps) and isinstance(captured, int) and captured > 0
        else None
    )
    status = "passed" if not reasons else (
        "pending_capture"
        if not evidence_path.is_file() or not video_ref.get("exists")
        else "failed"
    )
    return {
        "case_id": case_id,
        "kind": case.get("kind"),
        "status": status,
        "pass": status == "passed",
        "evidence": {**_sha_ref(evidence_path), "status": evidence.get("status")},
        "case": case_entry,
        "video": video_ref,
        "frames": {
            "requested_count": requested,
            "captured_count": captured,
            "dropped_count": dropped,
            "metadata": _sha_ref(metadata_path),
            "camera_ids": camera_ids,
            "first_timestamp_us": (evidence.get("frames") or {}).get("first_timestamp_us"),
            "last_timestamp_us": (evidence.get("frames") or {}).get("last_timestamp_us"),
            "rpc_latency_ms": {
                "p50": _percentile(latency_values, 0.50),
                "p95": _percentile(latency_values, 0.95),
                "sample_count": len(latency_values),
            },
            "camera_count": len(camera_ids),
            "camera_ids": camera_ids,
            "video_resolution": reported_output_size,
            "video_fps": video_fps,
            "duration_s": duration_s,
        },
        "quality": quality,
        "camera_quality": camera_quality,
        "motion": motion,
        "probe": dict(probe) if isinstance(probe, Mapping) else None,
        "reasons": reasons,
    }


def build_quality_report(
    manifest_path: Path,
    evidence_paths: Mapping[str, Path],
    *,
    artifact_path: Path | None = None,
    metrics_path: Path | None = None,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """Build and return a JSON-serialisable report without writing files."""

    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "nsb.nurec-interview-demo-manifest.v1":
        raise QualityReportError("unsupported interview-demo manifest schema")
    case_paths = _case_files(manifest, manifest_path)
    artifact_entry = manifest.get("artifact")
    if not isinstance(artifact_entry, Mapping):
        raise QualityReportError("manifest.artifact must be an object")
    expected_artifact_sha = artifact_entry.get("sha256")
    if not _is_sha256(expected_artifact_sha):
        raise QualityReportError("manifest artifact.sha256 is invalid")
    artifact = (
        artifact_path.resolve()
        if artifact_path is not None
        else _resolve_manifest_uri(
            artifact_entry.get("uri") or artifact_entry.get("path"),
            manifest_path,
            label="manifest artifact URI",
        )
    )
    artifact_ref = _sha_ref(artifact)
    artifact_reasons: list[str] = []
    if artifact_ref["exists"]:
        if artifact_ref["sha256"] != expected_artifact_sha:
            artifact_reasons.append("canonical USDZ SHA-256 does not match manifest")
        if artifact_ref["size_bytes"] != artifact_entry.get("size_bytes"):
            artifact_reasons.append("canonical USDZ size does not match manifest")
    else:
        artifact_reasons.append(f"canonical USDZ is unavailable: {artifact}")
    checkpoint_entry = manifest.get("checkpoint")
    checkpoint_ref = None
    checkpoint_reasons: list[str] = []
    if isinstance(checkpoint_entry, Mapping):
        checkpoint = _resolve_manifest_uri(
            checkpoint_entry.get("uri") or checkpoint_entry.get("path"),
            manifest_path,
            label="manifest checkpoint URI",
        )
        checkpoint_ref = _sha_ref(checkpoint)
        if not checkpoint_ref["exists"]:
            checkpoint_reasons.append(f"matching checkpoint is unavailable: {checkpoint}")
        else:
            if checkpoint_ref["sha256"] != checkpoint_entry.get("sha256"):
                checkpoint_reasons.append("matching checkpoint SHA-256 does not match manifest")
            if checkpoint_ref["size_bytes"] != checkpoint_entry.get("size_bytes"):
                checkpoint_reasons.append("matching checkpoint size does not match manifest")
    else:
        checkpoint_reasons.append("manifest.checkpoint is missing")
    selected_metrics_path = metrics_path.resolve() if metrics_path else _infer_metrics_path(artifact)
    metrics, metric_reasons = _load_metrics(selected_metrics_path)
    cases = [
        _validate_case_evidence(
            case_id,
            evidence_paths.get(case_id, manifest_path.parent / "cases" / case_id / "evidence.json"),
            case_paths[case_id],
            manifest_path,
            expected_artifact_sha,
        )
        for case_id in REQUIRED_CASES
    ]
    case_reasons = [reason for case in cases for reason in case["reasons"]]
    all_identity_ok = not artifact_reasons and not checkpoint_reasons
    all_cases_ok = all(case["pass"] for case in cases)
    metrics_ok = metrics is not None and not metric_reasons
    status = "passed" if all_identity_ok and all_cases_ok and metrics_ok else (
        "pending_capture" if any(case["status"] == "pending_capture" for case in cases) else "failed"
    )
    reasons = artifact_reasons + checkpoint_reasons + metric_reasons + case_reasons
    limitations = list(manifest.get("limitations") or [])
    limitations.extend(
        [
            "The report proves renderer RPC and file-level evidence only; it is not CARLA closed-loop acceptance.",
            "A pending_capture or failed report must not be used as a passed quality claim.",
        ]
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "pass": status == "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene": {
            "scene_id": manifest.get("scene_id"),
            "runtime_scene_id": manifest.get("runtime_scene_id"),
            "timestamp_range_us": manifest.get("scene_timestamp_range_us"),
        },
        "artifact": {
            "role": artifact_entry.get("role"),
            "uri": artifact_entry.get("uri") or artifact_entry.get("path"),
            "resolved_path": str(artifact),
            "sha256": expected_artifact_sha,
            "size_bytes": artifact_entry.get("size_bytes"),
            "observed": artifact_ref,
            "identity_verified": not artifact_reasons,
        },
        "checkpoint": {
            "role": checkpoint_entry.get("role") if isinstance(checkpoint_entry, Mapping) else None,
            "uri": checkpoint_entry.get("uri") if isinstance(checkpoint_entry, Mapping) else None,
            "sha256": checkpoint_entry.get("sha256") if isinstance(checkpoint_entry, Mapping) else None,
            "size_bytes": checkpoint_entry.get("size_bytes") if isinstance(checkpoint_entry, Mapping) else None,
            "observed": checkpoint_ref,
            "identity_verified": not checkpoint_reasons,
        },
        "code_commit": code_commit or _git_commit(),
        "nurec": dict(manifest.get("nurec") or {}),
        "runtime": dict(manifest.get("runtime") or {}),
        "track_inventory": dict(manifest.get("usd_z_validation") or {}),
        "reconstruction": {
            "training_recipe": artifact_entry.get("training_recipe"),
            "metrics": metrics,
            "metrics_status": (
                "passed"
                if metrics and not metric_reasons
                else ("unavailable" if not metrics else "failed")
            ),
            "config": _sha_ref(_infer_config_path(artifact)) if _infer_config_path(artifact) else None,
        },
        "cases": cases,
        "gates": {
            "artifact_identity": not artifact_reasons,
            "checkpoint_identity": not checkpoint_reasons,
            "case_capture": all_cases_ok,
            "metrics": metrics_ok,
            "report_pass": status == "passed",
        },
        "reasons": reasons,
        "limitations": list(dict.fromkeys(limitations)),
    }
    validate_quality_report(report)
    return report


def validate_quality_report(report: Mapping[str, Any]) -> None:
    """Validate the report's fail-closed invariants for downstream consumers."""

    if report.get("schema_version") != REPORT_SCHEMA:
        raise QualityReportError("unsupported quality report schema")
    status = report.get("status")
    if status not in {"passed", "pending_capture", "failed"}:
        raise QualityReportError("quality report status is invalid")
    if report.get("pass") != (status == "passed"):
        raise QualityReportError("quality report pass flag disagrees with status")
    cases = report.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, Mapping) for case in cases):
        raise QualityReportError("quality report cases are malformed")
    if {case.get("case_id") for case in cases} != set(REQUIRED_CASES):
        raise QualityReportError("quality report must contain exactly V01, V02, and V03")
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        raise QualityReportError("quality report gates are missing")
    if status == "passed" and not all(gates.get(name) is True for name in (
        "artifact_identity", "checkpoint_identity", "case_capture", "metrics", "report_pass"
    )):
        raise QualityReportError("passed report has an incomplete gate set")
    if status == "passed":
        required_quality_metrics = (
            "invalid_pixel_ratio",
            "dark_pixel_ratio",
            "laplacian_sharpness",
            "temporal_flicker",
        )
        for case in cases:
            quality = case.get("quality")
            if not isinstance(quality, Mapping):
                raise QualityReportError(f"{case.get('case_id')} quality measurements are missing")
            for name in required_quality_metrics:
                metric = quality.get(name)
                if not isinstance(metric, Mapping) or metric.get("available") is not True:
                    raise QualityReportError(
                        f"{case.get('case_id')} quality measurement is unavailable: {name}"
                    )
                if not _finite(metric.get("value")):
                    raise QualityReportError(
                        f"{case.get('case_id')} quality measurement is non-finite: {name}"
                    )
    if status != "passed" and report.get("pass") is True:
        raise QualityReportError("non-passed report cannot set pass=true")


def _parse_cli_evidence(values: list[str], root: Path | None) -> dict[str, Path]:
    result = discover_evidence(root) if root else {}
    base = root.resolve() if root else REPO_ROOT
    for raw in values:
        case_id, path = _parse_evidence_argument(raw, base=base)
        if case_id in result:
            raise QualityReportError(f"duplicate evidence for {case_id}")
        result[case_id] = path
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--evidence", action="append", default=[], metavar="CASE=PATH")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate",
        type=Path,
        metavar="REPORT",
        help="validate an existing report without rebuilding it",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate:
            report_path = _resolve(args.validate)
            report = _load_json(report_path)
            validate_quality_report(report)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "pass": report["pass"],
                        "report": str(report_path),
                    },
                    sort_keys=True,
                )
            )
            return 0 if report["pass"] else 2
        if args.output is None:
            parser.error("--output is required when --validate is not used")
        output = _resolve(args.output)
        if output.exists() and not args.overwrite:
            raise QualityReportError(f"refusing to overwrite existing output: {output}")
        root = _resolve(args.evidence_root) if args.evidence_root else None
        report = build_quality_report(
            _resolve(args.manifest),
            _parse_cli_evidence(args.evidence, root),
            artifact_path=_resolve(args.artifact) if args.artifact else None,
            metrics_path=_resolve(args.metrics) if args.metrics else None,
            code_commit=args.code_commit,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "pass": report["pass"], "output": str(output)}, sort_keys=True))
        return 0 if report["pass"] else 2
    except (OSError, QualityReportError, ValueError) as exc:
        print(f"generate_nurec_quality_report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
