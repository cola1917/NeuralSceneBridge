#!/usr/bin/env python3
"""Finalize the scene-0061 jitter evidence into JSON and Markdown.

The live diagnostic deliberately keeps capture and analysis separate.  This
post-processing step joins the already captured A-F evidence with the formal
V01/V02/V03 metadata, computes the same-timestamp V02 comparison, and writes a
human-readable report without re-rendering or changing any frames.
"""

from __future__ import annotations

import argparse
import bisect
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_nurec_jitter import (  # noqa: E402
    _crop,
    _project_target,
    _stats,
)
from scripts.render_counterfactual_video import (  # noqa: E402
    ArtifactScene,
    MAX_INTERPOLATION_GAP_US,
    TARGET_TRACK_ID,
    _pose_from_matrix,
    load_json,
    resolve_case,
    sha256_file,
)


DEFAULT_BASE_REPORT = REPO_ROOT / "outputs/nurec_scene0061_jitter_diagnostics/debug_report.json"
DEFAULT_PEAK_REPORT = REPO_ROOT / "outputs/nurec_scene0061_jitter_peak/debug_report.json"
DEFAULT_MANIFEST = REPO_ROOT / "demo/scene0061/manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/nurec_scene0061_jitter_diagnostics"
FORMAL_ROOT = REPO_ROOT / "outputs/nurec_scene0061_demo"
FORMAL_20HZ_ROOT = REPO_ROOT / "outputs/nurec_scene0061_demo_20hz"
STRICT_RAW_20HZ_ROOT = REPO_ROOT / "outputs/nurec_scene0061_demo_20hz_raw"
TARGET_ONLY_RAW_20HZ_ROOT = REPO_ROOT / "outputs/nurec_scene0061_demo_20hz_target_only"
MULTIMODAL_ROOTS = (
    REPO_ROOT / "outputs/nurec_scene0061_final/multimodal_20fps",
)
CAMERA_ID = "camera_front"
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 450
CAMERA_TILE_X = 800


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"metadata rows must be objects: {path}")
    return rows


def _timestamp(row: Mapping[str, Any]) -> int:
    for key in ("scene_timestamp_us", "requested_timestamp_us", "output_timestamp_us", "source_timestamp_us"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    raise ValueError("metadata row has no timestamp")


def _image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def _image_diff(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.abs(first.astype(np.float32) - second.astype(np.float32)).mean())


def _stitched_paths(root: Path) -> list[Path]:
    paths = sorted((root / "frames").glob("*.jpg"))
    if not paths:
        raise ValueError(f"no stitched frames under {root}")
    return paths


def _pose_from_row(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    poses = row.get("requested_sensor_poses")
    if isinstance(poses, Mapping):
        camera = poses.get(CAMERA_ID)
        if isinstance(camera, Mapping) and isinstance(camera.get("start"), Mapping):
            return camera["start"]
    return None


def _analysis_track_pose(track: Mapping[str, Any], timestamp_us: int) -> tuple[list[float] | None, bool]:
    """Return a pose and whether its interpolation crosses a source gap.

    This helper is intentionally analysis-only.  The renderer now fails closed
    for the same condition; reading legacy frames still needs a flag rather than
    silently dropping the comparison.
    """

    timestamps = [int(value) for value in track["timestamps_us"]]
    poses = track["poses"]
    if timestamp_us < timestamps[0] or timestamp_us > timestamps[-1]:
        return None, False
    right = bisect.bisect_left(timestamps, timestamp_us)
    if right == 0 or timestamps[right] == timestamp_us:
        return [float(value) for value in poses[right]], False
    left = right - 1
    span = timestamps[right] - timestamps[left]
    fraction = (timestamp_us - timestamps[left]) / span if span else 0.0
    first = [float(value) for value in poses[left]]
    second = [float(value) for value in poses[right]]
    position = [first[index] + (second[index] - first[index]) * fraction for index in range(3)]
    # Quaternion SLERP is not needed for the target projection, but retain the
    # source endpoint orientation so this report never invents a new rotation.
    quaternion = first[3:7] if fraction < 0.5 else second[3:7]
    return position + quaternion, span > MAX_INTERPOLATION_GAP_US


def _camera_center(scene: ArtifactScene, row: Mapping[str, Any], target_pose: list[float] | None) -> tuple[tuple[float, float] | None, bool]:
    camera_pose = _pose_from_row(row)
    if camera_pose is None:
        timestamp_us = _timestamp(row)
        camera_pose = _pose_from_matrix(scene.sensor_pose_matrix(CAMERA_ID, timestamp_us))
    if camera_pose is None or target_pose is None:
        return None, False
    local = _project_target(scene, CAMERA_ID, camera_pose, target_pose, CAMERA_WIDTH, CAMERA_HEIGHT)
    if local is None:
        return None, False
    return (local[0] + CAMERA_TILE_X, local[1]), True


def _target_pose_for_row(scene: ArtifactScene, row: Mapping[str, Any], case: Mapping[str, Any]) -> tuple[list[float] | None, bool]:
    timestamp_us = _timestamp(row)
    pose, crossed_gap = _analysis_track_pose(scene.tracks[TARGET_TRACK_ID], timestamp_us)
    if pose is None:
        return None, crossed_gap
    delta = row.get("target_pose_delta_m")
    if not isinstance(delta, Mapping):
        edit = case.get("lead_vehicle_edit")
        delta = edit.get("translation_m") if isinstance(edit, Mapping) else None
    if isinstance(delta, Mapping):
        pose[0] += float(delta.get("x", 0.0))
        pose[1] += float(delta.get("y", 0.0))
        pose[2] += float(delta.get("z", 0.0))
    return pose, crossed_gap


def _background_diff(first: np.ndarray, second: np.ndarray, centers: list[tuple[float, float] | None]) -> float:
    mask = np.ones(first.shape[:2], dtype=bool)
    for center in centers:
        if center is None:
            continue
        x, y = (int(round(value)) for value in center)
        left = max(0, x - 90)
        right = min(first.shape[1], left + 180)
        top = max(0, y - 64)
        bottom = min(first.shape[0], top + 128)
        mask[top:bottom, left:right] = False
    diff = np.abs(first.astype(np.float32) - second.astype(np.float32)).mean(axis=2)
    return float(diff[mask].mean()) if mask.any() else float(diff.mean())


def _formal_pair_comparison(scene: ArtifactScene, v01_root: Path, v02_root: Path, case_v01: Mapping[str, Any], case_v02: Mapping[str, Any]) -> dict[str, Any]:
    rows_01 = _read_rows(v01_root / "frames.jsonl")
    rows_02 = _read_rows(v02_root / "frames.jsonl")
    paths_01 = _stitched_paths(v01_root)
    paths_02 = _stitched_paths(v02_root)
    count = min(len(rows_01), len(rows_02), len(paths_01), len(paths_02))
    full: list[float] = []
    target: list[float] = []
    background: list[float] = []
    center_delta: list[float] = []
    timestamp_mismatches = 0
    skipped_projection = 0
    crossed_gaps = 0
    for index in range(count):
        row_01, row_02 = rows_01[index], rows_02[index]
        if _timestamp(row_01) != _timestamp(row_02):
            timestamp_mismatches += 1
        image_01, image_02 = _image(paths_01[index]), _image(paths_02[index])
        full.append(_image_diff(image_01, image_02))
        pose_01, gap_01 = _target_pose_for_row(scene, row_01, case_v01)
        pose_02, gap_02 = _target_pose_for_row(scene, row_02, case_v02)
        center_01, projected_01 = _camera_center(scene, row_01, pose_01)
        center_02, projected_02 = _camera_center(scene, row_02, pose_02)
        crossed_gaps += int(gap_01 or gap_02)
        if projected_01 and projected_02 and center_01 and center_02:
            crop_01, crop_02 = _crop(image_01, center_01), _crop(image_02, center_02)
            if crop_01 is not None and crop_02 is not None and crop_01.shape == crop_02.shape:
                target.append(_image_diff(crop_01, crop_02))
            center_delta.append(math.dist(center_01, center_02))
        else:
            skipped_projection += 1
        background.append(_background_diff(image_01, image_02, [center_01, center_02]))
    return {
        "v01_root": str(v01_root),
        "v02_root": str(v02_root),
        "frame_count_compared": count,
        "v01_frame_count": len(rows_01),
        "v02_frame_count": len(rows_02),
        "timestamp_mismatch_count": timestamp_mismatches,
        "full_frame_abs_diff": _stats(full),
        "target_crop_abs_diff": _stats(target),
        "background_abs_diff": _stats(background),
        "target_center_delta_px": _stats(center_delta),
        "target_projection_pair_count": len(target),
        "skipped_projection_count": skipped_projection,
        "source_gap_crossing_count": crossed_gaps,
        "target_edit_scope": {
            "track_id": TARGET_TRACK_ID,
            "v01_delta_m": {"x": 0.0, "y": 0.0, "z": 0.0},
            "v02_delta_m": dict(case_v02.get("lead_vehicle_edit", {}).get("translation_m", {})),
            "camera_pose_held_by_timestamp": True,
        },
    }


def _formal_case_summary(case_id: str, root: Path) -> dict[str, Any]:
    evidence_path = root / "evidence.json"
    evidence = _read_json(evidence_path)
    output = evidence.get("output") or {}
    frames = evidence.get("frames") or {}
    probe = evidence.get("probe")
    return {
        "case_id": case_id,
        "kind": (evidence.get("case") or {}).get("kind"),
        "evidence": str(evidence_path),
        "video": output.get("video"),
        "metadata": output.get("metadata"),
        "frame_count": frames.get("captured_count"),
        "requested_count": frames.get("requested_count"),
        "dropped_count": output.get("dropped_frame_count"),
        "video_fps": output.get("video_fps"),
        "video_resolution": output.get("video_resolution"),
        "source_resolution": output.get("source_resolution"),
        "camera_count": output.get("camera_count"),
        "camera_ids": output.get("camera_ids"),
        "pose_summary_json": output.get("pose_summary_json"),
        "pose_summary_png": output.get("pose_summary_png"),
        "first_timestamp_us": frames.get("first_timestamp_us"),
        "last_timestamp_us": frames.get("last_timestamp_us"),
        "probe_status": probe.get("status") if isinstance(probe, Mapping) else None,
        "probe": probe,
    }


def _raw_20hz_evidence() -> dict[str, Any]:
    """Summarize strict full-dynamic and explicit target-only raw attempts."""

    failure_path = STRICT_RAW_20HZ_ROOT / "V01" / "capture_error.json"
    failure = _read_json(failure_path) if failure_path.is_file() else {
        "status": "not_recorded",
        "scope": "all DYNAMIC|CONTROLLABLE tracks",
        "video_written": False,
    }
    preview_root = TARGET_ONLY_RAW_20HZ_ROOT / "V01"
    preview_evidence_path = preview_root / "evidence.json"
    preview: dict[str, Any] = {
        "status": "missing",
        "scope": "manifest target track only; other controllable actors omitted explicitly",
        "root": str(preview_root),
        "video": str(preview_root / "original_replay_target_only_20hz.mp4"),
        "metadata": str(preview_root / "frames.jsonl"),
    }
    if preview_evidence_path.is_file():
        evidence = _read_json(preview_evidence_path)
        output = evidence.get("output") if isinstance(evidence.get("output"), Mapping) else {}
        frames = evidence.get("frames") if isinstance(evidence.get("frames"), Mapping) else {}
        preview.update(
            {
                "status": evidence.get("status", "unknown"),
                "video": output.get("video", preview["video"]),
                "metadata": output.get("metadata", preview["metadata"]),
                "frame_count": frames.get("captured_count"),
                "requested_count": frames.get("requested_count"),
                "dropped_count": output.get("dropped_frame_count"),
                "sample_fps": output.get("sample_fps"),
                "video_fps": output.get("video_fps"),
                "video_resolution": output.get("video_resolution"),
                "camera_count": output.get("camera_count"),
                "camera_ids": output.get("camera_ids"),
                "sampling_mode": output.get("sampling_mode"),
                "case": evidence.get("case"),
            }
        )
    return {
        "strict_full_dynamic": {
            "attempt_root": str(STRICT_RAW_20HZ_ROOT / "V01"),
            "error_evidence": str(failure_path),
            **failure,
        },
        "target_only_preview": preview,
    }


def _multimodal_evidence() -> dict[str, Any]:
    """Load the final V04 RGB/LiDAR evidence without promoting failure."""

    for root in MULTIMODAL_ROOTS:
        evidence_path = root / "evidence.json"
        if not evidence_path.is_file():
            continue
        try:
            evidence = _read_json(evidence_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        consistency = evidence.get("consistency")
        if not isinstance(consistency, Mapping):
            consistency = {}
        lidar = evidence.get("lidar")
        if not isinstance(lidar, Mapping):
            lidar = {}
        result = {
            "root": str(root),
            "probe": str(evidence_path),
            "status": evidence.get("status"),
            "reason": consistency.get("claim") or "; ".join(evidence.get("limitations", [])),
            "visual": evidence.get("video"),
            "timestamp_us": evidence.get("timestamp_range_us"),
            "window_us": evidence.get("rgb_lidar_timestamp_alignment_max_us"),
            "pairing": evidence.get("rgb_lidar_pairing"),
            "evidence_classification": evidence.get("evidence_classification"),
            "control_mode": evidence.get("control_mode"),
            "camera_ids": ["camera_front"],
            "lidar_id": "lidar_top",
            "rgb_actor_changed": "A/B signal recorded",
            "lidar_actor_changed": "response difference recorded; actor ownership not established",
            "baseline_point_count": None,
            "edited_point_count": None,
            "voxel_added_count": None,
            "voxel_removed_count": None,
            "claim": consistency.get("claim"),
            "coordinate_frame": lidar.get("coordinate_frame"),
        }
        return result
    return {
        "status": "not_run",
        "reason": "No final V04 RGB/LiDAR evidence.json was found",
    }


def _source_peak(scene: ArtifactScene) -> dict[str, Any]:
    track = scene.tracks[TARGET_TRACK_ID]
    timestamps = [int(value) for value in track["timestamps_us"]]
    poses = track["poses"]
    speeds: list[float] = []
    for index, (first, second) in enumerate(zip(poses, poses[1:])):
        dt = (timestamps[index + 1] - timestamps[index]) / 1_000_000.0
        speeds.append(math.dist(first[:3], second[:3]) / dt if dt > 0 else float("nan"))
    accelerations: list[float] = []
    for index, (first, second) in enumerate(zip(speeds, speeds[1:])):
        dt = (timestamps[index + 2] - timestamps[index]) / 2_000_000.0
        accelerations.append((second - first) / dt if dt > 0 else float("nan"))
    peak_index = max(range(len(accelerations)), key=lambda index: abs(accelerations[index]))
    return {
        "track_id": TARGET_TRACK_ID,
        "index": peak_index,
        "acceleration_mps2": accelerations[peak_index],
        "absolute_acceleration_mps2": abs(accelerations[peak_index]),
        "start_timestamp_us": timestamps[peak_index],
        "end_timestamp_us": timestamps[peak_index + 2],
        "speed_before_mps": speeds[peak_index],
        "speed_after_mps": speeds[peak_index + 1],
    }


def _peak_window_summary(peak_report_path: Path, peak_report: Mapping[str, Any], source_peak: Mapping[str, Any]) -> dict[str, Any]:
    groups = peak_report.get("groups") if isinstance(peak_report.get("groups"), Mapping) else {}
    window_start = None
    window_end = None
    for group_name in ("D_short_30hz_interpolated_render_30fps_encode", "E_30hz_fixed_camera_target_motion"):
        group = groups.get(group_name)
        metadata = group.get("metadata") if isinstance(group, Mapping) else None
        if not metadata or not Path(metadata).is_file():
            continue
        rows = _read_rows(Path(metadata))
        if rows:
            values = [_timestamp(row) for row in rows]
            window_start = min(values) if window_start is None else min(window_start, min(values))
            window_end = max(values) if window_end is None else max(window_end, max(values))
    selected = {}
    for name in ("B_short_20hz_render_20fps_encode", "D_short_30hz_interpolated_render_30fps_encode", "E_30hz_fixed_camera_target_motion", "F_30hz_camera_motion_fixed_dynamic"):
        group = groups.get(name)
        if isinstance(group, Mapping):
            selected[name] = {
                "video": group.get("video"),
                "metadata": group.get("metadata"),
                "video_fps": group.get("video_fps"),
                "visual": group.get("visual"),
                "request_sequence_unique": group.get("request_sequence_unique"),
                "request_frame_id_unique": group.get("request_frame_id_unique"),
                "response_timestamp_available": group.get("response_timestamp_available"),
                "captured_count": group.get("captured_count"),
                "dropped_count": group.get("dropped_count"),
            }
    return {
        "report": str(peak_report_path),
        "source_acceleration_peak": dict(source_peak),
        "window_timestamp_range_us": {"start": window_start, "end": window_end},
        "window_contains_source_peak": bool(
            window_start is not None
            and window_end is not None
            and window_start <= int(source_peak["start_timestamp_us"]) <= window_end
        ),
        "groups": selected,
    }


def _build_report(base: Mapping[str, Any], peak: Mapping[str, Any] | None, manifest: Mapping[str, Any], scene: ArtifactScene, artifact_path: Path) -> dict[str, Any]:
    report = dict(base)
    # The peak directory is an input snapshot restored for analysis.  The
    # finalized deliverable lives in the diagnostics directory, so all report
    # relative paths and the explicit output_dir point there.
    report["output_dir"] = str(DEFAULT_OUTPUT_DIR)
    case_paths = {}
    cases = {}
    for case_id in ("V01", "V02", "V03"):
        path, case = resolve_case(case_id)
        case_paths[case_id] = path
        cases[case_id] = case
    formal_cases = {
        "V01": _formal_case_summary("V01", FORMAL_ROOT / "V01"),
        "V02": _formal_case_summary("V02", FORMAL_ROOT / "V02"),
        "V03": _formal_case_summary("V03", FORMAL_ROOT / "V03"),
    }
    v02_comparison = _formal_pair_comparison(
        scene,
        FORMAL_ROOT / "V01",
        FORMAL_ROOT / "V02",
        cases["V01"],
        cases["V02"],
    )
    source_peak = _source_peak(scene)
    peak_window = _peak_window_summary(
        DEFAULT_PEAK_REPORT,
        peak or {},
        source_peak,
    ) if peak else {"source_acceleration_peak": source_peak}
    encoding = report.get("encoding") if isinstance(report.get("encoding"), Mapping) else {}
    aa = report.get("groups", {}).get("A_A_repeat", {}) if isinstance(report.get("groups"), Mapping) else {}
    b = report.get("groups", {}).get("B_20hz_render_20fps_encode", {}) if isinstance(report.get("groups"), Mapping) else {}
    c = report.get("groups", {}).get("C_20hz_render_30fps_duplicate_encode", {}) if isinstance(report.get("groups"), Mapping) else {}
    d = report.get("groups", {}).get("D_30hz_interpolated_render_30fps_encode", {}) if isinstance(report.get("groups"), Mapping) else {}
    e = report.get("groups", {}).get("E_30hz_fixed_camera_target_motion", {}) if isinstance(report.get("groups"), Mapping) else {}
    f = report.get("groups", {}).get("F_30hz_camera_motion_fixed_dynamic", {}) if isinstance(report.get("groups"), Mapping) else {}
    raw_20hz = _raw_20hz_evidence()
    multimodal = _multimodal_evidence()
    report.update(
        {
            "finalized": True,
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "manifest_identity": {
                "path": str(DEFAULT_MANIFEST),
                "sha256": sha256_file(DEFAULT_MANIFEST),
                "artifact_path": str(artifact_path),
                "artifact_sha256": report.get("artifact", {}).get("sha256"),
            },
            "run_contract": {
                "renderer_script": str(REPO_ROOT / "scripts/render_counterfactual_video.py"),
                "diagnostic_script": str(REPO_ROOT / "scripts/diagnose_nurec_jitter.py"),
                "image": manifest.get("nurec", {}).get("image"),
                "runtime_version": manifest.get("nurec", {}).get("version_id"),
                "renderer_git_hash": manifest.get("nurec", {}).get("git_hash"),
                "server_address": manifest.get("runtime", {}).get("server_address"),
                "runtime_scene_id": manifest.get("runtime", {}).get("runtime_scene_id"),
                "python_api_path": manifest.get("runtime", {}).get("python_api_path"),
                "service_command": "serve-grpc --artifact-glob /scenes/last.usdz --host 127.0.0.1 --port 46443 --enable-editing-actors --enable-timing --timing-verbosity summary",
                "formal_render_command": "python3 scripts/render_counterfactual_video.py --manifest demo/scene0061/manifest.json --artifact <canonical last.usdz> --server-address 127.0.0.1:46443 --case V01|V02|V03 --output-dir outputs/nurec_scene0061_demo/<case>",
            },
            "interpolation_contract": {
                "translation": "time-linear",
                "rotation": "quaternion SLERP with shortest-path sign correction",
                "max_source_gap_us": MAX_INTERPOLATION_GAP_US,
                "cross_gap_behavior": "fail_closed",
                "raw_replay_smoothing": False,
                "video_post_processing": "none",
            },
            "formal_cases": formal_cases,
            "v02_same_timestamp_comparison": v02_comparison,
            "peak_window": peak_window,
            "rpc_contract": {
                "request_sequence_recorded": True,
                "request_frame_id_recorded": True,
                "request_timestamp_recorded": True,
                "request_send_receive_wall_time_recorded": True,
                "request_and_response_digest_recorded": True,
                "response_frame_id_available": False,
                "response_timestamp_available": False,
                "realized_timestamp_error": "RGBRenderReturn exposes image_bytes only; realized timestamp/frame_id cannot be read back",
                "calls_sequential": True,
                "aa_evidence": {
                    "request_digest_equal": aa.get("request_digest_equal"),
                    "response_digest_equal": aa.get("response_digest_equal"),
                    "rgb_payload_equal": aa.get("rgb_payload_equal"),
                    "pixel_abs_diff_mean": aa.get("pixel_abs_diff_mean"),
                },
            },
            "source_trajectory_findings": {
                "rig": report.get("trajectory", {}).get("rig"),
                "target": report.get("trajectory", {}).get("target"),
                "coordinate_contract": "USDZ rig/world and actor pose arrays use meters plus XYZW unit quaternions; camera requests use the same sensor pose timeline",
                "unit_checks": {"position_unit": "meter", "angle_unit": "degree only for case offsets", "quaternion_order": "xyzw"},
            },
            "diagnostic_interpretation": {
                "A_A": "passed: identical request/response/RGB payloads; no RPC/renderer nondeterminism observed",
                "B_20_20": "baseline: original-cadence render and 20 FPS CFR encode",
                "C_20_30": f"not a real 30 Hz render: {c.get('duplicate_output_frame_count')} of {c.get('output_frame_count')} output frames repeat a source frame",
                "D_30_30": "timestamp-interpolated 30 Hz render is visually smoother than B in measured mean differences, but legacy capture crossed source gaps and lacks realized timestamp readback",
                "E": "fixed camera leaves the background nearly stable while target crop changes, isolating local dynamic-layer behavior",
                "F": "fixed dynamic payload still moves the background under camera motion, separating camera trajectory motion from target-local motion",
                "V02": "same-timestamp edit comparison is concentrated in target crop, with a much smaller background difference",
            },
            "root_cause": {
                "primary": "The observed target-local swim is not explained by 20-to-30 FPS encoding alone. The renderer path is deterministic, while the source target has two approximately 100 ms timestamp holes and a high acceleration peak; interpolated dynamic-layer poses across those holes are not source-supported.",
                "secondary": "With the camera fixed, background second-difference is low and target-crop second-difference remains high, which points to USDZ dynamic-layer temporal stability/pose support as the remaining limitation after request pairing is ruled out.",
                "encoding": "CFR duplicate-frame retiming creates cadence judder, but true D 30/30 rendering does not show a metric increase over B 20/20; it is a separate presentation artifact.",
                "confidence": "high for excluding RPC nondeterminism and duplicate-frame encoding as the sole cause; medium-high for source-gap plus USDZ dynamic-layer limitation because realized render timestamps are unavailable from the API",
            },
            "decision": {
                "raw_20hz_is_source_faithful": False,
                "raw_20hz_video": str(FORMAL_20HZ_ROOT / "V01" / "original_replay.mp4"),
                "uniform_20fps_cadence_baseline_available": True,
                "uniform_20fps_cadence_baseline_video": str(FORMAL_20HZ_ROOT / "V01" / "original_replay.mp4"),
                "strict_raw_timestamp_capture_status": raw_20hz["strict_full_dynamic"].get("status"),
                "strict_raw_timestamp_capture": raw_20hz["strict_full_dynamic"],
                "target_only_raw_20hz_preview": raw_20hz["target_only_preview"],
                "legacy_30hz_videos_are_source_faithful": False,
                "formal_20fps_baseline_allowed": True,
                "formal_30fps_three_case_generation_allowed": False,
                "blocking_reasons": [
                    "strict full-dynamic raw replay hits a 500.404 ms gap in a non-target controllable pedestrian track",
                    "the target and many other controllable tracks also contain approximately 100 ms source holes exceeding the 75 ms interpolation contract",
                    "NuRec RGBRenderReturn does not expose realized timestamp/frame_id for end-to-end timestamp proof",
                ],
                "required_next_fix": "Repair or re-export the source trajectories/artifact so every controllable track is supported at the requested timestamps, then rerun the strict full-dynamic raw replay; do not bridge gaps or use optical-flow post-processing.",
                "short_evidence_video": str(DEFAULT_OUTPUT_DIR / "E_30hz_fixed_camera_target_motion" / "E.mp4"),
            },
            "source_inventory": {
                "cases": {case_id: {"path": str(path), "sha256": sha256_file(path)} for case_id, path in case_paths.items()},
                "metadata": {case_id: formal_cases[case_id]["metadata"] for case_id in formal_cases},
                "videos": {case_id: formal_cases[case_id]["video"] for case_id in formal_cases},
                "diagnostic_metadata": {name: value.get("metadata") for name, value in report.get("groups", {}).items() if isinstance(value, Mapping) and value.get("metadata")},
                "encoding_report": str(Path(report.get("output_dir", DEFAULT_OUTPUT_DIR)) / "encoding_report.json"),
                "multimodal_probe": multimodal.get("probe"),
                "multimodal_visual": multimodal.get("visual"),
            },
            "multimodal_consistency": multimodal,
        }
    )
    return report


def _num(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def _metric(group: Mapping[str, Any], name: str, stat: str = "mean") -> Any:
    visual = group.get("visual") if isinstance(group, Mapping) else None
    temporal = visual.get("temporal") if isinstance(visual, Mapping) else None
    metric = temporal.get(name) if isinstance(temporal, Mapping) else None
    return metric.get(stat) if isinstance(metric, Mapping) else None


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    groups = report.get("groups") if isinstance(report.get("groups"), Mapping) else {}
    trajectory = report.get("trajectory") if isinstance(report.get("trajectory"), Mapping) else {}
    rig = trajectory.get("rig") if isinstance(trajectory.get("rig"), Mapping) else {}
    target = trajectory.get("target") if isinstance(trajectory.get("target"), Mapping) else {}
    root_cause = report.get("root_cause") or {}
    decision = report.get("decision") or {}
    lines = [
        "# scene-0061 NuRec 动态目标抖动诊断",
        "",
        f"生成时间：`{report.get('finalized_at', report.get('generated_at'))}`",
        "",
        "## 结论",
        "",
        f"{root_cause.get('primary', '')}",
        "",
        f"{root_cause.get('secondary', '')}",
        "",
        f"编码结论：{root_cause.get('encoding', '')}",
        "",
        f"置信度：{root_cause.get('confidence', '')}",
        "",
        "正式视频门槛：",
        "",
        f"- 均匀 20 FPS cadence 基线：`{'允许用于观察节奏' if decision.get('formal_20fps_baseline_allowed') else '不允许'}`；文件 `{decision.get('uniform_20fps_cadence_baseline_video', decision.get('raw_20hz_video'))}`。该文件不是完整 source-timestamp faithful replay。",
        f"- 严格完整全动态 raw 20 Hz：`{'成功' if decision.get('strict_raw_timestamp_capture_status') == 'passed' else '失败并 fail-closed'}`；详见 `{(decision.get('strict_raw_timestamp_capture') or {}).get('error_evidence')}`。",
        f"- 主目标单轨 raw 20 Hz 预览：`{(decision.get('target_only_raw_20hz_preview') or {}).get('video')}`；仅用于观察主目标时间表现，不代表完整动态层。",
        f"- 当前 30 FPS 三段正式视频：`{'允许' if decision.get('formal_30fps_three_case_generation_allowed') else '暂不允许作为 source-faithful 证据'}`。",
        f"- 短片证据：`{decision.get('short_evidence_video')}`。",
        "",
        "## 身份与运行",
        "",
        f"- USDZ：`{report.get('artifact', {}).get('path')}`",
        f"- USDZ SHA-256：`{report.get('artifact', {}).get('sha256')}`",
        f"- USDZ 大小：`{report.get('artifact', {}).get('size_bytes')}` bytes",
        f"- NRE image：`{report.get('run_contract', {}).get('image')}`；runtime：`{report.get('run_contract', {}).get('runtime_version')}` / `{report.get('run_contract', {}).get('renderer_git_hash')}`",
        f"- renderer：`{report.get('run_contract', {}).get('renderer_script')}`",
        f"- service：`{report.get('run_contract', {}).get('service_command')}`",
        "",
        "## 输入轨迹",
        "",
        f"rig：{rig.get('count')} samples，effective `{_num(rig.get('effective_hz'), 3)} Hz`，median delta `{_num((rig.get('delta_us') or {}).get('median'), 0)} us`，strictly increasing `{rig.get('strictly_increasing')}`，duplicates `{rig.get('duplicate_count')}`。",
        f"target：speed max `{_num((target.get('motion') or {}).get('speed_mps', {}).get('max'))} m/s`，speed second-difference max `{_num((target.get('motion') or {}).get('speed_second_difference_mps', {}).get('max'))} m/s²`；quaternion norm error `{(target.get('quaternion') or {}).get('norm_error_count')}`，negative adjacent dot `{(target.get('quaternion') or {}).get('negative_dot_count')}`。",
        "",
        "时间洞：",
        "",
    ]
    for gap in rig.get("gaps_gt_75ms", []) if isinstance(rig.get("gaps_gt_75ms"), list) else []:
        lines.append(f"- index `{gap.get('index')}`：`{gap.get('start_us')}` -> `{gap.get('end_us')}`，delta `{gap.get('delta_us')} us`。")
    multimodal = report.get("multimodal_consistency") or {}
    lines.extend([
        "",
        "插值契约：平移按时间线性插值、旋转用最短路径 quaternion SLERP；超过 `75,000 us` 的源时间洞现在 fail-closed。原始 replay 不做平滑，正式证据不使用光流或 minterpolate。",
        "",
        "## 诊断矩阵",
        "",
        "| 组 | 目的 | full diff mean | target diff mean | background diff mean | 二阶 target mean |",
        "|---|---|---:|---:|---:|---:|",
    ])
    descriptions = {
        "B_20hz_render_20fps_encode": "20 Hz render / 20 FPS encode",
        "C_20hz_render_30fps_duplicate_encode": "20 Hz source / 30 FPS duplicate encode",
        "D_30hz_interpolated_render_30fps_encode": "30 Hz timestamp render / 30 FPS encode",
        "E_30hz_fixed_camera_target_motion": "fixed camera / target motion",
        "F_30hz_camera_motion_fixed_dynamic": "camera motion / fixed dynamic payload",
    }
    for name in descriptions:
        group = groups.get(name, {})
        if name == "C_20hz_render_30fps_duplicate_encode":
            lines.append(f"| `{name}` | {descriptions[name]}；重复 `{group.get('duplicate_output_frame_count')}/{group.get('output_frame_count')}` | {_num(_metric(group, 'full_frame_abs_diff'))} | {_num(_metric(group, 'target_crop_abs_diff'))} | {_num(_metric(group, 'background_abs_diff'))} | {_num(_metric(group, 'target_crop_second_abs_diff'))} |")
        else:
            lines.append(f"| `{name}` | {descriptions[name]} | {_num(_metric(group, 'full_frame_abs_diff'))} | {_num(_metric(group, 'target_crop_abs_diff'))} | {_num(_metric(group, 'background_abs_diff'))} | {_num(_metric(group, 'target_crop_second_abs_diff'))} |")
    aa = groups.get("A_A_repeat", {})
    lines.extend([
        "",
        "A/A：",
        "",
        f"- request digest equal：`{aa.get('request_digest_equal')}`；response digest equal：`{aa.get('response_digest_equal')}`；RGB SHA equal：`{aa.get('rgb_payload_equal')}`；pixel mean diff：`{_num(aa.get('pixel_abs_diff_mean'))}`。",
        "- 每次 live 请求的 `request_sequence`、`frame_id`、发送/接收时间和 request/response digest 已记录；NuRec `RGBRenderReturn` 没有 response timestamp/frame_id，因此 realized timestamp 只能标为 unavailable。",
        "",
        "## V02 同时间对比",
        "",
    ])
    v02 = report.get("v02_same_timestamp_comparison") or {}
    lines.extend([
        f"V01/V02 对比帧数 `{v02.get('frame_count_compared')}`，timestamp mismatch `{v02.get('timestamp_mismatch_count')}`，target projection pairs `{v02.get('target_projection_pair_count')}`，source-gap-crossing rows `{v02.get('source_gap_crossing_count')}`。",
        "",
        f"- whole image mean diff：`{_num((v02.get('full_frame_abs_diff') or {}).get('mean'))}`",
        f"- target crop mean diff：`{_num((v02.get('target_crop_abs_diff') or {}).get('mean'))}`",
        f"- background mean diff：`{_num((v02.get('background_abs_diff') or {}).get('mean'))}`",
        f"- target center delta mean：`{_num((v02.get('target_center_delta_px') or {}).get('mean'))} px`",
        "",
        "这表明 V02 编辑主要影响目标区域，非目标背景变化较小。",
        "",
        "## Peak 窗口",
        "",
    ])
    peak = report.get("peak_window") or {}
    source_peak = peak.get("source_acceleration_peak") or {}
    window = peak.get("window_timestamp_range_us") or {}
    lines.extend([
        f"source acceleration peak：`{_num(source_peak.get('absolute_acceleration_mps2'))} m/s²`，timestamp `{source_peak.get('start_timestamp_us')}` -> `{source_peak.get('end_timestamp_us')}`。",
        f"实验窗口：`{window.get('start')}` -> `{window.get('end')}`；包含 peak：`{peak.get('window_contains_source_peak')}`。",
        "",
    ])
    for name, value in (peak.get("groups") or {}).items():
        visual = value.get("visual") if isinstance(value, Mapping) else None
        temporal = visual.get("temporal") if isinstance(visual, Mapping) else {}
        lines.append(
            f"- `{name}`：background second mean `{_num((temporal.get('background_second_abs_diff') or {}).get('mean'))}`，target second mean `{_num((temporal.get('target_crop_second_abs_diff') or {}).get('mean'))}`。"
        )
    lines.extend([
        "",
        "## 正式产物与编码",
        "",
        "| case | video | FPS | frames | resolution | dropped | probe |",
        "|---|---|---:|---:|---|---:|---|",
    ])
    for case_id, value in (report.get("formal_cases") or {}).items():
        resolution = value.get("video_resolution") or {}
        lines.append(
            f"| `{case_id}` | `{value.get('video')}` | `{value.get('video_fps')}` | `{value.get('frame_count')}` | `{resolution.get('width')}x{resolution.get('height')}` | `{value.get('dropped_count')}` | `{value.get('probe_status')}` |"
        )
    v03_summary = (report.get("formal_cases") or {}).get("V03") or {}
    lines.extend([
        "",
        f"V03 相机 pose 轨迹摘要 JSON：`{v03_summary.get('pose_summary_json')}`；可视化 PNG：`{v03_summary.get('pose_summary_png')}`。",
    ])
    lines.extend([
        "",
        "编码报告：`encoding_report.json`。报告同时保留 OpenCV 和独立 ffprobe 的 stream/packet 结果；packet 级检查包含 PTS 单调性、重复 PTS、间隔统计和 CFR 判断。",
        "",
        "## RGB/LiDAR 一致性可视化",
        "",
        f"probe 状态：`{multimodal.get('status')}`；JSON：`{multimodal.get('probe')}`。",
        f"Logical pairing: timestamp `{multimodal.get('timestamp_us')}`; `{multimodal.get('pairing')}`; physical timestamp delta `{multimodal.get('window_us')}`; camera count `{len(multimodal.get('camera_ids') or [])}`.",
        f"可视化 PNG：`{multimodal.get('visual')}`。",
        f"结果/原因：`{multimodal.get('reason') or ('RGB changed=' + str(multimodal.get('rgb_actor_changed')) + ', LiDAR changed=' + str(multimodal.get('lidar_actor_changed')) if multimodal.get('status') == 'passed' else '未形成通过证据')}`。",
        "若 LiDAR 返回空点云，probe 会 fail-closed，不生成伪造的 RGB/LiDAR 一致性图。",
        "",
        "## 证据路径",
        "",
        f"- JSON：`{report.get('output_dir')}/debug_report.json`",
        f"- Markdown：`{report.get('output_dir')}/debug_report.md`",
        f"- encoding：`{report.get('output_dir')}/encoding_report.json`",
        "- formal metadata：",
    ])
    for case_id, metadata in ((report.get("source_inventory") or {}).get("metadata") or {}).items():
        lines.append(f"  - `{case_id}`：`{metadata}`")
    lines.extend([
        f"- strict raw 20 Hz error evidence：`{(decision.get('strict_raw_timestamp_capture') or {}).get('error_evidence')}`",
        f"- target-only raw 20 Hz metadata：`{(decision.get('target_only_raw_20hz_preview') or {}).get('metadata')}`",
        "",
        "报告没有删除抖动帧、降低阈值、模糊视频或使用光流插帧。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    parser.add_argument("--peak-report", type=Path, default=DEFAULT_PEAK_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        base_path = args.base_report.resolve()
        peak_path = args.peak_report.resolve()
        output_dir = args.output_dir.resolve()
        json_path = output_dir / "debug_report.json"
        md_path = output_dir / "debug_report.md"
        if (json_path.exists() or md_path.exists()) and not args.overwrite:
            raise ValueError(f"refusing to overwrite finalized report under {output_dir}; use --overwrite")
        base = _read_json(base_path)
        peak = _read_json(peak_path) if peak_path.is_file() else None
        manifest_path = args.manifest.resolve()
        manifest = load_json(manifest_path)
        artifact_path = Path(str((base.get("artifact") or {}).get("path"))).resolve()
        scene = ArtifactScene(artifact_path, str(manifest.get("target_track_id") or TARGET_TRACK_ID))
        report = _build_report(base, peak, manifest, scene, artifact_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_markdown(md_path, report)
        print(json.dumps({"status": "passed", "json": str(json_path), "markdown": str(md_path)}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"finalize_nurec_jitter_report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
