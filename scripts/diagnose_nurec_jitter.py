#!/usr/bin/env python3
"""Diagnose temporal jitter in the scene-0061 NuRec RGB path.

The diagnostic deliberately separates source cadence, timestamp interpolation,
RPC pairing, and MP4 encoding.  It writes short raw-camera frame sequences and
JSON metadata for live controls (A/A, E, F), plus offline analyses of the
existing 20/20 and 30/30 captures and a 20 Hz -> 30 FPS duplicate-frame encode.
"""

from __future__ import annotations

import argparse
import bisect
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_counterfactual_video import (
    ArtifactScene,
    RenderError,
    SensorsimClient,
    _camera_pose_pair,
    _dynamic_digest,
    _image_metrics,
    _make_output_dir,
    _matrix_from_pose,
    _pose_from_matrix,
    _slerp,
    _timestamp_values,
    _resolve_repo_path,
    canonical_digest,
    load_json,
    resolve_case,
    sha256_bytes,
    sha256_file,
    validate_manifest_identity,
    _encode_mp4,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "demo" / "scene0061" / "manifest.json"
DEFAULT_ARTIFACT = REPO_ROOT / (
    "outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/"
    "9aChcizbAsm4oDQKJMdBHM/artifacts/last.usdz"
)
DEFAULT_PYTHON_API_PATH = Path(
    "/home/cwadmin/sim-env/data/CARLA_0.9.16/PythonAPI/examples/nvidia/nurec"
)
TARGET_TRACK_ID = "c1958768d48640948f6053d04cffd35b"
CAMERA_ID = "camera_front"
WIDTH = 800
HEIGHT = 450
TARGET_CROP_WIDTH = 180
TARGET_CROP_HEIGHT = 128


class DiagnosticError(RuntimeError):
    """Raised when a diagnostic cannot produce trustworthy evidence."""


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _stats(values: Iterable[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if _finite(value)]
    if not clean:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(clean)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "count": len(clean),
        "min": ordered[0],
        "median": float(statistics.median(ordered)),
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "mean": float(statistics.mean(clean)),
    }


def _timestamp_stats(timestamps: list[int]) -> dict[str, Any]:
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    return {
        "count": len(timestamps),
        "strictly_increasing": all(delta > 0 for delta in deltas),
        "duplicate_count": sum(delta == 0 for delta in deltas),
        "delta_us": _stats(deltas),
        "effective_hz": (1_000_000.0 / statistics.mean(deltas)) if deltas else None,
        "gaps_gt_75ms": [
            {"index": index, "start_us": timestamps[index], "end_us": timestamps[index + 1], "delta_us": delta}
            for index, delta in enumerate(deltas)
            if delta > 75_000
        ],
    }


def _quat_stats(poses: list[list[float]]) -> dict[str, Any]:
    norms = [math.sqrt(sum(float(value) ** 2 for value in pose[3:7])) for pose in poses]
    dots: list[float] = []
    angles: list[float] = []
    for first, second in zip(poses, poses[1:]):
        first_norm = math.sqrt(sum(float(value) ** 2 for value in first[3:7])) or 1.0
        second_norm = math.sqrt(sum(float(value) ** 2 for value in second[3:7])) or 1.0
        dot = sum(float(a) * float(b) for a, b in zip(first[3:7], second[3:7])) / (first_norm * second_norm)
        dots.append(dot)
        angles.append(2.0 * math.acos(min(1.0, max(-1.0, abs(dot)))))
    return {
        "norm": _stats(norms),
        "norm_error_count": sum(abs(norm - 1.0) > 1e-4 for norm in norms),
        "adjacent_dot": _stats(dots),
        "negative_dot_count": sum(dot < 0.0 for dot in dots),
        "shortest_path_angle_deg": _stats(math.degrees(angle) for angle in angles),
    }


def _pose_motion_stats(timestamps: list[int], poses: list[list[float]]) -> dict[str, Any]:
    speeds: list[float] = []
    speed_deltas: list[float] = []
    accelerations: list[float] = []
    speed_dt_seconds: list[float] = []
    for index, (first, second) in enumerate(zip(poses, poses[1:])):
        dt = (timestamps[index + 1] - timestamps[index]) / 1_000_000.0
        if dt > 0.0:
            speeds.append(math.dist(first[:3], second[:3]) / dt)
    for index, (first, second) in enumerate(zip(speeds, speeds[1:])):
        dt = (timestamps[index + 2] - timestamps[index]) / 2_000_000.0
        if dt > 0.0:
            speed_deltas.append(second - first)
            speed_dt_seconds.append(dt)
            accelerations.append((second - first) / dt)
    return {
        "speed_mps": _stats(speeds),
        "speed_delta_mps": _stats(speed_deltas),
        "speed_second_difference_mps": _stats(accelerations),
    }


def _trajectory_summary(scene: ArtifactScene, target_track_id: str) -> dict[str, Any]:
    target = scene.tracks[target_track_id]
    target_timestamps = [int(value) for value in target["timestamps_us"]]
    target_poses = [[float(value) for value in pose] for pose in target["poses"]]
    track_rows: list[dict[str, Any]] = []
    for track_id, track in scene.tracks.items():
        timestamps = [int(value) for value in track["timestamps_us"]]
        poses = [[float(value) for value in pose] for pose in track["poses"]]
        track_rows.append(
            {
                "track_id": track_id,
                "flag": scene.track_flags[track_id],
                "label": scene.track_labels[track_id],
                "sample_count": len(timestamps),
                "timestamp": _timestamp_stats(timestamps),
                "pose_shape": sorted({len(pose) for pose in poses}),
                "quaternion": _quat_stats(poses),
            }
        )
    track_rows.sort(key=lambda row: row["track_id"])
    return {
        "sequence_id": scene.trajectory.get("sequence_id"),
        "rig": _timestamp_stats(scene.rig_timestamps),
        "target_track_id": target_track_id,
        "target": {
            "flag": scene.track_flags[target_track_id],
            "label": scene.track_labels[target_track_id],
            "timestamp": _timestamp_stats(target_timestamps),
            "quaternion": _quat_stats(target_poses),
            "motion": _pose_motion_stats(target_timestamps, target_poses),
        },
        "track_count": len(track_rows),
        "controllable_track_count": len(scene.controllable_track_ids),
        "controllable_track_ids_digest": canonical_digest(scene.controllable_track_ids),
        "track_rows": track_rows,
    }


def _inverse_rigid(matrix: list[list[float]]) -> list[list[float]]:
    rotation = [row[:3] for row in matrix[:3]]
    translation = [matrix[index][3] for index in range(3)]
    inverse = [[rotation[column][row] for column in range(3)] + [0.0] for row in range(3)]
    for row in range(3):
        inverse[row][3] = -sum(inverse[row][column] * translation[column] for column in range(3))
    return inverse + [[0.0, 0.0, 0.0, 1.0]]


def _transform_point(matrix: list[list[float]], point: list[float]) -> list[float]:
    return [
        sum(matrix[row][column] * point[column] for column in range(3)) + matrix[row][3]
        for row in range(3)
    ]


def _project_target(
    scene: ArtifactScene,
    camera_id: str,
    camera_pose: Mapping[str, Any],
    target_pose: list[float],
    width: int,
    height: int,
) -> tuple[float, float] | None:
    matrix = _matrix_from_pose(
        [camera_pose["position_m"][axis] for axis in ("x", "y", "z")],
        [camera_pose["orientation_xyzw"][axis] for axis in ("x", "y", "z", "w")],
    )
    camera_point = _transform_point(_inverse_rigid(matrix), [float(value) for value in target_pose[:3]])
    if camera_point[2] <= 0.1:
        return None
    intrinsics = scene.camera_intrinsics(camera_id)
    fx, fy = (float(value) for value in intrinsics["focal_length"][:2])
    cx, cy = (float(value) for value in intrinsics["principal_point"][:2])
    native_width, native_height = (float(value) for value in intrinsics["resolution"][:2])
    x = (fx * camera_point[0] / camera_point[2] + cx) * width / native_width
    y = (fy * camera_point[1] / camera_point[2] + cy) * height / native_height
    if x < -width or x > width * 2 or y < -height or y > height * 2:
        return None
    return x, y


def _crop(image: np.ndarray, center: tuple[float, float] | None) -> np.ndarray | None:
    if center is None:
        return None
    x, y = center
    half_width = TARGET_CROP_WIDTH // 2
    half_height = TARGET_CROP_HEIGHT // 2
    left = max(0, min(image.shape[1] - TARGET_CROP_WIDTH, int(round(x)) - half_width))
    top = max(0, min(image.shape[0] - TARGET_CROP_HEIGHT, int(round(y)) - half_height))
    return image[top : top + TARGET_CROP_HEIGHT, left : left + TARGET_CROP_WIDTH]


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DiagnosticError(f"cannot decode diagnostic frame: {path}")
    return image


def _frame_paths(root: Path) -> list[Path]:
    camera_dir = root / "camera_frames" / CAMERA_ID
    if camera_dir.is_dir():
        paths = sorted(camera_dir.glob("*.jpg"))
    else:
        paths = sorted((root / "frames").glob("*.jpg"))
    if not paths:
        raise DiagnosticError(f"no JPEG frames found under {root}")
    return paths


def _metadata_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "metadata.jsonl"
    if not path.is_file():
        path = root / "frames.jsonl"
    if not path.is_file():
        raise DiagnosticError(f"metadata JSONL is unavailable under {root}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise DiagnosticError(f"metadata rows are not objects: {path}")
    return rows


def _row_timestamp(row: Mapping[str, Any]) -> int:
    for key in ("scene_timestamp_us", "requested_timestamp_us", "output_timestamp_us", "source_timestamp_us"):
        if _finite(row.get(key)):
            return int(row[key])
    raise DiagnosticError("metadata row has no timestamp")


def _row_camera_pose(row: Mapping[str, Any], camera_id: str) -> Mapping[str, Any] | None:
    requested = row.get("requested_sensor_poses")
    if isinstance(requested, Mapping):
        camera = requested.get(camera_id)
        if isinstance(camera, Mapping) and isinstance(camera.get("start"), Mapping):
            return camera["start"]
    pose = row.get("camera_pose_start")
    return pose if isinstance(pose, Mapping) else None


def _row_target_pose(row: Mapping[str, Any]) -> list[float] | None:
    pose = row.get("target_pose_start")
    if isinstance(pose, list) and len(pose) == 7:
        return [float(value) for value in pose]
    return None


def _analysis_target_pose(
    scene: ArtifactScene, timestamp_us: int
) -> tuple[list[float] | None, bool]:
    """Reconstruct only the target projection pose for legacy video analysis.

    This is deliberately separate from the renderer's fail-closed interpolation:
    old captures may lack target pose fields in metadata, so analysis records a
    gap-crossing flag instead of allowing an unrelated actor to abort the report.
    """

    track = scene.tracks[TARGET_TRACK_ID]
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
    position = [
        first[index] + (second[index] - first[index]) * fraction
        for index in range(3)
    ]
    quaternion = _slerp(first[3:7], second[3:7], fraction)
    return position + quaternion, span > 75_000


def _image_diff(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.abs(first.astype(np.float32) - second.astype(np.float32)).mean())


def _visual_metrics(
    root: Path,
    scene: ArtifactScene,
    case: Mapping[str, Any],
    *,
    metadata_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = metadata_rows or _metadata_rows(root)
    paths = _frame_paths(root)
    if len(paths) != len(rows):
        raise DiagnosticError(f"frame/metadata count differs under {root}: {len(paths)}/{len(rows)}")
    images: list[np.ndarray] = []
    centers: list[tuple[float, float] | None] = []
    projection_gap_crossings = 0
    for row, path in zip(rows, paths):
        image = _read_image(path)
        images.append(image)
        timestamp = _row_timestamp(row)
        camera_pose = _row_camera_pose(row, CAMERA_ID)
        target_pose = _row_target_pose(row)
        if camera_pose is None:
            camera_matrix = scene.sensor_pose_matrix(CAMERA_ID, timestamp)
            camera_pose = _pose_from_matrix(camera_matrix)
        if target_pose is None:
            target_pose, crossed_gap = _analysis_target_pose(scene, timestamp)
            projection_gap_crossings += int(crossed_gap)
            edit = case.get("lead_vehicle_edit")
            delta = edit.get("translation_m") if isinstance(edit, Mapping) else None
            if target_pose is not None and isinstance(delta, Mapping):
                target_pose = list(target_pose)
                target_pose[0] += float(delta.get("x", 0.0))
                target_pose[1] += float(delta.get("y", 0.0))
                target_pose[2] += float(delta.get("z", 0.0))
        centers.append(
            _project_target(scene, CAMERA_ID, camera_pose, target_pose, image.shape[1], image.shape[0])
            if target_pose is not None
            else None
        )

    full_diffs: list[float] = []
    target_diffs: list[float] = []
    background_diffs: list[float] = []
    full_second_diffs: list[float] = []
    target_second_diffs: list[float] = []
    background_second_diffs: list[float] = []
    valid_target_pairs = 0
    for first, second, first_center, second_center in zip(images, images[1:], centers, centers[1:]):
        full_diffs.append(_image_diff(first, second))
        first_crop = _crop(first, first_center)
        second_crop = _crop(second, second_center or first_center)
        if first_crop is not None and second_crop is not None and first_crop.shape == second_crop.shape:
            target_diffs.append(_image_diff(first_crop, second_crop))
            valid_target_pairs += 1
        mask = np.ones(first.shape[:2], dtype=bool)
        if first_center is not None:
            x, y = (int(round(value)) for value in first_center)
            left = max(0, x - TARGET_CROP_WIDTH // 2)
            right = min(first.shape[1], left + TARGET_CROP_WIDTH)
            top = max(0, y - TARGET_CROP_HEIGHT // 2)
            bottom = min(first.shape[0], top + TARGET_CROP_HEIGHT)
            mask[top:bottom, left:right] = False
        diff = np.abs(first.astype(np.float32) - second.astype(np.float32)).mean(axis=2)
        background_diffs.append(float(diff[mask].mean()) if mask.any() else float(diff.mean()))

    for first, second, third, first_center, second_center, third_center in zip(
        images,
        images[1:],
        images[2:],
        centers,
        centers[1:],
        centers[2:],
    ):
        full_second_diffs.append(
            float(
                np.abs(
                    third.astype(np.float32)
                    - 2.0 * second.astype(np.float32)
                    + first.astype(np.float32)
                ).mean()
            )
        )
        first_crop = _crop(first, first_center)
        second_crop = _crop(second, second_center or first_center)
        third_crop = _crop(third, third_center or second_center or first_center)
        if first_crop is not None and second_crop is not None and third_crop is not None:
            target_second_diffs.append(
                float(
                    np.abs(
                        third_crop.astype(np.float32)
                        - 2.0 * second_crop.astype(np.float32)
                        + first_crop.astype(np.float32)
                    ).mean()
                )
            )
        mask = np.ones(first.shape[:2], dtype=bool)
        if first_center is not None:
            x, y = (int(round(value)) for value in first_center)
            left = max(0, x - TARGET_CROP_WIDTH // 2)
            right = min(first.shape[1], left + TARGET_CROP_WIDTH)
            top = max(0, y - TARGET_CROP_HEIGHT // 2)
            bottom = min(first.shape[0], top + TARGET_CROP_HEIGHT)
            mask[top:bottom, left:right] = False
        second_difference = np.abs(
            third.astype(np.float32)
            - 2.0 * second.astype(np.float32)
            + first.astype(np.float32)
        ).mean(axis=2)
        background_second_diffs.append(
            float(second_difference[mask].mean()) if mask.any() else float(second_difference.mean())
        )

    timestamp_values = [_row_timestamp(row) for row in rows]
    dt_seconds = [(b - a) / 1_000_000.0 for a, b in zip(timestamp_values, timestamp_values[1:])]
    center_velocity: list[float] = []
    center_acceleration: list[float] = []
    for index, (first, second) in enumerate(zip(centers, centers[1:])):
        if first is not None and second is not None and index < len(dt_seconds) and dt_seconds[index] > 0.0:
            center_velocity.append(math.dist(first, second) / dt_seconds[index])
    for first, second in zip(center_velocity, center_velocity[1:]):
        center_acceleration.append((second - first) / (statistics.mean(dt_seconds) or 1.0))
    return {
        "frame_count": len(images),
        "timestamp": _timestamp_stats(timestamp_values),
        "target_projection": {
            "available_frame_count": sum(center is not None for center in centers),
            "source_gap_crossing_count": projection_gap_crossings,
            "center_velocity_px_s": _stats(center_velocity),
            "center_acceleration_px_s2": _stats(center_acceleration),
        },
        "temporal": {
            "full_frame_abs_diff": _stats(full_diffs),
            "target_crop_abs_diff": _stats(target_diffs),
            "background_abs_diff": _stats(background_diffs),
            "full_frame_second_abs_diff": _stats(full_second_diffs),
            "target_crop_second_abs_diff": _stats(target_second_diffs),
            "background_second_abs_diff": _stats(background_second_diffs),
            "target_crop_pair_count": valid_target_pairs,
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_metadata(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _strip_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"rgb_bytes", "point_xyzs", "point_intensities"}}


def _dynamic_ids_digest(dynamic_objects: Iterable[Mapping[str, Any]]) -> str:
    return canonical_digest(sorted(str(item.get("track_id")) for item in dynamic_objects))


def _capture_frame(
    client: SensorsimClient,
    *,
    output_dir: Path,
    frame_index: int,
    frame_id: str,
    timestamp_us: int,
    end_timestamp_us: int,
    camera_pose_start: Mapping[str, Any],
    camera_pose_end: Mapping[str, Any],
    dynamic_objects: list[Mapping[str, Any]],
    width: int = WIDTH,
    height: int = HEIGHT,
) -> dict[str, Any]:
    result = client.render_rgb(
        camera_id=CAMERA_ID,
        width=width,
        height=height,
        start_us=timestamp_us,
        end_us=end_timestamp_us,
        start_pose=camera_pose_start,
        end_pose=camera_pose_end,
        dynamic_objects=dynamic_objects,
        frame_id=frame_id,
    )
    record: dict[str, Any] = {
        "frame_id": frame_id,
        "frame_index": frame_index,
        "camera_id": CAMERA_ID,
        "requested_timestamp_us": timestamp_us,
        "requested_logical_frame_end_us": end_timestamp_us,
        "requested_wire_timestamp_us": result.get("wire_frame_start_us"),
        "realized_timestamp_us": result.get("realized_timestamp_us"),
        "realized_timestamp_status": result.get("realized_timestamp_status"),
        "response_frame_id": result.get("response_frame_id"),
        "response_timestamp_us": result.get("response_timestamp_us"),
        "request_sequence": result.get("request_sequence"),
        "request_sent_unix_ns": result.get("request_sent_unix_ns"),
        "response_received_unix_ns": result.get("response_received_unix_ns"),
        "request_digest": result.get("request_digest"),
        "response_digest": result.get("response_digest"),
        "rgb_payload_sha256": result.get("rgb_payload_sha256"),
        "rpc_latency_ms": result.get("rpc_latency_ms"),
        "status": result.get("status"),
        "error": result.get("error"),
        "camera_pose_start": dict(camera_pose_start),
        "camera_pose_end": dict(camera_pose_end),
        "dynamic_object_digest": _dynamic_digest(dynamic_objects),
        "dynamic_object_ids_digest": _dynamic_ids_digest(dynamic_objects),
        "dynamic_object_count": len(dynamic_objects),
        "dynamic_object_track_ids": sorted(str(item.get("track_id")) for item in dynamic_objects),
    }
    target = next(
        (item for item in dynamic_objects if item.get("track_id") == TARGET_TRACK_ID),
        None,
    )
    if isinstance(target, Mapping) and isinstance(target.get("pose"), list):
        record["target_pose_start"] = [float(value) for value in target["pose"]]
        pose_pair = target.get("pose_pair")
        if isinstance(pose_pair, Mapping) and isinstance(pose_pair.get("end"), list):
            record["target_pose_end"] = [float(value) for value in pose_pair["end"]]
    if result.get("status") == "passed":
        frame_path = output_dir / "frames" / f"{frame_index:06d}.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(result["rgb_bytes"])
        record["frame_path"] = str(frame_path.relative_to(output_dir))
    return record


def _capture_sequence(
    client: SensorsimClient,
    scene: ArtifactScene,
    case: Mapping[str, Any],
    output_dir: Path,
    group: str,
    timestamps: list[int],
    *,
    fixed_camera_pose: Mapping[str, Any] | None = None,
    fixed_dynamic_objects: list[Mapping[str, Any]] | None = None,
    target_delta: Mapping[str, float] | None = None,
    video_fps: float | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for frame_index, timestamp_us in enumerate(timestamps):
        next_timestamp = timestamps[frame_index + 1] if frame_index + 1 < len(timestamps) else timestamp_us + 1
        if fixed_camera_pose is None:
            camera_start, camera_end = _camera_pose_pair(
                scene, case, CAMERA_ID, timestamp_us, next_timestamp, frame_index / max(1, len(timestamps) - 1)
            )
        else:
            camera_start = dict(fixed_camera_pose)
            camera_end = dict(fixed_camera_pose)
        dynamic = fixed_dynamic_objects
        if dynamic is None:
            dynamic = scene.dynamic_objects(
                timestamp_us,
                end_timestamp_us=next_timestamp,
                mode=str((case.get("dynamic_objects") or {}).get("mode", "controllable")),
                target_track_id=TARGET_TRACK_ID if target_delta else None,
                target_delta=target_delta,
            )
        rows.append(
            _capture_frame(
                client,
                output_dir=output_dir,
                frame_index=frame_index,
                frame_id=f"{group}:{frame_index}",
                timestamp_us=timestamp_us,
                end_timestamp_us=next_timestamp,
                camera_pose_start=camera_start,
                camera_pose_end=camera_end,
                dynamic_objects=dynamic,
            )
        )
    _write_metadata(output_dir / "metadata.jsonl", rows)
    passed = [row for row in rows if row.get("status") == "passed"]
    summary = {
        "group": group,
        "camera_id": CAMERA_ID,
        "requested_count": len(rows),
        "captured_count": len(passed),
        "dropped_count": len(rows) - len(passed),
        "request_sequence_unique": len({row.get("request_sequence") for row in rows}) == len(rows),
        "request_frame_id_unique": len({row.get("frame_id") for row in rows}) == len(rows),
        "response_frame_id_available": any(row.get("response_frame_id") is not None for row in rows),
        "response_timestamp_available": any(row.get("response_timestamp_us") is not None for row in rows),
        "dynamic_digest_unique": len({row.get("dynamic_object_digest") for row in rows}) == len(rows),
        "metadata": str(output_dir / "metadata.jsonl"),
    }
    if video_fps is not None and len(passed) == len(rows) and rows:
        video_path = output_dir / f"{group}.mp4"
        _encode_mp4([output_dir / row["frame_path"] for row in rows], video_path, video_fps)
        summary["video_fps"] = video_fps
        summary["video"] = str(video_path)
    _write_json(output_dir / "capture_summary.json", summary)
    return summary


def _capture_aa(
    client: SensorsimClient,
    scene: ArtifactScene,
    output_dir: Path,
    timestamp_us: int,
) -> dict[str, Any]:
    pose = _pose_from_matrix(scene.sensor_pose_matrix(CAMERA_ID, timestamp_us))
    dynamic = scene.dynamic_objects(timestamp_us, end_timestamp_us=timestamp_us + 1, mode="controllable")
    first = _capture_frame(
        client,
        output_dir=output_dir / "first",
        frame_index=0,
        frame_id="A:first",
        timestamp_us=timestamp_us,
        end_timestamp_us=timestamp_us + 1,
        camera_pose_start=pose,
        camera_pose_end=pose,
        dynamic_objects=dynamic,
    )
    second = _capture_frame(
        client,
        output_dir=output_dir / "repeat",
        frame_index=0,
        frame_id="A:repeat",
        timestamp_us=timestamp_us,
        end_timestamp_us=timestamp_us + 1,
        camera_pose_start=pose,
        camera_pose_end=pose,
        dynamic_objects=dynamic,
    )
    first_image = _read_image(output_dir / "first" / first["frame_path"])
    second_image = _read_image(output_dir / "repeat" / second["frame_path"])
    diff = _image_diff(first_image, second_image)
    rows = [first, second]
    _write_metadata(output_dir / "metadata.jsonl", rows)
    summary = {
        "group": "A_A_repeat",
        "timestamp_us": timestamp_us,
        "request_digest_equal": first.get("request_digest") == second.get("request_digest"),
        "dynamic_digest_equal": first.get("dynamic_object_digest") == second.get("dynamic_object_digest"),
        "rgb_payload_equal": first.get("rgb_payload_sha256") == second.get("rgb_payload_sha256"),
        "response_digest_equal": first.get("response_digest") == second.get("response_digest"),
        "pixel_abs_diff_mean": diff,
        "metadata": str(output_dir / "metadata.jsonl"),
        "rows": rows,
    }
    _write_json(output_dir / "capture_summary.json", summary)
    return summary


def _retime_20_to_30(source_root: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = _metadata_rows(source_root)
    source_paths = _frame_paths(source_root)
    source_timestamps = [_row_timestamp(row) for row in source_rows]
    start_us, end_us = source_timestamps[0], source_timestamps[-1]
    target_timestamps = [start_us]
    index = 1
    while start_us + round(index * 1_000_000 / 30.0) < end_us:
        target_timestamps.append(start_us + round(index * 1_000_000 / 30.0))
        index += 1
    if target_timestamps[-1] != end_us:
        target_timestamps.append(end_us)
    output_frames = output_dir / "frames"
    output_frames.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_indices: list[int] = []
    for frame_index, timestamp_us in enumerate(target_timestamps):
        source_index = min(range(len(source_timestamps)), key=lambda item: abs(source_timestamps[item] - timestamp_us))
        source_indices.append(source_index)
        target_path = output_frames / f"{frame_index:06d}.jpg"
        shutil.copy2(source_paths[source_index], target_path)
        rows.append(
            {
                "frame_id": f"C:{frame_index}",
                "frame_index": frame_index,
                "output_timestamp_us": timestamp_us,
                "source_frame_index": source_index,
                "source_timestamp_us": source_timestamps[source_index],
                "duplicate_from_previous": frame_index > 0 and source_index == source_indices[-2],
                "frame_path": str(target_path.relative_to(output_dir)),
            }
        )
    _write_metadata(output_dir / "metadata.jsonl", rows)
    video_path = output_dir / "20hz_render_30fps_duplicate_encode.mp4"
    _encode_mp4([output_dir / row["frame_path"] for row in rows], video_path, 30.0)
    summary = {
        "group": "C_20hz_render_30fps_duplicate_encode",
        "source_root": str(source_root),
        "source_frame_count": len(source_paths),
        "output_frame_count": len(rows),
        "unique_source_frames_used": len(set(source_indices)),
        "duplicate_output_frame_count": sum(row["duplicate_from_previous"] for row in rows),
        "output_fps": 30.0,
        "output_video": str(video_path),
        "metadata": str(output_dir / "metadata.jsonl"),
    }
    _write_json(output_dir / "capture_summary.json", summary)
    return summary


def _existing_capture_summary(
    scene: ArtifactScene,
    case: Mapping[str, Any],
    root: Path,
    group: str,
    *,
    video_fps: float | None = None,
) -> dict[str, Any]:
    metrics = _visual_metrics(root, scene, case)
    rows = _metadata_rows(root)
    summary = {
        "group": group,
        "root": str(root),
        "video_fps": video_fps,
        "metadata": str(root / "frames.jsonl"),
        "visual": metrics,
        "request_metadata_quality": {
            "frame_id_available": all(row.get("frame_id") is not None for row in rows),
            "request_sequence_available": all(row.get("request_sequence") is not None for row in rows),
            "response_timestamp_available": any(row.get("response_timestamp_us") is not None for row in rows),
            "legacy_metadata_rows": sum(row.get("request_sequence") is None for row in rows),
        },
    }
    _write_json(root.parent / f"{group}_analysis.json", summary)
    return summary


def _video_probe(path: Path, ffprobe_path: Path | str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    capture.release()
    result["opencv"] = {
        "opened": bool(fps > 0.0 and frame_count > 0),
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": frame_count / fps if fps > 0.0 else None,
        "width": width,
        "height": height,
        "fourcc": fourcc,
    }
    configured_ffprobe = str(ffprobe_path) if ffprobe_path else os.environ.get("NSB_FFPROBE_PATH")
    ffprobe = configured_ffprobe or shutil.which("ffprobe")
    if ffprobe:
        stream_completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,time_base",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        packet_completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "packet=pts_time,dts_time,duration_time,flags",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        packet_analysis: dict[str, Any] = {
            "available": packet_completed.returncode == 0,
            "returncode": packet_completed.returncode,
            "packet_count": 0,
            "duplicate_pts_count": 0,
            "non_monotonic_pts_count": 0,
            "pts_interval_s": _stats([]),
            "cfr": False,
        }
        if packet_completed.returncode == 0:
            try:
                packet_payload = json.loads(packet_completed.stdout or "{}")
                packets = packet_payload.get("packets") if isinstance(packet_payload, Mapping) else []
                pts_values = [
                    float(packet.get("pts_time"))
                    for packet in packets
                    if isinstance(packet, Mapping) and packet.get("pts_time") is not None
                ]
                intervals = [right - left for left, right in zip(pts_values, pts_values[1:])]
                duplicate_pts = sum(abs(delta) <= 1e-9 for delta in intervals)
                non_monotonic = sum(delta < -1e-9 for delta in intervals)
                finite_intervals = [delta for delta in intervals if math.isfinite(delta) and delta > 0.0]
                packet_analysis.update(
                    {
                        "packet_count": len(packets),
                        "pts_count": len(pts_values),
                        "duplicate_pts_count": duplicate_pts,
                        "non_monotonic_pts_count": non_monotonic,
                        "pts_interval_s": _stats(finite_intervals),
                        "first_pts_time": pts_values[0] if pts_values else None,
                        "last_pts_time": pts_values[-1] if pts_values else None,
                    }
                )
                if finite_intervals:
                    median_interval = statistics.median(finite_intervals)
                    tolerance = max(1e-6, abs(median_interval) * 1e-3)
                    packet_analysis["cfr"] = (
                        non_monotonic == 0
                        and duplicate_pts == 0
                        and all(abs(delta - median_interval) <= tolerance for delta in finite_intervals)
                    )
            except (TypeError, ValueError, json.JSONDecodeError):
                packet_analysis["parse_error"] = True
        result["ffprobe"] = {
            "available": True,
            "path": str(ffprobe),
            "returncode": stream_completed.returncode,
            "stdout": stream_completed.stdout,
            "stderr": stream_completed.stderr,
            "packet_analysis": packet_analysis,
        }
    else:
        result["ffprobe"] = {
            "available": False,
            "reason": "ffprobe executable is unavailable on this host",
        }
    gst = shutil.which("gst-discoverer-1.0")
    if gst:
        completed = subprocess.run(
            [gst, str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        result["gst_discoverer"] = {
            "available": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--server-address", default="127.0.0.1:46443")
    parser.add_argument("--runtime-scene-id", default="scene-0061")
    parser.add_argument("--python-api-path", type=Path, default=DEFAULT_PYTHON_API_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-timestamp", type=int)
    parser.add_argument("--duration-us", type=int, default=2_000_000)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument(
        "--ffprobe-path",
        type=Path,
        help="explicit ffprobe executable for independent encoding evidence",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _resolve_repo_path(args.manifest)
    manifest = load_json(manifest_path)
    artifact, artifact_entry, checkpoint, checkpoint_entry = validate_manifest_identity(manifest, args.artifact)
    required_track = str(manifest.get("target_track_id") or TARGET_TRACK_ID)
    if required_track != TARGET_TRACK_ID:
        raise DiagnosticError("manifest target_track_id does not match the canonical interview-demo target")
    scene = ArtifactScene(artifact, required_track)
    case_path, case = resolve_case("V01")
    output_dir = _resolve_repo_path(args.output_dir)
    _make_output_dir(output_dir, bool(args.overwrite))
    trajectory = _trajectory_summary(scene, required_track)
    groups: dict[str, Any] = {}
    existing_20 = REPO_ROOT / "outputs/nurec_scene0061_demo_20hz/V01"
    existing_30 = REPO_ROOT / "outputs/nurec_scene0061_demo/V01"
    if existing_20.is_dir():
        groups["B_20hz_render_20fps_encode"] = _existing_capture_summary(scene, case, existing_20, "B_20hz_render_20fps_encode", video_fps=20.0)
        c_root = output_dir / "C_20hz_render_30fps_duplicate_encode"
        c_capture = _retime_20_to_30(existing_20, c_root)
        c_capture["visual"] = _visual_metrics(c_root, scene, case)
        groups["C_20hz_render_30fps_duplicate_encode"] = c_capture
    if existing_30.is_dir():
        groups["D_30hz_interpolated_render_30fps_encode"] = _existing_capture_summary(scene, case, existing_30, "D_30hz_interpolated_render_30fps_encode", video_fps=30.0)

    client: SensorsimClient | None = None
    try:
        if not args.skip_live:
            client = SensorsimClient(args.server_address, args.runtime_scene_id, _resolve_repo_path(args.python_api_path), args.timeout_sec)
            runtime_inventory = client.inventory
            case_range = case.get("timestamp_range_us") or {}
            start_us = int(args.start_timestamp if args.start_timestamp is not None else case_range["start"])
            end_us = min(int(case_range["end"]), start_us + int(args.duration_us))
            short_case = dict(case)
            short_case["timestamp_range_us"] = {"start": start_us, "end": end_us}
            timestamps_20 = _timestamp_values(scene, short_case, None, None, None, 20.0)
            timestamps_30 = _timestamp_values(scene, short_case, None, None, None, 30.0)
            aa_dir = output_dir / "A_A_repeat"
            groups["A_A_repeat"] = _capture_aa(client, scene, aa_dir, start_us)
            aa_rows = _metadata_rows(aa_dir)
            groups["A_A_repeat"]["visual"] = _visual_metrics(
                aa_dir / "first",
                scene,
                case,
                metadata_rows=[aa_rows[0]],
            )
            b_dir = output_dir / "B_short_20hz_render_20fps_encode"
            groups["B_short_20hz_render_20fps_encode"] = _capture_sequence(
                client,
                scene,
                short_case,
                b_dir,
                "B20",
                timestamps_20,
                video_fps=20.0,
            )
            groups["B_short_20hz_render_20fps_encode"]["visual"] = _visual_metrics(b_dir, scene, case)
            d_dir = output_dir / "D_short_30hz_interpolated_render_30fps_encode"
            groups["D_short_30hz_interpolated_render_30fps_encode"] = _capture_sequence(
                client,
                scene,
                short_case,
                d_dir,
                "D30",
                timestamps_30,
                video_fps=30.0,
            )
            groups["D_short_30hz_interpolated_render_30fps_encode"]["visual"] = _visual_metrics(d_dir, scene, case)
            e_dir = output_dir / "E_30hz_fixed_camera_target_motion"
            fixed_pose = _pose_from_matrix(scene.sensor_pose_matrix(CAMERA_ID, start_us))
            groups["E_30hz_fixed_camera_target_motion"] = _capture_sequence(
                client,
                scene,
                short_case,
                e_dir,
                "E",
                timestamps_30,
                fixed_camera_pose=fixed_pose,
                target_delta={"x": 0.5, "y": 0.0, "z": 0.0},
                video_fps=30.0,
            )
            groups["E_30hz_fixed_camera_target_motion"]["visual"] = _visual_metrics(e_dir, scene, case)
            f_dir = output_dir / "F_30hz_camera_motion_fixed_dynamic"
            fixed_dynamic = scene.dynamic_objects(start_us, end_timestamp_us=start_us + 1, mode="controllable")
            groups["F_30hz_camera_motion_fixed_dynamic"] = _capture_sequence(
                client,
                scene,
                short_case,
                f_dir,
                "F",
                timestamps_30,
                fixed_dynamic_objects=fixed_dynamic,
                video_fps=30.0,
            )
            groups["F_30hz_camera_motion_fixed_dynamic"]["visual"] = _visual_metrics(f_dir, scene, case)
    finally:
        if client is not None:
            client.close()

    video_reports: dict[str, Any] = {}
    for group_name, group in groups.items():
        video_value = group.get("video") if isinstance(group, Mapping) else None
        if video_value:
            video_reports[group_name] = _video_probe(Path(str(video_value)), args.ffprobe_path)
    for group_name, video_path in {
        "B_20hz_render_20fps_encode": existing_20 / "original_replay.mp4",
        "D_30hz_interpolated_render_30fps_encode": existing_30 / "original_replay.mp4",
    }.items():
        video_reports.setdefault(group_name, _video_probe(video_path, args.ffprobe_path))
    _write_json(output_dir / "encoding_report.json", video_reports)

    report = {
        "schema_version": "nsb.nurec-jitter-diagnostic.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "case": {"path": str(case_path), "sha256": sha256_file(case_path), "case_id": case.get("case_id")},
        "artifact": {
            "path": str(artifact),
            "sha256": artifact_entry.get("sha256"),
            "size_bytes": artifact_entry.get("size_bytes"),
            "checkpoint": {
                "path": str(checkpoint) if checkpoint else None,
                "sha256": checkpoint_entry.get("sha256") if checkpoint_entry else None,
            },
        },
        "runtime": locals().get("runtime_inventory"),
        "trajectory": trajectory,
        "groups": groups,
        "encoding": video_reports,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "debug_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        run_diagnostics(args)
    except (DiagnosticError, RenderError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"diagnose_nurec_jitter: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
