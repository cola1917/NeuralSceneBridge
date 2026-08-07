#!/usr/bin/env python3
"""Build an audited 20 FPS RGB/NuRec-LiDAR alignment video for scene-0061."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_counterfactual_video import (
    ArtifactScene,
    DEFAULT_PYTHON_API_PATH,
    RenderError,
    SensorsimClient,
    _encode_mp4,
    _pose_from_matrix,
    load_json,
)


ARTIFACT = REPO_ROOT / (
    "outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/"
    "9aChcizbAsm4oDQKJMdBHM/artifacts/last.usdz"
)
TARGET_TRACK_ID = "c1958768d48640948f6053d04cffd35b"
SENSOR_TO_BEV_AXES = np.eye(3, dtype=np.float32)
RESPONSE_TO_ARTIFACT_AXES = np.asarray(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
TARGET_HALF_LENGTH_M = 3.0
TARGET_HALF_WIDTH_M = 1.5
TARGET_HALF_HEIGHT_M = 1.0
VOXEL_M = 0.10


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value.get("status") == "passed" and not value.get("dropped"):
                rows.append(value)
    return rows


def _valid_candidates(
    scene: ArtifactScene,
    baseline_rows: list[dict[str, Any]],
    edited_rows: list[dict[str, Any]],
    window_us: int,
    target_delta: dict[str, float],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    edited_by_timestamp = {int(row["scene_timestamp_us"]): row for row in edited_rows}
    valid = []
    rejected = []
    for baseline in baseline_rows:
        timestamp_us = int(baseline["scene_timestamp_us"])
        edited = edited_by_timestamp.get(timestamp_us)
        if edited is None:
            rejected.append({"timestamp_us": timestamp_us, "reason": "edited RGB missing"})
            continue
        try:
            scene.dynamic_objects(
                timestamp_us,
                end_timestamp_us=timestamp_us + window_us,
                mode="controllable",
            )
            scene.dynamic_objects(
                timestamp_us,
                end_timestamp_us=timestamp_us + window_us,
                mode="controllable",
                target_track_id=TARGET_TRACK_ID,
                target_delta=target_delta,
            )
        except RenderError as exc:
            rejected.append({"timestamp_us": timestamp_us, "reason": str(exc)})
            continue
        valid.append((baseline, edited))
    return valid, rejected


def _select_uniform(
    candidates: list[tuple[dict[str, Any], dict[str, Any]]], count: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(candidates) < count:
        raise RenderError(
            f"only {len(candidates)} source-supported timestamps; need {count}"
        )
    indices = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(np.int64)
    if len(set(int(value) for value in indices)) != count:
        raise RenderError("uniform timestamp selection produced duplicate indices")
    return [candidates[int(index)] for index in indices]


def _normalize_xyzi(response: dict[str, Any]) -> tuple[np.ndarray, bytes]:
    # NRE 26.04 render.py already applies T_nre_sensor_end before constructing
    # LidarRenderReturn. V04 retains that sensor-local response basis; the
    # camera projection later maps it through the artifact LiDAR calibration.
    points = np.asarray(response["point_xyzs"], dtype=np.float32).reshape((-1, 3))
    intensities = np.asarray(response["point_intensities"], dtype=np.float32)
    sensor_points = points @ SENSOR_TO_BEV_AXES.T
    xyzi = np.column_stack((sensor_points, intensities)).astype("<f4", copy=False)
    return xyzi, xyzi.tobytes()


def _target_object(
    scene: ArtifactScene,
    timestamp_us: int,
    target_delta: dict[str, float] | None,
) -> dict[str, Any]:
    objects = scene.dynamic_objects(
        timestamp_us,
        mode="controllable",
        target_track_id=TARGET_TRACK_ID,
        target_delta=target_delta,
    )
    target = next(
        (item for item in objects if item.get("track_id") == TARGET_TRACK_ID), None
    )
    if target is None:
        raise RenderError(f"target pose is unavailable at {timestamp_us}")
    return target


def _rotation_matrix_from_xyzw(quaternion: list[float]) -> np.ndarray:
    if len(quaternion) != 4:
        raise RenderError("target quaternion must contain four values")
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _target_response_pose(
    scene: ArtifactScene,
    timestamp_us: int,
    target_delta: dict[str, float] | None,
) -> tuple[np.ndarray, float]:
    target = _target_object(scene, timestamp_us, target_delta)
    matrix = np.asarray(
        scene.lidar_pose_matrix("lidar_top", timestamp_us), dtype=np.float64
    )
    world = np.asarray(target["pose"][:3], dtype=np.float64)
    native = matrix[:3, :3].T @ (world - matrix[:3, 3])
    # The artifact calibration basis is x-right/y-forward. The NRE LiDAR
    # response is x-forward/y-right, so swap the horizontal components.
    position = np.asarray([native[1], native[0], native[2]], dtype=np.float32)
    target_rotation = _rotation_matrix_from_xyzw(target["pose"][3:7])
    target_forward_native = matrix[:3, :3].T @ target_rotation[:, 0]
    target_forward_response = np.asarray(
        [target_forward_native[1], target_forward_native[0]], dtype=np.float64
    )
    yaw_rad = math.atan2(
        float(target_forward_response[1]), float(target_forward_response[0])
    )
    return position, yaw_rad


def _target_response_position(
    scene: ArtifactScene,
    timestamp_us: int,
    target_delta: dict[str, float] | None,
) -> np.ndarray:
    position, _ = _target_response_pose(scene, timestamp_us, target_delta)
    return position


def _target_roi(
    points: np.ndarray,
    target_response: np.ndarray,
    target_yaw_rad: float = 0.0,
) -> np.ndarray:
    delta = points[:, :2] - target_response[:2]
    cos_yaw = math.cos(target_yaw_rad)
    sin_yaw = math.sin(target_yaw_rad)
    target_forward = delta[:, 0] * cos_yaw + delta[:, 1] * sin_yaw
    target_right = -delta[:, 0] * sin_yaw + delta[:, 1] * cos_yaw
    return (
        (np.abs(target_forward) <= TARGET_HALF_LENGTH_M)
        & (np.abs(target_right) <= TARGET_HALF_WIDTH_M)
    )


def _bev(
    points: np.ndarray,
    title: str,
    *,
    target_response: np.ndarray,
    reference_response: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    width, height = 800, 450
    image = np.full((height, width, 3), (18, 16, 14), dtype=np.uint8)
    x_min, x_max = -10.0, 60.0
    y_min, y_max = -30.0, 30.0
    xyz = points[:, :3]
    mask = (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] <= y_max)
    )
    visible = xyz[mask]
    px = ((visible[:, 1] - y_min) / (y_max - y_min) * (width - 1)).astype(np.int32)
    py = ((x_max - visible[:, 0]) / (x_max - x_min) * (height - 1)).astype(np.int32)
    image[py, px] = (190, 196, 202)

    target_mask = mask & _target_roi(xyz, target_response)
    target_points = xyz[target_mask]
    target_px = (
        (target_points[:, 1] - y_min) / (y_max - y_min) * (width - 1)
    ).astype(np.int32)
    target_py = (
        (x_max - target_points[:, 0]) / (x_max - x_min) * (height - 1)
    ).astype(np.int32)
    for point_x, point_y in zip(target_px, target_py):
        cv2.circle(image, (int(point_x), int(point_y)), 2, (0, 220, 255), -1)

    def pixel(position: np.ndarray) -> tuple[int, int]:
        return (
            int(round((float(position[1]) - y_min) / (y_max - y_min) * (width - 1))),
            int(round((x_max - float(position[0])) / (x_max - x_min) * (height - 1))),
        )

    def footprint(position: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        left_top = pixel(
            np.asarray(
                [
                    position[0] + TARGET_HALF_LENGTH_M,
                    position[1] - TARGET_HALF_WIDTH_M,
                ]
            )
        )
        right_bottom = pixel(
            np.asarray(
                [
                    position[0] - TARGET_HALF_LENGTH_M,
                    position[1] + TARGET_HALF_WIDTH_M,
                ]
            )
        )
        cv2.rectangle(image, left_top, right_bottom, color, thickness, cv2.LINE_AA)

    if reference_response is not None:
        footprint(reference_response, (150, 150, 150), 1)
    footprint(target_response, (0, 220, 255), 2)
    target_pixel = pixel(target_response)
    cv2.drawMarker(
        image, target_pixel, (0, 220, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA
    )
    origin_x = int((0.0 - y_min) / (y_max - y_min) * (width - 1))
    origin_y = int((x_max - 0.0) / (x_max - x_min) * (height - 1))
    cv2.drawMarker(image, (origin_x, origin_y), (80, 210, 120), cv2.MARKER_CROSS, 14, 2)
    cv2.arrowedLine(
        image,
        (origin_x, origin_y - 4),
        (origin_x, origin_y - 48),
        (80, 210, 120),
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )
    cv2.putText(image, "FRONT +x", (origin_x + 8, origin_y - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 230, 140), 1, cv2.LINE_AA)
    cv2.putText(image, "LEFT -y", (18, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 190), 1, cv2.LINE_AA)
    cv2.putText(image, "RIGHT +y", (width - 92, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 190), 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (width - 1, 30), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"target pose | {len(target_points)} returns in outlined ROI",
        (10, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return image, int(len(target_points))


def _target_world_cuboid(
    scene: ArtifactScene,
    timestamp_us: int,
    target_delta: dict[str, float] | None,
) -> np.ndarray:
    target = _target_object(scene, timestamp_us, target_delta)
    center = np.asarray(target["pose"][:3], dtype=np.float64)
    rotation = _rotation_matrix_from_xyzw(target["pose"][3:7])
    local = np.asarray(
        [
            [TARGET_HALF_LENGTH_M, -TARGET_HALF_WIDTH_M, -TARGET_HALF_HEIGHT_M],
            [TARGET_HALF_LENGTH_M, TARGET_HALF_WIDTH_M, -TARGET_HALF_HEIGHT_M],
            [-TARGET_HALF_LENGTH_M, TARGET_HALF_WIDTH_M, -TARGET_HALF_HEIGHT_M],
            [-TARGET_HALF_LENGTH_M, -TARGET_HALF_WIDTH_M, -TARGET_HALF_HEIGHT_M],
            [TARGET_HALF_LENGTH_M, -TARGET_HALF_WIDTH_M, TARGET_HALF_HEIGHT_M],
            [TARGET_HALF_LENGTH_M, TARGET_HALF_WIDTH_M, TARGET_HALF_HEIGHT_M],
            [-TARGET_HALF_LENGTH_M, TARGET_HALF_WIDTH_M, TARGET_HALF_HEIGHT_M],
            [-TARGET_HALF_LENGTH_M, -TARGET_HALF_WIDTH_M, TARGET_HALF_HEIGHT_M],
        ],
        dtype=np.float64,
    )
    return local @ rotation.T + center


def _camera_project_world(
    scene: ArtifactScene,
    world_points: np.ndarray,
    timestamp_us: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsics = scene.camera_intrinsics("camera_front")
    resolution = intrinsics.get("resolution") or [width, height]
    scale_x = width / float(resolution[0])
    scale_y = height / float(resolution[1])
    focal = intrinsics.get("focal_length") or [1.0, 1.0]
    principal = intrinsics.get("principal_point") or [width / 2.0, height / 2.0]
    fx, fy = float(focal[0]) * scale_x, float(focal[1]) * scale_y
    cx, cy = float(principal[0]) * scale_x, float(principal[1]) * scale_y
    pose = np.asarray(scene.sensor_pose_matrix("camera_front", timestamp_us), dtype=np.float64)
    world = np.asarray(world_points, dtype=np.float64).reshape((-1, 3))
    camera = (world - pose[:3, 3]) @ pose[:3, :3]
    depth = camera[:, 2]
    uv = np.empty((len(camera), 2), dtype=np.float32)
    uv[:, 0] = cx + fx * camera[:, 0] / np.maximum(depth, 1e-6)
    uv[:, 1] = cy + fy * camera[:, 1] / np.maximum(depth, 1e-6)
    return camera, uv, depth > 0.2


def _draw_projected_cuboid(
    image: np.ndarray,
    uv: np.ndarray,
    in_front: np.ndarray,
    color: tuple[int, int, int],
) -> bool:
    height, width = image.shape[:2]
    in_view = (
        in_front
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    if int(in_view.sum()) < 4:
        return False
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for first, second in edges:
        if in_front[first] and in_front[second]:
            start = tuple(int(round(value)) for value in uv[first])
            end = tuple(int(round(value)) for value in uv[second])
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)
    return True


def _cell_keys(points: np.ndarray) -> set[tuple[int, int, int]]:
    cells = np.floor(points[:, :3] / VOXEL_M).astype(np.int64)
    return {tuple(int(value) for value in row) for row in cells}


def _camera_lidar(
    scene: ArtifactScene,
    points: np.ndarray,
    title: str,
    *,
    lidar_timestamp_us: int,
    camera_timestamp_us: int,
    target_response: np.ndarray,
    target_yaw_rad: float,
    highlight_cells: set[tuple[int, int, int]] | None = None,
    second_points: np.ndarray | None = None,
    second_cells: set[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, int]:
    width, height = 800, 450
    image = np.zeros((height, width, 3), dtype=np.uint8)
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    lidar_pose = np.asarray(
        scene.lidar_pose_matrix("lidar_top", lidar_timestamp_us), dtype=np.float64
    )
    artifact_xyz = xyz @ RESPONSE_TO_ARTIFACT_AXES.T
    world = artifact_xyz @ lidar_pose[:3, :3].T + lidar_pose[:3, 3]
    camera, uv, in_front = _camera_project_world(
        scene, world, camera_timestamp_us, width, height
    )
    visible = (
        in_front
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )

    def draw_layer(
        layer_points: np.ndarray,
        layer_camera: np.ndarray,
        layer_uv: np.ndarray,
        layer_visible: np.ndarray,
        layer_cells: set[tuple[int, int, int]] | None,
        signal_color: tuple[int, int, int],
        layer_dim: bool,
    ) -> None:
        indices = np.flatnonzero(layer_visible)
        if len(indices):
            indices = indices[np.argsort(layer_camera[indices, 2])[::-1]]
        for index in indices:
            pixel = (
                int(np.clip(round(float(layer_uv[index, 0])), 0, width - 1)),
                int(np.clip(round(float(layer_uv[index, 1])), 0, height - 1)),
            )
            radius = 1
            color = tuple(int(255.0 * (0.28 if layer_dim else 0.45)) for _ in range(3))
            if layer_cells is not None:
                key = tuple(
                    int(value)
                    for value in np.floor(layer_points[index, :3] / VOXEL_M).astype(np.int64)
                )
                if key in layer_cells:
                    color = signal_color
                    radius = 2
            cv2.circle(image, pixel, radius, color, -1, cv2.LINE_AA)

    draw_layer(points, camera, uv, visible, highlight_cells, (0, 220, 255), False)
    if second_points is not None:
        second_xyz = np.asarray(second_points[:, :3], dtype=np.float32)
        second_world = (
            second_xyz @ RESPONSE_TO_ARTIFACT_AXES.T @ lidar_pose[:3, :3].T
            + lidar_pose[:3, 3]
        )
        second_camera, second_uv, second_front = _camera_project_world(
            scene, second_world, camera_timestamp_us, width, height
        )
        second_visible = (
            second_front
            & (second_uv[:, 0] >= 0)
            & (second_uv[:, 0] < width)
            & (second_uv[:, 1] >= 0)
            & (second_uv[:, 1] < height)
        )
        draw_layer(
            second_points,
            second_camera,
            second_uv,
            second_visible,
            second_cells,
            (255, 100, 30),
            True,
        )

    target_mask = _target_roi(xyz, target_response, target_yaw_rad)
    target_return_count = int((visible & target_mask).sum())
    cv2.rectangle(image, (0, 0), (width - 1, 74), (5, 7, 10), -1)
    cv2.putText(
        image,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return image, target_return_count


def _label_rgb(image: np.ndarray, title: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 30), (0, 0, 0), -1)
    cv2.putText(
        result,
        title,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def _read_front(root: Path, row: dict[str, Any]) -> tuple[np.ndarray, Path]:
    relative = (row.get("camera_frame_paths") or {}).get("camera_front")
    if not relative:
        raise RenderError("camera_front frame path missing")
    path = root / str(relative)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (450, 800):
        raise RenderError(f"invalid camera_front image: {path}")
    return image, path


def _decode_rgb(response: dict[str, Any], label: str) -> tuple[np.ndarray, bytes]:
    if response.get("status") != "passed":
        raise RenderError(f"{label} RGB render failed: {response.get('error')}")
    body = bytes(response.get("rgb_bytes") or b"")
    image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (450, 800):
        raise RenderError(f"{label} RGB response is not an 800x450 JPEG")
    return image, body


def render(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.resume:
        if not (output_dir / "frames.jsonl").is_file():
            raise RenderError("V04 resume requires an existing frames.jsonl")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RenderError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    lidar_dir = output_dir / "lidar"
    rgb_dir = output_dir / "rgb"
    frames_dir.mkdir(exist_ok=True)
    lidar_dir.mkdir(exist_ok=True)
    rgb_dir.mkdir(exist_ok=True)

    scene = ArtifactScene(ARTIFACT, TARGET_TRACK_ID)
    scene.max_actor_interpolation_gap_us = int(args.max_interpolation_gap_us)
    scene.max_rig_interpolation_gap_us = int(args.max_interpolation_gap_us)
    model = scene.lidar_model("lidar_top")
    frequency = float(model["parameters"]["spinning_frequency_hz"])
    window_us = int(round(1_000_000.0 / frequency))
    case = load_json(REPO_ROOT / "demo/scene0061/cases/original_replay.json")
    edit_case = load_json(REPO_ROOT / "demo/scene0061/cases/lead_vehicle_edit.json")
    edit_definition = edit_case.get("lead_vehicle_edit") or {}
    raw_target_delta = edit_definition.get("translation_m") or {}
    if not isinstance(raw_target_delta, Mapping):
        raise RenderError("lead_vehicle_edit.translation_m must be an object")
    target_delta = {
        axis: float(raw_target_delta.get(axis, 0.0))
        for axis in ("x", "y", "z")
    }
    target_delta_frame = str(edit_definition.get("translation_frame", "world"))
    if target_delta_frame != "world":
        raise RenderError(
            "V04 currently requires lead_vehicle_edit.translation_frame=world"
        )
    case_range = case["timestamp_range_us"]
    first_us = int(case_range["start"])
    last_start_us = min(int(case_range["end"]), scene.rig_timestamps[-1]) - window_us
    selected = [
        int(round(value))
        for value in np.linspace(first_us, last_start_us, int(args.frame_count))
    ]
    rejected = []
    for timestamp_us in selected:
        try:
            scene.dynamic_objects(
                timestamp_us,
                end_timestamp_us=timestamp_us + window_us,
                mode="controllable",
            )
            scene.dynamic_objects(
                timestamp_us,
                end_timestamp_us=timestamp_us + window_us,
                mode="controllable",
                target_track_id=TARGET_TRACK_ID,
                target_delta=target_delta,
            )
            scene.rig_pose_matrix(timestamp_us)
            scene.rig_pose_matrix(timestamp_us + window_us)
        except RenderError as exc:
            rejected.append({"timestamp_us": timestamp_us, "reason": str(exc)})
    if rejected:
        raise RenderError(f"V04 preflight rejected {len(rejected)} timestamps")
    preflight = {
        "schema_version": "nsb.v04-preflight.v1",
        "candidate_count": len(selected),
        "rejected_count": len(rejected),
        "selected_count": len(selected),
        "rejected": rejected,
        "actor_interpolation_max_gap_us": scene.max_actor_interpolation_gap_us,
        "rig_interpolation_max_gap_us": scene.max_rig_interpolation_gap_us,
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    client = SensorsimClient(
        str(args.server_address),
        "scene-0061",
        Path(args.python_api_path),
        float(args.timeout_sec),
    )
    records = []
    if args.resume:
        records = _load_rows(output_dir / "frames.jsonl")
        if len(records) > len(selected):
            raise RenderError("V04 resume contains too many frames")
        for expected_index, record in enumerate(records):
            if int(record.get("frame_index", -1)) != expected_index:
                raise RenderError("V04 resume frame indices are not contiguous")
            if int(record.get("timestamp_us", -1)) != selected[expected_index]:
                raise RenderError(f"V04 resume timestamp differs at frame {expected_index}")
            for relative in (
                record.get("rgb", {}).get("original_path"),
                record.get("rgb", {}).get("edited_path"),
                record.get("lidar", {}).get("original_path"),
                record.get("lidar", {}).get("edited_path"),
                f"frames/{expected_index:06d}.jpg",
            ):
                if not relative or not (output_dir / relative).is_file():
                    raise RenderError(f"V04 resume payload is missing: {relative}")
    try:
        for frame_index in range(len(records), len(selected)):
            timestamp_us = selected[frame_index]
            end_us = timestamp_us + window_us
            mid_us = (timestamp_us + end_us) // 2
            baseline_objects = scene.dynamic_objects(
                timestamp_us, end_timestamp_us=end_us, mode="controllable"
            )
            edited_objects = scene.dynamic_objects(
                timestamp_us,
                end_timestamp_us=end_us,
                mode="controllable",
                target_track_id=TARGET_TRACK_ID,
                target_delta=target_delta,
            )
            start_pose = _pose_from_matrix(scene.lidar_pose_matrix("lidar_top", timestamp_us))
            end_pose = _pose_from_matrix(scene.lidar_pose_matrix("lidar_top", end_us))
            rgb_start_pose = _pose_from_matrix(
                scene.sensor_pose_matrix("camera_front", timestamp_us)
            )
            rgb_end_pose = _pose_from_matrix(
                scene.sensor_pose_matrix("camera_front", end_us)
            )
            started = time.perf_counter()
            baseline_rgb_response = client.render_rgb(
                camera_id="camera_front",
                width=800,
                height=450,
                start_us=timestamp_us,
                end_us=end_us,
                start_pose=rgb_start_pose,
                end_pose=rgb_end_pose,
                dynamic_objects=baseline_objects,
                frame_id=f"V04:{frame_index}:original:camera_front",
            )
            edited_rgb_response = client.render_rgb(
                camera_id="camera_front",
                width=800,
                height=450,
                start_us=timestamp_us,
                end_us=end_us,
                start_pose=rgb_start_pose,
                end_pose=rgb_end_pose,
                dynamic_objects=edited_objects,
                frame_id=f"V04:{frame_index}:edited:camera_front",
            )
            baseline_lidar = client.render_lidar(
                lidar_id="lidar_top",
                device_type="PANDAR128",
                start_us=timestamp_us,
                end_us=end_us,
                start_pose=start_pose,
                end_pose=end_pose,
                dynamic_objects=baseline_objects,
            )
            edited_lidar = client.render_lidar(
                lidar_id="lidar_top",
                device_type="PANDAR128",
                start_us=timestamp_us,
                end_us=end_us,
                start_pose=start_pose,
                end_pose=end_pose,
                dynamic_objects=edited_objects,
            )
            if baseline_lidar.get("status") != "passed" or edited_lidar.get("status") != "passed":
                raise RenderError(
                    f"LiDAR render failed at {timestamp_us}: "
                    f"{baseline_lidar.get('error')} / {edited_lidar.get('error')}"
                )
            baseline_points, baseline_bytes = _normalize_xyzi(baseline_lidar)
            edited_points, edited_bytes = _normalize_xyzi(edited_lidar)
            baseline_cells = _cell_keys(baseline_points)
            edited_cells = _cell_keys(edited_points)
            added_cells = edited_cells - baseline_cells
            removed_cells = baseline_cells - edited_cells
            baseline_target_response, baseline_target_yaw = _target_response_pose(
                scene, end_us, None
            )
            edited_target_response, edited_target_yaw = _target_response_pose(
                scene, end_us, target_delta
            )
            baseline_bin = lidar_dir / f"{frame_index:06d}_original.xyzi.bin"
            edited_bin = lidar_dir / f"{frame_index:06d}_edited.xyzi.bin"
            baseline_bin.write_bytes(baseline_bytes)
            edited_bin.write_bytes(edited_bytes)
            baseline_rgb, baseline_rgb_bytes = _decode_rgb(
                baseline_rgb_response, "original"
            )
            edited_rgb, edited_rgb_bytes = _decode_rgb(
                edited_rgb_response, "edited"
            )
            baseline_rgb_path = rgb_dir / f"{frame_index:06d}_original.jpg"
            edited_rgb_path = rgb_dir / f"{frame_index:06d}_edited.jpg"
            baseline_rgb_path.write_bytes(baseline_rgb_bytes)
            edited_rgb_path.write_bytes(edited_rgb_bytes)
            baseline_lidar_view, baseline_target_return_count = _camera_lidar(
                scene,
                baseline_points,
                "Original LiDAR | camera_front projection | no target bbox",
                lidar_timestamp_us=end_us,
                camera_timestamp_us=mid_us,
                target_response=baseline_target_response,
                target_yaw_rad=baseline_target_yaw,
            )
            edited_lidar_view, edited_target_return_count = _camera_lidar(
                scene,
                edited_points,
                "Edited LiDAR overlay | gray=unchanged | yellow=added | blue=removed",
                lidar_timestamp_us=end_us,
                camera_timestamp_us=mid_us,
                target_response=edited_target_response,
                target_yaw_rad=edited_target_yaw,
                highlight_cells=added_cells,
                second_points=baseline_points,
                second_cells=removed_cells,
            )
            grid = np.vstack(
                (
                    np.hstack(
                        (
                            _label_rgb(baseline_rgb, "Original camera_front RGB"),
                            baseline_lidar_view,
                        )
                    ),
                    np.hstack(
                        (
                            _label_rgb(edited_rgb, "Lead-vehicle edit camera_front RGB"),
                            edited_lidar_view,
                        )
                    ),
                )
            )
            cv2.putText(
                grid,
                f"V04 | 20 FPS | t={timestamp_us}",
                (1170, 438),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            frame_path = frames_dir / f"{frame_index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RenderError(f"cannot write V04 frame {frame_index}")
            record = {
                "frame_index": frame_index,
                "status": "passed",
                "dropped": False,
                "timestamp_us": timestamp_us,
                "lidar_window_end_us": end_us,
                "rgb_render_timestamp_us": mid_us,
                "target_delta_m": target_delta,
                "target_delta_frame": target_delta_frame,
                "rgb": {
                    "original_path": str(baseline_rgb_path.relative_to(output_dir)),
                    "original_sha256": sha256_bytes(baseline_rgb_bytes),
                    "original_request_sha256": baseline_rgb_response.get("request_digest"),
                    "edited_path": str(edited_rgb_path.relative_to(output_dir)),
                    "edited_sha256": sha256_bytes(edited_rgb_bytes),
                    "edited_request_sha256": edited_rgb_response.get("request_digest"),
                },
                "lidar": {
                    "original_path": str(baseline_bin.relative_to(output_dir)),
                    "original_sha256": sha256_bytes(baseline_bytes),
                    "original_point_count": len(baseline_points),
                    "edited_path": str(edited_bin.relative_to(output_dir)),
                    "edited_sha256": sha256_bytes(edited_bytes),
                    "edited_point_count": len(edited_points),
                    "response_encoding": baseline_lidar.get("response_encoding"),
                    "response_axis_convention": "x_forward_y_right_z_up",
                    "lidar_reference_timestamp_us": end_us,
                    "camera_projection_timestamp_us": mid_us,
                    "projection": "camera_front_perspective",
                    "response_to_artifact_axes": RESPONSE_TO_ARTIFACT_AXES.reshape(-1).tolist(),
                    "roi_semantics": "oriented geometry ROI; returns are not actor-owned labels",
                    "original_target_response_position_m": baseline_target_response.tolist(),
                    "edited_target_response_position_m": edited_target_response.tolist(),
                    "original_target_response_yaw_deg": math.degrees(baseline_target_yaw),
                    "edited_target_response_yaw_deg": math.degrees(edited_target_yaw),
                    "original_target_roi_return_count": baseline_target_return_count,
                    "edited_target_roi_return_count": edited_target_return_count,
                },
                "rgb_lidar_timestamp_delta_us": None,
                "rgb_lidar_pairing": "same logical render window; RGB midpoint, LiDAR end-of-spin",
                "rpc_latency_ms": (time.perf_counter() - started) * 1000.0,
            }
            records.append(record)
            with (output_dir / "frames.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if (frame_index + 1) % 25 == 0:
                checkpoint = {
                    "schema_version": "nsb.v04-checkpoint.v1",
                    "status": "in_progress",
                    "completed_frames": frame_index + 1,
                    "last_timestamp_us": timestamp_us,
                }
                (output_dir / "checkpoint.json").write_text(
                    json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(checkpoint, sort_keys=True), flush=True)
    finally:
        client.close()

    video_path = output_dir / "V04_multimodal_alignment_20fps.mp4"
    _encode_mp4(
        [frames_dir / f"{index:06d}.jpg" for index in range(len(records))],
        video_path,
        20.0,
    )
    result = {
        "schema_version": "nsb.v04-multimodal-diagnostic.v2",
        "status": "passed",
        "evidence_classification": "open_loop_renderer_diagnostic",
        "control_mode": "none",
        "video": str(video_path),
        "video_sha256": sha256_file(video_path),
        "frame_count": len(records),
        "fps": 20.0,
        "duration_s": len(records) / 20.0,
        "timestamp_range_us": [records[0]["timestamp_us"], records[-1]["timestamp_us"]],
        "source_frame_policy": "385 approximately 20 Hz live RGB/LiDAR windows uniformly spanning the artifact-supported interval",
        "artifact": {
            "path": str(ARTIFACT),
            "sha256": sha256_file(ARTIFACT),
            "size_bytes": ARTIFACT.stat().st_size,
        },
        "lidar": {
            "source": "live SensorsimService.render_lidar",
            "device_type": "PANDAR128",
            "coordinate_frame": "nre_lidar_sensor_local_end_of_spin",
            "axis_convention": "x_forward_y_right_z_up",
            "server_transform": "pc_nre transformed by T_nre_sensor_end before LidarRenderReturn",
            "response_to_artifact_axes": RESPONSE_TO_ARTIFACT_AXES.reshape(-1).tolist(),
            "projection": "camera_front perspective using camera intrinsics",
            "camera_reference_timestamp": "RGB wire midpoint; LiDAR points remain in end-of-spin coordinates",
            "target_annotation": "oriented target cuboid projected into camera_front",
            "roi_semantics": "oriented geometry ROI; highlighted returns are not actor-owned labels",
            "server_implementation": "nre/render/render.py: pc_sensor = transform_point_cloud(pc_nre, T_nre_sensor_end)",
        },
        "rgb_lidar_timestamp_alignment_max_us": None,
        "rgb_lidar_pairing": "same logical render window; RGB midpoint, LiDAR end-of-spin",
        "target_delta_m": target_delta,
        "target_delta_frame": target_delta_frame,
        "rejected_source_timestamp_count": len(rejected),
        "playback_only": True,
        "max_interpolation_gap_us": int(args.max_interpolation_gap_us),
        "limitations": [
            "V04 is a uniform 20 FPS playback baseline over approximately 20 Hz live render windows.",
            "Actor and rig gaps up to the declared playback-only interpolation limit are interpolated.",
            "RGB is sampled at each logical window midpoint; LiDAR is projected from its end-of-spin frame into that camera pose.",
            "The highlighted LiDAR ROI is geometric evidence, not per-point actor ownership.",
            "This variant renders baseline and edited requests only; no A/A repeat control was captured.",
            "RGB/LiDAR timestamp fields describe logical-window pairing, not zero-microsecond physical alignment.",
            "No source LiDAR, static point-cloud copying, optical flow, or synthetic points were used.",
        ],
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "checkpoint.json").write_text(
        json.dumps(
            {"schema_version": "nsb.v04-checkpoint.v1", "status": "completed", "completed_frames": len(records)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs/nurec_scene0061_final/multimodal_20fps"),
    )
    parser.add_argument("--server-address", default="127.0.0.1:46443")
    parser.add_argument("--python-api-path", default=str(DEFAULT_PYTHON_API_PATH))
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--frame-count", type=int, default=385)
    parser.add_argument("--max-interpolation-gap-us", type=int, default=600_000)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
