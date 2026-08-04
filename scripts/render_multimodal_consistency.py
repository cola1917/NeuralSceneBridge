#!/usr/bin/env python3
"""Capture a synchronized RGB/LiDAR actor-change probe.

The probe deliberately keeps the six-camera RGB grid and the LiDAR response on
one logical time window.  It emits immutable RGB/JPEG and float32 XYZI payloads
alongside a four-panel PNG so a reviewer can see the same vehicle edit in both
modalities without treating a point-count change alone as proof of alignment.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_counterfactual_video import (  # noqa: E402
    ArtifactScene,
    DEFAULT_PYTHON_API_PATH,
    RenderError,
    SensorsimClient,
    _camera_grid_for_case,
    _camera_ids_for_case,
    _camera_pose_pair,
    _dynamic_digest,
    _make_output_dir,
    _non_target_digest,
    _pose_from_matrix,
    _resolve_repo_path,
    _stitch_camera_frames,
    canonical_digest,
    load_json,
    resolve_case,
    sha256_bytes,
    validate_manifest_identity,
)


SOURCE_SCHEMA = "rgb_lidar_actor_change_source_report.v1"
PROBE_SCHEMA = "nsb.rgb-lidar-consistency-probe.v1"
DEFAULT_LIDAR_ID = "lidar_top"
DEFAULT_DEVICE_TYPE = "PANDAR128"


def _file_ref(path: Path, *, root: Path, kind: str, encoding: str) -> dict[str, Any]:
    body = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "kind": kind,
        "encoding": encoding,
        "sha256": sha256_bytes(body),
        "size_bytes": len(body),
    }


def _pose_digest(pair: Mapping[str, Any]) -> str:
    return canonical_digest(pair)


def _world_to_sensor(matrix: list[list[float]], point: list[float]) -> list[float]:
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    translation = np.asarray(matrix, dtype=np.float64)[:3, 3]
    local = rotation.T @ (np.asarray(point, dtype=np.float64) - translation)
    return [float(value) for value in local]


def _point_cloud(response: Mapping[str, Any]) -> np.ndarray:
    xyz = np.asarray(response.get("point_xyzs") or [], dtype=np.float32)
    if xyz.size == 0 or xyz.size % 3:
        raise RenderError("LiDAR response has no complete XYZ point cloud")
    points = xyz.reshape((-1, 3))
    intensities = np.asarray(response.get("point_intensities") or [], dtype=np.float32)
    if intensities.size != len(points):
        raise RenderError("LiDAR response intensity count differs from point count")
    return np.column_stack((points, intensities))


def _voxel_signature(points: np.ndarray, voxel_m: float = 0.10) -> set[tuple[int, int, int]]:
    if not math.isfinite(voxel_m) or voxel_m <= 0.0:
        raise RenderError("voxel size must be positive and finite")
    cells = np.floor(points[:, :3] / float(voxel_m)).astype(np.int64)
    return {tuple(int(value) for value in row) for row in cells}


def _render_rgb_grid(
    client: SensorsimClient,
    scene: ArtifactScene,
    case: Mapping[str, Any],
    camera_ids: list[str],
    *,
    width: int,
    height: int,
    start_us: int,
    end_us: int,
    dynamic_objects: list[Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, Any], np.ndarray]:
    bodies: dict[str, bytes] = {}
    responses: dict[str, Any] = {}
    for camera_id in camera_ids:
        start_pose, end_pose = _camera_pose_pair(
            scene, case, camera_id, start_us, end_us, 0.0
        )
        response = client.render_rgb(
            camera_id=camera_id,
            width=width,
            height=height,
            start_us=start_us,
            end_us=end_us,
            start_pose=start_pose,
            end_pose=end_pose,
            dynamic_objects=dynamic_objects,
        )
        responses[camera_id] = response
        if response.get("status") != "passed":
            raise RenderError(
                f"RGB {camera_id} failed in multimodal probe: {response.get('error')}"
            )
        bodies[camera_id] = response["rgb_bytes"]
    stitched = _stitch_camera_frames(
        bodies,
        camera_ids,
        width=width,
        height=height,
        columns=int((case.get("camera_grid") or {}).get("columns", 3)),
        rows=int((case.get("camera_grid") or {}).get("rows", 2)),
        label_cameras=bool((case.get("camera_grid") or {}).get("label_cameras", True)),
    )
    image = cv2.imdecode(np.frombuffer(stitched, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RenderError("cannot decode stitched RGB probe image")
    return bodies, responses, image


def _render_lidar(
    client: SensorsimClient,
    scene: ArtifactScene,
    *,
    lidar_id: str,
    device_type: str,
    start_us: int,
    end_us: int,
    dynamic_objects: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_matrix = scene.lidar_pose_matrix(lidar_id, start_us)
    end_matrix = scene.lidar_pose_matrix(lidar_id, end_us)
    response = client.render_lidar(
        lidar_id=lidar_id,
        device_type=device_type,
        start_us=start_us,
        end_us=end_us,
        start_pose=_pose_from_matrix(start_matrix),
        end_pose=_pose_from_matrix(end_matrix),
        dynamic_objects=dynamic_objects,
    )
    if response.get("status") != "passed":
        raise RenderError(f"LiDAR render failed: {response.get('error')}")
    return response, {
        "start": _pose_from_matrix(start_matrix),
        "end": _pose_from_matrix(end_matrix),
    }


def _draw_bev(
    canvas: np.ndarray,
    points: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
    target_sensor: list[float] | None = None,
    target_color: tuple[int, int, int] = (255, 0, 255),
    point_radius: int = 1,
) -> None:
    xmin, xmax, ymin, ymax = bounds
    height, width = canvas.shape[:2]
    span_x = max(1e-6, xmax - xmin)
    span_y = max(1e-6, ymax - ymin)
    xy = points[:, :2]
    visible = (
        (xy[:, 0] >= xmin)
        & (xy[:, 0] <= xmax)
        & (xy[:, 1] >= ymin)
        & (xy[:, 1] <= ymax)
    )
    for x, y in xy[visible]:
        pixel_x = int(round((float(x) - xmin) / span_x * (width - 1)))
        pixel_y = int(round((ymax - float(y)) / span_y * (height - 1)))
        cv2.circle(canvas, (pixel_x, pixel_y), point_radius, color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (width - 1, 34), (8, 10, 15), -1)
    cv2.putText(canvas, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 240, 245), 1, cv2.LINE_AA)
    if target_sensor is not None:
        x, y = target_sensor[:2]
        pixel_x = int(round((x - xmin) / span_x * (width - 1)))
        pixel_y = int(round((ymax - y) / span_y * (height - 1)))
        cv2.drawMarker(canvas, (pixel_x, pixel_y), target_color, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)


def _bev_bounds(points: np.ndarray, target_points: list[list[float]]) -> tuple[float, float, float, float]:
    all_points = [points[:, :2]]
    if target_points:
        all_points.append(np.asarray(target_points, dtype=np.float32)[:, :2])
    xy = np.concatenate(all_points, axis=0)
    low = np.percentile(xy, 1.0, axis=0)
    high = np.percentile(xy, 99.0, axis=0)
    center = (low + high) / 2.0
    span = max(float(high[0] - low[0]), float(high[1] - low[1]), 10.0)
    span *= 1.12
    return (
        float(center[0] - span / 2.0),
        float(center[0] + span / 2.0),
        float(center[1] - span / 2.0),
        float(center[1] + span / 2.0),
    )


def _compose_visual(
    baseline_rgb: np.ndarray,
    moved_rgb: np.ndarray,
    baseline_points: np.ndarray,
    moved_points: np.ndarray,
    *,
    target_baseline_sensor: list[float],
    target_moved_sensor: list[float],
    baseline_count: int,
    moved_count: int,
    timestamp_us: int,
    window_us: int,
) -> np.ndarray:
    rgb_height, rgb_width = baseline_rgb.shape[:2]
    panel_width = max(rgb_width // 2, 1200)
    panel_height = max(int(round(panel_width * rgb_height / max(1, rgb_width))), 450)
    output = np.zeros((panel_height * 2, panel_width * 2, 3), dtype=np.uint8)
    baseline_rgb = cv2.resize(baseline_rgb, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    moved_rgb = cv2.resize(moved_rgb, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    output[:panel_height, :panel_width] = baseline_rgb
    output[:panel_height, panel_width:] = moved_rgb
    target_points = [target_baseline_sensor, target_moved_sensor]
    bounds = _bev_bounds(np.concatenate((baseline_points, moved_points), axis=0), target_points)
    baseline_bev = np.full((panel_height, panel_width, 3), (18, 23, 30), dtype=np.uint8)
    overlay_bev = baseline_bev.copy()
    _draw_bev(
        baseline_bev,
        baseline_points,
        bounds=bounds,
        color=(215, 220, 225),
        label=f"LiDAR baseline | {baseline_count:,} points | sensor-local BEV",
        target_sensor=target_baseline_sensor,
    )
    _draw_bev(
        overlay_bev,
        baseline_points,
        bounds=bounds,
        color=(255, 75, 30),
        label=f"LiDAR edit orange / baseline blue | {moved_count:,} edited points",
        target_sensor=target_baseline_sensor,
        target_color=(255, 0, 255),
    )
    xmin, xmax, ymin, ymax = bounds
    span_x = xmax - xmin
    span_y = ymax - ymin
    for x, y in moved_points[:, :2]:
        px = int(round((float(x) - xmin) / span_x * (panel_width - 1)))
        py = int(round((ymax - float(y)) / span_y * (panel_height - 1)))
        cv2.circle(overlay_bev, (px, py), 1, (35, 135, 255), -1, cv2.LINE_AA)
    cv2.putText(
        overlay_bev,
        f"window {window_us / 1000.0:.1f} ms | t={timestamp_us}",
        (12, panel_height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 220, 230),
        1,
        cv2.LINE_AA,
    )
    output[panel_height:, :panel_width] = baseline_bev
    output[panel_height:, panel_width:] = overlay_bev
    for x in (panel_width,):
        cv2.line(output, (x, 0), (x, output.shape[0] - 1), (80, 90, 100), 2)
    cv2.line(output, (0, panel_height), (output.shape[1] - 1, panel_height), (80, 90, 100), 2)
    return output


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "demo/scene0061/manifest.json")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--case", default="V02")
    parser.add_argument("--server-address")
    parser.add_argument("--runtime-scene-id")
    parser.add_argument("--python-api-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamp-us", type=int)
    parser.add_argument("--window-us", type=int)
    parser.add_argument("--lidar-id", default=DEFAULT_LIDAR_ID)
    parser.add_argument("--device-type", default=DEFAULT_DEVICE_TYPE)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--camera-ids")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _resolve_repo_path(args.manifest)
    manifest = load_json(manifest_path)
    _, case = resolve_case(args.case)
    if case.get("case_id") != "V02" or not isinstance(case.get("lead_vehicle_edit"), Mapping):
        raise RenderError("multimodal consistency probe requires the V02 lead_vehicle_edit case")
    artifact, artifact_entry, _, _ = validate_manifest_identity(manifest, args.artifact)
    target_track_id = str(manifest.get("target_track_id") or "")
    scene = ArtifactScene(artifact, target_track_id)
    camera_ids = _camera_ids_for_case(case, args.camera_ids)
    _camera_grid_for_case(case, camera_ids)
    width = int((case.get("resolution") or {}).get("width", 800))
    height = int((case.get("resolution") or {}).get("height", 450))
    timestamp_us = int(args.timestamp_us if args.timestamp_us is not None else case.get("probe_timestamp_us"))
    if args.window_us is None:
        model = scene.lidar_model(args.lidar_id)
        frequency = float((model.get("parameters") or {}).get("spinning_frequency_hz", 20.0))
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise RenderError("artifact LiDAR spinning frequency is invalid")
        window_us = max(1, int(round(1_000_000.0 / frequency)))
    else:
        window_us = int(args.window_us)
    if window_us <= 0:
        raise RenderError("LiDAR window must be positive")
    end_timestamp_us = timestamp_us + window_us
    if end_timestamp_us > scene.rig_timestamps[-1]:
        raise RenderError("multimodal probe window exceeds artifact rig trajectory")
    dynamic_cfg = case.get("dynamic_objects") or {}
    dynamic_mode = str(dynamic_cfg.get("mode", "controllable"))
    edit_cfg = case["lead_vehicle_edit"]
    target_delta = edit_cfg.get("translation_m") or {}
    baseline_objects = scene.dynamic_objects(
        timestamp_us, end_timestamp_us=end_timestamp_us, mode=dynamic_mode
    )
    moved_objects = scene.dynamic_objects(
        timestamp_us,
        end_timestamp_us=end_timestamp_us,
        mode=dynamic_mode,
        target_track_id=target_track_id,
        target_delta=target_delta,
    )
    output_dir = _resolve_repo_path(args.output_dir)
    _make_output_dir(output_dir, bool(args.overwrite))
    (output_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "lidar").mkdir(parents=True, exist_ok=True)
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), Mapping) else {}
    server_address = str(args.server_address or runtime.get("server_address") or "127.0.0.1:46443")
    runtime_scene_id = str(args.runtime_scene_id or runtime.get("runtime_scene_id") or case.get("runtime_scene_id") or "scene-0061")
    python_api_path = _resolve_repo_path(
        args.python_api_path or DEFAULT_PYTHON_API_PATH
    )
    client = SensorsimClient(server_address, runtime_scene_id, python_api_path, float(args.timeout_sec))
    try:
        try:
            baseline_rgb_bodies, baseline_rgb_responses, baseline_rgb = _render_rgb_grid(
                client, scene, case, camera_ids, width=width, height=height,
                start_us=timestamp_us, end_us=end_timestamp_us, dynamic_objects=baseline_objects,
            )
            repeat_rgb_bodies, _, _ = _render_rgb_grid(
                client, scene, case, camera_ids, width=width, height=height,
                start_us=timestamp_us, end_us=end_timestamp_us, dynamic_objects=baseline_objects,
            )
            moved_rgb_bodies, moved_rgb_responses, moved_rgb = _render_rgb_grid(
                client, scene, case, camera_ids, width=width, height=height,
                start_us=timestamp_us, end_us=end_timestamp_us, dynamic_objects=moved_objects,
            )
            baseline_lidar, baseline_lidar_pose = _render_lidar(
                client, scene, lidar_id=args.lidar_id, device_type=args.device_type,
                start_us=timestamp_us, end_us=end_timestamp_us, dynamic_objects=baseline_objects,
            )
            repeat_lidar, _ = _render_lidar(
                client, scene, lidar_id=args.lidar_id, device_type=args.device_type,
                start_us=timestamp_us, end_us=end_timestamp_us, dynamic_objects=baseline_objects,
            )
            moved_lidar, moved_lidar_pose = _render_lidar(
                client, scene, lidar_id=args.lidar_id, device_type=args.device_type,
                start_us=timestamp_us, end_us=end_timestamp_us, dynamic_objects=moved_objects,
            )
        except RenderError as exc:
            unavailable = {
                "schema_version": PROBE_SCHEMA,
                "status": "unavailable",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "reason": str(exc),
                "artifact_sha256": artifact_entry.get("sha256"),
                "scene_id": str(case.get("scene_id") or runtime_scene_id),
                "case_id": str(case.get("case_id")),
                "target_track_id": target_track_id,
                "timestamp_us": timestamp_us,
                "window_us": window_us,
                "camera_ids": camera_ids,
                "lidar_id": args.lidar_id,
                "device_type": str(args.device_type).upper(),
                "coordinate_frame": "sensor_local_end_of_spin",
                "axis_convention": "unverified_sensor_local_xy",
                "baseline_dynamic_digest": _dynamic_digest(baseline_objects),
                "edited_dynamic_digest": _dynamic_digest(moved_objects),
                "non_target_actors_unchanged": _non_target_digest(
                    baseline_objects, target_track_id
                )
                == _non_target_digest(moved_objects, target_track_id),
            }
            (output_dir / "multimodal_consistency_probe.json").write_text(
                json.dumps(unavailable, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
    finally:
        client.close()

    baseline_points = _point_cloud(baseline_lidar)
    moved_points = _point_cloud(moved_lidar)
    baseline_voxels = _voxel_signature(baseline_points)
    moved_voxels = _voxel_signature(moved_points)
    voxel_added = len(moved_voxels - baseline_voxels)
    voxel_removed = len(baseline_voxels - moved_voxels)
    target_pose = next((item for item in baseline_objects if item.get("track_id") == target_track_id), None)
    moved_target_pose = next((item for item in moved_objects if item.get("track_id") == target_track_id), None)
    if not isinstance(target_pose, Mapping) or not isinstance(moved_target_pose, Mapping):
        raise RenderError("target track is absent from synchronized dynamic object payload")
    base_pose = target_pose.get("pose_pair", {}).get("start")
    moved_pose = moved_target_pose.get("pose_pair", {}).get("start")
    if not isinstance(base_pose, list) or not isinstance(moved_pose, list):
        raise RenderError("target pose pair is malformed")
    lidar_matrix = scene.lidar_pose_matrix(args.lidar_id, timestamp_us)
    target_baseline_sensor = _world_to_sensor(lidar_matrix, [float(value) for value in base_pose[:3]])
    target_moved_sensor = _world_to_sensor(lidar_matrix, [float(value) for value in moved_pose[:3]])
    visual = _compose_visual(
        baseline_rgb, moved_rgb, baseline_points, moved_points,
        target_baseline_sensor=target_baseline_sensor,
        target_moved_sensor=target_moved_sensor,
        baseline_count=len(baseline_points), moved_count=len(moved_points),
        timestamp_us=timestamp_us, window_us=window_us,
    )
    visual_path = output_dir / "multimodal_consistency_probe.png"
    if not cv2.imwrite(str(visual_path), visual):
        raise RenderError("failed to write multimodal probe visual")

    rgb_refs: dict[str, dict[str, dict[str, Any]]] = {"baseline": {}, "edited": {}}
    for camera_id in camera_ids:
        baseline_path = output_dir / "rgb" / f"baseline_{camera_id}.jpg"
        edited_path = output_dir / "rgb" / f"edited_{camera_id}.jpg"
        baseline_path.write_bytes(baseline_rgb_bodies[camera_id])
        edited_path.write_bytes(moved_rgb_bodies[camera_id])
        rgb_refs["baseline"][camera_id] = _file_ref(
            baseline_path, root=output_dir, kind="rgb", encoding="jpeg"
        )
        rgb_refs["edited"][camera_id] = _file_ref(
            edited_path, root=output_dir, kind="rgb", encoding="jpeg"
        )
    baseline_lidar_path = output_dir / "lidar" / "baseline.xyzi.bin"
    edited_lidar_path = output_dir / "lidar" / "edited.xyzi.bin"
    baseline_lidar_path.write_bytes(baseline_lidar["xyzi_bytes"])
    edited_lidar_path.write_bytes(moved_lidar["xyzi_bytes"])
    lidar_refs = {
        "baseline": _file_ref(
            baseline_lidar_path, root=output_dir, kind="lidar", encoding="float32_xyzi_little_endian"
        ),
        "edited": _file_ref(
            edited_lidar_path, root=output_dir, kind="lidar", encoding="float32_xyzi_little_endian"
        ),
    }
    rgb_baseline_hashes = [rgb_refs["baseline"][camera_id]["sha256"] for camera_id in camera_ids]
    rgb_edited_hashes = [rgb_refs["edited"][camera_id]["sha256"] for camera_id in camera_ids]
    rgb_repeat_hashes = [sha256_bytes(repeat_rgb_bodies[camera_id]) for camera_id in camera_ids]
    rgb_actor_changed = rgb_baseline_hashes != rgb_edited_hashes
    lidar_actor_changed = lidar_refs["baseline"]["sha256"] != lidar_refs["edited"]["sha256"]
    rgb_repeatable = rgb_baseline_hashes == rgb_repeat_hashes
    lidar_repeatable = baseline_lidar["lidar_payload_sha256"] == repeat_lidar["lidar_payload_sha256"]
    target_changed = _dynamic_digest(baseline_objects) != _dynamic_digest(moved_objects)
    non_target_unchanged = _non_target_digest(baseline_objects, target_track_id) == _non_target_digest(moved_objects, target_track_id)
    same_pose = _pose_digest(baseline_lidar_pose) == _pose_digest(moved_lidar_pose)
    passed = all((
        rgb_actor_changed, lidar_actor_changed, rgb_repeatable, lidar_repeatable,
        target_changed, non_target_unchanged, same_pose,
        len(baseline_points) > 0, len(moved_points) > 0,
    ))
    frame_range = {
        phase: {
            "start_frame_id": 0,
            "end_frame_id": 0,
            "frame_count": 1,
            "start_timestamp_sec": (timestamp_us - scene.rig_timestamps[0]) / 1_000_000.0,
            "end_timestamp_sec": (end_timestamp_us - scene.rig_timestamps[0]) / 1_000_000.0,
        }
        for phase in ("baseline", "edited")
    }
    source_report = {
        "schema_version": SOURCE_SCHEMA,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "scene_id": str(case.get("scene_id") or runtime_scene_id),
            "case_id": str(case.get("case_id")),
            "artifact_sha256": artifact_entry.get("sha256"),
        },
        "target_track_id": target_track_id,
        "frame_range": frame_range,
        "payloads": {
            "rgb": {
                "baseline": [rgb_refs["baseline"][camera_id] for camera_id in camera_ids],
                "edited": [rgb_refs["edited"][camera_id] for camera_id in camera_ids],
            },
            "lidar": {"baseline": [lidar_refs["baseline"]], "edited": [lidar_refs["edited"]]},
        },
        "change_flags": {
            "rgb_actor_changed": rgb_actor_changed,
            "lidar_actor_changed": lidar_actor_changed,
        },
    }
    source_path = output_dir / "rgb_lidar_actor_change_source_report.json"
    source_path.write_text(json.dumps(source_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    probe = {
        "schema_version": PROBE_SCHEMA,
        "status": "passed" if passed else "failed",
        "generated_at": source_report["generated_at"],
        "source_report": source_report,
        "source_report_path": str(source_path),
        "visual_path": str(visual_path),
        "artifact_sha256": artifact_entry.get("sha256"),
        "scene_id": str(case.get("scene_id") or runtime_scene_id),
        "case_id": str(case.get("case_id")),
        "timestamp_us": timestamp_us,
        "window_us": window_us,
        "camera_ids": camera_ids,
        "lidar_id": args.lidar_id,
        "device_type": str(args.device_type).upper(),
        "coordinate_frame": "sensor_local_end_of_spin",
        "axis_convention": "sensor_local_xy_unverified_without_native_carla_anchor",
        "target_track_id": target_track_id,
        "target_delta_m": dict(target_delta),
        "target_baseline_sensor_xyz": target_baseline_sensor,
        "target_edited_sensor_xyz": target_moved_sensor,
        "same_logical_time_window": True,
        "same_lidar_sensor_pose": same_pose,
        "baseline_dynamic_digest": _dynamic_digest(baseline_objects),
        "edited_dynamic_digest": _dynamic_digest(moved_objects),
        "non_target_actors_unchanged": non_target_unchanged,
        "rgb_actor_changed": rgb_actor_changed,
        "lidar_actor_changed": lidar_actor_changed,
        "rgb_a_a_repeatable": rgb_repeatable,
        "lidar_a_a_repeatable": lidar_repeatable,
        "baseline_point_count": len(baseline_points),
        "edited_point_count": len(moved_points),
        "point_count_delta": len(moved_points) - len(baseline_points),
        "voxel_size_m": 0.10,
        "voxel_added_count": voxel_added,
        "voxel_removed_count": voxel_removed,
        "baseline_lidar_payload_sha256": lidar_refs["baseline"]["sha256"],
        "edited_lidar_payload_sha256": lidar_refs["edited"]["sha256"],
        "rgb_baseline_payload_sha256_by_camera": dict(zip(camera_ids, rgb_baseline_hashes)),
        "rgb_edited_payload_sha256_by_camera": dict(zip(camera_ids, rgb_edited_hashes)),
        "render_requests": {
            "baseline_rgb": {camera_id: baseline_rgb_responses[camera_id].get("request_digest") for camera_id in camera_ids},
            "edited_rgb": {camera_id: moved_rgb_responses[camera_id].get("request_digest") for camera_id in camera_ids},
            "baseline_lidar": baseline_lidar.get("request_digest"),
            "edited_lidar": moved_lidar.get("request_digest"),
        },
    }
    probe_path = output_dir / "multimodal_consistency_probe.json"
    probe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return probe


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        result = run_probe(args)
    except (RenderError, OSError, ValueError) as exc:
        print(f"render_multimodal_consistency: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
