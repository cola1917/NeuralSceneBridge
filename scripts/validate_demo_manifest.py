#!/usr/bin/env python3
"""Validate the lightweight interview demo manifest without runtime assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_CASES = {"V01", "V02", "V03"}
EXPECTED_CAMERA_IDS = [
    "camera_front_left",
    "camera_front",
    "camera_front_right",
    "camera_back_left",
    "camera_back",
    "camera_back_right",
]


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _sha256(value: Any, path: str) -> str:
    digest = _nonempty(value, path)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{path} must be 64 lowercase hex characters")
    return digest


def _token(value: Any, path: str) -> str:
    token = _nonempty(value, path)
    if len(token) != 32 or any(char not in "0123456789abcdef" for char in token):
        raise ValueError(f"{path} must be a 32-character lowercase hex token")
    return token


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _load_case(root: Path, path: str, expected_case_id: str, target_track_id: str) -> None:
    case_path = (root / path).resolve()
    try:
        case_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"case path escapes manifest directory: {path}") from exc
    case = _mapping(json.loads(case_path.read_text(encoding="utf-8")), path)
    if case.get("schema_version") != "nsb.nurec-counterfactual-case.v1":
        raise ValueError(f"{path} has an unsupported case schema")
    if case.get("case_id") != expected_case_id:
        raise ValueError(f"{path} has the wrong case id")
    if case.get("scene_id") != "scene-0061" or case.get("runtime_scene_id") != "scene-0061":
        raise ValueError(f"{path} is not bound to scene-0061")
    if case.get("camera_ids") != EXPECTED_CAMERA_IDS:
        raise ValueError(f"{path} must use the canonical six-camera order")
    resolution = _mapping(case.get("resolution"), f"{path}.resolution")
    _positive_int(resolution.get("width"), f"{path}.resolution.width")
    _positive_int(resolution.get("height"), f"{path}.resolution.height")
    quality = _mapping(case.get("quality"), f"{path}.quality")
    if quality.get("require_contiguous_frames") is not True:
        raise ValueError(f"{path} must require contiguous frames")
    if case.get("video_fps") != 30.0:
        raise ValueError(f"{path} must remain at 30 FPS")
    if expected_case_id == "V02":
        edit = _mapping(case.get("lead_vehicle_edit"), f"{path}.lead_vehicle_edit")
        if edit.get("track_id") != target_track_id:
            raise ValueError(f"{path} edits a track other than the manifest target")
    if expected_case_id == "V03":
        sweep = _mapping(case.get("camera_sweep"), f"{path}.camera_sweep")
        if sweep.get("apply_to_all_cameras") is not True:
            raise ValueError(f"{path} must apply the sweep to all cameras")
        if "intrinsics" in case or "intrinsics" in sweep:
            raise ValueError(f"{path} must not edit camera intrinsics")


def validate_manifest(path: Path) -> dict[str, Any]:
    resolved_manifest = path.resolve()
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    manifest_root = resolved_manifest.parents[2]
    root = _mapping(payload, "manifest")
    if root.get("schema_version") != "nsb.nurec-interview-demo-manifest.v1":
        raise ValueError("unsupported demo manifest schema")
    scene_id = _nonempty(root.get("scene_id"), "scene_id")
    if scene_id != "scene-0061" or root.get("runtime_scene_id") != scene_id:
        raise ValueError("manifest must be bound to scene-0061")
    timestamp_range = _mapping(root.get("scene_timestamp_range_us"), "scene_timestamp_range_us")
    if timestamp_range.get("stop", 0) <= timestamp_range.get("start", 0):
        raise ValueError("scene timestamp range must be positive")

    for role in ("artifact", "checkpoint"):
        item = _mapping(root.get(role), role)
        _nonempty(item.get("uri"), f"{role}.uri")
        _sha256(item.get("sha256"), f"{role}.sha256")
        _positive_int(item.get("size_bytes"), f"{role}.size_bytes")

    runtime = _mapping(root.get("runtime"), "runtime")
    if runtime.get("runtime_scene_id") != scene_id:
        raise ValueError("runtime must use the manifest scene")
    resolution = _mapping(runtime.get("resolution"), "runtime.resolution")
    _positive_int(resolution.get("width"), "runtime.resolution.width")
    _positive_int(resolution.get("height"), "runtime.resolution.height")

    target_track_id = _token(root.get("target_track_id"), "target_track_id")

    case_files = root.get("case_files")
    if not isinstance(case_files, list) or len(case_files) != len(REQUIRED_CASES):
        raise ValueError("case_files must contain exactly V01, V02, and V03")
    expected_paths = {
        "demo/scene0061/cases/original_replay.json": "V01",
        "demo/scene0061/cases/lead_vehicle_edit.json": "V02",
        "demo/scene0061/cases/camera_pose_sweep.json": "V03",
    }
    if set(case_files) != set(expected_paths):
        raise ValueError("case_files do not match the canonical demo cases")
    for path, case_id in expected_paths.items():
        _load_case(manifest_root, path, case_id, target_track_id)

    limitations = root.get("limitations")
    if not isinstance(limitations, list) or not any(
        "does not claim CARLA closed-loop behavior" in str(item) for item in limitations
    ):
        raise ValueError("manifest must disclose that this demo is not CARLA closed-loop evidence")
    return {
        "status": "valid",
        "scene_id": scene_id,
        "case_count": len(case_files),
        "artifact_roles": ["artifact", "checkpoint"],
        "quality_report": _nonempty(root.get("quality_report"), "quality_report"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_manifest(args.manifest.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"DEMO MANIFEST INVALID: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
