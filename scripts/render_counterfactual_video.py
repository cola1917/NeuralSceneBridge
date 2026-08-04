#!/usr/bin/env python3
"""Render the scene-0061 NuRec interview demo without CARLA.

The script talks directly to NVIDIA's SensorsimService protobuf API.  Camera
trajectories, camera calibration, and dynamic tracks are read from the locked
USDZ so that the request coordinate frame is the same one used by the trained
artifact.  Every run writes an immutable evidence directory and only encodes
an MP4 after all requested frames have been captured.
"""

from __future__ import annotations

import argparse
import bisect
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import site
import statistics
import struct
import sys
import time
import uuid
import zipfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "demo" / "scene0061" / "manifest.json"
DEFAULT_PYTHON_API_PATH = Path(
    "/home/cwadmin/sim-env/data/CARLA_0.9.16/PythonAPI/examples/nvidia/nurec"
)
TARGET_TRACK_ID = "c1958768d48640948f6053d04cffd35b"
VIDEO_NAMES = {
    "V01": "original_replay.mp4",
    "V02": "lead_vehicle_edit.mp4",
    "V03": "camera_pose_sweep.mp4",
}
FORMAL_CAMERA_ORDER = (
    "camera_front_left",
    "camera_front",
    "camera_front_right",
    "camera_back_left",
    "camera_back",
    "camera_back_right",
)
DEFAULT_CAMERA_GRID = {"columns": 3, "rows": 2, "label_cameras": True}
MAX_INTERPOLATION_GAP_US = 75_000
CASE_FILE_NAMES = {
    "V01": "original_replay.json",
    "V02": "lead_vehicle_edit.json",
    "V03": "camera_pose_sweep.json",
}
SHA256_HEX = 64


class RenderError(RuntimeError):
    """Raised for a fail-closed configuration or capture error."""


def _read_protobuf_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift <= 63:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise RenderError("LiDAR response contains a truncated protobuf varint")


def _protobuf_wire_fields(data: bytes) -> dict[int, list[tuple[int, int | bytes]]]:
    fields: dict[int, list[tuple[int, int | bytes]]] = {}
    offset = 0
    while offset < len(data):
        key, offset = _read_protobuf_varint(data, offset)
        field_number = key >> 3
        wire_type = key & 7
        if field_number < 1 or field_number > (1 << 29) - 1:
            raise RenderError(f"LiDAR response has invalid protobuf field {field_number}")
        if wire_type == 0:
            value, offset = _read_protobuf_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise RenderError("LiDAR response has a truncated fixed64 field")
            value = data[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _read_protobuf_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise RenderError("LiDAR response has a truncated bytes field")
            value = data[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise RenderError("LiDAR response has a truncated fixed32 field")
            value = data[offset:end]
            offset = end
        else:
            raise RenderError(
                f"LiDAR response uses unsupported protobuf wire type {wire_type}"
            )
        fields.setdefault(field_number, []).append((wire_type, value))
    return fields


def _decode_lidar_response(response: Any) -> tuple[list[float], list[float], str]:
    """Decode legacy repeated floats and NRE 26.04 buffered LiDAR replies."""

    xyz = [float(value) for value in getattr(response, "point_xyzs", ())]
    intensities = [float(value) for value in getattr(response, "point_intensities", ())]
    if xyz or intensities:
        encoding = "legacy_repeated_float"
    else:
        point_count = int(getattr(response, "num_points", 0) or 0)
        xyz_buffer = bytes(getattr(response, "point_xyzs_buffer", b"") or b"")
        intensity_buffer = bytes(
            getattr(response, "point_intensities_buffer", b"") or b""
        )
        if not (point_count or xyz_buffer or intensity_buffer):
            serializer = getattr(response, "SerializeToString", None)
            if not callable(serializer):
                raise RenderError("LiDAR response is not a protobuf message")
            fields = _protobuf_wire_fields(serializer())
            counts = fields.get(3, [])
            xyz_buffers = fields.get(4, [])
            intensity_buffers = fields.get(5, [])
            if (
                len(counts) != 1
                or len(xyz_buffers) != 1
                or len(intensity_buffers) != 1
                or counts[0][0] != 0
                or xyz_buffers[0][0] != 2
                or intensity_buffers[0][0] != 2
                or not isinstance(counts[0][1], int)
                or not isinstance(xyz_buffers[0][1], bytes)
                or not isinstance(intensity_buffers[0][1], bytes)
            ):
                raise RenderError(
                    "LiDAR response contains neither legacy points nor valid NRE 26.04 buffers"
                )
            point_count = counts[0][1]
            xyz_buffer = xyz_buffers[0][1]
            intensity_buffer = intensity_buffers[0][1]
            encoding = "nre_26_04_unknown_buffers"
        else:
            encoding = "nre_26_04_buffers"
        if point_count <= 0:
            raise RenderError("LiDAR response num_points is empty")
        if len(xyz_buffer) != point_count * 12:
            raise RenderError("LiDAR response XYZ buffer size differs from point count")
        if len(intensity_buffer) != point_count * 4:
            raise RenderError("LiDAR response intensity buffer size differs from point count")
        xyz = list(struct.unpack(f"<{point_count * 3}f", xyz_buffer))
        intensities = list(struct.unpack(f"<{point_count}f", intensity_buffer))

    if len(xyz) % 3:
        raise RenderError("LiDAR response point_xyzs is not divisible by three")
    point_count = len(xyz) // 3
    if point_count <= 0:
        raise RenderError("LiDAR response point_xyzs is empty")
    if len(intensities) != point_count:
        raise RenderError(
            "LiDAR response intensity count differs from point count: "
            f"{len(intensities)}/{point_count}"
        )
    if not all(math.isfinite(value) for value in (*xyz, *intensities)):
        raise RenderError("LiDAR response contains non-finite values")
    return xyz, intensities, encoding


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _resolve_repo_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == SHA256_HEX and all(
        char in "0123456789abcdef" for char in value
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RenderError(f"JSON root must be an object: {path}")
    return value


def resolve_case(case_arg: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(case_arg).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [REPO_ROOT / path, REPO_ROOT / "demo" / "scene0061" / "cases" / path]
        )
        if path.suffix != ".json":
            candidates.append(
                REPO_ROOT / "demo" / "scene0061" / "cases" / f"{path}.json"
            )
            case_file_name = CASE_FILE_NAMES.get(str(path))
            if case_file_name:
                candidates.append(
                    REPO_ROOT / "demo" / "scene0061" / "cases" / case_file_name
                )
    for candidate in candidates:
        if candidate.is_file():
            payload = load_json(candidate.resolve())
            return candidate.resolve(), payload
    raise RenderError(f"case file does not exist: {case_arg}")


def validate_manifest_identity(
    manifest: Mapping[str, Any], artifact_override: Path | None
) -> tuple[Path, dict[str, Any], Path | None, dict[str, Any] | None]:
    artifact_entry = manifest.get("artifact")
    if not isinstance(artifact_entry, Mapping):
        raise RenderError("manifest.artifact is required")
    artifact_value = artifact_override or artifact_entry.get("uri") or artifact_entry.get("path")
    if not artifact_value:
        raise RenderError("manifest artifact URI is empty")
    artifact = _resolve_repo_path(str(artifact_value))
    if not artifact.is_file():
        raise RenderError(
            "canonical USDZ is unavailable; configure the remote artifact path "
            f"before rendering: {artifact}"
        )
    expected_hash = artifact_entry.get("sha256")
    expected_size = artifact_entry.get("size_bytes")
    if not _is_sha256(expected_hash):
        raise RenderError("manifest artifact.sha256 must be a lowercase SHA-256")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise RenderError("manifest artifact.size_bytes must be a positive integer")
    actual_size = artifact.stat().st_size
    actual_hash = sha256_file(artifact)
    if actual_size != expected_size or actual_hash != expected_hash:
        raise RenderError(
            "canonical USDZ identity mismatch: "
            f"expected {expected_hash}/{expected_size}, got {actual_hash}/{actual_size}"
        )

    checkpoint_entry = manifest.get("checkpoint")
    checkpoint: Path | None = None
    if isinstance(checkpoint_entry, Mapping):
        checkpoint_value = checkpoint_entry.get("uri") or checkpoint_entry.get("path")
        if checkpoint_value:
            checkpoint = _resolve_repo_path(str(checkpoint_value))
            if not checkpoint.is_file():
                raise RenderError(f"manifest checkpoint is unavailable: {checkpoint}")
            if not _is_sha256(checkpoint_entry.get("sha256")):
                raise RenderError("manifest checkpoint.sha256 must be a lowercase SHA-256")
            if checkpoint.stat().st_size != checkpoint_entry.get("size_bytes"):
                raise RenderError("manifest checkpoint size does not match the file")
            if sha256_file(checkpoint) != checkpoint_entry.get("sha256"):
                raise RenderError("manifest checkpoint SHA-256 does not match the file")
    return artifact, dict(artifact_entry), checkpoint, (
        dict(checkpoint_entry) if isinstance(checkpoint_entry, Mapping) else None
    )


def _sequence_chunks(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RenderError("sequence_tracks.json must contain an object")
    if isinstance(payload.get("tracks_data"), dict):
        return [payload]
    chunks = [
        value
        for value in payload.values()
        if isinstance(value, dict) and isinstance(value.get("tracks_data"), dict)
    ]
    if not chunks:
        raise RenderError("sequence_tracks.json contains no tracks_data chunk")
    return chunks


class ArtifactScene:
    """Minimal reader for the USDZ records needed by the demo."""

    def __init__(self, artifact: Path, required_track_id: str) -> None:
        self.max_actor_interpolation_gap_us = MAX_INTERPOLATION_GAP_US
        self.max_rig_interpolation_gap_us = MAX_INTERPOLATION_GAP_US
        try:
            archive = zipfile.ZipFile(artifact, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise RenderError(f"cannot open USDZ artifact {artifact}: {exc}") from exc
        with archive:
            try:
                self.rig = json.loads(archive.read("rig_trajectories.json"))
                sequence = json.loads(archive.read("sequence_tracks.json"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RenderError(
                    "USDZ must contain valid rig_trajectories.json and sequence_tracks.json"
                ) from exc
        if not isinstance(self.rig, dict):
            raise RenderError("rig_trajectories.json must contain an object")
        trajectories = self.rig.get("rig_trajectories")
        if not isinstance(trajectories, list) or len(trajectories) != 1:
            raise RenderError("USDZ must contain exactly one rig trajectory")
        self.trajectory = trajectories[0]
        if not isinstance(self.trajectory, dict):
            raise RenderError("rig trajectory must be an object")
        self.rig_timestamps = [
            int(value) for value in self.trajectory.get("T_rig_world_timestamps_us", [])
        ]
        self.rig_matrices = self.trajectory.get("T_rig_worlds") or []
        if len(self.rig_timestamps) != len(self.rig_matrices) or not self.rig_timestamps:
            raise RenderError("USDZ rig trajectory timestamps and poses are invalid")
        self.camera_calibrations = self.rig.get("camera_calibrations")
        if not isinstance(self.camera_calibrations, dict):
            raise RenderError("USDZ camera_calibrations is missing")
        self.lidar_calibrations = self.rig.get("lidar_calibrations") or {}
        if not isinstance(self.lidar_calibrations, dict):
            raise RenderError("USDZ lidar_calibrations must be an object when present")
        self.tracks: dict[str, dict[str, Any]] = {}
        self.track_flags: dict[str, str] = {}
        self.track_labels: dict[str, str] = {}
        for chunk in _sequence_chunks(sequence):
            data = chunk["tracks_data"]
            ids = data.get("tracks_id")
            poses = data.get("tracks_poses")
            timestamps = data.get("tracks_timestamps_us")
            flags = data.get("tracks_flags")
            labels = data.get("tracks_label_class")
            if not all(isinstance(value, list) for value in (ids, poses, timestamps, flags, labels)):
                raise RenderError("USDZ track arrays must be lists")
            lengths = {len(ids), len(poses), len(timestamps), len(flags), len(labels)}
            if len(lengths) != 1:
                raise RenderError("USDZ track arrays have different lengths")
            for index, raw_id in enumerate(ids):
                track_id = str(raw_id).strip()
                if not track_id or track_id in self.tracks:
                    raise RenderError(f"duplicate or empty USDZ track ID: {track_id!r}")
                track_timestamps = [int(value) for value in timestamps[index]]
                track_poses = poses[index]
                if len(track_timestamps) != len(track_poses) or not track_timestamps:
                    raise RenderError(f"invalid pose series for USDZ track {track_id}")
                if any(
                    not isinstance(pose, list) or len(pose) != 7 for pose in track_poses
                ):
                    raise RenderError(f"USDZ track {track_id} contains malformed poses")
                self.tracks[track_id] = {
                    "track_id": track_id,
                    "timestamps_us": track_timestamps,
                    "poses": track_poses,
                }
                self.track_flags[track_id] = str(flags[index])
                self.track_labels[track_id] = str(labels[index])
        if required_track_id not in self.tracks:
            raise RenderError(
                "required target track is absent from canonical USDZ: "
                f"{required_track_id}"
            )
        self.controllable_track_ids = sorted(
            track_id
            for track_id, flag in self.track_flags.items()
            if flag == "DYNAMIC|CONTROLLABLE"
        )
        if required_track_id not in self.controllable_track_ids:
            raise RenderError(
                "required target track is not controllable in canonical USDZ: "
                f"{required_track_id}"
            )

    def nearest_rig_index(self, timestamp_us: int) -> int:
        return min(
            range(len(self.rig_timestamps)),
            key=lambda index: abs(self.rig_timestamps[index] - timestamp_us),
        )

    def rig_pose_matrix(self, timestamp_us: int) -> list[list[float]]:
        return _interpolate_matrix(
            self.rig_timestamps,
            self.rig_matrices,
            timestamp_us,
            max_gap_us=self.max_rig_interpolation_gap_us,
        )

    def camera_extrinsic(self, camera_id: str) -> list[list[float]]:
        key = f"{camera_id}@{self.trajectory.get('sequence_id', 'scene-0061')}"
        calibration = self.camera_calibrations.get(key)
        if not isinstance(calibration, Mapping):
            raise RenderError(f"USDZ has no calibration for camera {camera_id}")
        matrix = calibration.get("T_sensor_rig")
        if not isinstance(matrix, list):
            raise RenderError(f"USDZ camera calibration has no T_sensor_rig: {camera_id}")
        return _copy_matrix(matrix)

    def camera_intrinsics(self, camera_id: str) -> Mapping[str, Any]:
        key = f"{camera_id}@{self.trajectory.get('sequence_id', 'scene-0061')}"
        calibration = self.camera_calibrations.get(key)
        model = calibration.get("camera_model") if isinstance(calibration, Mapping) else None
        parameters = model.get("parameters") if isinstance(model, Mapping) else None
        if not isinstance(parameters, Mapping):
            raise RenderError(f"USDZ camera intrinsics are missing: {camera_id}")
        return parameters

    def sensor_pose_matrix(self, camera_id: str, timestamp_us: int) -> list[list[float]]:
        return _mat_mul(self.rig_pose_matrix(timestamp_us), self.camera_extrinsic(camera_id))

    def lidar_extrinsic(self, lidar_id: str = "lidar_top") -> list[list[float]]:
        key = f"{lidar_id}@{self.trajectory.get('sequence_id', 'scene-0061')}"
        calibration = self.lidar_calibrations.get(key)
        if not isinstance(calibration, Mapping):
            raise RenderError(f"USDZ has no calibration for LiDAR {lidar_id}")
        matrix = calibration.get("T_sensor_rig")
        if not isinstance(matrix, list):
            raise RenderError(f"USDZ LiDAR calibration has no T_sensor_rig: {lidar_id}")
        return _copy_matrix(matrix)

    def lidar_model(self, lidar_id: str = "lidar_top") -> Mapping[str, Any]:
        key = f"{lidar_id}@{self.trajectory.get('sequence_id', 'scene-0061')}"
        calibration = self.lidar_calibrations.get(key)
        model = calibration.get("lidar_model") if isinstance(calibration, Mapping) else None
        if not isinstance(model, Mapping):
            raise RenderError(f"USDZ LiDAR model is missing: {lidar_id}")
        return model

    def lidar_pose_matrix(self, lidar_id: str, timestamp_us: int) -> list[list[float]]:
        return _mat_mul(self.rig_pose_matrix(timestamp_us), self.lidar_extrinsic(lidar_id))

    def dynamic_objects(
        self,
        timestamp_us: int,
        *,
        end_timestamp_us: int | None = None,
        mode: str = "controllable",
        target_track_id: str | None = None,
        target_delta: Mapping[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        if mode == "all":
            selected = sorted(self.tracks)
        elif mode in {"controllable", "all_controllable"}:
            selected = self.controllable_track_ids
        elif mode == "target_only":
            selected = [target_track_id] if target_track_id else []
        else:
            raise RenderError(f"unsupported dynamic_objects.mode: {mode}")
        result: list[dict[str, Any]] = []
        for track_id in selected:
            if not track_id or track_id not in self.tracks:
                raise RenderError(f"dynamic object track is unavailable: {track_id}")
            start_pose = _interpolate_track_pose(
                self.tracks[track_id],
                timestamp_us,
                max_gap_us=self.max_actor_interpolation_gap_us,
            )
            if start_pose is None:
                continue
            end_pose = (
                _interpolate_track_pose(
                    self.tracks[track_id],
                    end_timestamp_us,
                    max_gap_us=self.max_actor_interpolation_gap_us,
                )
                if end_timestamp_us is not None
                else None
            )
            if end_pose is None:
                end_pose = list(start_pose)
            if target_track_id and track_id == target_track_id and target_delta:
                start_pose = _translate_pose(start_pose, target_delta)
                end_pose = _translate_pose(end_pose, target_delta)
            item: dict[str, Any] = {"track_id": track_id, "pose": start_pose}
            if end_timestamp_us is not None:
                item["pose_pair"] = {"start": start_pose, "end": end_pose}
            result.append(item)
        return result


def _copy_matrix(matrix: Any) -> list[list[float]]:
    if not isinstance(matrix, list) or len(matrix) != 4:
        raise RenderError("pose matrix must be a 4x4 list")
    result = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 4:
            raise RenderError("pose matrix must be a 4x4 list")
        result.append([float(value) for value in row])
    return result


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)]
        for row in range(4)
    ]


def _translate_pose(pose: list[float], delta: Mapping[str, float]) -> list[float]:
    translated = list(pose)
    translated[0] += float(delta.get("x", 0.0))
    translated[1] += float(delta.get("y", 0.0))
    translated[2] += float(delta.get("z", 0.0))
    return translated


def _rotation_z(angle_deg: float) -> list[list[float]]:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _pose_from_matrix(matrix: list[list[float]]) -> dict[str, Any]:
    rotation = matrix
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2][1] - rotation[1][2]) / scale
        qy = (rotation[0][2] - rotation[2][0]) / scale
        qz = (rotation[1][0] - rotation[0][1]) / scale
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        qw = (rotation[2][1] - rotation[1][2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0][1] + rotation[1][0]) / scale
        qz = (rotation[0][2] + rotation[2][0]) / scale
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        qw = (rotation[0][2] - rotation[2][0]) / scale
        qx = (rotation[0][1] + rotation[1][0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1][2] + rotation[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        qw = (rotation[1][0] - rotation[0][1]) / scale
        qx = (rotation[0][2] + rotation[2][0]) / scale
        qy = (rotation[1][2] + rotation[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    return {
        "position_m": {"x": matrix[0][3], "y": matrix[1][3], "z": matrix[2][3]},
        "orientation_xyzw": {
            "x": qx / norm,
            "y": qy / norm,
            "z": qz / norm,
            "w": qw / norm,
        },
    }


def _matrix_from_pose(position: list[float], quaternion_xyzw: list[float]) -> list[list[float]]:
    """Build a rigid transform from a position and normalized XYZW quaternion."""

    if len(position) != 3 or len(quaternion_xyzw) != 4:
        raise RenderError("pose position/quaternion has an invalid shape")
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w), float(position[0])],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w), float(position[1])],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y), float(position[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _interpolate_matrix(
    timestamps: list[int],
    matrices: list[Any],
    timestamp_us: int,
    *,
    max_gap_us: int = MAX_INTERPOLATION_GAP_US,
) -> list[list[float]]:
    """Interpolate a rig pose without crossing an unobserved time gap."""

    if not timestamps or len(timestamps) != len(matrices):
        raise RenderError("rig trajectory timestamps and poses are invalid")
    if timestamp_us <= timestamps[0]:
        return _copy_matrix(matrices[0])
    if timestamp_us >= timestamps[-1]:
        return _copy_matrix(matrices[-1])
    right = bisect.bisect_left(timestamps, timestamp_us)
    if timestamps[right] == timestamp_us:
        return _copy_matrix(matrices[right])
    left = right - 1
    span = timestamps[right] - timestamps[left]
    if span > max_gap_us:
        raise RenderError(
            "refusing to interpolate rig pose across a source time gap: "
            f"{span}us between {timestamps[left]} and {timestamps[right]} "
            f"(limit {max_gap_us}us)"
        )
    fraction = (timestamp_us - timestamps[left]) / span if span else 0.0
    first = _pose_from_matrix(_copy_matrix(matrices[left]))
    second = _pose_from_matrix(_copy_matrix(matrices[right]))
    first_position = first["position_m"]
    second_position = second["position_m"]
    position = [
        float(first_position[axis])
        + (float(second_position[axis]) - float(first_position[axis])) * fraction
        for axis in ("x", "y", "z")
    ]
    first_quaternion = first["orientation_xyzw"]
    second_quaternion = second["orientation_xyzw"]
    quaternion = _slerp(
        [float(first_quaternion[axis]) for axis in ("x", "y", "z", "w")],
        [float(second_quaternion[axis]) for axis in ("x", "y", "z", "w")],
        fraction,
    )
    return _matrix_from_pose(position, quaternion)


def _interpolate_track_pose(
    track: Mapping[str, Any],
    timestamp_us: int,
    *,
    max_gap_us: int = MAX_INTERPOLATION_GAP_US,
) -> list[float] | None:
    timestamps = [int(value) for value in track["timestamps_us"]]
    poses = track["poses"]
    if timestamp_us < timestamps[0] or timestamp_us > timestamps[-1]:
        return None
    right = bisect.bisect_left(timestamps, timestamp_us)
    if right == 0 or timestamps[right] == timestamp_us:
        return [float(value) for value in poses[right]]
    left = right - 1
    span = timestamps[right] - timestamps[left]
    if span > max_gap_us:
        track_id = str(track.get("track_id", "unknown"))
        raise RenderError(
            "refusing to interpolate actor pose across a source time gap: "
            f"track {track_id} "
            f"{span}us between {timestamps[left]} and {timestamps[right]} "
            f"(limit {max_gap_us}us)"
        )
    fraction = (timestamp_us - timestamps[left]) / span if span else 0.0
    first = [float(value) for value in poses[left]]
    second = [float(value) for value in poses[right]]
    position = [first[index] + (second[index] - first[index]) * fraction for index in range(3)]
    quaternion = _slerp(first[3:7], second[3:7], fraction)
    return position + quaternion


def _slerp(first: list[float], second: list[float], fraction: float) -> list[float]:
    a = [float(value) for value in first]
    b = [float(value) for value in second]
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = [-value for value in b]
        dot = -dot
    if dot > 0.9995:
        result = [a[index] + fraction * (b[index] - a[index]) for index in range(4)]
        norm = math.sqrt(sum(value * value for value in result)) or 1.0
        return [value / norm for value in result]
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta) or 1.0
    left = math.sin((1.0 - fraction) * theta) / sin_theta
    right = math.sin(fraction * theta) / sin_theta
    return [left * a[index] + right * b[index] for index in range(4)]


def _apply_camera_offset(
    matrix: list[list[float]],
    *,
    translation_m: Mapping[str, float] | None = None,
    yaw_deg: float = 0.0,
    frame: str = "world",
) -> list[list[float]]:
    result = _copy_matrix(matrix)
    translation = translation_m or {}
    delta = [float(translation.get(axis, 0.0)) for axis in ("x", "y", "z")]
    yaw_rotation = _rotation_z(yaw_deg)
    if frame == "sensor":
        for row in range(3):
            result[row][3] += sum(result[row][column] * delta[column] for column in range(3))
        rotated = _mat_mul(result, yaw_rotation)
        for row in range(3):
            result[row][:3] = rotated[row][:3]
    elif frame == "world":
        for row in range(3):
            result[row][3] += delta[row]
        rotated = _mat_mul(yaw_rotation, result)
        for row in range(3):
            result[row][:3] = rotated[row][:3]
    else:
        raise RenderError("camera offset frame must be world or sensor")
    return result


def _dynamic_digest(objects: Iterable[Mapping[str, Any]]) -> str:
    normalized = []
    for item in objects:
        pose_pair = item.get("pose_pair")
        if isinstance(pose_pair, Mapping):
            normalized.append(
                {
                    "track_id": str(item["track_id"]),
                    "pose_pair": {
                        endpoint: [float(value) for value in pose_pair[endpoint]]
                        for endpoint in ("start", "end")
                    },
                }
            )
        else:
            normalized.append(
                {
                    "track_id": str(item["track_id"]),
                    "pose": [float(value) for value in item["pose"]],
                }
            )
    normalized.sort(key=lambda item: item["track_id"])
    return canonical_digest(normalized)


def _non_target_digest(objects: Iterable[Mapping[str, Any]], target_track_id: str) -> str:
    return _dynamic_digest(item for item in objects if item["track_id"] != target_track_id)


def _load_runtime_modules(python_api_path: Path) -> tuple[Any, Any, Any, Any]:
    # ROS exports /usr/lib/python3/dist-packages through PYTHONPATH on this
    # host.  That path contains an older protobuf package which shadows the
    # active conda environment and cannot import generated NuRec protos.
    # Prefer the interpreter's own site-packages before importing grpc.
    current_site_packages: list[str] = []
    try:
        current_site_packages.extend(str(value) for value in site.getsitepackages())
    except (AttributeError, TypeError):
        pass
    try:
        user_site = site.getusersitepackages()
    except (AttributeError, TypeError):
        user_site = None
    if isinstance(user_site, str):
        current_site_packages.append(user_site)
    for package_path in reversed(dict.fromkeys(current_site_packages)):
        if not Path(package_path).is_dir():
            continue
        while package_path in sys.path:
            sys.path.remove(package_path)
        sys.path.insert(0, package_path)
    path = str(python_api_path.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        grpc = importlib.import_module("grpc")
        protobuf = importlib.import_module("nre.grpc.protos.sensorsim_pb2")
        common = importlib.import_module("nre.grpc.protos.common_pb2")
        stub_module = importlib.import_module("nre.grpc.protos.sensorsim_pb2_grpc")
    except ImportError as exc:
        raise RenderError(
            "NuRec protobuf runtime is unavailable; set --python-api-path to the "
            "installed nurec Python API"
        ) from exc
    return grpc, protobuf, common, stub_module


class SensorsimClient:
    def __init__(self, address: str, runtime_scene_id: str, python_api_path: Path, timeout: float) -> None:
        grpc, protobuf, common, stub_module = _load_runtime_modules(python_api_path)
        self.protobuf = protobuf
        self.common = common
        self.timeout = timeout
        self.runtime_scene_id = runtime_scene_id
        self.channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ],
        )
        self.stub = stub_module.SensorsimServiceStub(self.channel)
        self.address = address
        self.request_sequence = 0
        self.inventory = self._query_inventory()
        self.camera_intrinsics = self._camera_intrinsics()

    def close(self) -> None:
        close = getattr(self.channel, "close", None)
        if callable(close):
            close()

    def _query_inventory(self) -> dict[str, Any]:
        empty = self.common.Empty()
        version = self.stub.get_version(empty, timeout=self.timeout)
        scenes = self.stub.get_available_scenes(empty, timeout=self.timeout)
        if self.runtime_scene_id not in list(scenes.scene_ids):
            raise RenderError(
                f"runtime scene {self.runtime_scene_id!r} is not advertised by {self.address}"
            )
        cameras = self.stub.get_available_cameras(
            self.protobuf.AvailableCamerasRequest(scene_id=self.runtime_scene_id),
            timeout=self.timeout,
        )
        rows = []
        for camera in cameras.available_cameras:
            rows.append(
                {
                    "logical_id": str(camera.logical_id),
                    "trajectory_idx": int(camera.trajectory_idx),
                    "resolution_w": int(camera.intrinsics.resolution_w),
                    "resolution_h": int(camera.intrinsics.resolution_h),
                }
            )
        api = version.grpc_api_version
        return {
            "target": self.address,
            "runtime_scene_id": self.runtime_scene_id,
            "available_scene_ids": sorted(str(value) for value in scenes.scene_ids),
            "renderer": {
                "version_id": str(version.version_id),
                "git_hash": str(version.git_hash),
                "grpc_api_version": {
                    "major": int(api.major),
                    "minor": int(api.minor),
                    "patch": int(api.patch),
                },
            },
            "cameras": sorted(rows, key=lambda item: item["logical_id"]),
        }

    def _camera_intrinsics(self) -> dict[str, Any]:
        response = self.stub.get_available_cameras(
            self.protobuf.AvailableCamerasRequest(scene_id=self.runtime_scene_id),
            timeout=self.timeout,
        )
        return {str(item.logical_id): item.intrinsics for item in response.available_cameras}

    def _pose_message(self, pose: Mapping[str, Any]) -> Any:
        position = pose["position_m"]
        orientation = pose["orientation_xyzw"]
        return self.common.Pose(
            vec=self.common.Vec3(
                x=float(position["x"]), y=float(position["y"]), z=float(position["z"])
            ),
            quat=self.common.Quat(
                w=float(orientation["w"]),
                x=float(orientation["x"]),
                y=float(orientation["y"]),
                z=float(orientation["z"]),
            ),
        )

    def _dynamic_object_messages(self, dynamic_objects: Iterable[Mapping[str, Any]]) -> list[Any]:
        messages = []
        for item in dynamic_objects:
            pose_pair_value = item.get("pose_pair")
            if isinstance(pose_pair_value, Mapping):
                start_track_pose = pose_pair_value.get("start")
                end_track_pose = pose_pair_value.get("end")
            else:
                start_track_pose = item.get("pose")
                end_track_pose = start_track_pose
            if not isinstance(start_track_pose, list) or not isinstance(end_track_pose, list):
                raise RenderError(f"dynamic object {item.get('track_id')} has no valid pose pair")
            messages.append(
                self.protobuf.DynamicObject(
                    track_id=str(item["track_id"]),
                    pose_pair=self.protobuf.PosePair(
                        start_pose=self._pose_message(_pose_from_track(start_track_pose)),
                        end_pose=self._pose_message(_pose_from_track(end_track_pose)),
                    ),
                )
            )
        return messages

    def render_rgb(
        self,
        *,
        camera_id: str,
        width: int,
        height: int,
        start_us: int,
        end_us: int,
        start_pose: Mapping[str, Any],
        end_pose: Mapping[str, Any],
        dynamic_objects: list[Mapping[str, Any]],
        frame_id: str | None = None,
    ) -> dict[str, Any]:
        if camera_id not in self.camera_intrinsics:
            raise RenderError(f"runtime does not advertise camera {camera_id}")
        logical_end_us = int(max(start_us + 1, end_us))
        # A 40k temporal Gaussian field should be sampled instantaneously.
        # Keep the logical pose interval in the sensor/actor PosePairs, but
        # send the interval midpoint as a one-microsecond RGB render window.
        wire_start_us = int((int(start_us) + logical_end_us) // 2)
        wire_end_us = wire_start_us + 1
        pose_pair = self.protobuf.PosePair(
            start_pose=self._pose_message(start_pose), end_pose=self._pose_message(end_pose)
        )
        request = self.protobuf.RGBRenderRequest(
            scene_id=self.runtime_scene_id,
            resolution_h=int(height),
            resolution_w=int(width),
            camera_intrinsics=self.camera_intrinsics[camera_id],
            frame_start_us=wire_start_us,
            frame_end_us=wire_end_us,
            sensor_pose=pose_pair,
            dynamic_objects=self._dynamic_object_messages(dynamic_objects),
            image_format=self.protobuf.JPEG,
            image_quality=95.0,
        )
        request_digest = sha256_bytes(request.SerializeToString())
        self.request_sequence = int(getattr(self, "request_sequence", 0)) + 1
        request_sequence = self.request_sequence
        request_sent_unix_ns = time.time_ns()
        started = time.perf_counter()
        try:
            response = self.stub.render_rgb(request, timeout=self.timeout)
            response_received_unix_ns = time.time_ns()
            body = bytes(response.image_bytes)
            if not body:
                raise RenderError("RGB response image_bytes is empty")
            width_observed, height_observed = _jpeg_dimensions(body)
            if width_observed != width or height_observed != height:
                raise RenderError(
                    f"RGB response dimensions {(width_observed, height_observed)} "
                    f"!= requested {(width, height)}"
                )
            return {
                "status": "passed",
                "request_digest": request_digest,
                "response_digest": sha256_bytes(response.SerializeToString()),
                "rgb_payload_sha256": sha256_bytes(body),
                "rgb_bytes": body,
                "decoded_width": width_observed,
                "decoded_height": height_observed,
                "logical_frame_start_us": int(start_us),
                "logical_frame_end_us": logical_end_us,
                "wire_frame_start_us": wire_start_us,
                "wire_frame_end_us": wire_end_us,
                "request_frame_id": frame_id,
                "request_sequence": request_sequence,
                "request_sent_unix_ns": request_sent_unix_ns,
                "response_received_unix_ns": response_received_unix_ns,
                "response_frame_id": None,
                "response_timestamp_us": None,
                "realized_timestamp_us": None,
                "realized_timestamp_status": "unavailable: RGBRenderReturn exposes image_bytes only",
                "rpc_latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        except Exception as exc:
            response_received_unix_ns = time.time_ns()
            return {
                "status": "error",
                "request_digest": request_digest,
                "error": f"{type(exc).__name__}: {exc}",
                "request_frame_id": frame_id,
                "request_sequence": request_sequence,
                "request_sent_unix_ns": request_sent_unix_ns,
                "response_received_unix_ns": response_received_unix_ns,
                "response_frame_id": None,
                "response_timestamp_us": None,
                "realized_timestamp_us": None,
                "realized_timestamp_status": "unavailable: RGBRenderReturn exposes image_bytes only",
                "rpc_latency_ms": (time.perf_counter() - started) * 1000.0,
            }

    def render_lidar(
        self,
        *,
        lidar_id: str,
        device_type: str,
        start_us: int,
        end_us: int,
        start_pose: Mapping[str, Any],
        end_pose: Mapping[str, Any],
        dynamic_objects: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        device_name = str(device_type).upper()
        if device_name not in {"PANDAR128", "AT128"}:
            raise RenderError("LiDAR device type must be PANDAR128 or AT128")
        logical_end_us = int(max(start_us + 1, end_us))
        request = self.protobuf.LidarRenderRequest(
            scene_id=self.runtime_scene_id,
            lidar_config=self.protobuf.LidarSpec(
                lidar_type=getattr(self.protobuf, device_name)
            ),
            frame_start_us=int(start_us),
            frame_end_us=logical_end_us,
            sensor_pose=self.protobuf.PosePair(
                start_pose=self._pose_message(start_pose),
                end_pose=self._pose_message(end_pose),
            ),
            dynamic_objects=self._dynamic_object_messages(dynamic_objects),
        )
        request_digest = sha256_bytes(request.SerializeToString())
        started = time.perf_counter()
        try:
            response = self.stub.render_lidar(request, timeout=self.timeout)
            xyz, intensities, response_encoding = _decode_lidar_response(response)
            point_count = len(intensities)

            xyzi = bytearray()
            for index in range(point_count):
                xyzi.extend(
                    struct.pack(
                        "<4f",
                        xyz[index * 3],
                        xyz[index * 3 + 1],
                        xyz[index * 3 + 2],
                        intensities[index],
                    )
                )
            xyzi_bytes = bytes(xyzi)
            return {
                "status": "passed",
                "request_digest": request_digest,
                "response_digest": sha256_bytes(response.SerializeToString()),
                "lidar_payload_sha256": sha256_bytes(xyzi_bytes),
                "xyzi_bytes": xyzi_bytes,
                "point_xyzs": xyz,
                "point_intensities": intensities,
                "point_count": point_count,
                "response_encoding": response_encoding,
                "logical_frame_start_us": int(start_us),
                "logical_frame_end_us": logical_end_us,
                "rpc_latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        except Exception as exc:
            return {
                "status": "error",
                "request_digest": request_digest,
                "error": f"{type(exc).__name__}: {exc}",
                "rpc_latency_ms": (time.perf_counter() - started) * 1000.0,
            }


def _pose_from_track(pose: list[float]) -> dict[str, Any]:
    if not isinstance(pose, list) or len(pose) != 7:
        raise RenderError("dynamic object pose must contain 7 values")
    return {
        "position_m": {"x": pose[0], "y": pose[1], "z": pose[2]},
        "orientation_xyzw": {"x": pose[3], "y": pose[4], "z": pose[5], "w": pose[6]},
    }


def _jpeg_dimensions(body: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(BytesIO(body)) as image:
            image.verify()
        with Image.open(BytesIO(body)) as image:
            return int(image.width), int(image.height)
    except Exception as exc:
        raise RenderError(f"RGB response is not a decodable image: {exc}") from exc


def _image_metrics(body: bytes) -> dict[str, float]:
    from PIL import Image

    import numpy as np

    import cv2

    with Image.open(BytesIO(body)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "invalid_pixel_ratio": 0.0,
        "dark_pixel_ratio": float((gray < 8).mean()),
        "laplacian_sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_luma": float(gray.mean()),
    }


def _apply_sweep_offset(case: Mapping[str, Any], progress: float) -> tuple[dict[str, float], float]:
    sweep = case.get("camera_sweep")
    if not isinstance(sweep, Mapping):
        return {"x": 0.0, "y": 0.0, "z": 0.0}, 0.0
    profile = str(sweep.get("profile", "sinusoidal"))
    amount = math.sin(math.pi * progress) if profile == "sinusoidal" else progress
    translation = sweep.get("translation_m") or {}
    translation = {
        axis: float(translation.get(axis, 0.0)) * amount for axis in ("x", "y", "z")
    }
    yaw = float(sweep.get("yaw_deg", 0.0)) * amount
    return translation, yaw


def _validate_sweep_bounds(case: Mapping[str, Any]) -> None:
    sweep = case.get("camera_sweep")
    if not isinstance(sweep, Mapping):
        return
    translation = sweep.get("translation_m") or {}
    magnitude = math.sqrt(sum(float(translation.get(axis, 0.0)) ** 2 for axis in ("x", "y", "z")))
    rotation = max(abs(float(sweep.get(key, 0.0))) for key in ("roll_deg", "pitch_deg", "yaw_deg"))
    limits = case.get("limits") or {}
    max_translation = float(limits.get("max_translation_m", 0.2))
    max_rotation = float(limits.get("max_rotation_deg", 2.0))
    if magnitude > max_translation + 1e-9:
        raise RenderError(f"camera sweep translation {magnitude:.6f}m exceeds {max_translation:.6f}m")
    if rotation > max_rotation + 1e-9:
        raise RenderError(f"camera sweep rotation {rotation:.6f}deg exceeds {max_rotation:.6f}deg")
    if case.get("intrinsics_edit"):
        raise RenderError("camera pose case must not modify camera intrinsics")


def _camera_ids_for_case(case: Mapping[str, Any], override: str | None) -> list[str]:
    """Resolve a deterministic camera list, retaining the ClosedLoopBench order."""

    if override:
        raw_values: object = override.replace(",", " ").split()
    else:
        raw_values = case.get("camera_ids") or [case.get("camera_id") or "camera_front"]
    if not isinstance(raw_values, (list, tuple)):
        raise RenderError("camera_ids must be a list or comma-separated override")
    camera_ids = [str(value).split("@", 1)[0].strip() for value in raw_values]
    if not camera_ids or any(not value for value in camera_ids):
        raise RenderError("camera_ids must contain non-empty values")
    if len(set(camera_ids)) != len(camera_ids):
        raise RenderError("camera_ids must not contain duplicates")
    return camera_ids


def _camera_grid_for_case(
    case: Mapping[str, Any], camera_ids: list[str]
) -> tuple[int, int, bool]:
    config = case.get("camera_grid")
    if config is None:
        return len(camera_ids), 1, False
    if not isinstance(config, Mapping):
        raise RenderError("camera_grid must be an object")
    columns = int(config.get("columns", DEFAULT_CAMERA_GRID["columns"]))
    rows = int(config.get("rows", DEFAULT_CAMERA_GRID["rows"]))
    label_cameras = bool(config.get("label_cameras", True))
    if columns <= 0 or rows <= 0 or len(camera_ids) != columns * rows:
        raise RenderError(
            f"camera grid {columns}x{rows} cannot contain {len(camera_ids)} cameras"
        )
    if len(camera_ids) == len(FORMAL_CAMERA_ORDER) and tuple(camera_ids) != FORMAL_CAMERA_ORDER:
        raise RenderError(
            "formal six-camera output must use front-left, front, front-right, "
            "back-left, back, back-right order"
        )
    return columns, rows, label_cameras


def _timestamp_values(
    scene: ArtifactScene,
    case: Mapping[str, Any],
    start_override: int | None,
    end_override: int | None,
    frame_step_override: int | None,
    sample_fps_override: float | None = None,
) -> list[int]:
    case_range = case.get("timestamp_range_us") or {}
    start = int(start_override if start_override is not None else case_range.get("start", scene.rig_timestamps[0]))
    end = int(end_override if end_override is not None else case_range.get("end", scene.rig_timestamps[-1]))
    if start > end:
        raise RenderError("start timestamp must not exceed end timestamp")
    if frame_step_override is None and sample_fps_override is None:
        configured_fps = case.get("sample_fps")
        if configured_fps is not None:
            sample_fps_override = float(configured_fps)
    if frame_step_override is None and sample_fps_override is not None:
        sample_fps = float(sample_fps_override)
        if not math.isfinite(sample_fps) or sample_fps <= 0.0:
            raise RenderError("sample-fps must be positive and finite")
        period_us = 1_000_000.0 / sample_fps
        values = [start]
        index = 1
        while start + round(index * period_us) < end:
            values.append(start + round(index * period_us))
            index += 1
        if values[-1] != end:
            values.append(end)
        if values[-1] > scene.rig_timestamps[-1]:
            raise RenderError("sampled timestamp range exceeds the artifact rig trajectory")
        return values
    step = int(frame_step_override if frame_step_override is not None else case.get("frame_step", 1))
    if step <= 0:
        raise RenderError("frame-step must be positive")
    indices = [
        index
        for index, timestamp in enumerate(scene.rig_timestamps)
        if start <= timestamp <= end
    ]
    if not indices:
        raise RenderError("requested timestamp range has no artifact rig poses")
    return [scene.rig_timestamps[index] for index in indices[::step]]


def _make_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise RenderError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise RenderError(
                    f"refusing to overwrite non-empty output directory: {path}; "
                    "choose a new run directory or pass --overwrite explicitly"
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _resume_capture(
    output_dir: Path,
    metadata_path: Path,
    timestamps: list[int],
    camera_ids: list[str],
    *,
    width: int,
    height: int,
    output_width: int,
    output_height: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    if not output_dir.is_dir() or not metadata_path.is_file():
        raise RenderError("resume requires an existing output directory with frames.jsonl")

    records: list[dict[str, Any]] = []
    frame_paths: list[Path] = []
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RenderError(f"cannot read resume metadata {metadata_path}: {exc}") from exc
    if not lines:
        raise RenderError("resume metadata is empty")
    if len(lines) > len(timestamps):
        raise RenderError("resume metadata contains more frames than the requested capture")

    def checked_jpeg(relative_value: object, expected_size: tuple[int, int]) -> Path:
        if not isinstance(relative_value, str) or not relative_value:
            raise RenderError("resume metadata contains an invalid JPEG path")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RenderError(f"resume JPEG path must stay inside the output directory: {relative}")
        path = output_dir / relative
        if not path.is_file():
            raise RenderError(f"resume JPEG is missing: {path}")
        observed = _jpeg_dimensions(path.read_bytes())
        if observed != expected_size:
            raise RenderError(
                f"resume JPEG dimensions {observed} != {expected_size}: {path}"
            )
        return path

    for expected_index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RenderError(
                f"resume metadata line {expected_index + 1} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise RenderError(f"resume metadata line {expected_index + 1} is not an object")
        if int(record.get("frame_index", -1)) != expected_index:
            raise RenderError("resume frame indices are not contiguous from zero")
        if int(record.get("scene_timestamp_us", -1)) != int(timestamps[expected_index]):
            raise RenderError(f"resume timestamp differs at frame {expected_index}")
        if record.get("status") != "passed" or bool(record.get("dropped")):
            raise RenderError(f"resume frame {expected_index} is not a passed frame")
        if list(record.get("camera_ids") or []) != camera_ids:
            raise RenderError(f"resume camera order differs at frame {expected_index}")
        frame_path = checked_jpeg(
            record.get("frame_path"), (output_width, output_height)
        )
        camera_paths = record.get("camera_frame_paths")
        if not isinstance(camera_paths, Mapping):
            raise RenderError(f"resume camera paths are missing at frame {expected_index}")
        for camera_id in camera_ids:
            checked_jpeg(camera_paths.get(camera_id), (width, height))
        records.append(record)
        frame_paths.append(frame_path)
    return records, frame_paths


def _stitch_camera_frames(
    bodies: Mapping[str, bytes],
    camera_ids: list[str],
    *,
    width: int,
    height: int,
    columns: int,
    rows: int,
    label_cameras: bool,
    pose_overlay: Mapping[str, Any] | None = None,
    pose_progress: float | None = None,
    pose_frame: str = "sensor",
    pose_profile: str = "sinusoidal",
    pose_camera_count: int | None = None,
) -> bytes:
    """Create the 3x2 presentation grid used by ClosedLoopBench."""

    import cv2
    import numpy as np

    cells: list[Any] = []
    for camera_id in camera_ids:
        encoded = np.frombuffer(bodies[camera_id], dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            observed = None if image is None else (image.shape[1], image.shape[0])
            raise RenderError(
                f"camera {camera_id} decoded dimensions {observed} != {(width, height)}"
            )
        cell = image.copy()
        if label_cameras:
            overlay = cell.copy()
            cv2.rectangle(overlay, (0, 0), (width - 1, 28), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.62, cell, 0.38, 0.0, cell)
            cv2.putText(
                cell,
                camera_id,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cells.append(cell)
    grid_rows = [
        np.concatenate(cells[row * columns : (row + 1) * columns], axis=1)
        for row in range(rows)
    ]
    grid = np.concatenate(grid_rows, axis=0)
    if pose_overlay is not None:
        _draw_pose_sweep_overlay(
            grid,
            pose_overlay,
            progress=pose_progress,
            frame=pose_frame,
            profile=pose_profile,
            camera_count=pose_camera_count or len(camera_ids),
        )
    ok, encoded_grid = cv2.imencode(
        ".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )
    if not ok:
        raise RenderError("failed to encode stitched camera grid")
    return bytes(encoded_grid)


def _draw_pose_sweep_overlay(
    grid: Any,
    offset: Mapping[str, Any],
    *,
    progress: float | None,
    frame: str,
    profile: str,
    camera_count: int,
) -> None:
    """Draw an explicit pose-sweep readout without changing the grid size."""

    import cv2

    height, width = grid.shape[:2]
    translation = offset.get("translation_m")
    if not isinstance(translation, Mapping):
        translation = {}
    values = {
        axis: float(translation.get(axis, 0.0)) for axis in ("x", "y", "z")
    }
    yaw = float(offset.get("yaw_deg", 0.0))
    progress_value = min(1.0, max(0.0, float(progress or 0.0)))
    # Keep the strip inside the existing 900 px output contract.  A solid
    # dark backing makes the numbers readable over both sky and road pixels.
    strip_height = 62
    top = max(0, height - strip_height)
    overlay = grid.copy()
    cv2.rectangle(overlay, (0, top), (width - 1, height - 1), (8, 14, 20), -1)
    cv2.addWeighted(overlay, 0.82, grid, 0.18, 0.0, grid)

    title = (
        f"CAMERA POSE SWEEP  |  {camera_count} CAMERAS  |  "
        f"{str(frame).upper()} FRAME  |  {str(profile).upper()}"
    )
    details = (
        f"dx {values['x']:+.3f} m   dy {values['y']:+.3f} m   "
        f"dz {values['z']:+.3f} m   yaw {yaw:+.3f} deg   "
        f"progress {progress_value * 100:05.1f}%"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        grid,
        title,
        (18, top + 23),
        font,
        0.62,
        (245, 250, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        grid,
        details,
        (18, top + 49),
        font,
        0.66,
        (96, 226, 255),
        1,
        cv2.LINE_AA,
    )

    # A compact progress rail makes the sinusoidal sweep visible even when
    # the camera view itself contains few static landmarks.
    rail_left = max(0, width - 430)
    rail_right = width - 24
    rail_y = top + 48
    cv2.line(grid, (rail_left, rail_y), (rail_right, rail_y), (95, 105, 115), 3, cv2.LINE_AA)
    marker_x = rail_left + int(round((rail_right - rail_left) * progress_value))
    cv2.circle(grid, (marker_x, rail_y), 7, (0, 214, 255), -1, cv2.LINE_AA)


def _encode_mp4(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    import cv2

    if not frame_paths:
        raise RenderError("cannot encode a video with no frames")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RenderError(f"cannot decode first captured frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RenderError("OpenCV could not open an MP4 writer")
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (height, width):
                raise RenderError(f"captured frame has invalid dimensions: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RenderError("MP4 encoder produced no output")


def _pose_sweep_samples(frame_records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = list(frame_records)
    denominator = max(1, len(records) - 1)
    samples: list[dict[str, Any]] = []
    for row in records:
        offset = row.get("camera_sweep_offset")
        if not isinstance(offset, Mapping):
            continue
        raw_translation = offset.get("translation_m")
        translation = raw_translation if isinstance(raw_translation, Mapping) else {}
        values = {
            axis: float(translation.get(axis, 0.0)) for axis in ("x", "y", "z")
        }
        frame_index = int(row.get("frame_index", len(samples)))
        samples.append(
            {
                "frame_index": frame_index,
                "progress": frame_index / denominator,
                "translation_m": values,
                "yaw_deg": float(offset.get("yaw_deg", 0.0)),
            }
        )
    return samples


def _write_pose_sweep_summary(
    output_dir: Path,
    case: Mapping[str, Any],
    frame_records: Iterable[Mapping[str, Any]],
    video_path: Path,
) -> tuple[Path, Path | None] | None:
    """Persist inspectable numeric and graphical summaries for a pose sweep."""

    sweep = case.get("camera_sweep")
    if not isinstance(sweep, Mapping):
        return None
    samples = _pose_sweep_samples(frame_records)
    if not samples:
        return None
    translation_magnitude = [
        math.sqrt(sum(sample["translation_m"][axis] ** 2 for axis in ("x", "y", "z")))
        for sample in samples
    ]
    max_translation_index = max(range(len(samples)), key=translation_magnitude.__getitem__)
    max_yaw_index = max(range(len(samples)), key=lambda index: abs(samples[index]["yaw_deg"]))
    max_translation = samples[max_translation_index]
    max_yaw = samples[max_yaw_index]
    summary = {
        "schema_version": "nsb.nurec-camera-pose-sweep-summary.v1",
        "case_id": case.get("case_id"),
        "video": str(video_path),
        "camera_ids": list(case.get("camera_ids") or []),
        "camera_count": len(case.get("camera_ids") or []),
        "frame": str(sweep.get("frame", "world")),
        "profile": str(sweep.get("profile", "sinusoidal")),
        "apply_to_all_cameras": bool(sweep.get("apply_to_all_cameras", False)),
        "sample_count": len(samples),
        "start": samples[0],
        "midpoint": samples[len(samples) // 2],
        "end": samples[-1],
        "max_translation": {
            "magnitude_m": translation_magnitude[max_translation_index],
            "sample": max_translation,
        },
        "max_yaw": {
            "absolute_deg": abs(max_yaw["yaw_deg"]),
            "sample": max_yaw,
        },
        "trajectory": samples,
    }
    json_path = output_dir / "pose_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    png_path: Path | None = output_dir / "pose_sweep_summary.png"
    try:
        import cv2
        import numpy as np

        canvas = np.full((760, 1600, 3), (24, 30, 37), dtype=np.uint8)
        cv2.putText(
            canvas,
            "CAMERA POSE SWEEP SUMMARY",
            (52, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (245, 250, 255),
            2,
            cv2.LINE_AA,
        )
        subtitle = (
            f"{summary['profile']} / {summary['frame']} frame / "
            f"{summary['camera_count']} cameras / {summary['sample_count']} samples"
        )
        cv2.putText(
            canvas,
            subtitle,
            (54, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (160, 178, 192),
            1,
            cv2.LINE_AA,
        )

        def draw_plot(
            rect: tuple[int, int, int, int],
            title: str,
            series: list[tuple[str, list[float], tuple[int, int, int]]],
            unit: str,
        ) -> None:
            left, top, plot_width, plot_height = rect
            right = left + plot_width
            bottom = top + plot_height
            cv2.rectangle(canvas, (left, top), (right, bottom), (67, 78, 90), 1)
            cv2.putText(
                canvas,
                title,
                (left + 14, top - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.67,
                (225, 235, 242),
                1,
                cv2.LINE_AA,
            )
            all_values = [value for _, values, _ in series for value in values]
            limit = max(0.01, max(abs(value) for value in all_values) * 1.15)
            zero_y = top + plot_height // 2
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                y = top + int(round(plot_height * fraction))
                cv2.line(canvas, (left, y), (right, y), (45, 54, 64), 1)
            cv2.line(canvas, (left, zero_y), (right, zero_y), (95, 105, 115), 1)
            cv2.putText(
                canvas,
                f"+{limit:.3f} {unit}",
                (right - 160, top + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (150, 165, 178),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"-{limit:.3f} {unit}",
                (right - 160, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (150, 165, 178),
                1,
                cv2.LINE_AA,
            )
            for series_index, (label, values, color) in enumerate(series):
                points = []
                for index, value in enumerate(values):
                    x = left + int(round(index * plot_width / max(1, len(values) - 1)))
                    y = zero_y - int(round(value / limit * (plot_height / 2)))
                    points.append((x, y))
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
                legend_x = left + 16 + 105 * series_index
                cv2.line(canvas, (legend_x, bottom + 25), (legend_x + 24, bottom + 25), color, 3, cv2.LINE_AA)
                cv2.putText(canvas, label, (legend_x + 32, bottom + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 1, cv2.LINE_AA)

        draw_plot(
            (70, 185, 1460, 210),
            "Translation offset",
            [
                ("dx", [sample["translation_m"]["x"] for sample in samples], (96, 226, 255)),
                ("dy", [sample["translation_m"]["y"] for sample in samples], (104, 235, 145)),
                ("dz", [sample["translation_m"]["z"] for sample in samples], (220, 180, 90)),
            ],
            "m",
        )
        draw_plot(
            (70, 500, 1460, 150),
            "Yaw offset",
            [("yaw", [sample["yaw_deg"] for sample in samples], (0, 214, 255))],
            "deg",
        )
        max_text = (
            f"MAX TRANSLATION {summary['max_translation']['magnitude_m']:.3f} m  |  "
            f"MAX YAW {summary['max_yaw']['absolute_deg']:.3f} deg  |  "
            f"MIDPOINT dx {summary['midpoint']['translation_m']['x']:+.3f} m, "
            f"dy {summary['midpoint']['translation_m']['y']:+.3f} m"
        )
        cv2.putText(canvas, max_text, (70, 718), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (96, 226, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(png_path), canvas):
            png_path = None
    except Exception:
        png_path = None
    return json_path, png_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, help="canonical USDZ; defaults to manifest")
    parser.add_argument("--server-address", help="SensorsimService host:port")
    parser.add_argument("--case", required=True, help="case ID or JSON path")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-id", help="override case camera ID")
    parser.add_argument(
        "--camera-ids",
        help="override case camera IDs with a comma/space-separated list",
    )
    parser.add_argument("--start-timestamp", type=int)
    parser.add_argument("--end-timestamp", type=int)
    parser.add_argument("--frame-step", type=int)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="validate and continue a contiguous frames.jsonl capture",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runtime-scene-id")
    parser.add_argument("--python-api-path", type=Path)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--max-actor-interpolation-gap-us",
        type=int,
        default=MAX_INTERPOLATION_GAP_US,
        help=(
            "maximum actor and rig source-pose gap allowed for interpolation; "
            "values above the strict 75 ms default are playback-only"
        ),
    )
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--fps", type=float, help="encoded video FPS; defaults to case sample_fps")
    parser.add_argument("--probe-only", action="store_true")
    return parser


def _camera_pose_pair(
    scene: ArtifactScene,
    case: Mapping[str, Any],
    camera_id: str,
    timestamp_us: int,
    end_timestamp_us: int,
    progress: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_matrix = scene.sensor_pose_matrix(camera_id, timestamp_us)
    end_matrix = scene.sensor_pose_matrix(camera_id, end_timestamp_us)
    sweep = case.get("camera_sweep")
    if isinstance(sweep, Mapping):
        translation, yaw = _apply_sweep_offset(case, progress)
        frame = str(sweep.get("frame", "world"))
        start_matrix = _apply_camera_offset(
            start_matrix, translation_m=translation, yaw_deg=yaw, frame=frame
        )
        end_matrix = _apply_camera_offset(
            end_matrix, translation_m=translation, yaw_deg=yaw, frame=frame
        )
    return _pose_from_matrix(start_matrix), _pose_from_matrix(end_matrix)


def _probe_responses(
    client: SensorsimClient,
    scene: ArtifactScene,
    camera_ids: list[str],
    *,
    width: int,
    height: int,
    timestamp_us: int,
    dynamic_objects: list[Mapping[str, Any]],
    offset: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    responses: dict[str, dict[str, Any]] = {}
    offset = offset or {}
    for camera_id in camera_ids:
        matrix = scene.sensor_pose_matrix(camera_id, timestamp_us)
        if offset:
            matrix = _apply_camera_offset(
                matrix,
                translation_m=offset.get("translation_m") or {},
                yaw_deg=float(offset.get("yaw_deg", 0.0)),
                frame=str(offset.get("frame", "world")),
            )
        pose = _pose_from_matrix(matrix)
        responses[camera_id] = client.render_rgb(
            camera_id=camera_id,
            width=width,
            height=height,
            start_us=timestamp_us,
            end_us=timestamp_us + 1,
            start_pose=pose,
            end_pose=pose,
            dynamic_objects=dynamic_objects,
            frame_id=f"probe:{timestamp_us}:{camera_id}",
        )
    return responses


def render_case(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = _resolve_repo_path(args.manifest)
    manifest = load_json(manifest_path)
    case_path, case = resolve_case(args.case)
    if case.get("schema_version") != "nsb.nurec-counterfactual-case.v1":
        raise RenderError(f"unsupported case schema: {case.get('schema_version')!r}")
    case_id = str(case.get("case_id") or "")
    if case_id not in VIDEO_NAMES:
        raise RenderError(f"unsupported case_id: {case_id}")
    if args.camera_id and args.camera_ids:
        raise RenderError("--camera-id and --camera-ids cannot be used together")
    artifact, artifact_entry, checkpoint, checkpoint_entry = validate_manifest_identity(
        manifest, args.artifact
    )
    required_track = str(manifest.get("target_track_id") or TARGET_TRACK_ID)
    if required_track != TARGET_TRACK_ID:
        raise RenderError("manifest target_track_id does not match the interview-demo target")
    scene = ArtifactScene(artifact, required_track)
    if args.max_actor_interpolation_gap_us < MAX_INTERPOLATION_GAP_US:
        raise RenderError(
            f"max-actor-interpolation-gap-us must be at least {MAX_INTERPOLATION_GAP_US}"
        )
    scene.max_actor_interpolation_gap_us = int(args.max_actor_interpolation_gap_us)
    scene.max_rig_interpolation_gap_us = int(args.max_actor_interpolation_gap_us)
    _validate_sweep_bounds(case)
    camera_override = args.camera_ids or args.camera_id
    camera_ids = _camera_ids_for_case(case, camera_override)
    columns, rows, label_cameras = _camera_grid_for_case(case, camera_ids)
    for camera_id in camera_ids:
        camera_key = f"{camera_id}@{scene.trajectory.get('sequence_id', 'scene-0061')}"
        if camera_key not in scene.camera_calibrations:
            raise RenderError(f"case camera is absent from artifact: {camera_id}")
    timestamps = _timestamp_values(
        scene,
        case,
        args.start_timestamp,
        args.end_timestamp,
        args.frame_step,
        args.sample_fps,
    )
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), Mapping) else {}
    server_address = str(args.server_address or runtime.get("server_address") or "127.0.0.1:46443")
    runtime_scene_id = str(
        args.runtime_scene_id
        or runtime.get("runtime_scene_id")
        or case.get("runtime_scene_id")
        or "scene-0061"
    )
    python_api_path = _resolve_repo_path(
        args.python_api_path or os.environ.get("NUREC_PYTHON_API_PATH") or DEFAULT_PYTHON_API_PATH
    )
    output_dir = _resolve_repo_path(args.output_dir)
    if args.resume:
        if not output_dir.is_dir():
            raise RenderError(f"resume output directory does not exist: {output_dir}")
    else:
        _make_output_dir(output_dir, bool(args.overwrite))
    frames_dir = output_dir / "frames"
    camera_frames_dir = output_dir / "camera_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    camera_frames_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "frames.jsonl"
    evidence_path = output_dir / "evidence.json"
    video_name = str(case.get("video_name") or VIDEO_NAMES[case_id])
    video_path = output_dir / video_name
    width = int((case.get("resolution") or {}).get("width", 800))
    height = int((case.get("resolution") or {}).get("height", 450))
    if width <= 0 or height <= 0:
        raise RenderError("case resolution must be positive")
    output_width, output_height = width * columns, height * rows
    sample_fps = float(
        args.sample_fps
        if args.sample_fps is not None
        else case.get("sample_fps", 0.0)
    )
    sampling_mode = "uniform_timestamp_sampling"
    if args.frame_step is not None:
        deltas = [
            right - left
            for left, right in zip(timestamps, timestamps[1:])
            if right > left
        ]
        sample_fps = 1_000_000.0 / statistics.median(deltas) if deltas else 0.0
        sampling_mode = "artifact_timestamps"
    video_fps = float(
        args.fps
        if args.fps is not None
        else case.get("video_fps", sample_fps if sample_fps > 0.0 else 20.0)
    )
    if not math.isfinite(video_fps) or video_fps <= 0.0:
        raise RenderError("fps must be positive and finite")
    dynamic_cfg = case.get("dynamic_objects") or {}
    dynamic_mode = str(dynamic_cfg.get("mode", "controllable"))
    edit_cfg = case.get("lead_vehicle_edit") if case_id == "V02" else None
    # A target-only replay is an explicit diagnostic variant.  It keeps the
    # source target pose timeline while avoiding unsupported interpolation of
    # unrelated controllable actors; the evidence records this reduced scope.
    target_track_id = (
        required_track
        if isinstance(edit_cfg, Mapping) or dynamic_mode == "target_only"
        else None
    )
    if isinstance(edit_cfg, Mapping) and str(edit_cfg.get("track_id")) != required_track:
        raise RenderError("V02 may only edit the manifest target track")
    target_delta = (edit_cfg or {}).get("translation_m") if isinstance(edit_cfg, Mapping) else None
    if target_delta is not None and not isinstance(target_delta, Mapping):
        raise RenderError("lead_vehicle_edit.translation_m must be an object")
    capture_info = {
        "camera_ids": list(camera_ids),
        "camera_order": list(camera_ids),
        "camera_count": len(camera_ids),
        "source_resolution": {"width": width, "height": height},
        "video_resolution": {"width": output_width, "height": output_height},
        "grid": {"columns": columns, "rows": rows, "label_cameras": label_cameras},
        "sample_fps": sample_fps if sample_fps > 0.0 else None,
        "sampling_mode": sampling_mode,
        "video_fps": video_fps,
        "actor_interpolation": {
            "max_gap_us": scene.max_actor_interpolation_gap_us,
            "strict_default_gap_us": MAX_INTERPOLATION_GAP_US,
            "playback_only_relaxation": (
                scene.max_actor_interpolation_gap_us > MAX_INTERPOLATION_GAP_US
            ),
        },
        "rig_interpolation": {
            "max_gap_us": scene.max_rig_interpolation_gap_us,
            "strict_default_gap_us": MAX_INTERPOLATION_GAP_US,
            "playback_only_relaxation": (
                scene.max_rig_interpolation_gap_us > MAX_INTERPOLATION_GAP_US
            ),
        },
        "pose_overlay": {
            "enabled": isinstance(case.get("camera_sweep"), Mapping),
            "target": "stitched_grid",
            "raw_camera_frames_unchanged": True,
        },
    }
    sweep_config = case.get("camera_sweep")

    if args.resume:
        frame_records, frame_paths = _resume_capture(
            output_dir,
            metadata_path,
            timestamps,
            camera_ids,
            width=width,
            height=height,
            output_width=output_width,
            output_height=output_height,
        )
    else:
        frame_records = []
        frame_paths = []
    resumed_frame_count = len(frame_records)
    capture_info["resumed_frame_count"] = resumed_frame_count

    client = SensorsimClient(server_address, runtime_scene_id, python_api_path, float(args.timeout_sec))
    dropped = 0
    probe_evidence: dict[str, Any] | None = None
    try:
        if case_id == "V02":
            probe_timestamp = int(case.get("probe_timestamp_us", timestamps[0]))
            baseline_objects = scene.dynamic_objects(probe_timestamp, mode=dynamic_mode)
            moved_objects = scene.dynamic_objects(
                probe_timestamp,
                mode=dynamic_mode,
                target_track_id=target_track_id,
                target_delta=target_delta,
            )
            a1 = _probe_responses(
                client,
                scene,
                camera_ids,
                width=width,
                height=height,
                timestamp_us=probe_timestamp,
                dynamic_objects=baseline_objects,
            )
            a2 = _probe_responses(
                client,
                scene,
                camera_ids,
                width=width,
                height=height,
                timestamp_us=probe_timestamp,
                dynamic_objects=baseline_objects,
            )
            b = _probe_responses(
                client,
                scene,
                camera_ids,
                width=width,
                height=height,
                timestamp_us=probe_timestamp,
                dynamic_objects=moved_objects,
            )
            a_ok = all(response.get("status") == "passed" for response in a1.values()) and all(
                response.get("status") == "passed" for response in a2.values()
            )
            b_ok = all(response.get("status") == "passed" for response in b.values())
            aa_digest_equal = all(
                a1[camera_id].get("request_digest") == a2[camera_id].get("request_digest")
                for camera_id in camera_ids
            )
            aa_rgb_equal = all(
                a1[camera_id].get("rgb_payload_sha256") == a2[camera_id].get("rgb_payload_sha256")
                for camera_id in camera_ids
            )
            rgb_response_changed = any(
                a1[camera_id].get("rgb_payload_sha256") != b[camera_id].get("rgb_payload_sha256")
                for camera_id in camera_ids
            )
            target_pose_changed = _dynamic_digest(baseline_objects) != _dynamic_digest(moved_objects)
            non_target_actors_unchanged = _non_target_digest(baseline_objects, required_track) == _non_target_digest(
                moved_objects, required_track
            )
            probe_evidence = {
                "probe_timestamp_us": probe_timestamp,
                "camera_ids": list(camera_ids),
                "baseline_dynamic_digest": _dynamic_digest(baseline_objects),
                "moved_dynamic_digest": _dynamic_digest(moved_objects),
                "baseline_non_target_digest": _non_target_digest(baseline_objects, required_track),
                "moved_non_target_digest": _non_target_digest(moved_objects, required_track),
                "aa_request_digest_equal": aa_digest_equal,
                "aa_rgb_repeatable": aa_rgb_equal,
                "rgb_response_changed": rgb_response_changed,
                "target_pose_changed": target_pose_changed,
                "non_target_actors_unchanged": non_target_actors_unchanged,
                "status": "passed"
                if all(
                    [
                        a_ok,
                        b_ok,
                        aa_digest_equal,
                        aa_rgb_equal,
                        rgb_response_changed,
                        target_pose_changed,
                        non_target_actors_unchanged,
                    ]
                )
                else "failed",
                "responses": {
                    "A": {key: _strip_response(value) for key, value in a1.items()},
                    "A_repeat": {key: _strip_response(value) for key, value in a2.items()},
                    "B": {key: _strip_response(value) for key, value in b.items()},
                },
            }
            if probe_evidence["status"] != "passed":
                raise RenderError("V02 A/A/B probe failed; refusing to encode an edited video")
        elif case_id == "V03":
            probe_evidence = _run_camera_probes(
                client,
                scene,
                case,
                camera_ids,
                width,
                height,
                timestamps[0],
                dynamic_mode,
            )
            if probe_evidence["status"] != "passed":
                raise RenderError("V03 camera direction probes failed")
        if args.probe_only:
            return _finalize_evidence(
                output_dir,
                evidence_path,
                manifest_path,
                case_path,
                case,
                artifact,
                artifact_entry,
                checkpoint,
                checkpoint_entry,
                client.inventory,
                frame_records,
                frame_paths,
                dropped,
                probe_evidence,
                None,
                capture_info,
            )
        metadata_mode = "a" if args.resume else "w"
        with metadata_path.open(metadata_mode, encoding="utf-8") as metadata_handle:
            for frame_index in range(resumed_frame_count, len(timestamps)):
                timestamp_us = timestamps[frame_index]
                next_timestamp = (
                    timestamps[frame_index + 1]
                    if frame_index + 1 < len(timestamps)
                    else timestamp_us + 1
                )
                actor_end_timestamp = next_timestamp
                dynamic_objects = scene.dynamic_objects(
                    timestamp_us,
                    end_timestamp_us=actor_end_timestamp,
                    mode=dynamic_mode,
                    target_track_id=target_track_id,
                    target_delta=target_delta,
                )
                progress = frame_index / max(1, len(timestamps) - 1)
                dynamic_digest = _dynamic_digest(dynamic_objects)
                record: dict[str, Any] = {
                    "frame_index": frame_index,
                    "scene_timestamp_us": timestamp_us,
                    "camera_id": camera_ids[0],
                    "camera_ids": list(camera_ids),
                    "dynamic_object_digest": dynamic_digest,
                    "dynamic_object_count": len(dynamic_objects),
                    "target_track_id": required_track,
                    "target_pose_delta_m": dict(target_delta or {"x": 0.0, "y": 0.0, "z": 0.0}),
                    "camera_sweep_offset": {
                        "translation_m": _apply_sweep_offset(case, progress)[0],
                        "yaw_deg": _apply_sweep_offset(case, progress)[1],
                    },
                }
                frame_started = time.perf_counter()
                camera_responses: dict[str, dict[str, Any]] = {}
                camera_metrics: dict[str, dict[str, float]] = {}
                camera_bodies: dict[str, bytes] = {}
                camera_pose_pairs: dict[str, dict[str, Any]] = {}
                for camera_id in camera_ids:
                    start_pose, end_pose = _camera_pose_pair(
                        scene,
                        case,
                        camera_id,
                        timestamp_us,
                        next_timestamp,
                        progress,
                    )
                    camera_pose_pairs[camera_id] = {"start": start_pose, "end": end_pose}
                    response = client.render_rgb(
                        camera_id=camera_id,
                        width=width,
                        height=height,
                        start_us=timestamp_us,
                        end_us=next_timestamp,
                        start_pose=start_pose,
                        end_pose=end_pose,
                        dynamic_objects=dynamic_objects,
                        frame_id=f"{case_id}:{frame_index}:{camera_id}",
                    )
                    camera_responses[camera_id] = response
                    if response.get("status") == "passed":
                        camera_bodies[camera_id] = response["rgb_bytes"]
                        camera_metrics[camera_id] = _image_metrics(response["rgb_bytes"])
                frame_latency_ms = (time.perf_counter() - frame_started) * 1000.0
                record["requested_sensor_poses"] = camera_pose_pairs
                record["requested_sensor_pose"] = camera_pose_pairs[camera_ids[0]]
                record["camera_responses"] = {
                    camera_id: _strip_response(response)
                    for camera_id, response in camera_responses.items()
                }
                record["camera_metrics"] = camera_metrics
                record["camera_rpc_latency_ms"] = {
                    camera_id: response.get("rpc_latency_ms")
                    for camera_id, response in camera_responses.items()
                }
                record["rpc_latency_ms"] = frame_latency_ms
                record["frame_rpc_latency_ms"] = frame_latency_ms
                record["requested_timestamp_us"] = int(timestamp_us)
                record["requested_logical_frame_end_us"] = int(next_timestamp)
                record["requested_wire_timestamp_us"] = int((timestamp_us + next_timestamp) // 2)
                record["realized_timestamp_us"] = None
                record["realized_timestamp_status"] = (
                    "unavailable: RGBRenderReturn exposes image_bytes only"
                )
                if len(camera_bodies) != len(camera_ids):
                    dropped += 1
                    record["status"] = "error"
                    record["dropped"] = True
                    record["drop_reason"] = "one or more camera RPCs failed"
                else:
                    try:
                        stitched = _stitch_camera_frames(
                            camera_bodies,
                            camera_ids,
                            width=width,
                            height=height,
                            columns=columns,
                            rows=rows,
                            label_cameras=label_cameras,
                            pose_overlay=(
                                record["camera_sweep_offset"]
                                if isinstance(case.get("camera_sweep"), Mapping)
                                else None
                            ),
                            pose_progress=progress,
                            pose_frame=str(
                                sweep_config.get("frame", "sensor")
                                if isinstance(sweep_config, Mapping)
                                else "sensor"
                            ),
                            pose_profile=str(
                                sweep_config.get("profile", "sinusoidal")
                                if isinstance(sweep_config, Mapping)
                                else "sinusoidal"
                            ),
                            pose_camera_count=len(camera_ids),
                        )
                        metrics = _image_metrics(stitched)
                        record.update(metrics)
                        record["decoded_width"] = output_width
                        record["decoded_height"] = output_height
                        if metrics["mean_luma"] <= 0.5:
                            dropped += 1
                            record["status"] = "error"
                            record["dropped"] = True
                            record["drop_reason"] = "black stitched frame"
                        else:
                            frame_path = frames_dir / f"{frame_index:06d}.jpg"
                            frame_path.write_bytes(stitched)
                            frame_paths.append(frame_path)
                            camera_frame_paths: dict[str, str] = {}
                            for camera_id in camera_ids:
                                camera_path = camera_frames_dir / camera_id / f"{frame_index:06d}.jpg"
                                camera_path.parent.mkdir(parents=True, exist_ok=True)
                                camera_path.write_bytes(camera_bodies[camera_id])
                                camera_frame_paths[camera_id] = str(camera_path.relative_to(output_dir))
                            record["frame_path"] = str(frame_path.relative_to(output_dir))
                            record["camera_frame_paths"] = camera_frame_paths
                            record["status"] = "passed"
                            record["dropped"] = False
                    except RenderError as exc:
                        dropped += 1
                        record["status"] = "error"
                        record["dropped"] = True
                        record["drop_reason"] = str(exc)
                metadata_handle.write(json.dumps(record, sort_keys=True) + "\n")
                metadata_handle.flush()
                frame_records.append(record)
        if dropped or not frame_paths:
            raise RenderError(f"capture produced {dropped} dropped/invalid frames")
        _encode_mp4(frame_paths, video_path, video_fps)
        pose_summary = _write_pose_sweep_summary(
            output_dir,
            case,
            frame_records,
            video_path,
        )
        if pose_summary:
            capture_info = dict(capture_info)
            capture_info["pose_summary_json"] = str(pose_summary[0])
            capture_info["pose_summary_png"] = str(pose_summary[1]) if pose_summary[1] else None
        return _finalize_evidence(
            output_dir,
            evidence_path,
            manifest_path,
            case_path,
            case,
            artifact,
            artifact_entry,
            checkpoint,
            checkpoint_entry,
            client.inventory,
            frame_records,
            frame_paths,
            dropped,
            probe_evidence,
            video_path,
            capture_info,
        )
    finally:
        client.close()


def _strip_response(response: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in response.items() if key != "rgb_bytes"}


def _run_camera_probes(
    client: SensorsimClient,
    scene: ArtifactScene,
    case: Mapping[str, Any],
    camera_ids: list[str],
    width: int,
    height: int,
    timestamp_us: int,
    dynamic_mode: str,
) -> dict[str, Any]:
    objects = scene.dynamic_objects(timestamp_us, mode=dynamic_mode)
    probes = case.get("probe_offsets") or [
        {"name": "+x", "translation_m": {"x": 0.05}},
        {"name": "+y", "translation_m": {"y": 0.05}},
        {"name": "+yaw", "yaw_deg": 0.5},
    ]
    rows = []
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise RenderError("camera probe entries must be objects")
        responses = _probe_responses(
            client,
            scene,
            camera_ids,
            width=width,
            height=height,
            timestamp_us=timestamp_us,
            dynamic_objects=objects,
            offset=probe,
        )
        body_hashes = {
            camera_id: response.get("rgb_payload_sha256")
            for camera_id, response in responses.items()
        }
        rows.append(
            {
                "name": probe.get("name"),
                "responses": {
                    camera_id: _strip_response(response)
                    for camera_id, response in responses.items()
                },
                "rgb_payload_sha256_by_camera": body_hashes,
                "status": "passed"
                if all(response.get("status") == "passed" for response in responses.values())
                else "failed",
            }
        )
    passed = all(row["status"] == "passed" for row in rows)
    probe_digests = [
        canonical_digest(row["rgb_payload_sha256_by_camera"])
        for row in rows
    ]
    changed = len(set(probe_digests)) == len(probe_digests)
    return {
        "status": "passed" if passed and changed else "failed",
        "timestamp_us": timestamp_us,
        "camera_ids": list(camera_ids),
        "probes": rows,
    }


def _finalize_evidence(
    output_dir: Path,
    evidence_path: Path,
    manifest_path: Path,
    case_path: Path,
    case: Mapping[str, Any],
    artifact: Path,
    artifact_entry: Mapping[str, Any],
    checkpoint: Path | None,
    checkpoint_entry: Mapping[str, Any] | None,
    inventory: Mapping[str, Any],
    frame_records: list[Mapping[str, Any]],
    frame_paths: list[Path],
    dropped: int,
    probe_evidence: Mapping[str, Any] | None,
    video_path: Path | None,
    capture_info: Mapping[str, Any],
) -> dict[str, Any]:
    passed_records = [record for record in frame_records if record.get("status") == "passed" and not record.get("dropped")]
    latencies = sorted(float(record["rpc_latency_ms"]) for record in passed_records if record.get("rpc_latency_ms") is not None)
    evidence = {
        "schema_version": "nsb.nurec-counterfactual-run.v1",
        "status": "passed" if video_path and len(passed_records) == len(frame_records) and dropped == 0 else "probe_only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": {
            "case_id": case.get("case_id"),
            "path": str(case_path),
            "sha256": sha256_file(case_path),
            "kind": case.get("kind"),
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "artifact": {
            "uri": artifact_entry.get("uri") or artifact_entry.get("path"),
            "resolved_path": str(artifact),
            "sha256": artifact_entry.get("sha256"),
            "size_bytes": artifact_entry.get("size_bytes"),
        },
        "checkpoint": (
            {
                "uri": checkpoint_entry.get("uri") or checkpoint_entry.get("path"),
                "resolved_path": str(checkpoint),
                "sha256": checkpoint_entry.get("sha256"),
                "size_bytes": checkpoint_entry.get("size_bytes"),
            }
            if checkpoint and checkpoint_entry
            else None
        ),
        "runtime": dict(inventory),
        "output": {
            "directory": str(output_dir),
            "video": str(video_path) if video_path else None,
            "frame_count": len(frame_paths),
            "dropped_frame_count": dropped,
            "metadata": str(output_dir / "frames.jsonl"),
            "camera_ids": list(capture_info.get("camera_ids") or []),
            "camera_count": int(capture_info.get("camera_count", 0)),
            "source_resolution": capture_info.get("source_resolution"),
            "video_resolution": capture_info.get("video_resolution"),
            "grid": capture_info.get("grid"),
            "sample_fps": capture_info.get("sample_fps"),
            "sampling_mode": capture_info.get("sampling_mode"),
            "video_fps": capture_info.get("video_fps"),
            "actor_interpolation": capture_info.get("actor_interpolation"),
            "rig_interpolation": capture_info.get("rig_interpolation"),
            "resumed_frame_count": capture_info.get("resumed_frame_count", 0),
            "pose_overlay": capture_info.get("pose_overlay"),
            "pose_summary_json": capture_info.get("pose_summary_json"),
            "pose_summary_png": capture_info.get("pose_summary_png"),
        },
        "frames": {
            "requested_count": len(frame_records),
            "captured_count": len(frame_paths),
            "first_timestamp_us": frame_records[0].get("scene_timestamp_us") if frame_records else None,
            "last_timestamp_us": frame_records[-1].get("scene_timestamp_us") if frame_records else None,
            "rpc_latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
        },
        "probe": dict(probe_evidence) if probe_evidence else None,
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(math.ceil(fraction * len(values)) - 1)))
    return float(values[index])


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        evidence = render_case(args)
    except (RenderError, OSError, ValueError) as exc:
        print(f"render_counterfactual_video: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
