#!/usr/bin/env python3
"""Validate and package the ten scene-0061 playback videos."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/nurec_scene0061_final"
ARCHIVE = ROOT / "outputs/nurec_scene0061_final_10videos.zip"
ARTIFACT_SHA = "69e2c36e31e113f9ad66968a0a0a4243c7989dae5582a91a43d150051eaf98b4"
HARMONIZER_SHA = "ece8e2daa914e8c2a027a2da94e0eb2064491d5b3fd8514009fae9a442e06e90"
TIMESTAMP_RANGE = [1532402927598150, 1532402946797517]
CAMERAS = [
    "camera_front_left",
    "camera_front",
    "camera_front_right",
    "camera_back_left",
    "camera_back",
    "camera_back_right",
]


VIDEOS = [
    ("V01", "videos_20fps/V01_original_replay_20fps.mp4", 20, 385, False),
    ("V02", "videos_20fps/V02_lead_vehicle_edit_20fps.mp4", 20, 385, False),
    ("V03", "videos_20fps/V03_camera_pose_sweep_20fps.mp4", 20, 385, False),
    ("V01", "videos_30fps/V01_original_replay_30fps.mp4", 30, 577, False),
    ("V02", "videos_30fps/V02_lead_vehicle_edit_30fps.mp4", 30, 577, False),
    ("V03", "videos_30fps/V03_camera_pose_sweep_30fps.mp4", 30, 577, False),
    ("V01", "videos_30fps_harmonizer/V01_original_replay_30fps_harmonizer.mp4", 30, 577, True),
    ("V02", "videos_30fps_harmonizer/V02_lead_vehicle_edit_30fps_harmonizer.mp4", 30, 577, True),
    ("V03", "videos_30fps_harmonizer/V03_camera_pose_sweep_30fps_harmonizer.mp4", 30, 577, True),
    ("V04", "multimodal_20fps/V04_multimodal_alignment_20fps.mp4", 20, 385, False),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        result = {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()
    result["duration_s"] = result["frame_count"] / result["fps"]
    return result


def sampled_quality(path: Path, sample_step: int = 10) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    dark: list[float] = []
    invalid: list[float] = []
    sharpness: list[float] = []
    flicker: list[float] = []
    previous: np.ndarray | None = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % sample_step == 0:
                reduced = cv2.resize(frame, (320, 120), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
                dark.append(float(np.mean(gray <= 5)))
                invalid.append(float(np.mean(~np.isfinite(reduced))))
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                if previous is not None:
                    flicker.append(float(np.mean(cv2.absdiff(gray, previous))))
                previous = gray
            index += 1
    finally:
        capture.release()
    if not sharpness:
        raise RuntimeError(f"no decodable frames in {path}")
    return {
        "sample_step": sample_step,
        "sample_count": len(sharpness),
        "dark_pixel_ratio_mean": statistics.fmean(dark),
        "invalid_pixel_ratio_mean": statistics.fmean(invalid),
        "sharpness_laplacian_mean": statistics.fmean(sharpness),
        "temporal_flicker_mean": statistics.fmean(flicker) if flicker else 0.0,
    }


def source_policy(case_id: str, fps: int, harmonizer: bool) -> str:
    if case_id == "V04":
        return "385 approximately 20 Hz synchronized live RGB/LiDAR render windows"
    suffix = " with NVIDIA Harmonizer post-processing" if harmonizer else ""
    return f"independent uniform {fps} Hz NuRec timestamp requests{suffix}"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    entries = []
    quality = {}
    for case_id, relative, fps, expected_frames, harmonizer in VIDEOS:
        path = FINAL / relative
        if not path.is_file():
            raise RuntimeError(f"required video is missing: {path}")
        metadata = probe(path)
        expected_size = (1600, 900) if case_id == "V04" else (2400, 900)
        observed = (
            metadata["fps"],
            metadata["frame_count"],
            metadata["width"],
            metadata["height"],
        )
        expected = (float(fps), expected_frames, *expected_size)
        if observed != expected:
            raise RuntimeError(f"video contract mismatch for {path}: {observed} != {expected}")
        entry = {
            "case_id": case_id,
            "path": f"outputs/nurec_scene0061_final/{relative}",
            **metadata,
            "sample_fps": fps,
            "video_fps": fps,
            "source_frame_policy": source_policy(case_id, fps, harmonizer),
            "camera_ids": ["camera_front"] if case_id == "V04" else CAMERAS,
            "artifact_sha256": ARTIFACT_SHA,
            "output_sha256": sha256_file(path),
            "output_size_bytes": path.stat().st_size,
            "harmonizer_enabled": harmonizer,
            "harmonizer_checkpoint_sha256": HARMONIZER_SHA if harmonizer else None,
            "timestamp_range_us": TIMESTAMP_RANGE,
            "repeated_source_frame_count": 0,
            "playback_only": True,
            "max_actor_interpolation_gap_us": 600_000,
            "max_rig_interpolation_gap_us": 600_000,
            "limitations": [
                "Playback-only uniform cadence; not a source-timestamp-faithful full-dynamic replay.",
                "Source trajectory gaps and lead-vehicle dynamic-layer instability are not repaired.",
                "NVIDIA Harmonizer is RGB appearance post-processing only."
                if harmonizer
                else "No optical-flow interpolation or trajectory repair is applied.",
            ],
            "status": "completed",
        }
        entries.append(entry)
        quality[relative] = {**metadata, **sampled_quality(path)}

    v04_evidence = FINAL / "multimodal_20fps/evidence.json"
    if not v04_evidence.is_file():
        raise RuntimeError("V04 evidence.json is missing")
    v04 = json.loads(v04_evidence.read_text(encoding="utf-8"))
    if v04.get("status") != "passed" or v04.get("rgb_lidar_timestamp_alignment_max_us") != 0:
        raise RuntimeError("V04 evidence does not prove RGB/LiDAR alignment")
    v04_entry = next(item for item in entries if item["case_id"] == "V04")
    v04_entry["timestamp_range_us"] = v04["timestamp_range_us"]
    v04_entry["limitations"] = v04["limitations"]

    manifest = {
        "schema_version": "nsb.scene0061-final-video-manifest.v2",
        "generated_at": generated_at,
        "status": "completed",
        "video_count": len(entries),
        "playback_only": True,
        "videos": entries,
    }
    report = {
        "schema_version": "nsb.scene0061-final-quality.v2",
        "generated_at": generated_at,
        "status": "passed",
        "video_count": len(entries),
        "videos": quality,
        "V04": {
            "rgb_lidar_timestamp_alignment_max_us": 0,
            "evidence": "outputs/nurec_scene0061_final/multimodal_20fps/evidence.json",
        },
        "limitations": [
            "All videos use the declared 600 ms playback-only actor and rig interpolation limit.",
            "Harmonizer does not fix actor trajectories or temporal Gaussian instability.",
        ],
    }
    write_json(FINAL / "video_manifest.json", manifest)
    write_json(FINAL / "quality_report.json", report)

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    archive_files = [FINAL / relative for _, relative, _, _, _ in VIDEOS]
    archive_files += [
        FINAL / "README.md",
        FINAL / "video_manifest.json",
        FINAL / "quality_report.json",
        FINAL / "artifact_manifest.json",
        v04_evidence,
    ]
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in archive_files:
            bundle.write(path, path.relative_to(FINAL.parent))
    with zipfile.ZipFile(ARCHIVE, "r") as bundle:
        bad = bundle.testzip()
        mp4_count = sum(name.endswith(".mp4") for name in bundle.namelist())
        if bad is not None or mp4_count != 10:
            raise RuntimeError(f"archive verification failed: bad={bad}, mp4_count={mp4_count}")
    print(json.dumps({
        "archive": str(ARCHIVE),
        "sha256": sha256_file(ARCHIVE),
        "size_bytes": ARCHIVE.stat().st_size,
        "video_count": 10,
        "status": "passed",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
