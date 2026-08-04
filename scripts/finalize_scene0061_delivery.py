#!/usr/bin/env python3
"""Package existing scene-0061 frame evidence into audited final videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = REPO_ROOT / "outputs" / "nurec_scene0061_final"
CAMERA_ORDER = (
    "camera_front_left",
    "camera_front",
    "camera_front_right",
    "camera_back_left",
    "camera_back",
    "camera_back_right",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _overlay(frame: np.ndarray, label: str) -> np.ndarray:
    result = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    x = max(8, result.shape[1] - text_width - 24)
    y = 42
    cv2.rectangle(
        result,
        (x - 10, y - text_height - 9),
        (result.shape[1] - 8, y + baseline + 7),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        result,
        label,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return result


def _video_writer(path: Path, fps: float, shape: tuple[int, int]) -> cv2.VideoWriter:
    width, height = shape
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open MP4 writer: {path}")
    return writer


def encode_frames(
    frame_paths: Iterable[Path], output: Path, *, fps: float, label: str
) -> dict[str, object]:
    paths = list(frame_paths)
    if not paths:
        raise RuntimeError(f"no frames for {output}")
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot decode frame: {paths[0]}")
    height, width = first.shape[:2]
    writer = _video_writer(output, fps, (width, height))
    try:
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError(f"invalid or inconsistent frame: {path}")
            writer.write(_overlay(frame, label))
    finally:
        writer.release()
    return video_record(output, fps=fps, frame_count=len(paths), width=width, height=height)


def encode_harmonizer_grid(
    source_root: Path, output: Path, *, fps: float, label: str
) -> dict[str, object]:
    camera_frames = {
        camera: sorted((source_root / camera).glob("*.jpg"))
        for camera in CAMERA_ORDER
    }
    counts = {camera: len(paths) for camera, paths in camera_frames.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values()), 0):
        raise RuntimeError(f"incomplete Harmonizer frame grid: {counts}")
    frame_count = next(iter(counts.values()))
    writer = _video_writer(output, fps, (2400, 900))
    try:
        for index in range(frame_count):
            cells = []
            for camera in CAMERA_ORDER:
                frame = cv2.imread(str(camera_frames[camera][index]), cv2.IMREAD_COLOR)
                if frame is None or frame.shape[:2] != (450, 800):
                    raise RuntimeError(
                        f"invalid Harmonizer frame: {camera_frames[camera][index]}"
                    )
                cells.append(frame)
            grid = np.vstack((np.hstack(cells[:3]), np.hstack(cells[3:])))
            writer.write(_overlay(grid, label))
    finally:
        writer.release()
    return video_record(
        output, fps=fps, frame_count=frame_count, width=2400, height=900
    )


def video_record(
    path: Path, *, fps: float, frame_count: int, width: int, height: int
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    try:
        observed = {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()
    expected = {
        "fps": float(fps),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
    }
    if observed != expected:
        raise RuntimeError(f"encoded stream mismatch for {path}: {observed} != {expected}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        **observed,
        "duration_s": frame_count / fps,
    }


def write_checkpoint(records: list[dict[str, object]]) -> None:
    path = FINAL_ROOT / "packaging_checkpoint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": "nsb.scene0061-packaging-checkpoint.v1", "videos": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    jobs = [
        (
            REPO_ROOT / "outputs/nurec_scene0061_demo_20hz/V01/frames",
            FINAL_ROOT / "videos_20fps/V01_original_replay_20fps.mp4",
            20.0,
            "V01 | 20 FPS | raw cadence baseline",
        ),
        *[
            (
                REPO_ROOT / f"outputs/nurec_scene0061_demo/{case}/frames",
                FINAL_ROOT / f"videos_30fps/{case}_{name}_30fps.mp4",
                30.0,
                f"{case} | 30 FPS | raw NuRec RGB",
            )
            for case, name in (
                ("V01", "original_replay"),
                ("V02", "lead_vehicle_edit"),
                ("V03", "camera_pose_sweep"),
            )
        ],
    ]
    records: list[dict[str, object]] = []
    for source, output, fps, label in jobs:
        if output.exists() and not args.overwrite:
            raise RuntimeError(f"output exists; pass --overwrite: {output}")
        record = encode_frames(sorted(source.glob("*.jpg")), output, fps=fps, label=label)
        records.append(record)
        write_checkpoint(records)
        print(json.dumps(record, sort_keys=True), flush=True)

    harmonizer_source = Path(
        "/home/cwadmin/workspace/ClosedLoopBench/outputs/scene-0061-final-closure-v2/"
        "formal_acceptance/harmonizer_ab/harmonized_577_800x450"
    )
    harmonizer_output = (
        FINAL_ROOT
        / "videos_30fps_harmonizer/V01_original_replay_30fps_harmonizer.mp4"
    )
    if harmonizer_output.exists() and not args.overwrite:
        raise RuntimeError(f"output exists; pass --overwrite: {harmonizer_output}")
    record = encode_harmonizer_grid(
        harmonizer_source,
        harmonizer_output,
        fps=30.0,
        label="V01 | 30 FPS | NVIDIA Harmonizer",
    )
    records.append(record)
    write_checkpoint(records)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
