#!/usr/bin/env python3
"""Compose synchronized raw/Harmonizer front-camera comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/nurec_scene0061_final"
CASES = (
    (
        "V01",
        "ORIGINAL REPLAY",
        FINAL / "videos_30fps/V01_original_replay_30fps.mp4",
        FINAL / "videos_30fps_harmonizer/V01_original_replay_30fps_harmonizer.mp4",
    ),
    (
        "V02",
        "LEAD-VEHICLE EDIT",
        FINAL / "videos_30fps/V02_lead_vehicle_edit_30fps.mp4",
        FINAL / "videos_30fps_harmonizer/V02_lead_vehicle_edit_30fps_harmonizer.mp4",
    ),
    (
        "V03",
        "CAMERA POSE SWEEP",
        FINAL / "videos_30fps/V03_camera_pose_sweep_30fps.mp4",
        FINAL / "videos_30fps_harmonizer/V03_camera_pose_sweep_30fps_harmonizer.mp4",
    ),
)
SOURCE_SIZE = (2400, 900)
CELL_SIZE = (800, 450)
FRONT_CROP = (800, 0, 1600, 450)
EXPECTED_FPS = 30.0
EXPECTED_FRAMES = 577


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(capture: cv2.VideoCapture, path: Path) -> dict[str, int | float]:
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {path}")
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    expected = (*SOURCE_SIZE, EXPECTED_FPS, EXPECTED_FRAMES)
    observed = (
        result["width"],
        result["height"],
        result["fps"],
        result["frame_count"],
    )
    if observed != expected:
        raise RuntimeError(f"source contract mismatch for {path}: {observed} != {expected}")
    return result


def _front_cell(frame: np.ndarray, row: str, case_id: str, title: str) -> np.ndarray:
    x0, y0, x1, y1 = FRONT_CROP
    cell = frame[y0:y1, x0:x1].copy()
    overlay = cell.copy()
    cv2.rectangle(overlay, (0, 0), (CELL_SIZE[0] - 1, 31), (5, 8, 11), -1)
    cv2.addWeighted(overlay, 0.88, cell, 0.12, 0.0, cell)
    color = (245, 245, 245) if row == "RAW" else (80, 225, 255)
    cv2.putText(
        cell,
        f"{row} | {case_id} {title}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )
    return cell


def compose(output: Path, manifest_path: Path, overwrite: bool) -> dict[str, object]:
    if output.exists() and not overwrite:
        raise RuntimeError(f"output exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    sources = []
    captures: list[cv2.VideoCapture] = []
    for case_id, title, raw_path, harmonizer_path in CASES:
        for kind, path in (("raw", raw_path), ("harmonizer", harmonizer_path)):
            capture = cv2.VideoCapture(str(path))
            metadata = _probe(capture, path)
            captures.append(capture)
            sources.append(
                {
                    "case_id": case_id,
                    "kind": kind,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    **metadata,
                }
            )

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        EXPECTED_FPS,
        (CELL_SIZE[0] * 3, CELL_SIZE[1] * 2),
    )
    if not writer.isOpened():
        for capture in captures:
            capture.release()
        raise RuntimeError(f"cannot open video writer: {output}")

    difference_sum = [0.0, 0.0, 0.0]
    try:
        for frame_index in range(EXPECTED_FRAMES):
            frames = []
            for capture in captures:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"source decode failed at frame {frame_index}")
                frames.append(frame)

            raw_cells = []
            harmonizer_cells = []
            for case_index, (case_id, title, _, _) in enumerate(CASES):
                raw = _front_cell(frames[case_index * 2], "RAW", case_id, title)
                harmonized = _front_cell(
                    frames[case_index * 2 + 1], "NVIDIA HARMONIZER", case_id, title
                )
                raw_cells.append(raw)
                harmonizer_cells.append(harmonized)
                difference_sum[case_index] += float(
                    np.mean(cv2.absdiff(raw[32:], harmonized[32:]))
                )
            writer.write(
                np.vstack((np.hstack(raw_cells), np.hstack(harmonizer_cells)))
            )
    finally:
        writer.release()
        for capture in captures:
            capture.release()

    verification = cv2.VideoCapture(str(output))
    try:
        output_metadata = {
            "width": int(verification.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(verification.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(verification.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(verification.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        verification.release()
    expected_output = (2400, 900, EXPECTED_FPS, EXPECTED_FRAMES)
    observed_output = tuple(output_metadata[key] for key in ("width", "height", "fps", "frame_count"))
    if observed_output != expected_output:
        raise RuntimeError(f"output contract mismatch: {observed_output} != {expected_output}")

    result: dict[str, object] = {
        "schema_version": "nsb.harmonizer-comparison.v1",
        "status": "passed",
        "layout": {
            "columns": ["V01 original replay", "V02 lead-vehicle edit", "V03 camera pose sweep"],
            "top_row": "raw camera_front",
            "bottom_row": "NuRec RGB with NVIDIA Harmonizer post-processing camera_front",
            "source_front_crop_xyxy": list(FRONT_CROP),
        },
        "output": {
            "path": str(output.relative_to(ROOT)),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            **output_metadata,
            "duration_s": EXPECTED_FRAMES / EXPECTED_FPS,
        },
        "sources": sources,
        "raw_harmonizer_mean_abs_difference": {
            case_id: difference_sum[index] / EXPECTED_FRAMES
            for index, (case_id, _, _, _) in enumerate(CASES)
        },
        "limitations": [
            "Harmonizer is RGB appearance post-processing and does not repair actor trajectories.",
            "Panels are synchronized by the common frame index and 30 FPS source contract.",
        ],
    }
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=FINAL / "V05_raw_vs_harmonizer_front_3x2_30fps.mp4",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FINAL / "V05_raw_vs_harmonizer_front_3x2_30fps.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = compose(args.output.resolve(), args.manifest.resolve(), args.overwrite)
    print(json.dumps(result["output"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
