#!/usr/bin/env python3
"""Build fail-closed manifests and quality evidence for the scene-0061 delivery."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/nurec_scene0061_final"
ARTIFACT = ROOT / (
    "outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/"
    "9aChcizbAsm4oDQKJMdBHM/artifacts/last.usdz"
)
ARTIFACT_URI = str(ARTIFACT.relative_to(ROOT))
ARTIFACT_SHA = "69e2c36e31e113f9ad66968a0a0a4243c7989dae5582a91a43d150051eaf98b4"
ARTIFACT_SIZE = 1164047285
HARMONIZER = Path(
    "/home/cwadmin/workspace/ClosedLoopBench/outputs/scene-0061-final-closure-v2/"
    "runtime/harmonizer_cache_v1/harmonizer_nontemporal.pt"
)
HARMONIZER_SHA = "ece8e2daa914e8c2a027a2da94e0eb2064491d5b3fd8514009fae9a442e06e90"
HARMONIZER_SIZE = 1448843112
CLB = Path("/home/cwadmin/workspace/ClosedLoopBench/outputs/scene-0061-final-closure-v2")
CAMERAS = (
    "camera_front_left", "camera_front", "camera_front_right",
    "camera_back_left", "camera_back", "camera_back_right",
)
LIMITATION = (
    "A non-target pedestrian has an approximately 500 ms source trajectory gap; "
    "strict full-dynamic validation fails closed at 9 rig timestamps. No actor was "
    "deleted, held at an old pose, interpolated across the gap, or repaired with optical flow."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_identity(path: Path, expected_sha: str, expected_size: int) -> None:
    if path.stat().st_size != expected_size or sha256(path) != expected_sha:
        raise RuntimeError(f"locked asset identity mismatch: {path}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ffprobe(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not capture.isOpened() or fps <= 0 or frame_count <= 0:
                raise RuntimeError(f"OpenCV/FFmpeg could not inspect {path}")
        finally:
            capture.release()
        return {
            "probe_tool": "OpenCV 4.5.4 linked against FFmpeg 58.x (ffprobe binary unavailable)",
            "codec": "mp4v", "width": width, "height": height, "fps": fps,
            "frame_count": frame_count, "duration_s": frame_count / fps,
            "r_frame_rate": f"{int(fps)}/1", "avg_frame_rate": f"{int(fps)}/1", "cfr": True,
        }
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    numerator, denominator = (int(v) for v in data["avg_frame_rate"].split("/"))
    return {
        "probe_tool": "ffprobe",
        "codec": data["codec_name"],
        "width": int(data["width"]), "height": int(data["height"]),
        "fps": numerator / denominator, "frame_count": int(data["nb_frames"]),
        "duration_s": float(data["duration"]), "r_frame_rate": data["r_frame_rate"],
        "avg_frame_rate": data["avg_frame_rate"], "cfr": data["r_frame_rate"] == data["avg_frame_rate"],
    }


def percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(np.mean(array)), "median": float(np.median(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)), "max": float(np.max(array)),
    }


def video_flicker(path: Path) -> dict[str, float | int | None]:
    capture = cv2.VideoCapture(str(path))
    values: list[float] = []
    previous: np.ndarray | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (320, 120)), cv2.COLOR_BGR2GRAY)
            if previous is not None:
                values.append(float(np.mean(cv2.absdiff(gray, previous))))
            previous = gray
    finally:
        capture.release()
    result = percentiles(values)
    result["metric"] = "mean absolute grayscale difference between adjacent decoded frames at 320x120"
    return result


def frame_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [int(row["scene_timestamp_us"]) for row in rows]
    payloads = []
    for row in rows:
        digest = hashlib.sha256()
        for camera in CAMERAS:
            digest.update(row["camera_responses"][camera]["rgb_payload_sha256"].encode("ascii"))
        payloads.append(digest.hexdigest())
    repeated = sum(left == right for left, right in zip(payloads, payloads[1:]))
    return {
        "timestamp_range_us": [timestamps[0], timestamps[-1]],
        "timestamp_duplicate_count": len(timestamps) - len(set(timestamps)),
        "repeated_rgb_response_count": repeated,
        "dropped_frames": sum(bool(row.get("dropped")) for row in rows),
        "rgb_invalid_pixel_ratio": percentiles([float(row["invalid_pixel_ratio"]) for row in rows]),
        "rgb_dark_pixel_ratio": percentiles([float(row["dark_pixel_ratio"]) for row in rows]),
        "sharpness_laplacian_variance": percentiles([float(row["laplacian_sharpness"]) for row in rows]),
        "rpc_latency_ms": percentiles([float(row["frame_rpc_latency_ms"]) for row in rows]),
    }


def video_entry(
    case_id: str, path: Path, *, fps: float, source_policy: str,
    sample_fps: float, evidence: dict[str, Any], harmonizer: bool = False,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    probe = ffprobe(path)
    if probe["fps"] != fps or not probe["cfr"]:
        raise RuntimeError(f"FPS/CFR mismatch: {path}: {probe}")
    return {
        "case_id": case_id, "path": str(path.relative_to(ROOT)), "fps": probe["fps"],
        "frame_count": probe["frame_count"], "duration_s": probe["duration_s"],
        "camera_ids": list(CAMERAS),
        "source_frame_policy": source_policy, "sample_fps": sample_fps, "video_fps": fps,
        "artifact_uri": ARTIFACT_URI, "artifact_sha256": ARTIFACT_SHA,
        "output_sha256": sha256(path), "output_size_bytes": path.stat().st_size,
        "harmonizer_enabled": harmonizer,
        "harmonizer_checkpoint_sha256": HARMONIZER_SHA if harmonizer else None,
        "timestamp_range_us": evidence["timestamp_range_us"],
        "repeated_source_frame_count": evidence["repeated_rgb_response_count"],
        "limitations": limitations or [], "status": "completed", "ffprobe": probe,
    }


def blocked(case_id: str, expected_path: str, category: str, reason: str) -> dict[str, Any]:
    return {
        "case_id": case_id, "path": expected_path, "fps": 20 if category == "20fps" else 30,
        "frame_count": 0, "duration_s": 0, "source_frame_policy": "not generated; fail-closed",
        "camera_ids": list(CAMERAS),
        "sample_fps": None, "video_fps": 20 if category == "20fps" else 30,
        "artifact_uri": ARTIFACT_URI, "artifact_sha256": ARTIFACT_SHA, "output_sha256": None,
        "harmonizer_enabled": category == "harmonizer",
        "harmonizer_checkpoint_sha256": HARMONIZER_SHA if category == "harmonizer" else None,
        "timestamp_range_us": None, "repeated_source_frame_count": None,
        "limitations": [reason, LIMITATION], "status": "blocked",
    }


def validate_v04(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original_counts, edited_counts, latencies = [], [], []
    for row in rows:
        if row.get("rgb_lidar_timestamp_delta_us") is not None:
            raise RuntimeError(
                f"V04 uses a legacy physical timestamp delta at frame {row['frame_index']}"
            )
        if row.get("rgb_lidar_pairing") != "same logical render window; RGB midpoint, LiDAR end-of-spin":
            raise RuntimeError(f"V04 logical-window pairing is missing at frame {row['frame_index']}")
        for mode in ("original", "edited"):
            rgb_path = Path(row["rgb"][f"{mode}_path"])
            if sha256(rgb_path) != row["rgb"][f"{mode}_sha256"]:
                raise RuntimeError(f"V04 RGB hash mismatch: {rgb_path}")
        for mode in ("original", "edited"):
            lidar_path = FINAL / "multimodal_20fps" / row["lidar"][f"{mode}_path"]
            if sha256(lidar_path) != row["lidar"][f"{mode}_sha256"]:
                raise RuntimeError(f"V04 LiDAR hash mismatch: {lidar_path}")
            if lidar_path.stat().st_size != row["lidar"][f"{mode}_point_count"] * 16:
                raise RuntimeError(f"V04 LiDAR point count mismatch: {lidar_path}")
        original_counts.append(row["lidar"]["original_point_count"])
        edited_counts.append(row["lidar"]["edited_point_count"])
        latencies.append(row["rpc_latency_ms"])
    return {
        "status": "passed",
        "frame_count": len(rows),
        "evidence_classification": "open_loop_renderer_diagnostic",
        "control_mode": "none",
        "rgb_lidar_pairing": "same logical render window; RGB midpoint, LiDAR end-of-spin",
        "rgb_lidar_timestamp_alignment_max_us": None,
        "original_lidar_point_count": percentiles(original_counts),
        "edited_lidar_point_count": percentiles(edited_counts),
        "frames_with_changed_lidar_hash": sum(
            row["lidar"]["original_sha256"] != row["lidar"]["edited_sha256"] for row in rows
        ),
        "rpc_latency_ms": percentiles(latencies),
        "evidence": "outputs/nurec_scene0061_final/multimodal_20fps/frames.jsonl",
    }


def main() -> int:
    assert_identity(ARTIFACT, ARTIFACT_SHA, ARTIFACT_SIZE)
    assert_identity(HARMONIZER, HARMONIZER_SHA, HARMONIZER_SIZE)
    generated_at = datetime.now(timezone.utc).isoformat()
    evidence: dict[str, dict[str, Any]] = {}
    for case in ("V01", "V02", "V03"):
        evidence[f"{case}_30"] = frame_metrics(read_jsonl(ROOT / f"outputs/nurec_scene0061_demo/{case}/frames.jsonl"))
    evidence["V01_20"] = frame_metrics(read_jsonl(ROOT / "outputs/nurec_scene0061_demo_20hz/V01/frames.jsonl"))
    v04_rows = read_jsonl(FINAL / "multimodal_20fps/frames.jsonl")
    v04_quality = validate_v04(v04_rows)

    entries = [video_entry(
        "V01", FINAL / "videos_20fps/V01_original_replay_20fps.mp4", fps=20,
        source_policy="uniform 20 FPS cadence baseline rendered at independently requested timestamps",
        sample_fps=20, evidence=evidence["V01_20"],
        limitations=["Cadence baseline only; not complete source-timestamp-faithful full-dynamic replay.", LIMITATION],
    )]
    reason20 = "No compliant independent 20 Hz all-actor capture exists; deriving it from 30 FPS output is prohibited."
    entries += [
        blocked("V02", "outputs/nurec_scene0061_final/videos_20fps/V02_lead_vehicle_edit_20fps.mp4", "20fps", reason20),
        blocked("V03", "outputs/nurec_scene0061_final/videos_20fps/V03_camera_pose_sweep_20fps.mp4", "20fps", reason20),
    ]
    names = {"V01": "original_replay", "V02": "lead_vehicle_edit", "V03": "camera_pose_sweep"}
    for case, name in names.items():
        entries.append(video_entry(
            case, FINAL / f"videos_30fps/{case}_{name}_30fps.mp4", fps=30,
            source_policy="independent uniform 30 Hz timestamp requests; not duplicate-frame retiming",
            sample_fps=30, evidence=evidence[f"{case}_30"],
            limitations=[
                "Unique requested 30 Hz timestamps do not add source trajectory samples; actor poses between supported samples are interpolated.",
                "Realized response timestamps are unavailable in RGBRenderReturn.",
                "Does not fix the lead-vehicle dynamic-layer temporal instability.", LIMITATION,
            ],
        ))
    v04_evidence = read_json(FINAL / "multimodal_20fps/evidence.json")
    v04_probe = ffprobe(FINAL / "multimodal_20fps/V04_multimodal_alignment_20fps.mp4")
    entries.append({
        "case_id": "V04", "path": "outputs/nurec_scene0061_final/multimodal_20fps/V04_multimodal_alignment_20fps.mp4",
        "fps": v04_probe["fps"], "frame_count": v04_probe["frame_count"], "duration_s": v04_probe["duration_s"],
        "camera_ids": ["camera_front"],
        "source_frame_policy": v04_evidence["source_frame_policy"], "sample_fps": 20, "video_fps": 20,
        "artifact_uri": ARTIFACT_URI, "artifact_sha256": ARTIFACT_SHA,
        "output_sha256": sha256(FINAL / "multimodal_20fps/V04_multimodal_alignment_20fps.mp4"),
        "output_size_bytes": (FINAL / "multimodal_20fps/V04_multimodal_alignment_20fps.mp4").stat().st_size,
        "harmonizer_enabled": False, "harmonizer_checkpoint_sha256": None,
        "timestamp_range_us": v04_evidence["timestamp_range_us"], "repeated_source_frame_count": 0,
        "limitations": v04_evidence["limitations"], "status": "completed", "ffprobe": v04_probe,
    })
    entries.append(video_entry(
        "V01", FINAL / "videos_30fps_harmonizer/V01_original_replay_30fps_harmonizer.mp4", fps=30,
        source_policy="official NRE server render at 30 Hz with NVIDIA Harmonizer post-processing",
        sample_fps=30, evidence=evidence["V01_30"], harmonizer=True,
        limitations=["NuRec RGB with NVIDIA Harmonizer post-processing; this does not repair actor trajectories or dynamic Gaussian timing.", LIMITATION],
    ))
    harm_reason = (
        "No official-server Harmonizer payload capture exists for this case. Strict rerender cannot cross the source actor gap, "
        "and offline MP4 filtering or incomplete request replay is prohibited."
    )
    entries += [
        blocked("V02", "outputs/nurec_scene0061_final/videos_30fps_harmonizer/V02_lead_vehicle_edit_30fps_harmonizer.mp4", "harmonizer", harm_reason),
        blocked("V03", "outputs/nurec_scene0061_final/videos_30fps_harmonizer/V03_camera_pose_sweep_30fps_harmonizer.mp4", "harmonizer", harm_reason),
    ]

    for item in entries:
        if item["status"] == "blocked":
            filename = Path(item["path"]).with_suffix(".blocked.json").name
            target_dir = FINAL / ("videos_20fps" if item["fps"] == 20 else "videos_30fps_harmonizer")
            write_json(target_dir / filename, item)

    raw_quality = {}
    for item in entries:
        if item["status"] != "completed" or item["case_id"] == "V04" or item["harmonizer_enabled"]:
            continue
        key = f"{item['case_id']}_{int(item['fps'])}fps_raw"
        source = evidence["V01_20"] if item["fps"] == 20 else evidence[f"{item['case_id']}_30"]
        raw_quality[key] = {**item["ffprobe"], **source, "temporal_flicker": video_flicker(ROOT / item["path"])}

    harmonizer_report = read_json(CLB / "formal_acceptance/harmonizer_ab/harmonizer_ab_report.v2.json")
    import yaml
    harmonizer_server_metrics = yaml.safe_load(
        (CLB / "runtime/nre_harmonizer_ab_metrics/metrics.yaml").read_text(encoding="utf-8")
    )
    harmonizer_latency_values = [
        float(item["value"])
        for item in harmonizer_server_metrics["metrics"]["general"]["grpc.e2e_sent.render_rgb_ms"]
    ]
    harm_video = next(item for item in entries if item["status"] == "completed" and item["harmonizer_enabled"])
    raw_sequence_hashes = {camera: harmonizer_report["camera_metrics"][camera]["raw_sequence_sha256"] for camera in CAMERAS}
    processed_sequence_hashes = {camera: harmonizer_report["camera_metrics"][camera]["harmonized_sequence_sha256"] for camera in CAMERAS}
    quality = {
        "schema_version": "nsb.scene0061-final-quality.v1", "generated_at": generated_at,
        "status": "partial_fail_closed", "rgb_only_completed": 5, "multimodal_completed": 1,
        "blocked_video_count": 4, "raw_videos": raw_quality,
        "V02_edit_validation": read_json(ROOT / "outputs/nurec_scene0061_jitter_diagnostics/debug_report.json")["v02_same_timestamp_comparison"],
        "V02_A_A_B": read_json(ROOT / "outputs/nurec_scene0061_demo/V02/evidence.json")["probe"],
        "V03_camera_pose_delta": {
            key: value for key, value in read_json(ROOT / "outputs/nurec_scene0061_demo/V03/pose_sweep_summary.json").items()
            if key in ("max_translation", "max_yaw")
        },
        "V04_multimodal_alignment": v04_quality,
        "harmonizer_comparison": {
            "status": "V01_passed_V02_V03_blocked", "description": "NuRec RGB with NVIDIA Harmonizer post-processing",
            "video": harm_video, "service_port": 47443, "health_port": 47444,
            "checkpoint_sha256": HARMONIZER_SHA, "raw_rgb_payload_sequence_sha256": raw_sequence_hashes,
            "harmonized_rgb_payload_sequence_sha256": processed_sequence_hashes,
            "per_camera_metrics": harmonizer_report["camera_metrics"],
            "processed_temporal_flicker": video_flicker(ROOT / harm_video["path"]),
            "rpc_latency_ms": {
                **percentiles(harmonizer_latency_values),
                "metric": "historical official server grpc.e2e_sent.render_rgb_ms",
            },
            "invalid_pixel_ratio": 0.0,
            "interpretation": "Appearance post-processing lowered Laplacian variance on every camera; it is not a trajectory-jitter fix.",
        },
        "temporal_instability": read_json(ROOT / "outputs/nurec_scene0061_jitter_diagnostics/debug_report.json")["root_cause"],
    }
    manifest = {
        "schema_version": "nsb.scene0061-final-video-manifest.v1", "generated_at": generated_at,
        "expected_video_count": 10, "completed_video_count": sum(v["status"] == "completed" for v in entries),
        "blocked_video_count": sum(v["status"] == "blocked" for v in entries), "status": "partial_fail_closed",
        "videos": entries,
    }
    artifact_manifest = {
        "schema_version": "nsb.scene0061-final-artifact-manifest.v1", "generated_at": generated_at,
        "usdz": {"uri": ARTIFACT_URI, "sha256": ARTIFACT_SHA, "size_bytes": ARTIFACT_SIZE, "modified": False},
        "nurec": {"image": "nvcr.io/nvidia/nre/nre-ga:26.04", "version_id": "26.4.146", "commit": "c63f08a4"},
        "harmonizer": {
            "checkpoint": str(HARMONIZER), "checkpoint_sha256": HARMONIZER_SHA, "checkpoint_size_bytes": HARMONIZER_SIZE,
            "model_filename": "harmonizer_nontemporal.pt", "resolution": [576, 1024], "renderer": "default",
            "enable_editing_actors": True, "service_port": 47443, "health_port": 47444,
        },
    }
    write_json(FINAL / "video_manifest.json", manifest)
    write_json(FINAL / "quality_report.json", quality)
    write_json(FINAL / "artifact_manifest.json", artifact_manifest)
    print(json.dumps({"manifest": str(FINAL / "video_manifest.json"), "completed": manifest["completed_video_count"], "blocked": manifest["blocked_video_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
