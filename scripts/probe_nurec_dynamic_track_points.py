#!/usr/bin/env python3
"""Probe NuRec's dynamic-track point extraction without training a model.

Run this script inside the pinned NuRec image with the same parsed training
configuration and NCore dataset that a failed training attempt used. It keeps
the vendor implementation intact and records the rows it returns before the
Gaussian initializer can silently replace missing observations with random
points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Protocol


class _PointCloud(Protocol):
    n_points: int


class _TrackPointCloud(Protocol):
    track_id: str
    point_cloud: _PointCloud


def summarize_track_point_clouds(
    rows: Iterable[_TrackPointCloud], track_ids: set[str]
) -> dict[str, Any]:
    """Summarize emitted rows, retaining zero-point rows as evidence."""

    tracks = {
        track_id: {
            "track_id": track_id,
            "emitted_row_count": 0,
            "nonempty_row_count": 0,
            "point_count": 0,
        }
        for track_id in sorted(track_ids)
    }
    unexpected_track_ids: set[str] = set()
    for row in rows:
        track_id = str(row.track_id)
        if track_id not in tracks:
            unexpected_track_ids.add(track_id)
            continue
        record = tracks[track_id]
        count = int(row.point_cloud.n_points)
        record["emitted_row_count"] += 1
        record["nonempty_row_count"] += int(count > 0)
        record["point_count"] += count

    return {
        "tracks": list(tracks.values()),
        "unexpected_track_ids": sorted(unexpected_track_ids),
        "emitted_row_count": sum(int(item["emitted_row_count"]) for item in tracks.values()),
        "nonempty_row_count": sum(int(item["nonempty_row_count"]) for item in tracks.values()),
        "point_count": sum(int(item["point_count"]) for item in tracks.values()),
    }


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NuRec dynamic-track extraction")


def probe(
    *,
    parsed_config: Path,
    track_ids: set[str],
    keep_all_track_poses: bool,
    step_frame: int,
) -> dict[str, Any]:
    """Run matching extraction modes against the NuRec/NCore implementation."""

    _require_cuda()
    from omegaconf import OmegaConf
    import torch
    from nre.config.dataset import NCoreDatasetConfig
    from nre.config.sensor import LidarModelsConfig
    from nre.datasets.ncore import NCOREDataSource
    from nre.datasets.tracks import CuboidTracks

    raw_config = OmegaConf.to_container(OmegaConf.load(parsed_config), resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("parsed NuRec config must be a mapping")
    dataset_raw = raw_config.get("dataset")
    sensor_raw = raw_config.get("sensor")
    if not isinstance(dataset_raw, dict) or not isinstance(sensor_raw, dict):
        raise ValueError("parsed NuRec config requires dataset and sensor mappings")
    dataset_config = NCoreDatasetConfig.model_validate(dataset_raw)
    lidar_models_raw = sensor_raw.get("lidar_models")
    lidar_models = (
        LidarModelsConfig.model_validate(lidar_models_raw)
        if isinstance(lidar_models_raw, dict)
        else None
    )
    source = NCOREDataSource(dataset_config, lidar_models)
    source._maybe_init_worker()

    all_tracks = source.get_cuboid_tracks(dynamic_only=False, world_frame=False)
    available_track_ids = set(all_tracks.tracks_id)
    missing_track_ids = sorted(track_ids - available_track_ids)
    if missing_track_ids:
        raise ValueError("requested tracks absent from NuRec datasource: " + ", ".join(missing_track_ids))
    selected_tracks = CuboidTracks.Ops.subset_from_tracks_id(all_tracks, sorted(track_ids))

    modes = []
    for return_color in (False, True):
        rows = source.get_track_point_clouds(
            selected_tracks,
            cuboid_dim_scale_factor=1.0,
            lidar_ids=list(dataset_config.lidar_ids),
            camera_ids=None,
            return_color=return_color,
            keep_all_track_poses=keep_all_track_poses,
            step_frame=step_frame,
            device=torch.device("cuda"),
        )
        summary = summarize_track_point_clouds(rows, track_ids)
        summary["return_color"] = return_color
        summary["status"] = "passed" if summary["point_count"] > 0 else "failed"
        modes.append(summary)

    return {
        "schema_version": "nurec_dynamic_track_point_probe.v1",
        "status": "passed" if all(mode["status"] == "passed" for mode in modes) else "failed",
        "parsed_config": str(parsed_config),
        "requested_track_ids": sorted(track_ids),
        "available_track_count": len(available_track_ids),
        "selected_track_pose_count": int(len(selected_tracks.tracks_timestamps_us)),
        "keep_all_track_poses": keep_all_track_poses,
        "step_frame": step_frame,
        "modes": modes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-config", required=True, type=Path)
    parser.add_argument("--track-id", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keep-all-track-poses", action="store_true")
    parser.add_argument("--step-frame", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if args.step_frame < 1:
        parser.error("--step-frame must be positive")

    try:
        report = probe(
            parsed_config=args.parsed_config,
            track_ids={str(track_id) for track_id in args.track_id},
            keep_all_track_poses=args.keep_all_track_poses,
            step_frame=args.step_frame,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        report = {
            "schema_version": "nurec_dynamic_track_point_probe.v1",
            "status": "failed",
            "detail": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
