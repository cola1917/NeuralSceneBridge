#!/usr/bin/env python3
"""Validate per-frame NuRec LiDAR metrics and point-cloud evidence.

NRE 26.04 contains a vendor bug in ``BaseSystem.collect_metric``: the
``is_lidar`` argument is unconditionally reset to ``False``.  LiDAR metrics
are therefore written below ``per_camera`` and ``per_lidar`` remains empty.
This validator fails closed on that layout unless the caller explicitly
allows the known 26.04 defect.  The exception is reported, never normalized
silently.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml


DEFAULT_REQUIRED_METRICS = (
    "test/chamfer_distance",
    "test/raydrop_accuracy",
)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _metric_samples(
    sensor_groups: dict[str, Any], metric_name: str
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sequence in sensor_groups.values():
        sequence_map = _mapping(sequence, "metrics.per_sequence.<sequence>")
        for sensor_metrics in sequence_map.values():
            sensor_map = _mapping(sensor_metrics, "metrics sensor entry")
            raw_samples = sensor_map.get(metric_name, [])
            if not isinstance(raw_samples, list):
                raise ValueError(f"{metric_name} samples must be a list")
            for sample in raw_samples:
                samples.append(_mapping(sample, f"{metric_name} sample"))
    return samples


def _sensor_groups(per_sequence: dict[str, Any], group: str) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for sequence_id, sequence in per_sequence.items():
        sequence_map = _mapping(sequence, f"metrics.per_sequence.{sequence_id}")
        sensors = _mapping(
            sequence_map.get(group, {}),
            f"metrics.per_sequence.{sequence_id}.{group}",
        )
        grouped[str(sequence_id)] = sensors
    return grouped


def _validate_samples(
    samples: Iterable[dict[str, Any]], metric_name: str, minimum: int
) -> int:
    materialized = list(samples)
    if len(materialized) < minimum:
        raise ValueError(
            f"{metric_name} has {len(materialized)} frame samples; expected at least {minimum}"
        )
    frame_ids: set[int] = set()
    for sample in materialized:
        frame_id = sample.get("unique_frame_idx")
        timestamp_begin = sample.get("timestamp_us_begin")
        timestamp_end = sample.get("timestamp_us_end")
        value = sample.get("value")
        if not isinstance(frame_id, int):
            raise ValueError(f"{metric_name} sample lacks integer unique_frame_idx")
        if not isinstance(timestamp_begin, int) or not isinstance(timestamp_end, int):
            raise ValueError(f"{metric_name} frame {frame_id} lacks integer timestamps")
        if timestamp_end < timestamp_begin:
            raise ValueError(f"{metric_name} frame {frame_id} has reversed timestamps")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{metric_name} frame {frame_id} has non-numeric value")
        if not math.isfinite(float(value)):
            raise ValueError(f"{metric_name} frame {frame_id} has non-finite value")
        frame_ids.add(frame_id)
    if len(frame_ids) < minimum:
        raise ValueError(
            f"{metric_name} has {len(frame_ids)} unique frames; expected at least {minimum}"
        )
    return len(frame_ids)


def _ply_pair_count(validation_dir: Path) -> tuple[int, list[str], list[str]]:
    point_cloud_dir = validation_dir / "pred_pc"
    predicted: dict[str, Path] = {}
    ground_truth: dict[str, Path] = {}
    if point_cloud_dir.is_dir():
        for path in point_cloud_dir.glob("*output.ply"):
            predicted[path.name[: -len("output.ply")]] = path
        for path in point_cloud_dir.glob("*output_gt.ply"):
            ground_truth[path.name[: -len("output_gt.ply")]] = path

    for path in (*predicted.values(), *ground_truth.values()):
        if path.stat().st_size <= 0:
            raise ValueError(f"empty LiDAR point cloud: {path}")

    missing_gt = sorted(set(predicted) - set(ground_truth))
    missing_prediction = sorted(set(ground_truth) - set(predicted))
    return len(set(predicted) & set(ground_truth)), missing_gt, missing_prediction


def validate(args: argparse.Namespace) -> str:
    with args.metrics.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    root = _mapping(payload, "metrics document")
    aggregated = _mapping(root.get("aggregated_metrics", {}), "aggregated_metrics")
    metrics = _mapping(root.get("metrics", {}), "metrics")
    per_sequence = _mapping(metrics.get("per_sequence", {}), "metrics.per_sequence")
    if not per_sequence:
        raise ValueError("metrics.per_sequence is empty")

    for metric_name in args.required_metrics:
        aggregate = _mapping(
            aggregated.get(metric_name), f"aggregated_metrics.{metric_name}"
        )
        value = aggregate.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"aggregated {metric_name} has non-numeric value")
        if not math.isfinite(float(value)):
            raise ValueError(f"aggregated {metric_name} has non-finite value")

    lidar_groups = _sensor_groups(per_sequence, "per_lidar")
    camera_groups = _sensor_groups(per_sequence, "per_camera")
    lidar_samples = {
        name: _metric_samples(lidar_groups, name) for name in args.required_metrics
    }
    camera_samples = {
        name: _metric_samples(camera_groups, name) for name in args.required_metrics
    }

    if all(lidar_samples.values()):
        classification = "native_per_lidar"
        selected = lidar_samples
    elif all(camera_samples.values()) and not any(lidar_samples.values()):
        classification = "nre_26_04_vendor_grouping_bug"
        if not args.allow_nre_2604_lidar_grouping_bug:
            raise ValueError(
                "NRE 26.04 LiDAR grouping bug detected: required LiDAR metrics are "
                "under per_camera while per_lidar is empty; set "
                "--allow-nre-2604-lidar-grouping-bug only for an explicitly audited 26.04 run"
            )
        selected = camera_samples
    else:
        locations = {
            name: {
                "per_lidar": len(lidar_samples[name]),
                "per_camera": len(camera_samples[name]),
            }
            for name in args.required_metrics
        }
        raise ValueError(f"incomplete or mixed LiDAR metric grouping: {locations}")

    frame_counts = {
        name: _validate_samples(samples, name, args.min_frame_samples)
        for name, samples in selected.items()
    }

    pair_count, missing_gt, missing_prediction = _ply_pair_count(args.metrics.parent)
    if missing_gt or missing_prediction:
        raise ValueError(
            "unpaired LiDAR PLY files: "
            f"missing_gt={missing_gt[:10]}, missing_prediction={missing_prediction[:10]}"
        )
    if pair_count < args.min_ply_pairs:
        raise ValueError(
            f"LiDAR PLY pair count is {pair_count}; expected at least {args.min_ply_pairs}"
        )

    return (
        "NUREC LIDAR VALIDATION EVIDENCE OK\n"
        f"  classification: {classification}\n"
        f"  frame samples: {frame_counts}\n"
        f"  predicted/GT PLY pairs: {pair_count}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--required-metrics", default=",".join(DEFAULT_REQUIRED_METRICS))
    parser.add_argument("--min-frame-samples", type=int, default=1)
    parser.add_argument("--min-ply-pairs", type=int, default=1)
    parser.add_argument("--allow-nre-2604-lidar-grouping-bug", action="store_true")
    args = parser.parse_args(argv)
    args.required_metrics = tuple(
        item.strip() for item in args.required_metrics.split(",") if item.strip()
    )
    if not args.required_metrics:
        parser.error("--required-metrics must contain at least one metric")
    if args.min_frame_samples < 1:
        parser.error("--min-frame-samples must be positive")
    if args.min_ply_pairs < 1:
        parser.error("--min-ply-pairs must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        print(validate(args))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"NuRec LiDAR evidence gate failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
