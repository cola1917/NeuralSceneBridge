#!/usr/bin/env python3
"""Fail closed unless an NCore sequence contains usable NuRec actor tracks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class TrackSummary:
    track_id: str
    class_id: str
    source: str
    observation_count: int
    start_timestamp_us: int
    stop_timestamp_us: int
    displacement_m: float
    median_speed_ms: float
    eligible: bool


def source_name(source: Any) -> str:
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return name
    rendered = str(source)
    return rendered.rsplit(".", 1)[-1] if "." in rendered else rendered


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def summarize_tracks(
    observations: Iterable[Any],
    *,
    accepted_sources: set[str],
    accepted_classes: set[str],
    min_observations: int,
    min_displacement_m: float,
    min_median_speed_ms: float,
) -> tuple[list[TrackSummary], Counter[str], Counter[str]]:
    by_track: dict[str, list[Any]] = defaultdict(list)
    sources: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    for observation in observations:
        current_source = source_name(observation.source)
        sources[current_source] += 1
        classes[str(observation.class_id)] += 1
        if current_source in accepted_sources and str(observation.class_id) in accepted_classes:
            by_track[str(observation.track_id)].append(observation)

    summaries: list[TrackSummary] = []
    for track_id, items in by_track.items():
        items.sort(key=lambda item: int(item.timestamp_us))
        centroids = [tuple(float(value) for value in item.bbox3.centroid) for item in items]
        # A coordinate span is robust when a track returns close to its start.
        axes = list(zip(*centroids))
        displacement_m = math.sqrt(
            sum((max(axis) - min(axis)) ** 2 for axis in axes)
        )
        speeds: list[float] = []
        for left, right, left_xyz, right_xyz in zip(
            items, items[1:], centroids, centroids[1:]
        ):
            elapsed_s = (int(right.timestamp_us) - int(left.timestamp_us)) / 1_000_000
            if elapsed_s > 0:
                speeds.append(_distance(left_xyz, right_xyz) / elapsed_s)
        median_speed_ms = statistics.median(speeds) if speeds else 0.0
        eligible = (
            len(items) >= min_observations
            and displacement_m >= min_displacement_m
            and median_speed_ms >= min_median_speed_ms
        )
        summaries.append(
            TrackSummary(
                track_id=track_id,
                class_id=str(items[0].class_id),
                source=source_name(items[0].source),
                observation_count=len(items),
                start_timestamp_us=int(items[0].timestamp_us),
                stop_timestamp_us=int(items[-1].timestamp_us),
                displacement_m=round(displacement_m, 6),
                median_speed_ms=round(median_speed_ms, 6),
                eligible=eligible,
            )
        )
    summaries.sort(key=lambda item: (item.class_id, item.track_id))
    return summaries, sources, classes


def parse_csv(value: str) -> set[str]:
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return result


def load_observations(manifest: Path) -> list[Any]:
    try:
        from ncore.impl.data.v4.components import (  # type: ignore[import-not-found]
            CuboidsComponent,
            SequenceComponentGroupsReader,
        )
    except ImportError as exc:  # pragma: no cover - exercised in the NCore image
        raise RuntimeError("the NCore Python runtime is required") from exc

    reader = SequenceComponentGroupsReader([manifest])
    cuboid_readers = reader.open_component_readers(CuboidsComponent.Reader)
    if not cuboid_readers:
        raise ValueError("manifest has no CuboidsComponent")
    observations: list[Any] = []
    for cuboid_reader in cuboid_readers.values():
        observations.extend(cuboid_reader.get_observations())
    if not observations:
        raise ValueError("CuboidsComponent has no observations")
    return observations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--accepted-sources", type=parse_csv, required=True)
    parser.add_argument("--vehicle-classes", type=parse_csv, default={"automobile"})
    parser.add_argument("--pedestrian-classes", type=parse_csv, default={"pedestrian"})
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--min-displacement-m", type=float, default=1.0)
    parser.add_argument("--min-median-speed-ms", type=float, default=0.1)
    parser.add_argument("--min-eligible-vehicles", type=int, default=1)
    parser.add_argument("--min-eligible-pedestrians", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    observations = load_observations(manifest)
    accepted_classes = args.vehicle_classes | args.pedestrian_classes
    tracks, sources, classes = summarize_tracks(
        observations,
        accepted_sources=args.accepted_sources,
        accepted_classes=accepted_classes,
        min_observations=args.min_observations,
        min_displacement_m=args.min_displacement_m,
        min_median_speed_ms=args.min_median_speed_ms,
    )
    eligible = [track for track in tracks if track.eligible]
    vehicles = [track for track in eligible if track.class_id in args.vehicle_classes]
    pedestrians = [track for track in eligible if track.class_id in args.pedestrian_classes]
    report = {
        "schema_version": 1,
        "manifest": str(manifest),
        "observation_count": len(observations),
        "observation_sources": dict(sorted(sources.items())),
        "observation_classes": dict(sorted(classes.items())),
        "contract": {
            "accepted_sources": sorted(args.accepted_sources),
            "vehicle_classes": sorted(args.vehicle_classes),
            "pedestrian_classes": sorted(args.pedestrian_classes),
            "min_observations": args.min_observations,
            "min_displacement_m": args.min_displacement_m,
            "min_median_speed_ms": args.min_median_speed_ms,
        },
        "eligible_vehicle_count": len(vehicles),
        "eligible_pedestrian_count": len(pedestrians),
        "eligible_tracks": [asdict(track) for track in eligible],
        "pass": (
            len(vehicles) >= args.min_eligible_vehicles
            and len(pedestrians) >= args.min_eligible_pedestrians
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["pass"]:
        raise SystemExit(
            "dynamic-track gate failed: "
            f"eligible vehicles={len(vehicles)} (need {args.min_eligible_vehicles}), "
            f"pedestrians={len(pedestrians)} (need {args.min_eligible_pedestrians})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

