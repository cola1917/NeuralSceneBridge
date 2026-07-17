#!/usr/bin/env python3
"""Fail-closed validation for dynamic actor tracks embedded in a NuRec USDZ."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import zipfile


DEFAULT_VEHICLE_CLASSES = (
    "automobile,bus,heavy_truck,Other Vehicle - Construction Vehicle,"
    "Emergency Vehicle,trailer"
)
DEFAULT_PEDESTRIAN_CLASSES = "pedestrian"


class ValidationError(ValueError):
    """Raised when a USDZ does not satisfy the dynamic-track contract."""


def _csv(value: str) -> set[str]:
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result:
        raise argparse.ArgumentTypeError("class list must not be empty")
    return result


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{location} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{location} must be finite, got {value!r}")
    return number


def _sequence_chunks(payload: object) -> list[tuple[str, dict]]:
    if not isinstance(payload, dict):
        raise ValidationError("sequence_tracks.json must contain a JSON object")
    if "tracks_data" in payload:
        return [("$", payload)]
    chunks = [
        (str(name), chunk)
        for name, chunk in payload.items()
        if isinstance(chunk, dict) and "tracks_data" in chunk
    ]
    if not chunks:
        raise ValidationError("sequence_tracks.json contains no track chunks")
    return chunks


def validate_usdz(
    usdz_path: Path,
    *,
    min_total: int,
    min_vehicles: int,
    min_pedestrians: int,
    vehicle_classes: set[str],
    pedestrian_classes: set[str],
) -> dict:
    try:
        with zipfile.ZipFile(usdz_path) as archive:
            matches = [
                info
                for info in archive.infolist()
                if Path(info.filename).name == "sequence_tracks.json"
            ]
            if len(matches) != 1:
                raise ValidationError(
                    "USDZ must contain exactly one sequence_tracks.json; "
                    f"found {[item.filename for item in matches]!r}"
                )
            info = matches[0]
            try:
                payload = json.loads(archive.read(info).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValidationError(f"invalid sequence_tracks.json: {exc}") from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"cannot read USDZ archive {usdz_path}: {exc}") from exc

    labels: Counter[str] = Counter()
    seen_ids: set[str] = set()
    total_tracks = 0
    chunks = _sequence_chunks(payload)
    array_names = (
        "tracks_id",
        "tracks_poses",
        "tracks_timestamps_us",
        "tracks_label_class",
        "tracks_flags",
    )

    for chunk_name, chunk in chunks:
        tracks_data = chunk.get("tracks_data")
        cuboid_data = chunk.get("cuboidtracks_data")
        if not isinstance(tracks_data, dict):
            raise ValidationError(f"chunk {chunk_name}: tracks_data must be an object")
        if not isinstance(cuboid_data, dict):
            raise ValidationError(f"chunk {chunk_name}: cuboidtracks_data must be an object")

        arrays: dict[str, list] = {}
        for name in array_names:
            value = tracks_data.get(name)
            if not isinstance(value, list):
                raise ValidationError(f"chunk {chunk_name}: {name} must be an array")
            arrays[name] = value
        dims = cuboid_data.get("cuboids_dims")
        if not isinstance(dims, list):
            raise ValidationError(f"chunk {chunk_name}: cuboids_dims must be an array")

        lengths = {name: len(value) for name, value in arrays.items()}
        lengths["cuboids_dims"] = len(dims)
        if len(set(lengths.values())) != 1:
            raise ValidationError(f"chunk {chunk_name}: track array lengths differ: {lengths}")

        for index in range(lengths["tracks_id"]):
            location = f"chunk {chunk_name} track[{index}]"
            raw_id = arrays["tracks_id"][index]
            track_id = str(raw_id).strip()
            if not track_id:
                raise ValidationError(f"{location}: track id must not be empty")
            if track_id in seen_ids:
                raise ValidationError(f"{location}: duplicate global track id {track_id!r}")
            seen_ids.add(track_id)

            label = arrays["tracks_label_class"][index]
            if not isinstance(label, str) or not label.strip():
                raise ValidationError(f"{location}: label class must be a non-empty string")
            label = label.strip()
            labels[label] += 1

            poses = arrays["tracks_poses"][index]
            timestamps = arrays["tracks_timestamps_us"][index]
            if not isinstance(poses, list) or not poses:
                raise ValidationError(f"{location}: poses must be a non-empty array")
            if not isinstance(timestamps, list) or len(timestamps) != len(poses):
                raise ValidationError(
                    f"{location}: timestamps and poses must have equal non-zero lengths"
                )
            previous_timestamp: int | None = None
            for observation, (pose, timestamp) in enumerate(zip(poses, timestamps)):
                obs_location = f"{location} observation[{observation}]"
                if not isinstance(pose, list) or len(pose) != 7:
                    raise ValidationError(f"{obs_location}: pose must contain 7 numbers")
                for coordinate, value in enumerate(pose):
                    _finite_number(value, f"{obs_location} pose[{coordinate}]")
                if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                    raise ValidationError(f"{obs_location}: timestamp must be an integer")
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    raise ValidationError(f"{location}: timestamps must be strictly increasing")
                previous_timestamp = timestamp

            track_dims = dims[index]
            if not isinstance(track_dims, list) or len(track_dims) != 3:
                raise ValidationError(f"{location}: cuboid dimensions must contain 3 numbers")
            if any(
                _finite_number(value, f"{location} dimension[{axis}]") <= 0
                for axis, value in enumerate(track_dims)
            ):
                raise ValidationError(f"{location}: cuboid dimensions must be positive")
            total_tracks += 1

    vehicle_count = sum(labels[name] for name in vehicle_classes)
    pedestrian_count = sum(labels[name] for name in pedestrian_classes)
    failures = []
    if total_tracks < min_total:
        failures.append(f"tracks {total_tracks} < required {min_total}")
    if vehicle_count < min_vehicles:
        failures.append(f"vehicle tracks {vehicle_count} < required {min_vehicles}")
    if pedestrian_count < min_pedestrians:
        failures.append(f"pedestrian tracks {pedestrian_count} < required {min_pedestrians}")
    if failures:
        raise ValidationError("dynamic-track thresholds failed: " + "; ".join(failures))

    return {
        "schema": "nsb.nurec-usdz-dynamic-tracks",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "usdz": str(usdz_path.resolve()),
        "sequence_tracks_member": info.filename,
        "sequence_tracks_compressed_bytes": info.compress_size,
        "sequence_tracks_uncompressed_bytes": info.file_size,
        "chunk_count": len(chunks),
        "track_count": total_tracks,
        "vehicle_track_count": vehicle_count,
        "pedestrian_track_count": pedestrian_count,
        "label_counts": dict(sorted(labels.items())),
        "thresholds": {
            "min_total": min_total,
            "min_vehicles": min_vehicles,
            "min_pedestrians": min_pedestrians,
        },
        "pass": True,
    }


def _nonnegative(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usdz", type=Path)
    parser.add_argument("--min-total", type=_nonnegative, default=1)
    parser.add_argument("--min-vehicles", type=_nonnegative, default=1)
    parser.add_argument("--min-pedestrians", type=_nonnegative, default=1)
    parser.add_argument("--vehicle-classes", type=_csv, default=_csv(DEFAULT_VEHICLE_CLASSES))
    parser.add_argument(
        "--pedestrian-classes", type=_csv, default=_csv(DEFAULT_PEDESTRIAN_CLASSES)
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    try:
        report = validate_usdz(
            args.usdz,
            min_total=args.min_total,
            min_vehicles=args.min_vehicles,
            min_pedestrians=args.min_pedestrians,
            vehicle_classes=args.vehicle_classes,
            pedestrian_classes=args.pedestrian_classes,
        )
    except ValidationError as exc:
        print(f"NuRec USDZ dynamic-track validation failed: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
