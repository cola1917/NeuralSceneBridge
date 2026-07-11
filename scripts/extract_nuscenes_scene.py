#!/usr/bin/env python3
"""Extract one nuScenes scene into a self-contained dataset tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


TABLES = [
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "map",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
]


def load_table(meta_dir: Path, name: str) -> list[dict[str, Any]]:
    path = meta_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing table: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dump_table(meta_dir: Path, name: str, rows: list[dict[str, Any]]) -> None:
    path = meta_dir / f"{name}.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def index_by_token(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["token"]: row for row in rows}


def extract_scene(
    source_root: Path,
    output_root: Path,
    scene_name: str,
    version: str,
    force: bool,
) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    meta_src = source_root / version
    if not meta_src.is_dir():
        raise FileNotFoundError(f"missing metadata directory: {meta_src}")

    tables = {name: load_table(meta_src, name) for name in TABLES}
    scenes = [row for row in tables["scene"] if row["name"] == scene_name]
    if not scenes:
        available = ", ".join(sorted(row["name"] for row in tables["scene"]))
        raise ValueError(f"scene not found: {scene_name}; available: {available}")
    if len(scenes) != 1:
        raise ValueError(f"expected one scene named {scene_name}, found {len(scenes)}")
    scene = scenes[0]

    samples = [row for row in tables["sample"] if row["scene_token"] == scene["token"]]
    sample_tokens = {row["token"] for row in samples}
    if not sample_tokens:
        raise ValueError(f"scene {scene_name} has no samples")

    sample_data = [row for row in tables["sample_data"] if row["sample_token"] in sample_tokens]
    sample_annotations = [
        row for row in tables["sample_annotation"] if row["sample_token"] in sample_tokens
    ]

    ego_pose_tokens = {row["ego_pose_token"] for row in sample_data}
    calibrated_sensor_tokens = {row["calibrated_sensor_token"] for row in sample_data}
    ego_poses = [row for row in tables["ego_pose"] if row["token"] in ego_pose_tokens]
    calibrated_sensors = [
        row for row in tables["calibrated_sensor"] if row["token"] in calibrated_sensor_tokens
    ]

    sensor_tokens = {row["sensor_token"] for row in calibrated_sensors}
    sensors = [row for row in tables["sensor"] if row["token"] in sensor_tokens]

    instance_tokens = {row["instance_token"] for row in sample_annotations}
    instances = [row for row in tables["instance"] if row["token"] in instance_tokens]
    category_tokens = {row["category_token"] for row in instances}
    categories = [row for row in tables["category"] if row["token"] in category_tokens]

    attribute_tokens: set[str] = set()
    for row in sample_annotations:
        attribute_tokens.update(row.get("attribute_tokens", []))
    attributes = [row for row in tables["attribute"] if row["token"] in attribute_tokens]

    visibility_tokens = {row["visibility_token"] for row in sample_annotations}
    visibilities = [row for row in tables["visibility"] if row["token"] in visibility_tokens]

    logs = [row for row in tables["log"] if row["token"] == scene["log_token"]]
    if not logs:
        raise ValueError(f"missing log for scene {scene_name}: {scene['log_token']}")
    log = logs[0]

    maps = [row for row in tables["map"] if scene["log_token"] in row.get("log_tokens", [])]
    for row in maps:
        row["log_tokens"] = [scene["log_token"]]

    # Keep sample prev/next chains intact within the extracted scene.
    sample_by_token = index_by_token(samples)
    for row in samples:
        if row["prev"] and row["prev"] not in sample_by_token:
            row["prev"] = ""
        if row["next"] and row["next"] not in sample_by_token:
            row["next"] = ""

    sample_data_by_token = index_by_token(sample_data)
    for row in sample_data:
        if row["prev"] and row["prev"] not in sample_data_by_token:
            row["prev"] = ""
        if row["next"] and row["next"] not in sample_data_by_token:
            row["next"] = ""

    ann_by_token = index_by_token(sample_annotations)
    for row in sample_annotations:
        if row["prev"] and row["prev"] not in ann_by_token:
            row["prev"] = ""
        if row["next"] and row["next"] not in ann_by_token:
            row["next"] = ""

    # Instance first/last annotation tokens must stay inside the subset.
    for row in instances:
        if row["first_annotation_token"] not in ann_by_token:
            raise ValueError(
                f"instance {row['token']} first_annotation_token missing from subset"
            )
        if row["last_annotation_token"] not in ann_by_token:
            raise ValueError(
                f"instance {row['token']} last_annotation_token missing from subset"
            )

    if output_root.exists():
        if not force:
            raise FileExistsError(f"output already exists: {output_root} (pass --force)")
        shutil.rmtree(output_root)

    meta_dst = output_root / version
    meta_dst.mkdir(parents=True, exist_ok=True)

    filtered = {
        "attribute": attributes,
        "calibrated_sensor": calibrated_sensors,
        "category": categories,
        "ego_pose": ego_poses,
        "instance": instances,
        "log": logs,
        "map": maps,
        "sample": samples,
        "sample_annotation": sample_annotations,
        "sample_data": sample_data,
        "scene": scenes,
        "sensor": sensors,
        "visibility": visibilities,
    }
    for name, rows in filtered.items():
        dump_table(meta_dst, name, rows)

    marker = output_root / f".{version}.txt"
    marker.write_text(f"extracted from {source_root.name} for {scene_name}\n", encoding="utf-8")

    copied = 0
    missing: list[str] = []
    bytes_copied = 0
    for row in sample_data:
        rel = Path(row["filename"])
        src = source_root / rel
        dst = output_root / rel
        if not src.exists():
            missing.append(str(rel))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        bytes_copied += src.stat().st_size

    for row in maps:
        rel = Path(row["filename"])
        src = source_root / rel
        dst = output_root / rel
        if not src.exists():
            missing.append(str(rel))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        bytes_copied += src.stat().st_size

    # Optional expansion JSON used by some map tooling.
    location = log.get("location")
    if location:
        expansion = source_root / "maps" / f"{location}.json"
        if expansion.exists():
            dst = output_root / "maps" / expansion.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(expansion, dst)
            copied += 1
            bytes_copied += expansion.stat().st_size

    if missing:
        preview = "\n  ".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} source files missing, first ones:\n  {preview}"
        )

    print(f"Extracted scene: {scene_name}")
    print(f"  source: {source_root}")
    print(f"  output: {output_root}")
    print(f"  samples: {len(samples)}")
    print(f"  sample_data: {len(sample_data)}")
    print(f"  sample_annotations: {len(sample_annotations)}")
    print(f"  files copied: {copied}")
    print(f"  size: {bytes_copied / (1024 * 1024):.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/nuscenes-mini"),
        help="Full nuScenes root to extract from.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/nuscenes-mini-scene-0061"),
        help="Destination dataset root.",
    )
    parser.add_argument("--scene-name", default="scene-0061")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    args = parser.parse_args()
    extract_scene(
        source_root=args.source_root,
        output_root=args.output_root,
        scene_name=args.scene_name,
        version=args.version,
        force=args.force,
    )


if __name__ == "__main__":
    main()
