#!/usr/bin/env python3
"""Validate an NCore manifest and all component-store checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    stores = manifest.get("component_stores")
    if manifest.get("version") != "v4":
        raise ValueError(f"unsupported NCore version: {manifest.get('version')!r}")
    if not isinstance(stores, list) or not stores:
        raise ValueError("manifest has no component_stores")

    seen: set[str] = set()
    component_groups: set[str] = set()
    total_bytes = 0
    for store in stores:
        relative_path = store.get("path")
        expected_md5 = store.get("md5")
        components = store.get("components")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("component store has an invalid path")
        if relative_path in seen:
            raise ValueError(f"duplicate component store: {relative_path}")
        if not isinstance(expected_md5, str) or len(expected_md5) != 32:
            raise ValueError(f"invalid MD5 for {relative_path}")
        if not isinstance(components, dict) or not components:
            raise ValueError(f"no components declared for {relative_path}")

        store_path = manifest_path.parent / relative_path
        if not store_path.is_file():
            raise FileNotFoundError(store_path)
        actual_md5 = md5(store_path)
        if actual_md5 != expected_md5.lower():
            raise ValueError(
                f"checksum mismatch for {relative_path}: "
                f"expected {expected_md5}, got {actual_md5}"
            )

        seen.add(relative_path)
        component_groups.update(components)
        total_bytes += store_path.stat().st_size
        print(f"OK  {relative_path}  {actual_md5}")

    interval = manifest.get("sequence_timestamp_interval_us", {})
    start = interval.get("start")
    stop = interval.get("stop")
    if not isinstance(start, int) or not isinstance(stop, int) or stop <= start:
        raise ValueError("invalid sequence timestamp interval")

    print(
        "NCORE OUTPUT VALIDATION OK "
        f"(sequence={manifest.get('sequence_id')}, stores={len(stores)}, "
        f"components={','.join(sorted(component_groups))}, bytes={total_bytes})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
