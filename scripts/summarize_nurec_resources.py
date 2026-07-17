#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def summarize(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"resource log contains no samples: {path}")

    def values(key: str) -> list[float]:
        return [value for row in rows if (value := _number(row, key)) is not None]

    started = datetime.fromisoformat(rows[0]["timestamp_utc"].replace("Z", "+00:00"))
    finished = datetime.fromisoformat(rows[-1]["timestamp_utc"].replace("Z", "+00:00"))
    gpu_used = values("gpu_memory_used_mib")
    gpu_total = values("gpu_memory_total_mib")
    gpu_utilization = values("gpu_utilization_percent")
    host_available = values("host_memory_available_kib")
    disk_available = values("disk_available_kib")
    return {
        "schema_version": 1,
        "sample_count": len(rows),
        "started_at_utc": rows[0]["timestamp_utc"],
        "finished_at_utc": rows[-1]["timestamp_utc"],
        "observed_duration_seconds": (finished - started).total_seconds(),
        "gpu_memory_peak_mib": max(gpu_used) if gpu_used else None,
        "gpu_memory_total_mib": max(gpu_total) if gpu_total else None,
        "gpu_utilization_peak_percent": max(gpu_utilization) if gpu_utilization else None,
        "host_memory_available_min_kib": min(host_available) if host_available else None,
        "disk_available_min_kib": min(disk_available) if disk_available else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("resource_log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.resource_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
