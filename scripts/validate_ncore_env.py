from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable


REQUIRED_DATA_FILES = [
    "attribute.json",
    "calibrated_sensor.json",
    "category.json",
    "ego_pose.json",
    "instance.json",
    "log.json",
    "map.json",
    "sample.json",
    "sample_annotation.json",
    "sample_data.json",
    "scene.json",
    "sensor.json",
    "visibility.json",
]


MODULE_CHECKS = [
    ("ncore", "nvidia-ncore"),
    ("nuscenes", "nuscenes-devkit"),
    ("pyquaternion", "pyquaternion"),
    ("upath", "universal-pathlib"),
    ("numpy", "numpy"),
    ("click", "click"),
    ("tqdm", "tqdm"),
    ("tools.data_converter.nuscenes.main", None),
    ("tools.data_converter.nuscenes.converter", None),
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def package_version(dist_name: str | None) -> str:
    if not dist_name:
        return "vendored"
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def ensure_converter_root(converter_root: Path) -> Path:
    converter_root = converter_root.resolve()
    if not converter_root.exists():
        fail(f"converter root does not exist: {converter_root}")
    expected = converter_root / "tools" / "data_converter" / "nuscenes" / "main.py"
    if not expected.exists():
        fail(f"missing converter entry point: {expected}")
    if str(converter_root) not in sys.path:
        sys.path.insert(0, str(converter_root))
    return converter_root


def check_imports() -> None:
    print("Checking Python imports:")
    for module_name, dist_name in MODULE_CHECKS:
        importlib.import_module(module_name)
        print(f"  OK {module_name} ({package_version(dist_name)})")


def run_converter_help(converter_root: Path) -> None:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(converter_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    cmd = [sys.executable, "-m", "tools.data_converter.nuscenes.main", "--help"]
    result = subprocess.run(
        cmd,
        cwd=str(converter_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        fail(f"converter CLI help failed with exit code {result.returncode}")
    first_line = result.stdout.splitlines()[0] if result.stdout else "<empty help>"
    print(f"Converter CLI help: {first_line}")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_sensor_lookup(root: Path, version: str) -> tuple[dict[str, str], dict[str, dict]]:
    version_root = root / version
    calibrated = read_json(version_root / "calibrated_sensor.json")
    sensors = read_json(version_root / "sensor.json")
    cal_to_sensor = {row["token"]: row["sensor_token"] for row in calibrated}
    sensor_by_token = {row["token"]: row for row in sensors}
    return cal_to_sensor, sensor_by_token


def summarize_scene(root: Path, version: str, scene_name: str) -> dict[str, int | float | str]:
    version_root = root / version
    scenes = read_json(version_root / "scene.json")
    samples = read_json(version_root / "sample.json")
    sample_data = read_json(version_root / "sample_data.json")
    cal_to_sensor, sensor_by_token = build_sensor_lookup(root, version)

    scene = next((row for row in scenes if row.get("name") == scene_name), None)
    if not scene:
        available = ", ".join(row.get("name", "<unnamed>") for row in scenes[:10])
        fail(f"scene {scene_name!r} not found. First scenes: {available}")

    scene_samples = [row for row in samples if row["scene_token"] == scene["token"]]
    if not scene_samples:
        fail(f"scene {scene_name!r} has no samples")
    scene_samples.sort(key=lambda row: row["timestamp"])
    sample_tokens = {row["token"] for row in scene_samples}

    camera_keyframes = 0
    lidar_keyframes = 0
    radar_keyframes = 0
    missing_files: list[str] = []
    total_sample_data = 0
    for row in sample_data:
        if row["sample_token"] not in sample_tokens:
            continue
        total_sample_data += 1
        file_path = root / row["filename"]
        if not file_path.exists():
            missing_files.append(row["filename"])
        sensor = sensor_by_token[cal_to_sensor[row["calibrated_sensor_token"]]]
        if row["is_key_frame"] and sensor["modality"] == "camera":
            camera_keyframes += 1
        elif row["is_key_frame"] and sensor["modality"] == "lidar":
            lidar_keyframes += 1
        elif row["is_key_frame"] and sensor["modality"] == "radar":
            radar_keyframes += 1

    if missing_files:
        preview = ", ".join(missing_files[:5])
        fail(f"{len(missing_files)} sample_data files are missing, first missing: {preview}")

    duration_seconds = (scene_samples[-1]["timestamp"] - scene_samples[0]["timestamp"]) / 1_000_000.0
    return {
        "scene": scene_name,
        "samples": len(scene_samples),
        "duration_seconds": round(duration_seconds, 2),
        "sample_data_rows": total_sample_data,
        "camera_keyframes": camera_keyframes,
        "lidar_keyframes": lidar_keyframes,
        "radar_keyframes": radar_keyframes,
    }


def check_nuscenes_root(root: Path, version: str, scene_name: str) -> None:
    root = root.resolve()
    print(f"Checking nuScenes root: {root}")
    if not root.exists():
        fail(f"nuScenes root does not exist: {root}")
    for dirname in ["samples", "sweeps", "maps", version]:
        path = root / dirname
        if not path.exists():
            fail(f"missing nuScenes directory: {path}")
    for filename in REQUIRED_DATA_FILES:
        path = root / version / filename
        if not path.exists():
            fail(f"missing nuScenes metadata file: {path}")
    summary = summarize_scene(root, version, scene_name)
    print("nuScenes scene summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def check_ncore_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        path = path.resolve()
        if not path.exists():
            fail(f"NCore output path does not exist: {path}")
        if path.suffix == ".json":
            data = read_json(path)
            print(f"Checked NCore JSON: {path}")
            print(f"  top-level keys: {', '.join(sorted(data.keys())[:12])}")
        else:
            print(f"Checked NCore output path exists: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the NCore converter environment and optional nuScenes data.")
    parser.add_argument("--converter-root", default="third_party/ncore_converter", help="Path containing the vendored tools/ package.")
    parser.add_argument("--nuscenes-root", help="Optional nuScenes dataset root to validate.")
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes metadata version.")
    parser.add_argument("--scene-name", default="scene-0061", help="nuScenes scene name to validate.")
    parser.add_argument("--ncore-output", action="append", default=[], help="Optional NCore JSON or store path to check.")
    parser.add_argument("--skip-imports", action="store_true", help="Skip Python package and converter import checks.")
    parser.add_argument("--skip-cli-help", action="store_true", help="Skip running converter --help.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        converter_root = ensure_converter_root(Path(args.converter_root))
        if not args.skip_imports:
            check_imports()
        else:
            print("Skipping Python import checks.")
        if not args.skip_imports and not args.skip_cli_help:
            run_converter_help(converter_root)
        elif args.skip_imports and not args.skip_cli_help:
            print("Skipping converter CLI help because imports are skipped.")
        if args.nuscenes_root:
            check_nuscenes_root(Path(args.nuscenes_root), args.version, args.scene_name)
        if args.ncore_output:
            check_ncore_outputs(Path(path) for path in args.ncore_output)
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
