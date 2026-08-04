#!/usr/bin/env python3
"""Build a deterministic Reconstruction Package from explicit NuRec artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import sys
from typing import Any


_CONTRACT_SRC = Path(__file__).resolve().parents[2] / "SceneExchangeContracts" / "src"
if str(_CONTRACT_SRC) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_SRC))

from scene_exchange_contracts import validate_document


SCHEMA_VERSION = "reconstruction_package.v1"
ROLE_MEDIA_TYPES = {
    "ncore_manifest": "application/json",
    "ncore_store": "application/octet-stream",
    "nurec_usdz": "model/vnd.usdz+zip",
    "nurec_checkpoint": "application/octet-stream",
    "nurec_config": "application/yaml",
    "build_log": "text/plain",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(package_dir: Path, role: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        relative = resolved.relative_to(package_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact must be contained by package directory: {resolved}") from exc
    media_type = ROLE_MEDIA_TYPES.get(role) or mimetypes.guess_type(resolved.name)[0]
    return {
        "role": role,
        "path": relative,
        "media_type": media_type or "application/octet-stream",
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def build_reconstruction_package(
    *,
    output: Path,
    scene_token: str | None = None,
    scene_name: str | None = None,
    dataset_version: str | None = None,
    artifacts: list[tuple[str, Path]],
    backend: str = "nurec",
    backend_version: str | None = None,
    requested_window: tuple[float, float] | None = None,
    actual_window: tuple[float, float] | None = None,
    scenario_ir_path: Path | None = None,
) -> dict[str, Any]:
    scenario_ir_identity = None
    if scenario_ir_path is not None:
        scenario_ir, scenario_ir_sha256 = _load_scenario_ir(scenario_ir_path)
        ir_source = scenario_ir["source"]
        ir_scene_token = scenario_ir["scenario_id"]
        ir_scene_name = ir_source["scene_name"]
        ir_dataset_version = ir_source["version"]
        ir_window = scenario_ir["windows"]["reconstruction"]
        ir_requested_window = (float(ir_window["start_sec"]), float(ir_window["end_sec"]))
        for label, explicit, derived in (
            ("scene_token", scene_token, ir_scene_token),
            ("scene_name", scene_name, ir_scene_name),
            ("dataset_version", dataset_version, ir_dataset_version),
        ):
            if explicit is not None and explicit != derived:
                raise ValueError(f"{label} does not match Scenario IR")
        if requested_window is not None and tuple(map(float, requested_window)) != ir_requested_window:
            raise ValueError("requested_window does not match Scenario IR reconstruction window")
        scene_token = ir_scene_token
        scene_name = ir_scene_name
        dataset_version = ir_dataset_version
        requested_window = ir_requested_window
        scenario_ir_identity = {
            "schema_version": "scenario_ir.v1",
            "scenario_id": ir_scene_token,
            "sha256": scenario_ir_sha256,
        }

    if scene_token is None or scene_name is None:
        raise ValueError(
            "scene_token and scene_name are required unless scenario_ir_path is provided"
        )
    if dataset_version is None:
        dataset_version = "v1.0-mini"
    if len(scene_token) != 32 or any(char not in "0123456789abcdef" for char in scene_token):
        raise ValueError("scene_token must be 32 lowercase hexadecimal characters")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = [_artifact(output.parent, role, path) for role, path in artifacts]
    roles = [item["role"] for item in inventory]
    if len(roles) != len(set(roles)):
        raise ValueError("artifact roles must be unique")
    warnings = []
    coverage_mode = "full_scene"
    if requested_window is not None and actual_window is not None:
        if requested_window != actual_window:
            warnings.append("requested reconstruction window was not cropped; full-scene coverage was produced")
        else:
            coverage_mode = "window"
    elif requested_window is not None:
        warnings.append("actual reconstruction coverage is unknown")
        coverage_mode = "unknown"
    package = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_token,
        **({"scenario_ir": scenario_ir_identity} if scenario_ir_identity is not None else {}),
        "source": {
            "dataset": "nuscenes",
            "dataset_version": dataset_version,
            "scene_name": scene_name,
            "scene_token": scene_token,
        },
        "backend": {
            "name": backend,
            "version": backend_version,
        },
        "coverage": {
            "mode": coverage_mode,
            "requested_window": _window(requested_window),
            "actual_window": _window(actual_window),
        },
        "artifacts": inventory,
        "alignment": {
            "log_coordinate_frame": "nuscenes_global",
            "visual_coordinate_frame": "nurec_output",
            "sim_from_log_transform": None,
            "status": "pending_runtime_alignment",
        },
        "warnings": warnings,
    }
    validate_document(package)
    output.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return package


def _load_scenario_ir(path: Path) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Scenario IR is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError("Scenario IR must be a JSON object")
    try:
        validate_document(value)
    except ValueError as exc:
        raise ValueError(f"Scenario IR validation failed: {exc}") from exc
    window = ((value.get("windows") or {}).get("reconstruction") or {})
    if float(window.get("end_sec", 0)) <= float(window.get("start_sec", 0)):
        raise ValueError("Scenario IR reconstruction window must have positive duration")
    return value, hashlib.sha256(raw).hexdigest()


def _window(value: tuple[float, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    start, end = value
    if start < 0 or end <= start:
        raise ValueError("window end must be greater than a non-negative start")
    return {"start_sec": float(start), "end_sec": float(end)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenario-ir", type=Path, help="Validated scenario_ir.v1 JSON from TriggerEngine.")
    parser.add_argument("--scene-token")
    parser.add_argument("--scene-name")
    parser.add_argument("--dataset-version")
    parser.add_argument("--backend", default="nurec")
    parser.add_argument("--backend-version")
    parser.add_argument("--artifact", action="append", nargs=2, metavar=("ROLE", "PATH"), default=[])
    parser.add_argument("--requested-window", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--actual-window", nargs=2, type=float, metavar=("START", "END"))
    args = parser.parse_args(argv)
    if args.scenario_ir is None and not all(
        value is not None for value in (args.scene_token, args.scene_name)
    ):
        parser.error("--scenario-ir or both --scene-token and --scene-name is required")
    package = build_reconstruction_package(
        output=args.output,
        scene_token=args.scene_token,
        scene_name=args.scene_name,
        dataset_version=args.dataset_version,
        artifacts=[(role, Path(path)) for role, path in args.artifact],
        backend=args.backend,
        backend_version=args.backend_version,
        requested_window=tuple(args.requested_window) if args.requested_window else None,
        actual_window=tuple(args.actual_window) if args.actual_window else None,
        scenario_ir_path=args.scenario_ir,
    )
    print(json.dumps({"package": str(args.output), "artifacts": len(package["artifacts"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
