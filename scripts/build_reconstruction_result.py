#!/usr/bin/env python3
"""Build NeuralSceneBridge's canonical reconstruction_result.v1 envelope."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


_CONTRACT_SRC = Path(__file__).resolve().parents[2] / "SceneExchangeContracts" / "src"
if str(_CONTRACT_SRC) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_SRC))

from scene_exchange_contracts import validate_artifact_reference, validate_shared_document


def build_reconstruction_result(
    request: dict[str, Any],
    *,
    reconstruction_package_reference: dict[str, Any],
    artifact_references: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    producer: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    validate_shared_document(request)
    if request.get("schema_version") != "reconstruction_request.v1":
        raise ValueError("request must use reconstruction_request.v1")
    payload = request["payload"]
    source = payload["source"]
    if source.get("scene_key_kind") != "token":
        raise ValueError("reconstruction requests must address the converter by scene token")
    if source.get("scene_token") != payload["scene_id"]:
        raise ValueError("source.scene_token must equal payload.scene_id")
    validate_artifact_reference(reconstruction_package_reference)
    if reconstruction_package_reference.get("role") != "reconstruction_package":
        raise ValueError("package reference role must be reconstruction_package")
    if reconstruction_package_reference.get("media_type") != "application/json":
        raise ValueError("Reconstruction Package must use application/json")

    artifacts = []
    for reference in [*artifact_references, reconstruction_package_reference]:
        validate_artifact_reference(reference)
        if not any(
            existing["role"] == reference["role"] and existing["path"] == reference["path"]
            for existing in artifacts
        ):
            artifacts.append(deepcopy(reference))
    roles = {reference["role"] for reference in artifacts}
    requested = set(payload["requested_products"])
    product_status = {
        "ncore_dataset": _product_status("ncore_dataset", requested, bool(roles & {"ncore_manifest", "ncore_store"})),
        "nurec_reconstruction": _product_status("nurec_reconstruction", requested, bool(roles & {"nurec_usdz", "nurec_checkpoint"})),
        "reconstruction_package": _product_status("reconstruction_package", requested, "reconstruction_package" in roles),
    }
    status = "succeeded" if all(
        value in {"succeeded", "not_requested"} for value in product_status.values()
    ) else "partial"
    metric_payload = deepcopy(metrics or {})
    metric_payload["duration_sec"] = _duration(started_at, finished_at)
    metric_payload.setdefault("backend", "nurec")
    result = {
        "protocol_version": "shared_exchange_protocol.v1",
        "schema_version": "reconstruction_result.v1",
        "message_id": f"msg-result-{payload['job_id']}",
        "message_type": "reconstruction.build.result",
        "created_at": finished_at,
        "producer": deepcopy(producer),
        "correlation": {
            "correlation_id": request["correlation"]["correlation_id"],
            "root_message_id": request["correlation"]["root_message_id"],
            "causation_message_id": request["message_id"],
        },
        "idempotency": {
            "key": f"reconstruction-result/{payload['scene_id']}/{payload.get('scene_version', 'v001')}",
            "scope": "scene",
        },
        "payload": {
            "job_id": payload["job_id"],
            "scene_id": payload["scene_id"],
            "scene_version": payload.get("scene_version", "v001"),
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "artifacts": artifacts,
            "product_status": product_status,
            "metrics": metric_payload,
            "warnings": list(warnings or []),
        },
    }
    validate_shared_document(result)
    return result


def _product_status(product: str, requested: set[str], present: bool) -> str:
    if product not in requested:
        return "not_requested"
    return "succeeded" if present else "failed"


def _duration(started_at: str, finished_at: str) -> float:
    parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    duration = (parse(finished_at) - parse(started_at)).total_seconds()
    if duration < 0:
        raise ValueError("finished_at precedes started_at")
    return duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--package-reference", required=True, type=Path)
    parser.add_argument("--artifact-reference", action="append", type=Path, default=[])
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--finished-at", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    result = build_reconstruction_result(
        load(args.request),
        reconstruction_package_reference=load(args.package_reference),
        artifact_references=[load(path) for path in args.artifact_reference],
        started_at=args.started_at,
        finished_at=args.finished_at,
        producer={"project": "NeuralSceneBridge", "component": "nurec-build-worker", "version": "1.0.0"},
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
