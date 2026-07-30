#!/usr/bin/env python3
"""Derive an aligned NuRec USDZ with selected source tracks controllable.

This is an artifact-native, immutable derivation. It never edits the source
USDZ and changes only ``tracks_flags`` entries identified by explicit track
IDs. The output ZIP keeps every member stored and 64-byte data aligned, as
required by USDZ consumers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any


ALIGNMENT = 64
SEQUENCE_MEMBER = "sequence_tracks.json"
TARGET_FLAG = "DYNAMIC|CONTROLLABLE"


class DerivationError(ValueError):
    """Raised when a source USDZ cannot be safely derived."""


def derive_usdz(source: Path, output: Path, track_ids: list[str]) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    targets = sorted(set(str(value).strip() for value in track_ids if str(value).strip()))
    if not targets:
        raise DerivationError("at least one non-empty --track-id is required")
    if source == output:
        raise DerivationError("output must differ from source")
    if output.exists():
        raise DerivationError(f"refusing to overwrite output: {output}")

    with zipfile.ZipFile(source, "r") as archive:
        members = archive.infolist()
        sequence_infos = [info for info in members if Path(info.filename).name == SEQUENCE_MEMBER]
        if len(sequence_infos) != 1:
            raise DerivationError("source USDZ must contain exactly one sequence_tracks.json")
        if any(info.compress_type != zipfile.ZIP_STORED for info in members):
            raise DerivationError("source USDZ has compressed members; refusing to repack")
        sequence_info = sequence_infos[0]
        sequence = _load_sequence(archive.read(sequence_info))
        changed = _mark_controllable(sequence, targets)
        payloads = {
            info.filename: (
                json.dumps(sequence, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                if info.filename == sequence_info.filename
                else archive.read(info)
            )
            for info in members
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for info in members:
            archive.writestr(_aligned_zip_info(archive, info), payloads[info.filename])

    _verify_output(output, targets)
    return {
        "schema_version": "nsb.nurec_controllable_tracks_usdz.v1",
        "source": _file_identity(source),
        "output": _file_identity(output),
        "target_flag": TARGET_FLAG,
        "modified_tracks": changed,
        "member_alignment_bytes": ALIGNMENT,
    }


def _load_sequence(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DerivationError("sequence_tracks.json is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or not value:
        raise DerivationError("sequence_tracks.json must be a non-empty object")
    return value


def _mark_controllable(sequence: dict[str, Any], targets: list[str]) -> list[dict[str, str]]:
    by_id: dict[str, tuple[list[Any], int]] = {}
    for chunk_name, chunk in sequence.items():
        if not isinstance(chunk, dict) or not isinstance(chunk.get("tracks_data"), dict):
            raise DerivationError(f"invalid tracks_data chunk: {chunk_name}")
        data = chunk["tracks_data"]
        ids = data.get("tracks_id")
        flags = data.get("tracks_flags")
        if not isinstance(ids, list) or not isinstance(flags, list) or len(ids) != len(flags):
            raise DerivationError(f"track IDs/flags mismatch in chunk: {chunk_name}")
        for index, track_id in enumerate(ids):
            value = str(track_id)
            if value in by_id:
                raise DerivationError(f"duplicate track ID in sequence: {value}")
            by_id[value] = (flags, index)
    missing = sorted(set(targets) - set(by_id))
    if missing:
        raise DerivationError("target tracks absent from sequence: " + ", ".join(missing))
    changed = []
    for track_id in targets:
        flags, index = by_id[track_id]
        previous = flags[index]
        if not isinstance(previous, str):
            raise DerivationError(f"track {track_id} has non-string flag")
        flags[index] = TARGET_FLAG
        changed.append({"track_id": track_id, "previous_flag": previous, "new_flag": TARGET_FLAG})
    return changed


def _aligned_zip_info(archive: zipfile.ZipFile, source: zipfile.ZipInfo) -> zipfile.ZipInfo:
    result = zipfile.ZipInfo(source.filename, date_time=source.date_time)
    result.compress_type = zipfile.ZIP_STORED
    result.external_attr = source.external_attr
    result.internal_attr = source.internal_attr
    result.create_system = source.create_system
    result.create_version = source.create_version
    result.extract_version = source.extract_version
    result.flag_bits = source.flag_bits & ~0x08
    result.comment = source.comment
    header_size = 30 + len(source.filename.encode("utf-8"))
    padding = (-(archive.fp.tell() + header_size)) % ALIGNMENT
    if 0 < padding < 4:
        padding += ALIGNMENT
    result.extra = (
        b"" if padding == 0 else struct.pack("<HH", 0xFFFF, padding - 4) + b"\0" * (padding - 4)
    )
    return result


def _verify_output(output: Path, targets: list[str]) -> None:
    with zipfile.ZipFile(output, "r") as archive:
        infos = archive.infolist()
        if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
            raise DerivationError("derived USDZ contains compressed members")
        for info in infos:
            offset = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra)
            if offset % ALIGNMENT != 0:
                raise DerivationError(f"derived USDZ member is not {ALIGNMENT}-byte aligned: {info.filename}")
        sequence_info = next(
            (info for info in infos if Path(info.filename).name == SEQUENCE_MEMBER), None
        )
        if sequence_info is None:
            raise DerivationError("derived USDZ lost sequence_tracks.json")
        sequence = _load_sequence(archive.read(sequence_info))
    observed = {
        str(track_id): flag
        for chunk in sequence.values()
        if isinstance(chunk, dict) and isinstance(chunk.get("tracks_data"), dict)
        for track_id, flag in zip(
            chunk["tracks_data"].get("tracks_id") or [],
            chunk["tracks_data"].get("tracks_flags") or [],
        )
    }
    missing = [track_id for track_id in targets if observed.get(track_id) != TARGET_FLAG]
    if missing:
        raise DerivationError("derived USDZ did not retain controllable flags: " + ", ".join(missing))


def _file_identity(path: Path) -> dict[str, Any]:
    body = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--track-id", action="append", default=[])
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.manifest.exists():
            raise DerivationError(f"refusing to overwrite manifest: {args.manifest}")
        result = derive_usdz(args.source, args.output, args.track_id)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, DerivationError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
