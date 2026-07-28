import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "derive_nurec_controllable_tracks_usdz.py"
SPEC = importlib.util.spec_from_file_location("derive_controllable", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _sequence() -> dict:
    return {
        "chunk": {
            "tracks_data": {
                "tracks_id": ["fixed", "already"],
                "tracks_flags": ["NONE", "DYNAMIC|CONTROLLABLE"],
            }
        }
    }


class DeriveControllableTracksUsdTests(unittest.TestCase):
    def test_derives_aligned_stored_usdz_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.usdz"
            output = root / "output.usdz"
            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("scene/sequence_tracks.json", json.dumps(_sequence()))
                archive.writestr("scene/other.bin", b"payload")

            result = MODULE.derive_usdz(source, output, ["fixed", "already"])

            self.assertEqual(result["modified_tracks"][0]["previous_flag"], "DYNAMIC|CONTROLLABLE")
            self.assertEqual(result["modified_tracks"][1]["previous_flag"], "NONE")
            with zipfile.ZipFile(source) as archive:
                original = json.loads(archive.read("scene/sequence_tracks.json"))
            self.assertEqual(original["chunk"]["tracks_data"]["tracks_flags"][0], "NONE")
            with zipfile.ZipFile(output) as archive:
                derived = json.loads(archive.read("scene/sequence_tracks.json"))
                self.assertEqual(
                    derived["chunk"]["tracks_data"]["tracks_flags"],
                    ["DYNAMIC|CONTROLLABLE", "DYNAMIC|CONTROLLABLE"],
                )
                for info in archive.infolist():
                    offset = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra)
                    self.assertEqual(offset % 64, 0)

    def test_rejects_unknown_track(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.usdz"
            with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("sequence_tracks.json", json.dumps(_sequence()))
            with self.assertRaisesRegex(MODULE.DerivationError, "absent"):
                MODULE.derive_usdz(source, root / "output.usdz", ["missing"])


if __name__ == "__main__":
    unittest.main()
