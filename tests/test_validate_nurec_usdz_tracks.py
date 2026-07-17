import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_nurec_usdz_tracks.py"


def _track(track_id, label, *, observations=2):
    timestamps = [1_000_000 + index * 50_000 for index in range(observations)]
    poses = [[float(index), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for index in range(observations)]
    return track_id, poses, timestamps, label, 0, [4.0, 2.0, 1.5]


def _payload(*tracks):
    columns = list(zip(*tracks)) if tracks else [()] * 6
    return {
        "dummy_chunk_id": {
            "tracks_data": {
                "tracks_id": list(columns[0]),
                "tracks_poses": list(columns[1]),
                "tracks_timestamps_us": list(columns[2]),
                "tracks_label_class": list(columns[3]),
                "tracks_flags": list(columns[4]),
            },
            "cuboidtracks_data": {"cuboids_dims": list(columns[5])},
        }
    }


class ValidateNuRecUsdzTracksTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.usdz = Path(self.tempdir.name) / "last.usdz"

    def tearDown(self):
        self.tempdir.cleanup()

    def _write(self, payload):
        with zipfile.ZipFile(self.usdz, "w") as archive:
            archive.writestr("scene/sequence_tracks.json", json.dumps(payload))

    def _run(self, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.usdz), *extra],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_nested_vehicle_and_pedestrian_tracks(self):
        self._write(_payload(_track("car-1", "automobile"), _track("ped-1", "pedestrian")))
        result = self._run("--min-total", "2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["track_count"], 2)
        self.assertEqual(report["vehicle_track_count"], 1)
        self.assertEqual(report["pedestrian_track_count"], 1)

    def test_rejects_empty_track_arrays(self):
        self._write(_payload())
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tracks 0 < required 1", result.stderr)

    def test_rejects_mismatched_parallel_arrays(self):
        payload = _payload(_track("car-1", "automobile"))
        payload["dummy_chunk_id"]["tracks_data"]["tracks_flags"] = []
        self._write(payload)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("track array lengths differ", result.stderr)

    def test_rejects_missing_required_actor_classes(self):
        self._write(_payload(_track("cone-1", "traffic_cone")))
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("vehicle tracks 0 < required 1", result.stderr)
        self.assertIn("pedestrian tracks 0 < required 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
