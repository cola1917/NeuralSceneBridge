from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

try:
    import yaml
except ModuleNotFoundError:
    yaml = types.ModuleType("yaml")
    yaml.safe_load = json.load
    yaml.safe_dump = json.dumps
    yaml.YAMLError = ValueError
    sys.modules["yaml"] = yaml

from scripts.validate_nurec_lidar_metrics import validate


class ValidateNuRecLidarMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.validation_dir = Path(self.tempdir.name) / "val"
        self.validation_dir.mkdir()
        self.metrics_path = self.validation_dir / "metrics.yaml"
        point_cloud_dir = self.validation_dir / "pred_pc"
        point_cloud_dir.mkdir()
        (point_cloud_dir / "000001output.ply").write_bytes(b"pred")
        (point_cloud_dir / "000001output_gt.ply").write_bytes(b"gt")

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _sample(name: str):
        return {
            "name": name,
            "unique_frame_idx": 1,
            "timestamp_us_begin": 100,
            "timestamp_us_end": 110,
            "value": 0.5,
        }

    def _write_metrics(self, group: str = "per_lidar", include_raydrop: bool = True):
        names = ["test/chamfer_distance"]
        if include_raydrop:
            names.append("test/raydrop_accuracy")
        sensor_metrics = {name: [self._sample(name)] for name in names}
        payload = {
            "aggregated_metrics": {
                name: {"aggregation_method": "mean", "value": 0.5}
                for name in names
            },
            "metrics": {
                "general": {},
                "per_sequence": {
                    "scene-0061": {
                        "per_camera": {"camera_back": sensor_metrics}
                        if group == "per_camera"
                        else {},
                        "per_lidar": {"lidar_top": sensor_metrics}
                        if group == "per_lidar"
                        else {},
                    }
                },
            },
        }
        self.metrics_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    def _args(self, allow_bug: bool = False):
        return argparse.Namespace(
            metrics=self.metrics_path,
            required_metrics=("test/chamfer_distance", "test/raydrop_accuracy"),
            min_frame_samples=1,
            min_ply_pairs=1,
            allow_nre_2604_lidar_grouping_bug=allow_bug,
        )

    def test_accepts_native_per_lidar_metrics_and_ply_pair(self):
        self._write_metrics()
        output = validate(self._args())
        self.assertIn("classification: native_per_lidar", output)
        self.assertIn("predicted/GT PLY pairs: 1", output)

    def test_rejects_nre_2604_grouping_bug_by_default(self):
        self._write_metrics(group="per_camera")
        with self.assertRaisesRegex(ValueError, "grouping bug detected"):
            validate(self._args())

    def test_explicitly_reports_allowed_nre_2604_grouping_bug(self):
        self._write_metrics(group="per_camera")
        output = validate(self._args(allow_bug=True))
        self.assertIn("classification: nre_26_04_vendor_grouping_bug", output)

    def test_rejects_missing_required_metric(self):
        self._write_metrics(include_raydrop=False)
        with self.assertRaisesRegex(ValueError, "aggregated_metrics.test/raydrop_accuracy"):
            validate(self._args())

    def test_rejects_unpaired_point_cloud(self):
        self._write_metrics()
        (self.validation_dir / "pred_pc" / "000001output_gt.ply").unlink()
        with self.assertRaisesRegex(ValueError, "unpaired LiDAR PLY files"):
            validate(self._args())


if __name__ == "__main__":
    unittest.main()
