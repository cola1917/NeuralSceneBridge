from __future__ import annotations

import unittest

from scripts.validate_ncore_output import validate_conversion_provenance


class ValidateNCoreOutputTests(unittest.TestCase):
    def test_accepts_matching_dense_provenance(self) -> None:
        validate_conversion_provenance(
            {
                "generic_meta_data": {
                    "cuboid_sampling": "lidar-sweeps",
                    "cuboid_label_source": "EXTERNAL",
                    "cuboid_class_schema": "nre-26.04-car2sim",
                    "conversion_provenance_version": 2,
                }
            },
            "lidar-sweeps",
        )

    def test_rejects_legacy_manifest_without_dense_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            validate_conversion_provenance(
                {"generic_meta_data": {"source_dataset": "nuscenes"}},
                "lidar-sweeps",
            )
