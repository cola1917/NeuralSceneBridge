from __future__ import annotations

import unittest

from third_party.ncore_converter.tools.data_converter.nuscenes.contract import (
    NUREC_DYNAMIC_DEFORMABLE_CLASSES,
    NUREC_DYNAMIC_RIGID_CLASSES,
    NUREC_TRACK_LABEL_SOURCE,
    NUSCENES_CATEGORY_MAP,
)


class NuScenesDynamicContractTests(unittest.TestCase):
    def test_actor_classes_match_nurec_layers(self) -> None:
        self.assertEqual(NUREC_TRACK_LABEL_SOURCE, "EXTERNAL")
        self.assertIn(NUSCENES_CATEGORY_MAP["vehicle.car"], NUREC_DYNAMIC_RIGID_CLASSES)
        self.assertIn(
            NUSCENES_CATEGORY_MAP["human.pedestrian.adult"],
            NUREC_DYNAMIC_DEFORMABLE_CLASSES,
        )

    def test_legacy_unmatched_vehicle_names_are_not_emitted(self) -> None:
        self.assertNotIn("car", NUSCENES_CATEGORY_MAP.values())
        self.assertNotIn("truck", NUSCENES_CATEGORY_MAP.values())
        self.assertNotIn("construction_vehicle", NUSCENES_CATEGORY_MAP.values())
        self.assertNotIn("emergency_vehicle", NUSCENES_CATEGORY_MAP.values())


if __name__ == "__main__":
    unittest.main()
