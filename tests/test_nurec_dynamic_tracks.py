from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.validate_ncore_dynamic_tracks import source_name, summarize_tracks


def observation(track_id: str, class_id: str, timestamp_us: int, x: float):
    return SimpleNamespace(
        track_id=track_id,
        class_id=class_id,
        source=SimpleNamespace(name="EXTERNAL"),
        timestamp_us=timestamp_us,
        bbox3=SimpleNamespace(centroid=(x, 0.0, 0.0)),
    )


class DynamicTrackValidationTests(unittest.TestCase):
    def test_source_name_prefers_enum_name(self) -> None:
        self.assertEqual(source_name(SimpleNamespace(name="EXTERNAL")), "EXTERNAL")

    def test_eligible_track_requires_matching_source_class_and_motion(self) -> None:
        tracks, sources, classes = summarize_tracks(
            [
                observation("car-1", "automobile", 0, 0.0),
                observation("car-1", "automobile", 1_000_000, 2.0),
                observation("legacy", "car", 0, 0.0),
                observation("legacy", "car", 1_000_000, 3.0),
            ],
            accepted_sources={"EXTERNAL"},
            accepted_classes={"automobile", "pedestrian"},
            min_observations=2,
            min_displacement_m=1.0,
            min_median_speed_ms=0.1,
        )
        self.assertEqual(sources, {"EXTERNAL": 4})
        self.assertEqual(classes, {"automobile": 2, "car": 2})
        self.assertEqual(len(tracks), 1)
        self.assertTrue(tracks[0].eligible)
        self.assertEqual(tracks[0].displacement_m, 2.0)

    def test_stationary_track_is_not_eligible(self) -> None:
        tracks, _, _ = summarize_tracks(
            [
                observation("ped-1", "pedestrian", 0, 1.0),
                observation("ped-1", "pedestrian", 1_000_000, 1.0),
            ],
            accepted_sources={"EXTERNAL"},
            accepted_classes={"pedestrian"},
            min_observations=2,
            min_displacement_m=1.0,
            min_median_speed_ms=0.1,
        )
        self.assertFalse(tracks[0].eligible)


if __name__ == "__main__":
    unittest.main()

