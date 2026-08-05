import importlib.util
import unittest

import numpy as np


_HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(_HAS_CV2, "requires OpenCV")
class RenderMultimodalConsistencyTests(unittest.TestCase):
    def test_v04_target_pose_maps_artifact_right_forward_to_response_forward_right(self):
        from scripts.render_multimodal_alignment_video import (
            TARGET_TRACK_ID,
            _target_response_position,
        )

        class Scene:
            @staticmethod
            def dynamic_objects(timestamp_us, **kwargs):
                self.assertEqual(timestamp_us, 100)
                delta = kwargs.get("target_delta") or {}
                return [{
                    "track_id": TARGET_TRACK_ID,
                    "pose": [
                        2.0 + float(delta.get("x", 0.0)),
                        10.0 + float(delta.get("y", 0.0)),
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                }]

            @staticmethod
            def lidar_pose_matrix(lidar_id, timestamp_us):
                self.assertEqual(lidar_id, "lidar_top")
                self.assertEqual(timestamp_us, 100)
                return np.eye(4).tolist()

        position = _target_response_position(Scene(), 100, None)
        np.testing.assert_allclose(position, [10.0, 2.0, 1.0])

    def test_v04_target_roi_selects_only_real_points_in_footprint(self):
        from scripts.render_multimodal_alignment_video import _target_roi

        points = np.asarray(
            [[10.0, -2.0, 0.0], [12.9, -0.6, 1.0], [13.1, -2.0, 0.0], [10.0, -0.4, 0.0]],
            dtype=np.float32,
        )
        selected = _target_roi(points, np.asarray([10.0, -2.0, 0.0], dtype=np.float32))
        self.assertEqual(selected.tolist(), [True, True, False, False])

    def test_v04_target_roi_rotates_with_target_yaw(self):
        from scripts.render_multimodal_alignment_video import _target_roi

        points = np.asarray(
            [[12.0, 0.0, 0.0], [10.0, 2.9, 0.0], [10.0, 1.4, 0.0], [8.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        selected = _target_roi(
            points,
            np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
            np.pi / 2.0,
        )
        self.assertEqual(selected.tolist(), [False, True, True, False])

    def test_v04_response_to_artifact_axes_swaps_horizontal_basis(self):
        from scripts.render_multimodal_alignment_video import RESPONSE_TO_ARTIFACT_AXES

        np.testing.assert_array_equal(
            RESPONSE_TO_ARTIFACT_AXES,
            np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )

    def test_v04_voxel_difference_removes_a_a_control_change(self):
        from scripts.render_multimodal_alignment_video import _voxel_difference

        baseline = np.asarray(
            [[1.01, 2.01, 0.0, 0.5], [4.01, 5.01, 0.0, 0.4]],
            dtype=np.float32,
        )
        control = np.asarray(
            [[1.01, 2.01, 0.0, 0.5], [4.11, 5.01, 0.0, 0.4]],
            dtype=np.float32,
        )
        edited = np.asarray(
            [[1.01, 2.01, 0.0, 0.5], [4.21, 5.01, 0.0, 0.4]],
            dtype=np.float32,
        )
        difference = _voxel_difference(baseline, control, edited)
        self.assertEqual(len(difference["control_added"]), 1)
        self.assertEqual(len(difference["signal_added"]), 1)
        self.assertEqual(len(difference["signal_removed"]), 0)

    def test_v04_rgb_difference_uses_repeat_control_threshold(self):
        from scripts.render_multimodal_alignment_video import _rgb_difference

        baseline = np.zeros((4, 4, 3), dtype=np.uint8)
        control = baseline.copy()
        edited = baseline.copy()
        edited[1, 2] = (30, 30, 30)
        difference = _rgb_difference(baseline, control, edited)
        self.assertEqual(difference["signal_pixel_count"], 1)
        self.assertEqual(difference["control_mean_abs_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
