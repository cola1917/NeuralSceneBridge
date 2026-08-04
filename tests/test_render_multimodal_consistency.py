import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np


class RenderMultimodalConsistencyTests(unittest.TestCase):
    def test_xyzi_voxel_signature_detects_a_local_actor_change(self):
        from scripts.render_multimodal_consistency import _voxel_signature

        baseline = np.asarray(
            [[1.01, 2.02, 0.0, 0.5], [4.01, 5.02, 0.0, 0.4]], dtype=np.float32
        )
        edited = np.asarray(
            [[1.11, 2.02, 0.0, 0.5], [4.01, 5.02, 0.0, 0.4]], dtype=np.float32
        )
        self.assertNotEqual(_voxel_signature(baseline), _voxel_signature(edited))

    def test_world_to_sensor_uses_rigid_inverse(self):
        from scripts.render_multimodal_consistency import _world_to_sensor

        matrix = [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 3], [0, 0, 0, 1]]
        self.assertEqual(_world_to_sensor(matrix, [11, 22, 3]), [1.0, 2.0, 0.0])

    def test_file_reference_is_relative_and_hash_bound(self):
        from scripts.render_multimodal_consistency import _file_ref

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            path = root / "lidar" / "baseline.xyzi.bin"
            path.parent.mkdir()
            payload = b"xyzi"
            path.write_bytes(payload)
            ref = _file_ref(
                path,
                root=root,
                kind="lidar",
                encoding="float32_xyzi_little_endian",
            )
            self.assertEqual(ref["path"], "lidar/baseline.xyzi.bin")
            self.assertEqual(ref["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(ref["size_bytes"], len(payload))

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


if __name__ == "__main__":
    unittest.main()
