import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile


_HAS_CV2 = importlib.util.find_spec("cv2") is not None


class RenderCounterfactualVideoTests(unittest.TestCase):
    def test_decodes_nre_260_buffered_lidar_with_legacy_proto(self):
        from scripts.render_counterfactual_video import _decode_lidar_response

        def varint(value):
            body = bytearray()
            while value > 0x7F:
                body.append((value & 0x7F) | 0x80)
                value >>= 7
            body.append(value)
            return bytes(body)

        def wire_varint(field, value):
            return varint(field << 3) + varint(value)

        def wire_bytes(field, value):
            return varint((field << 3) | 2) + varint(len(value)) + value

        xyz_buffer = struct.pack("<6f", 1, 2, 3, 4, 5, 6)
        intensity_buffer = struct.pack("<2f", 0.25, 0.75)
        serialized = (
            wire_varint(3, 2)
            + wire_bytes(4, xyz_buffer)
            + wire_bytes(5, intensity_buffer)
        )

        class LegacyParsedResponse:
            point_xyzs = []
            point_intensities = []

            @staticmethod
            def SerializeToString():
                return serialized

        xyz, intensities, encoding = _decode_lidar_response(LegacyParsedResponse())
        self.assertEqual(xyz, [1, 2, 3, 4, 5, 6])
        self.assertEqual(intensities, [0.25, 0.75])
        self.assertEqual(encoding, "nre_26_04_unknown_buffers")

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.usdz = self.root / "scene.usdz"
        rig = {
            "rig_trajectories": [
                {
                    "sequence_id": "scene-0061",
                    "T_rig_world_timestamps_us": [100, 200, 300],
                    "T_rig_worlds": [
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    ],
                }
            ],
            "camera_calibrations": {
                "camera_front@scene-0061": {
                    "T_sensor_rig": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "camera_model": {"parameters": {"fx": 1.0}},
                }
            },
            "lidar_calibrations": {
                "lidar_top@scene-0061": {
                    "T_sensor_rig": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "lidar_model": {"type": "row-offset-spinning", "parameters": {"spinning_frequency_hz": 20.0}},
                }
            },
        }
        sequence = {
            "chunk": {
                "tracks_data": {
                    "tracks_id": ["target", "static"],
                    "tracks_poses": [
                        [[0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 1, 0, 0, 0]],
                        [[5, 0, 0, 1, 0, 0, 0], [5, 0, 0, 1, 0, 0, 0]],
                    ],
                    "tracks_timestamps_us": [[100, 300], [100, 300]],
                    "tracks_flags": ["DYNAMIC|CONTROLLABLE", "STATIC|NONE"],
                    "tracks_label_class": ["automobile", "barrier"],
                }
            }
        }
        with zipfile.ZipFile(self.usdz, "w") as archive:
            archive.writestr("rig_trajectories.json", json.dumps(rig))
            archive.writestr("sequence_tracks.json", json.dumps(sequence))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_reads_tracks_and_interpolates_target_pose(self):
        from scripts.render_counterfactual_video import ArtifactScene

        scene = ArtifactScene(self.usdz, "target")
        self.assertEqual(scene.controllable_track_ids, ["target"])
        objects = scene.dynamic_objects(200, target_track_id="target", target_delta={"x": 0.5})
        self.assertEqual(objects[0]["track_id"], "target")
        self.assertAlmostEqual(objects[0]["pose"][0], 1.5)

    def test_dynamic_pose_pair_uses_the_requested_interval_and_target_delta(self):
        from scripts.render_counterfactual_video import ArtifactScene

        scene = ArtifactScene(self.usdz, "target")
        objects = scene.dynamic_objects(
            100,
            end_timestamp_us=300,
            target_track_id="target",
            target_delta={"x": 0.5},
        )
        self.assertEqual(objects[0]["pose"][0], 0.5)
        self.assertEqual(objects[0]["pose_pair"]["start"][0], 0.5)
        self.assertEqual(objects[0]["pose_pair"]["end"][0], 2.5)

    def test_target_only_mode_selects_only_the_manifest_target(self):
        from scripts.render_counterfactual_video import ArtifactScene

        scene = ArtifactScene(self.usdz, "target")
        objects = scene.dynamic_objects(200, mode="target_only", target_track_id="target")
        self.assertEqual([item["track_id"] for item in objects], ["target"])

    def test_interpolation_refuses_to_cross_large_source_time_gaps(self):
        from scripts.render_counterfactual_video import (
            RenderError,
            _interpolate_matrix,
            _interpolate_track_pose,
        )

        with self.assertRaisesRegex(RenderError, "source time gap"):
            _interpolate_track_pose(
                {
                    "timestamps_us": [0, 100_000],
                    "poses": [
                        [0, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 1, 0, 0, 0],
                    ],
                },
                50_000,
            )
        with self.assertRaisesRegex(RenderError, "source time gap"):
            _interpolate_matrix(
                [0, 100_000],
                [
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                ],
                50_000,
            )

    def test_playback_override_can_interpolate_a_declared_large_actor_gap(self):
        from scripts.render_counterfactual_video import (
            _interpolate_matrix,
            _interpolate_track_pose,
        )

        pose = _interpolate_track_pose(
            {
                "track_id": "playback-gap",
                "timestamps_us": [0, 500_000],
                "poses": [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            },
            250_000,
            max_gap_us=600_000,
        )

        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose[0], 0.5)
        matrix = _interpolate_matrix(
            [0, 500_000],
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ],
            250_000,
            max_gap_us=600_000,
        )
        self.assertAlmostEqual(matrix[0][3], 0.5)

    def test_reads_lidar_calibration_and_interpolates_sensor_pose(self):
        from scripts.render_counterfactual_video import ArtifactScene

        scene = ArtifactScene(self.usdz, "target")
        pose = scene.lidar_pose_matrix("lidar_top", 200)
        self.assertAlmostEqual(pose[0][3], 1.0)
        self.assertEqual(scene.lidar_model("lidar_top")["type"], "row-offset-spinning")

    def test_missing_target_track_fails_closed(self):
        from scripts.render_counterfactual_video import ArtifactScene, RenderError

        with self.assertRaises(RenderError):
            ArtifactScene(self.usdz, "missing-target")

    def test_bare_case_id_resolves_to_checked_in_case_file(self):
        from scripts.render_counterfactual_video import resolve_case

        path, case = resolve_case("V01")
        self.assertEqual(path.name, "original_replay.json")
        self.assertEqual(case["case_id"], "V01")

    def test_timestamp_selection_and_sensor_offset_are_bounded(self):
        from scripts.render_counterfactual_video import (
            ArtifactScene,
            RenderError,
            _apply_camera_offset,
            _timestamp_values,
            _validate_sweep_bounds,
        )

        scene = ArtifactScene(self.usdz, "target")
        case = {"timestamp_range_us": {"start": 100, "end": 300}, "frame_step": 2}
        self.assertEqual(_timestamp_values(scene, case, None, None, None), [100, 300])
        shifted = _apply_camera_offset(scene.sensor_pose_matrix("camera_front", 100), translation_m={"x": 0.2}, frame="sensor")
        self.assertAlmostEqual(shifted[0][3], 0.2)
        _validate_sweep_bounds({"camera_sweep": {"translation_m": {"x": 0.1}, "yaw_deg": 1.0}, "limits": {"max_translation_m": 0.2, "max_rotation_deg": 2.0}})
        with self.assertRaises(RenderError):
            _validate_sweep_bounds({"camera_sweep": {"translation_m": {"x": 0.3}}, "limits": {"max_translation_m": 0.2}})

        with self.assertRaises(RenderError):
            _validate_sweep_bounds({"camera_sweep": {}, "intrinsics_edit": {"focal_length": 1.0}})

    def test_sample_fps_keeps_real_time_and_camera_grid_contract(self):
        from scripts.render_counterfactual_video import (
            ArtifactScene,
            FORMAL_CAMERA_ORDER,
            _camera_grid_for_case,
            _camera_ids_for_case,
            _timestamp_values,
        )

        scene = ArtifactScene(self.usdz, "target")
        case = {
            "timestamp_range_us": {"start": 100, "end": 300},
            "sample_fps": 30.0,
            "camera_ids": list(FORMAL_CAMERA_ORDER),
            "camera_grid": {"columns": 3, "rows": 2, "label_cameras": True},
        }
        # The final endpoint is retained so the video covers the full requested window.
        self.assertEqual(_timestamp_values(scene, case, None, None, None), [100, 300])
        self.assertEqual(_camera_ids_for_case(case, None), list(FORMAL_CAMERA_ORDER))
        self.assertEqual(_camera_grid_for_case(case, list(FORMAL_CAMERA_ORDER)), (3, 2, True))

    @unittest.skipUnless(_HAS_CV2, "requires OpenCV")
    def test_stitched_grid_has_formal_dimensions(self):
        from scripts.render_counterfactual_video import (
            FORMAL_CAMERA_ORDER,
            _stitch_camera_frames,
            _jpeg_dimensions,
        )
        from PIL import Image
        from io import BytesIO

        bodies = {}
        for index, camera_id in enumerate(FORMAL_CAMERA_ORDER):
            output = BytesIO()
            Image.new("RGB", (2, 2), (40 + index, 50, 60)).save(output, format="JPEG")
            bodies[camera_id] = output.getvalue()
        grid = _stitch_camera_frames(
            bodies,
            list(FORMAL_CAMERA_ORDER),
            width=2,
            height=2,
            columns=3,
            rows=2,
            label_cameras=True,
        )
        self.assertEqual(_jpeg_dimensions(grid), (6, 4))

    @unittest.skipUnless(_HAS_CV2, "requires OpenCV")
    def test_pose_overlay_is_visible_without_changing_grid_dimensions(self):
        from scripts.render_counterfactual_video import (
            FORMAL_CAMERA_ORDER,
            _jpeg_dimensions,
            _stitch_camera_frames,
        )
        from PIL import Image
        from io import BytesIO

        bodies = {}
        for camera_id in FORMAL_CAMERA_ORDER:
            output = BytesIO()
            Image.new("RGB", (80, 40), (40, 50, 60)).save(output, format="JPEG")
            bodies[camera_id] = output.getvalue()
        base = _stitch_camera_frames(
            bodies,
            list(FORMAL_CAMERA_ORDER),
            width=80,
            height=40,
            columns=3,
            rows=2,
            label_cameras=True,
        )
        overlaid = _stitch_camera_frames(
            bodies,
            list(FORMAL_CAMERA_ORDER),
            width=80,
            height=40,
            columns=3,
            rows=2,
            label_cameras=True,
            pose_overlay={
                "translation_m": {"x": 0.12, "y": 0.06, "z": 0.0},
                "yaw_deg": 1.0,
            },
            pose_progress=0.5,
            pose_frame="sensor",
            pose_profile="sinusoidal",
        )
        self.assertEqual(_jpeg_dimensions(overlaid), (240, 80))
        self.assertNotEqual(base, overlaid)

    def test_non_target_digest_changes_when_non_target_actor_is_modified(self):
        from scripts.render_counterfactual_video import _dynamic_digest, _non_target_digest

        baseline = [
            {"track_id": "target", "pose": [0, 0, 0, 1, 0, 0, 0]},
            {"track_id": "other", "pose": [1, 0, 0, 1, 0, 0, 0]},
        ]
        target_only = [
            {"track_id": "target", "pose": [0.5, 0, 0, 1, 0, 0, 0]},
            {"track_id": "other", "pose": [1, 0, 0, 1, 0, 0, 0]},
        ]
        non_target_edit = [
            {"track_id": "target", "pose": [0.5, 0, 0, 1, 0, 0, 0]},
            {"track_id": "other", "pose": [2, 0, 0, 1, 0, 0, 0]},
        ]
        self.assertEqual(_non_target_digest(baseline, "target"), _non_target_digest(target_only, "target"))
        self.assertNotEqual(_non_target_digest(baseline, "target"), _non_target_digest(non_target_edit, "target"))
        self.assertNotEqual(_dynamic_digest(baseline), _dynamic_digest(target_only))

    def test_empty_or_wrong_size_rgb_payload_is_rejected(self):
        from scripts.render_counterfactual_video import RenderError, _jpeg_dimensions

        with self.assertRaises(RenderError):
            _jpeg_dimensions(b"")
        with self.assertRaises(RenderError):
            _jpeg_dimensions(b"not-an-image")

    def test_output_directory_requires_explicit_overwrite(self):
        from scripts.render_counterfactual_video import RenderError, _make_output_dir

        output = self.root / "existing"
        output.mkdir()
        (output / "evidence.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(RenderError):
            _make_output_dir(output, overwrite=False)
        _make_output_dir(output, overwrite=True)
        self.assertEqual(list(output.iterdir()), [])

    @unittest.skipUnless(_HAS_CV2, "requires OpenCV")
    def test_resume_capture_validates_contiguous_frames_and_timestamps(self):
        import cv2
        import numpy as np

        from scripts.render_counterfactual_video import RenderError, _resume_capture

        output = self.root / "resume"
        camera_path = output / "camera_frames/camera_front/000000.jpg"
        frame_path = output / "frames/000000.jpg"
        camera_path.parent.mkdir(parents=True)
        frame_path.parent.mkdir(parents=True)
        self.assertTrue(cv2.imwrite(str(camera_path), np.full((4, 6, 3), 90, np.uint8)))
        self.assertTrue(cv2.imwrite(str(frame_path), np.full((4, 6, 3), 90, np.uint8)))
        record = {
            "frame_index": 0,
            "scene_timestamp_us": 100,
            "camera_ids": ["camera_front"],
            "frame_path": "frames/000000.jpg",
            "camera_frame_paths": {
                "camera_front": "camera_frames/camera_front/000000.jpg"
            },
            "status": "passed",
            "dropped": False,
        }
        metadata = output / "frames.jsonl"
        metadata.write_text(json.dumps(record) + "\n", encoding="utf-8")

        records, paths = _resume_capture(
            output,
            metadata,
            [100, 200],
            ["camera_front"],
            width=6,
            height=4,
            output_width=6,
            output_height=4,
        )
        self.assertEqual(records, [record])
        self.assertEqual(paths, [frame_path])

        record["scene_timestamp_us"] = 101
        metadata.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaises(RenderError):
            _resume_capture(
                output,
                metadata,
                [100, 200],
                ["camera_front"],
                width=6,
                height=4,
                output_width=6,
                output_height=4,
            )

    def test_manifest_identity_rejects_changed_artifact(self):
        from scripts.render_counterfactual_video import RenderError, validate_manifest_identity

        manifest = {
            "artifact": {
                "uri": str(self.usdz),
                "sha256": hashlib.sha256(self.usdz.read_bytes()).hexdigest(),
                "size_bytes": self.usdz.stat().st_size,
            }
        }
        validate_manifest_identity(manifest, None)
        self.usdz.write_bytes(self.usdz.read_bytes() + b"changed")
        with self.assertRaises(RenderError):
            validate_manifest_identity(manifest, None)

    def test_runtime_module_loader_handles_ros_pythonpath_protobuf_shadowing(self):
        from scripts.render_counterfactual_video import _load_runtime_modules

        python_api = Path(
            "/home/cwadmin/sim-env/data/CARLA_0.9.16/PythonAPI/examples/nvidia/nurec"
        )
        if not python_api.is_dir():
            self.skipTest("local NuRec Python API is unavailable")
        _, sensorsim_pb2, _, _ = _load_runtime_modules(python_api)
        self.assertEqual(
            sensorsim_pb2.RGBRenderRequest.DESCRIPTOR.full_name,
            "nre.grpc.protos.sensorsim.RGBRenderRequest",
        )

    def test_rgb_request_uses_instant_midpoint_and_dynamic_pose_pair(self):
        from io import BytesIO

        from PIL import Image

        from scripts.render_counterfactual_video import SensorsimClient, _load_runtime_modules

        python_api = Path(
            "/home/cwadmin/sim-env/data/CARLA_0.9.16/PythonAPI/examples/nvidia/nurec"
        )
        if not python_api.is_dir():
            self.skipTest("local NuRec Python API is unavailable")
        grpc, sensorsim_pb2, common_pb2, _ = _load_runtime_modules(python_api)
        del grpc

        body = BytesIO()
        Image.new("RGB", (2, 2), (90, 90, 90)).save(body, format="JPEG")

        class Stub:
            def render_rgb(self, request, *, timeout):
                self.request = request
                return sensorsim_pb2.RGBRenderReturn(image_bytes=body.getvalue())

        client = object.__new__(SensorsimClient)
        client.protobuf = sensorsim_pb2
        client.common = common_pb2
        client.timeout = 1.0
        client.runtime_scene_id = "scene-0061"
        client.camera_intrinsics = {"camera_front": sensorsim_pb2.CameraSpec()}
        client.stub = Stub()
        response = client.render_rgb(
            camera_id="camera_front",
            width=2,
            height=2,
            start_us=100,
            end_us=200,
            start_pose={
                "position_m": {"x": 0, "y": 0, "z": 0},
                "orientation_xyzw": {"x": 0, "y": 0, "z": 0, "w": 1},
            },
            end_pose={
                "position_m": {"x": 1, "y": 0, "z": 0},
                "orientation_xyzw": {"x": 0, "y": 0, "z": 0, "w": 1},
            },
            dynamic_objects=[
                {
                    "track_id": "target",
                    "pose_pair": {
                        "start": [0, 0, 0, 1, 0, 0, 0],
                        "end": [1, 0, 0, 1, 0, 0, 0],
                    },
                }
            ],
        )
        request = client.stub.request
        self.assertEqual((request.frame_start_us, request.frame_end_us), (150, 151))
        self.assertEqual(request.dynamic_objects[0].pose_pair.start_pose.vec.x, 0.0)
        self.assertEqual(request.dynamic_objects[0].pose_pair.end_pose.vec.x, 1.0)
        self.assertEqual(response["status"], "passed")


if __name__ == "__main__":
    unittest.main()
