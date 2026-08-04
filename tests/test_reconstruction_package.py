import hashlib
import json
import tempfile
import unittest
from pathlib import Path


SCENE_TOKEN = "c" * 32


def _scenario_ir() -> dict:
    return {
        "schema_version": "scenario_ir.v1",
        "scenario_id": SCENE_TOKEN,
        "scenario_type": "interaction",
        "source": {
            "dataset": "nuscenes",
            "scene_id": SCENE_TOKEN,
            "version": "v1.0-mini",
            "scene_name": "scene-0061",
            "scene_token": SCENE_TOKEN,
            "sample_count": 2,
        },
        "coordinate_frame": {
            "name": "scene_local_ego_start",
            "units": {"position": "meter", "time": "second", "yaw": "degree"},
            "handedness": "right",
            "x_axis": "initial_ego_forward",
            "y_axis": "initial_ego_left",
            "origin_global_translation": [0.0, 0.0, 0.0],
            "origin_global_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "origin_global_yaw_deg": 0.0,
            "transform": "local_xy = R(-origin_yaw) * (global_xy - origin_xy)",
        },
        "windows": {
            "event": {"start_sec": 1.0, "end_sec": 2.0},
            "warmup": {"start_sec": 0.0, "end_sec": 1.0},
            "reconstruction": {"start_sec": 0.0, "end_sec": 2.0},
        },
        "ego": {
            "track_id": "ego",
            "initial_state": {},
            "reference_trajectory": [],
            "route": {},
        },
        "actors": [],
        "map_context": {"feature_counts": {}, "features": []},
        "sensors": {"available_capabilities": []},
        "events": {"trigger": {}, "mined_events": []},
        "data_requirements": {
            "reconstruction": {
                "required": ["camera_images", "camera_calibration", "ego_pose", "actor_tracks"]
            },
            "closed_loop": {"required": ["ego_initial_state", "actor_initial_states", "map_context"]},
        },
        "risk_metrics": {
            "trigger_time_sec": 1.0,
            "trigger_tag": None,
            "actor_count": 0,
            "ego_reference_state_count": 0,
        },
        "dataset_refs": {
            "source": {"dataset": "nuscenes", "root": None, "scene_id": SCENE_TOKEN},
            "sample_refs": {"status": "deferred", "refs": []},
            "index_refs": {"status": "deferred", "refs": []},
        },
        "evaluation": {"metrics": []},
        "variants": {"mvp": {}, "final_closed_loop": {}},
    }


class ReconstructionPackageTests(unittest.TestCase):
    def test_builds_hashed_inventory_and_records_full_scene_fallback(self):
        from scripts.build_reconstruction_package import build_reconstruction_package

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "reconstruction" / "last.usdz"
            artifact.parent.mkdir()
            artifact.write_bytes(b"usdz-fixture")
            output = root / "reconstruction_package.json"
            package = build_reconstruction_package(
                output=output,
                scene_token="c" * 32,
                scene_name="scene-0061",
                dataset_version="v1.0-mini",
                artifacts=[("nurec_usdz", artifact)],
                requested_window=(4.0, 10.0),
                actual_window=(0.0, 19.15),
            )

            self.assertEqual(package["scene_id"], "c" * 32)
            self.assertEqual(package["artifacts"][0]["path"], "reconstruction/last.usdz")
            self.assertEqual(len(package["artifacts"][0]["sha256"]), 64)
            self.assertEqual(package["alignment"]["status"], "pending_runtime_alignment")
            self.assertTrue(package["warnings"])
            self.assertTrue(output.is_file())

    def test_rejects_artifact_outside_package_directory(self):
        from scripts.build_reconstruction_package import build_reconstruction_package

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            artifact = Path(outside) / "last.ckpt"
            artifact.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "contained"):
                build_reconstruction_package(
                    output=root / "reconstruction_package.json",
                    scene_token="d" * 32,
                    scene_name="scene-0061",
                    dataset_version="v1.0-mini",
                    artifacts=[("nurec_checkpoint", artifact)],
                )

    def test_consumes_scenario_ir_and_derives_package_identity_and_window(self):
        from scripts.build_reconstruction_package import build_reconstruction_package

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ir_path = root / "scenario_ir.json"
            ir_bytes = (json.dumps(_scenario_ir(), indent=2) + "\n").encode("utf-8")
            ir_path.write_bytes(ir_bytes)
            artifact = root / "reconstruction" / "last.usdz"
            artifact.parent.mkdir()
            artifact.write_bytes(b"usdz-fixture")

            package = build_reconstruction_package(
                output=root / "reconstruction_package.json",
                artifacts=[("nurec_usdz", artifact)],
                scenario_ir_path=ir_path,
                actual_window=(0.0, 2.0),
            )

        self.assertEqual(package["scene_id"], SCENE_TOKEN)
        self.assertEqual(package["source"]["scene_name"], "scene-0061")
        self.assertEqual(package["source"]["dataset_version"], "v1.0-mini")
        self.assertEqual(package["coverage"]["mode"], "window")
        self.assertEqual(package["coverage"]["requested_window"], {"start_sec": 0.0, "end_sec": 2.0})
        self.assertEqual(package["scenario_ir"]["schema_version"], "scenario_ir.v1")
        self.assertEqual(package["scenario_ir"]["scenario_id"], SCENE_TOKEN)
        self.assertEqual(package["scenario_ir"]["sha256"], hashlib.sha256(ir_bytes).hexdigest())

    def test_rejects_scenario_ir_identity_override(self):
        from scripts.build_reconstruction_package import build_reconstruction_package

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ir_path = root / "scenario_ir.json"
            ir_path.write_text(json.dumps(_scenario_ir()), encoding="utf-8")
            artifact = root / "last.usdz"
            artifact.write_bytes(b"usdz-fixture")
            with self.assertRaisesRegex(ValueError, "scene_token does not match"):
                build_reconstruction_package(
                    output=root / "reconstruction_package.json",
                    scene_token="d" * 32,
                    artifacts=[("nurec_usdz", artifact)],
                    scenario_ir_path=ir_path,
                )


if __name__ == "__main__":
    unittest.main()
