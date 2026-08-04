import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class DemoManifestTests(unittest.TestCase):
    def test_interview_manifest_is_runtime_independent(self):
        from validate_demo_manifest import validate_manifest

        result = validate_manifest(ROOT / "demo" / "scene0061" / "manifest.json")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(result["scene_id"], "scene-0061")

    def test_case_definitions_keep_intrinsics_fixed(self):
        case = json.loads(
            (ROOT / "demo" / "scene0061" / "cases" / "camera_pose_sweep.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(case["dynamic_objects"]["fixed"])
        self.assertTrue(case["camera_sweep"]["apply_to_all_cameras"])
        self.assertNotIn("intrinsics", case)
        self.assertNotIn("intrinsics", case["camera_sweep"])


if __name__ == "__main__":
    unittest.main()
