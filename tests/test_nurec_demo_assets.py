import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "demo" / "scene0061" / "manifest.json"
CASES = ROOT / "demo" / "scene0061" / "cases"
REPORT = ROOT / "demo" / "scene0061" / "quality_report.json"


class NuRecDemoAssetTests(unittest.TestCase):
    FORMAL_CAMERAS = [
        "camera_front_left",
        "camera_front",
        "camera_front_right",
        "camera_back_left",
        "camera_back",
        "camera_back_right",
    ]

    def test_manifest_and_cases_have_the_checked_in_contract(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "nsb.nurec-interview-demo-manifest.v1")
        self.assertEqual(manifest["scene_id"], "scene-0061")
        self.assertEqual(manifest["target_track_id"], "c1958768d48640948f6053d04cffd35b")
        self.assertEqual(manifest["usd_z_validation"]["track_count"], 223)
        self.assertEqual(manifest["usd_z_validation"]["controllable_track_count"], 74)
        self.assertEqual(
            {Path(path).name for path in manifest["case_files"]},
            {"original_replay.json", "lead_vehicle_edit.json", "camera_pose_sweep.json"},
        )
        for path in sorted(CASES.glob("*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(case["schema_version"], "nsb.nurec-counterfactual-case.v1")
            self.assertIn(case["case_id"], {"V01", "V02", "V03"})
            self.assertEqual(case["scene_id"], "scene-0061")
            self.assertEqual(case["resolution"], {"width": 800, "height": 450})
            self.assertEqual(case["camera_ids"], self.FORMAL_CAMERAS)
            self.assertEqual(case["camera_grid"], {"columns": 3, "rows": 2, "label_cameras": True})
            self.assertEqual(case["video_resolution"], {"width": 2400, "height": 900})
            self.assertEqual(case["sample_fps"], 30.0)
            self.assertEqual(case["video_fps"], 30.0)
            self.assertNotIn("intrinsics_edit", case)

    def test_quality_report_template_and_current_snapshot_cannot_claim_pass_without_capture(self):
        template = json.loads((REPORT.parent / "quality_report.example.json").read_text(encoding="utf-8"))
        current = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(template["status"], "example")
        self.assertFalse(template["pass"])
        self.assertIn(current["status"], {"pending_capture", "failed", "passed"})
        if current["status"] != "passed":
            self.assertFalse(current["pass"])

    def test_tracked_files_do_not_include_large_runtime_assets(self):
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        large = []
        for raw_path in result.stdout.split(b"\0"):
            if not raw_path:
                continue
            path = ROOT / os.fsdecode(raw_path)
            if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
                large.append(str(path.relative_to(ROOT)))
        self.assertEqual(large, [])


if __name__ == "__main__":
    unittest.main()
