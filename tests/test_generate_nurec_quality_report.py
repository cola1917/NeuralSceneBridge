import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class GenerateNuRecQualityReportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.artifact = self.root / "artifact.usdz"
        self.checkpoint = self.root / "last.ckpt"
        self.artifact.write_bytes(b"canonical-usdz")
        self.checkpoint.write_bytes(b"matching-checkpoint")
        self.cases = {}
        for case_id, kind in (
            ("V01", "original_replay"),
            ("V02", "lead_vehicle_edit"),
            ("V03", "camera_pose_sweep"),
        ):
            path = self.root / "cases" / f"{case_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "nsb.nurec-counterfactual-case.v1",
                        "case_id": case_id,
                        "kind": kind,
                        "resolution": {"width": 2, "height": 2},
                        "video_name": {
                            "V01": "original_replay.mp4",
                            "V02": "lead_vehicle_edit.mp4",
                            "V03": "camera_pose_sweep.mp4",
                        }[case_id],
                    }
                ),
                encoding="utf-8",
            )
            self.cases[case_id] = path
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": "nsb.nurec-interview-demo-manifest.v1",
                    "scene_id": "scene-0061",
                    "runtime_scene_id": "scene-0061",
                    "artifact": {
                        "role": "canonical_nurec_usdz",
                        "uri": "artifact.usdz",
                        "sha256": _sha(self.artifact),
                        "size_bytes": self.artifact.stat().st_size,
                        "training_recipe": "fixture",
                    },
                    "checkpoint": {
                        "role": "matching_nurec_checkpoint",
                        "uri": "last.ckpt",
                        "sha256": _sha(self.checkpoint),
                        "size_bytes": self.checkpoint.stat().st_size,
                    },
                    "nurec": {"version_id": "26.4.146", "git_hash": "fixture"},
                    "runtime": {"server_address": "127.0.0.1:46443"},
                    "case_files": [f"cases/{case_id}.json" for case_id in ("V01", "V02", "V03")],
                    "limitations": [],
                }
            ),
            encoding="utf-8",
        )
        self.metrics = self.root / "metrics.yaml"
        self.metrics.write_text(
            "aggregated_metrics:\n"
            "  test/chamfer_distance:\n"
            "    value: 0.5\n"
            "  test/raydrop_accuracy:\n"
            "    value: 0.75\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _evidence(self, case_id: str, *, dropped: int = 0, artifact_sha: str | None = None):
        case_path = self.cases[case_id]
        case_root = self.root / "captures" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        video = case_root / {
            "V01": "original_replay.mp4",
            "V02": "lead_vehicle_edit.mp4",
            "V03": "camera_pose_sweep.mp4",
        }[case_id]
        video.write_bytes(f"video-{case_id}".encode())
        frame_paths = []
        for frame_index, value in enumerate((80, 100)):
            frame_path = case_root / "frames" / f"{frame_index:06d}.jpg"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), (value, value, value)).save(frame_path)
            frame_paths.append(frame_path)
        metadata = case_root / "frames.jsonl"
        rows = []
        for frame_index, frame_path in enumerate(frame_paths):
            row = {
                "frame_index": frame_index,
                "frame_path": str(frame_path.relative_to(case_root)),
                "decoded_width": 2,
                "decoded_height": 2,
                "camera_id": "camera_front",
                "rpc_latency_ms": 3.0 + frame_index,
                "invalid_pixel_ratio": 0.0,
                "dark_pixel_ratio": 0.0,
                "laplacian_sharpness": 2.0,
                "dropped": bool(dropped),
            }
            if case_id == "V02":
                row["target_pose_delta_m"] = {"x": 0.5, "y": 0.0, "z": 0.0}
            if case_id == "V03":
                row["camera_sweep_offset"] = {
                    "translation_m": {"x": 0.1, "y": 0.0, "z": 0.0},
                    "yaw_deg": 0.5,
                }
            rows.append(row)
        metadata.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        payload = {
            "schema_version": "nsb.nurec-counterfactual-run.v1",
            "status": "passed",
            "case": {"case_id": case_id, "sha256": _sha(case_path)},
            "manifest": {"sha256": _sha(self.manifest)},
            "artifact": {"sha256": artifact_sha or _sha(self.artifact)},
            "output": {
                "video": str(video),
                "frame_count": 2,
                "dropped_frame_count": dropped,
                "metadata": str(metadata),
            },
            "frames": {
                "requested_count": 2,
                "captured_count": 2,
                "dropped_frame_count": dropped,
                "first_timestamp_us": 1,
                "last_timestamp_us": 1,
            },
        }
        if case_id in {"V02", "V03"}:
            payload["probe"] = {"status": "passed"}
        path = case_root / "evidence.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _build(self, evidence):
        from scripts.generate_nurec_quality_report import build_quality_report

        return build_quality_report(
            self.manifest,
            evidence,
            metrics_path=self.metrics,
            code_commit="fixture-commit",
        )

    def test_missing_capture_is_pending_and_never_passes(self):
        report = self._build({})
        self.assertEqual(report["status"], "pending_capture")
        self.assertFalse(report["pass"])
        self.assertFalse(report["gates"]["case_capture"])

    def test_complete_capture_set_passes_only_with_matching_identity(self):
        evidence = {case_id: self._evidence(case_id) for case_id in ("V01", "V02", "V03")}
        report = self._build(evidence)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["pass"])
        self.assertTrue(all(case["pass"] for case in report["cases"]))
        self.assertEqual(report["reconstruction"]["metrics"]["aggregated"]["test/chamfer_distance"], 0.5)

    def test_stale_artifact_or_dropped_frame_fails_closed(self):
        evidence = {
            "V01": self._evidence("V01", artifact_sha="0" * 64),
            "V02": self._evidence("V02", dropped=1),
            "V03": self._evidence("V03"),
        }
        report = self._build(evidence)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["pass"])
        reasons = " ".join(report["reasons"])
        self.assertIn("artifact SHA-256", reasons)
        self.assertIn("dropped 1", reasons)

    def test_discovery_finds_case_evidence_under_root(self):
        from scripts.generate_nurec_quality_report import discover_evidence

        for case_id in ("V01", "V02", "V03"):
            self._evidence(case_id)
        found = discover_evidence(self.root / "captures")
        self.assertEqual(set(found), {"V01", "V02", "V03"})


if __name__ == "__main__":
    unittest.main()
