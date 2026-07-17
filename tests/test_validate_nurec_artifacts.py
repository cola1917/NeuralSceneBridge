import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import textwrap
import unittest
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_nurec_artifacts.sh"


@unittest.skipUnless(
    os.name != "nt" and shutil.which("bash"),
    "a native POSIX bash is required for launcher tests",
)
class ValidateNuRecArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.run_dir = self.root / "outputs" / "test-run"
        (self.run_dir / "artifacts").mkdir(parents=True)
        (self.run_dir / "config").mkdir()
        (self.run_dir / "checkpoints").mkdir()
        (self.run_dir / "artifacts" / "last.usdz").write_bytes(b"non-empty-usdz")
        self._write_config()
        self._write_checkpoint(1000)

        fake_modules = self.root / "fake_modules"
        fake_modules.mkdir()
        (fake_modules / "yaml.py").write_text(
            "import json\ndef safe_load(stream): return json.load(stream)\n",
            encoding="utf-8",
        )
        self.env = os.environ.copy()
        self.env.update(
            {
                "OUTPUT_DIR": str(self.root / "outputs"),
                "ENV_FILE": str(self.root / "missing.env"),
                "NUREC_VALIDATION_BACKEND": "host",
                "PYTHONPATH": str(fake_modules),
            }
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_config(
        self,
        *,
        cameras=None,
        samples_per_epoch=1000,
        max_epochs=1,
        lidar_rays=2048,
        lidar_ratio=1.0,
        lidar_loss=0.005,
        val_lidar=False,
    ):
        config = {
            "trainer": {"max_epochs": max_epochs},
            "dataset": {
                "camera_ids": cameras
                or ["camera_front", "camera_front_left", "camera_front_right"],
                "n_samples_per_epoch": samples_per_epoch,
                "lidar_ids": ["lidar_top"],
                "train_lidar_ids": ["lidar_top"],
                "n_train_sample_lidar_rays": lidar_rays,
                "val_lidar": val_lidar,
                "samplers": {
                    "batch_sampler": {"ratio_lidar_samples": lidar_ratio}
                },
            },
            "loss": {"lidar": {"lambda_": lidar_loss}},
        }
        (self.run_dir / "config" / "parsed.yaml").write_text(
            json.dumps(config), encoding="utf-8"
        )

    def _write_checkpoint(self, global_step):
        with zipfile.ZipFile(
            self.run_dir / "checkpoints" / "last.ckpt", "w"
        ) as archive:
            archive.writestr(
                "archive/data.pkl",
                # Protocol 2 emits BINPUT between the key and value, matching
                # the NuRec/Lightning checkpoint produced by the 26.04 image.
                pickle.dumps({"global_step": global_step}, protocol=2),
            )

    def _write_dynamic_usdz(self, labels):
        track_count = len(labels)
        payload = {
            "dummy_chunk_id": {
                "tracks_data": {
                    "tracks_id": [f"track-{index}" for index in range(track_count)],
                    "tracks_poses": [
                        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
                        for _ in labels
                    ],
                    "tracks_timestamps_us": [[1_000_000] for _ in labels],
                    "tracks_label_class": labels,
                    "tracks_flags": [0] * track_count,
                },
                "cuboidtracks_data": {
                    "cuboids_dims": [[4.0, 2.0, 1.5] for _ in labels]
                },
            }
        }
        with zipfile.ZipFile(
            self.run_dir / "artifacts" / "last.usdz", "w"
        ) as archive:
            archive.writestr("sequence_tracks.json", json.dumps(payload))

    def _run(self):
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=REPO_ROOT,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_exact_three_camera_1000_step_run(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("checkpoint global_step: 1000", result.stdout)
        self.assertIn("result: PASS", result.stdout)

    def test_rejects_checkpoint_at_wrong_step(self):
        self._write_checkpoint(999)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected global_step=1000, found 999", result.stdout)

    def test_rejects_wrong_camera_set(self):
        self._write_config(cameras=["camera_front"])
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("config gate failed: expected camera_ids", result.stdout)

    def test_rejects_wrong_config_sample_count(self):
        self._write_config(samples_per_epoch=999)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected n_samples_per_epoch=1000", result.stdout)

    def test_rejects_epoch_limited_config(self):
        self._write_config(max_epochs=2)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected max_epochs=1", result.stdout)

    def test_lidar_supervision_gate_accepts_multimodal_training_config(self):
        self.env["REQUIRE_LIDAR_SUPERVISION"] = "1"
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("config dataset.n_train_sample_lidar_rays: 2048.0", result.stdout)
        self.assertIn("config loss.lidar.lambda_: 0.005", result.stdout)

    def test_lidar_supervision_gate_rejects_zero_lidar_loss(self):
        self._write_config(lidar_loss=0.0)
        self.env["REQUIRE_LIDAR_SUPERVISION"] = "1"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("config lidar gate failed", result.stdout)
        self.assertIn("loss.lidar.lambda_", result.stdout)

    def test_lidar_supervision_gate_requires_requested_validation(self):
        self.env["REQUIRE_LIDAR_SUPERVISION"] = "1"
        self.env["EXPECTED_VAL_LIDAR"] = "1"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected dataset.val_lidar=True, found False", result.stdout)

    def test_rejects_empty_artifact(self):
        (self.run_dir / "artifacts" / "last.usdz").write_bytes(b"")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("empty USDZ", result.stdout)

    def test_dynamic_gate_accepts_vehicle_and_pedestrian_tracks(self):
        self._write_dynamic_usdz(["automobile", "pedestrian"])
        self.env["REQUIRE_DYNAMIC_TRACKS"] = "1"
        self.env["EXPECTED_MIN_USDZ_TRACKS"] = "2"
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"track_count": 2', result.stdout)

    def test_dynamic_gate_rejects_actor_empty_usdz(self):
        self._write_dynamic_usdz([])
        self.env["REQUIRE_DYNAMIC_TRACKS"] = "1"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("result: FAIL (dynamic-track gate)", result.stdout)

    def test_strict_formal_gate_rejects_multiple_run_directories(self):
        shutil.copytree(self.run_dir, self.root / "outputs" / "second-run")
        self.env["REQUIRE_SINGLE_RUN"] = "1"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected exactly one NuRec run, found 2", result.stderr)

    def test_auto_backend_prefers_official_image_over_partial_host_runtime(self):
        fake_bin = self.root / "fake_bin"
        fake_bin.mkdir()
        docker_log = self.root / "docker.log"
        fake_docker = fake_bin / "docker"
        fake_docker.write_text(
            textwrap.dedent(
                f"""
                #!/usr/bin/env bash
                printf '%s\\n' "$*" >> {str(docker_log)!r}
                if [[ "$1 $2" == "image inspect" ]]; then
                    exit 0
                fi
                if [[ "$1" == "run" ]]; then
                    echo "checkpoint global_step: 1000"
                    exit 0
                fi
                exit 1
                """
            ).lstrip(),
            encoding="utf-8",
        )
        fake_docker.chmod(0o755)
        self.env["NUREC_VALIDATION_BACKEND"] = "auto"
        self.env["PATH"] = f"{fake_bin}{os.pathsep}{self.env['PATH']}"

        result = self._run()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("metadata backend: docker", result.stdout)
        self.assertIn("run --rm", docker_log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
