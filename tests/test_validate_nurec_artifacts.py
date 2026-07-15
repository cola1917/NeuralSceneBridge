import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


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
        (fake_modules / "torch.py").write_text(
            textwrap.dedent(
                """
                import json
                def load(path, map_location=None, weights_only=False):
                    with open(path, encoding="utf-8") as stream:
                        return json.load(stream)
                """
            ),
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

    def _write_config(self, *, cameras=None, samples_per_epoch=1000, max_epochs=1):
        config = {
            "trainer": {"max_epochs": max_epochs},
            "dataset": {
                "camera_ids": cameras
                or ["camera_front", "camera_front_left", "camera_front_right"],
                "n_samples_per_epoch": samples_per_epoch,
            },
        }
        (self.run_dir / "config" / "parsed.yaml").write_text(
            json.dumps(config), encoding="utf-8"
        )

    def _write_checkpoint(self, global_step):
        (self.run_dir / "checkpoints" / "last.ckpt").write_text(
            json.dumps({"global_step": global_step}), encoding="utf-8"
        )

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

    def test_rejects_empty_artifact(self):
        (self.run_dir / "artifacts" / "last.usdz").write_bytes(b"")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("empty USDZ", result.stdout)


if __name__ == "__main__":
    unittest.main()
