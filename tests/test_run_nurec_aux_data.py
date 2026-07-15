import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_nurec_aux_data.sh"


@unittest.skipUnless(
    os.name != "nt" and shutil.which("bash"),
    "a native POSIX bash is required for launcher tests",
)
class RunNuRecAuxDataTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_dir = self.root / "ncore"
        self.dataset_dir.mkdir()
        self.docker_log = self.root / "docker-args.txt"

        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        fake_docker = fake_bin / "docker"
        fake_docker.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${DOCKER_LOG}\"\n",
            encoding="utf-8",
        )
        fake_docker.chmod(0o755)

        self.env = os.environ.copy()
        self.env.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{self.env['PATH']}",
                "DOCKER_LOG": str(self.docker_log),
                "NGC_API_KEY": "test-only-placeholder",
                "DATASET_DIR": str(self.dataset_dir),
            }
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _run(self, **overrides):
        env = self.env.copy()
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        args = self.docker_log.read_text(encoding="utf-8").splitlines()
        return result, args

    def test_nested_manifest_defaults_output_beside_manifest(self):
        result, args = self._run(DATASET_PATH="scene-0061/scene-0061.json")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected_output = self.dataset_dir / "scene-0061"
        self.assertTrue(expected_output.is_dir())
        self.assertIn(f"{expected_output}:/workdir/output", args)
        self.assertIn(
            "--dataset-path=/workdir/dataset/scene-0061/scene-0061.json", args
        )

    def test_flat_manifest_keeps_dataset_root_as_default_output(self):
        result, args = self._run(DATASET_PATH="scene-0061.json")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"{self.dataset_dir}:/workdir/output", args)

    def test_explicit_output_dir_overrides_nested_manifest_default(self):
        explicit_output = self.root / "custom-aux"
        result, args = self._run(
            DATASET_PATH="scene-0061/scene-0061.json",
            OUTPUT_DIR=str(explicit_output),
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(explicit_output.is_dir())
        self.assertIn(f"{explicit_output}:/workdir/output", args)

    def test_nested_shard_pattern_uses_its_parent_directory(self):
        result, args = self._run(SHARD_FILE_PATTERN="scene-0061/scene-0061.zarr.itar")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected_output = self.dataset_dir / "scene-0061"
        self.assertIn(f"{expected_output}:/workdir/output", args)
        self.assertIn(
            "--shard-file-pattern=/workdir/dataset/scene-0061/scene-0061.zarr.itar",
            args,
        )


if __name__ == "__main__":
    unittest.main()
