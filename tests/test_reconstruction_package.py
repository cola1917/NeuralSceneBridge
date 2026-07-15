import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
