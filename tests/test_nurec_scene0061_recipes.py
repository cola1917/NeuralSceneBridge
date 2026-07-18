from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "config" / "nurec-scene0061-renderable-lidar-smoke-v3.env"
SMOKE_RETRY = ROOT / "config" / "nurec-scene0061-renderable-lidar-smoke-v3-attempt002.env"
FORMAL = ROOT / "config" / "nurec-scene0061-renderable-lidar-formal-v3.env"


def _assignments(path):
    result = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


class NuRecScene0061RecipesTests(unittest.TestCase):
    def test_smoke_and_formal_are_renderable_lidar_recipes(self):
        for path in (SMOKE, SMOKE_RETRY, FORMAL):
            values = _assignments(path)
            self.assertEqual(values["REQUIRE_LIDAR_SUPERVISION"], "1")
            self.assertEqual(values["REQUIRE_RENDERABLE_LIDAR"], "1")
            self.assertEqual(values["LIDAR_INTENSITY_LOSS_WEIGHT"], "1.0")
            self.assertEqual(values["LIDAR_RAYDROP_LOSS_WEIGHT"], "0.1")
            self.assertEqual(values["EXPECTED_MIN_LIDAR_EXTRA_SIGNAL_DIM"], "3")
            self.assertEqual(values["REQUIRE_SINGLE_RUN"], "1")
            self.assertEqual(values["DATASET_DIR"], "outputs/ncore_dense_lidar_sweeps_v2/scene-0061")

    def test_smoke_and_formal_have_separate_immutable_outputs_and_budgets(self):
        smoke = _assignments(SMOKE)
        smoke_retry = _assignments(SMOKE_RETRY)
        formal = _assignments(FORMAL)
        self.assertNotEqual(smoke["OUTPUT_DIR"], formal["OUTPUT_DIR"])
        self.assertNotEqual(smoke["OUTPUT_DIR"], smoke_retry["OUTPUT_DIR"])
        self.assertTrue(smoke_retry["OUTPUT_DIR"].endswith("attempt_002"))
        self.assertEqual(smoke["SAMPLES_PER_EPOCH"], "100")
        self.assertEqual(smoke["EXPECTED_GLOBAL_STEP"], "100")
        self.assertEqual(formal["SAMPLES_PER_EPOCH"], "40000")
        self.assertEqual(formal["EXPECTED_GLOBAL_STEP"], "40000")
        self.assertEqual(formal["REQUIRE_LIDAR_VALIDATION_EVIDENCE"], "1")
        self.assertEqual(formal["ALLOW_NRE_2604_LIDAR_GROUPING_BUG"], "1")


if __name__ == "__main__":
    unittest.main()
