from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "config" / "nurec-scene0061-renderable-lidar-smoke-v3.env"
SMOKE_RETRY = ROOT / "config" / "nurec-scene0061-renderable-lidar-smoke-v3-attempt002.env"
FORMAL = ROOT / "config" / "nurec-scene0061-renderable-lidar-formal-v3.env"
M8_COVERAGE_SMOKE = ROOT / "config" / "nurec-scene0061-m8-track-coverage-smoke.env"
M8_THRESHOLD_SMOKE = ROOT / "config" / "nurec-scene0061-m8-dynamic-threshold-smoke.env"
M8_TARGETED_SMOKE = ROOT / "config" / "nurec-scene0061-m8-targeted-track-smoke.env"
M8_CHECKPOINT_FRIENDLY = ROOT / "config" / "nurec-scene0061-m8-full-track-checkpoint-friendly-attempt012.env"
M8_FULL_FORMAL = ROOT / "config" / "nurec-scene0061-m8-full-track-formal-allocator-attempt013.env"
M8_CLOSURE_PREFLIGHT = ROOT / "config" / "nurec-scene0061-m8-full-object-closure-preflight-attempt014.env"
M8_EXACT_REGISTRY_PREFLIGHT = ROOT / "config" / "nurec-scene0061-m8-exact-registry-track-preflight-attempt015.env"
M8_NCORE_ELIGIBLE_REGISTRY_PREFLIGHT = ROOT / "config" / "nurec-scene0061-m8-ncore-eligible-registry-preflight-attempt016.env"
M8_NCORE_ELIGIBLE_REGISTRY_FORMAL = ROOT / "config" / "nurec-scene0061-m8-ncore-eligible-registry-formal-attempt017.env"
M8_NCORE_ELIGIBLE_REGISTRY_TIME_WINDOW_PREFLIGHT = ROOT / "config" / "nurec-scene0061-m8-ncore-eligible-time-window-preflight-attempt018.env"
M8_NCORE_ELIGIBLE_REGISTRY_TIME_WINDOW_PREFLIGHT_RETRY = ROOT / "config" / "nurec-scene0061-m8-ncore-eligible-time-window-preflight-attempt019.env"


def _assignments(path):
    result = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("source "):
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

    def test_m8_coverage_smoke_retains_sparse_dynamic_tracks(self):
        values = _assignments(M8_COVERAGE_SMOKE)
        self.assertEqual(values["SAMPLES_PER_EPOCH"], "100")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_TRACK"], "1000")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_LAYER"], "1500000")
        self.assertEqual(values["DYNAMIC_TRACK_KEEP_ALL_POSES"], "1")

    def test_train_recipe_exposes_dynamic_classification_overrides(self):
        script = (ROOT / "scripts" / "run_nurec_train.sh").read_text(encoding="utf-8")
        for name in (
            "PYTORCH_CUDA_ALLOC_CONF",
            "CHECKPOINT_FRIENDLY_BACKWARD",
            "TRACK_MIN_DISTANCE_M",
            "TRACK_MIN_DISPLACEMENT_M",
            "TRACK_MIN_SPEED_MS",
            "TRACK_USE_DISPLACEMENT_AND_DISTANCE",
            "NCORE_MIN_TRACK_DISPLACEMENT_M",
            "NCORE_MIN_TRACK_SPEED_MS",
            "TRACK_VEHICLE_CLASSES",
            "TRACK_PEDESTRIAN_CLASSES",
            "DYNAMIC_TRACK_IDS",
            "DYNAMIC_TRACK_INIT_STEP_FRAME",
        ):
            self.assertIn(name, script)
        self.assertIn("dataset.cuboid_tracks_params.track_min_distance_m=", script)
        self.assertIn("dataset.cuboid_tracks_params.track_min_displacement_m=", script)
        self.assertIn("dataset.cuboid_tracks_params.track_min_speed_ms=", script)
        self.assertIn("--min-displacement-m", script)
        self.assertIn("--min-median-speed-ms", script)
        self.assertIn("--vehicle-classes \"${TRACK_VEHICLE_CLASSES}\"", script)
        self.assertIn("+model.layers.dynamic_deformables.tracks.ids=", script)
        self.assertIn("+model.layers.${layer}.initialization.step_frame=${DYNAMIC_TRACK_INIT_STEP_FRAME}", script)
        self.assertIn("--env PYTORCH_CUDA_ALLOC_CONF", script)
        self.assertIn("model.renderer.checkpoint_friendly_backward=true", script)

    def test_m8_threshold_smoke_is_isolated_and_zero_threshold(self):
        values = _assignments(M8_THRESHOLD_SMOKE)
        self.assertTrue(values["OUTPUT_DIR"].endswith("dynamic_threshold_smoke_attempt_001"))
        self.assertEqual(values["TRACK_MIN_DISTANCE_M"], "0")
        self.assertEqual(values["TRACK_MIN_DISPLACEMENT_M"], "0")
        self.assertEqual(values["TRACK_MIN_SPEED_MS"], "0")
        self.assertEqual(values["TRACK_USE_DISPLACEMENT_AND_DISTANCE"], "1")

    def test_m8_targeted_smoke_binds_the_six_missing_tracks(self):
        values = _assignments(M8_TARGETED_SMOKE)
        ids = values["DYNAMIC_TRACK_IDS"].split(",")
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6)
        self.assertTrue(values["OUTPUT_DIR"].endswith("targeted_track_smoke_attempt_003"))

    def test_m8_checkpoint_friendly_retry_preserves_full_scene_recipe(self):
        values = _assignments(M8_CHECKPOINT_FRIENDLY)
        self.assertEqual(values["CHECKPOINT_FRIENDLY_BACKWARD"], "1")
        self.assertEqual(values["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_TRACK"], "1500000")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_LAYER"], "9000000")
        self.assertEqual(values["DYNAMIC_TRACK_KEEP_ALL_POSES"], "1")
        self.assertEqual(values["REQUIRE_RENDERABLE_LIDAR"], "1")

    def test_m8_full_formal_requires_the_entire_eligible_dynamic_inventory(self):
        values = _assignments(M8_FULL_FORMAL)
        self.assertEqual(values["SAMPLES_PER_EPOCH"], "40000")
        self.assertEqual(values["EXPECTED_GLOBAL_STEP"], "40000")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_TRACK"], "1500000")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_LAYER"], "9000000")
        self.assertEqual(values["EXPECTED_MIN_USDZ_TRACKS"], "50")
        self.assertEqual(values["EXPECTED_MIN_USDZ_VEHICLES"], "7")
        self.assertEqual(values["EXPECTED_MIN_USDZ_PEDESTRIANS"], "43")
        self.assertEqual(values["REQUIRE_LIDAR_VALIDATION_EVIDENCE"], "1")

    def test_m8_closure_preflight_removes_default_dynamic_classifier_thresholds(self):
        values = _assignments(M8_CLOSURE_PREFLIGHT)
        self.assertEqual(values["SAMPLES_PER_EPOCH"], "1")
        self.assertEqual(values["TRACK_MIN_DISTANCE_M"], "0")
        self.assertEqual(values["TRACK_MIN_DISPLACEMENT_M"], "0")
        self.assertEqual(values["TRACK_MIN_SPEED_MS"], "0")
        self.assertEqual(values["TRACK_USE_DISPLACEMENT_AND_DISTANCE"], "1")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_TRACK"], "1500000")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_LAYER"], "9000000")
        self.assertIn("bicycle", values["TRACK_VEHICLE_CLASSES"])
        self.assertIn("motorcycle", values["TRACK_VEHICLE_CLASSES"])

    def test_m8_exact_registry_preflight_selects_all_m6_dynamic_tracks_by_layer(self):
        values = _assignments(M8_EXACT_REGISTRY_PREFLIGHT)
        rigid_ids = values["DYNAMIC_RIGID_TRACK_IDS"].split(",")
        deformable_ids = values["DYNAMIC_TRACK_IDS"].split(",")
        self.assertEqual(len(rigid_ids), 23)
        self.assertEqual(len(deformable_ids), 66)
        self.assertEqual(len(set(rigid_ids) | set(deformable_ids)), 89)
        self.assertFalse(set(rigid_ids) & set(deformable_ids))
        self.assertEqual(values["TRACK_MIN_DISPLACEMENT_M"], "0")
        self.assertEqual(values["NCORE_MIN_TRACK_DISPLACEMENT_M"], "0")

    def test_m8_ncore_eligible_registry_preflight_removes_only_nonreplayable_tracks(self):
        lines = M8_NCORE_ELIGIBLE_REGISTRY_PREFLIGHT.read_text(encoding="utf-8").splitlines()
        values = _assignments(M8_NCORE_ELIGIBLE_REGISTRY_PREFLIGHT)
        self.assertIn("source config/nurec-scene0061-m8-exact-registry-track-preflight-attempt015.env", lines)
        self.assertTrue(values["OUTPUT_DIR"].endswith("attempt_016"))
        ids = values["DYNAMIC_TRACK_IDS"].split(",")
        self.assertEqual(len(ids), 64)
        self.assertNotIn("4c8ffcdaddb44fc0b4d42faeae50f083", ids)
        self.assertNotIn("85d771dc8b2049fc9155b376a73b2121", ids)

    def test_m8_ncore_eligible_registry_formal_uses_time_windowed_track_initialization(self):
        lines = M8_NCORE_ELIGIBLE_REGISTRY_FORMAL.read_text(encoding="utf-8").splitlines()
        values = _assignments(M8_NCORE_ELIGIBLE_REGISTRY_FORMAL)
        self.assertIn(
            "source config/nurec-scene0061-m8-ncore-eligible-registry-preflight-attempt016.env",
            lines,
        )
        self.assertTrue(values["OUTPUT_DIR"].endswith("attempt_017"))
        self.assertEqual(values["SAMPLES_PER_EPOCH"], "40000")
        self.assertEqual(values["DYNAMIC_TRACK_KEEP_ALL_POSES"], "0")
        self.assertEqual(values["DYNAMIC_TRACK_INIT_STEP_FRAME"], "1")
        self.assertEqual(values["EXPECTED_MIN_USDZ_TRACKS"], "87")
        self.assertEqual(values["EXPECTED_MIN_USDZ_VEHICLES"], "23")
        self.assertEqual(values["EXPECTED_MIN_USDZ_PEDESTRIANS"], "64")

    def test_m8_time_window_preflight_covers_all_tracks_with_a_bounded_point_budget(self):
        lines = M8_NCORE_ELIGIBLE_REGISTRY_TIME_WINDOW_PREFLIGHT.read_text(encoding="utf-8").splitlines()
        values = _assignments(M8_NCORE_ELIGIBLE_REGISTRY_TIME_WINDOW_PREFLIGHT)
        self.assertIn(
            "source config/nurec-scene0061-m8-ncore-eligible-registry-preflight-attempt016.env",
            lines,
        )
        self.assertTrue(values["OUTPUT_DIR"].endswith("attempt_018"))
        self.assertEqual(values["SAMPLES_PER_EPOCH"], "1")
        self.assertEqual(values["DYNAMIC_TRACK_KEEP_ALL_POSES"], "0")
        self.assertEqual(values["DYNAMIC_TRACK_INIT_STEP_FRAME"], "1")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_TRACK"], "100000")
        self.assertEqual(values["DYNAMIC_TRACK_POINTS_PER_LAYER"], "2000000")

    def test_m8_time_window_preflight_retry_keeps_the_same_immutable_contract(self):
        lines = M8_NCORE_ELIGIBLE_REGISTRY_TIME_WINDOW_PREFLIGHT_RETRY.read_text(encoding="utf-8").splitlines()
        values = _assignments(M8_NCORE_ELIGIBLE_REGISTRY_TIME_WINDOW_PREFLIGHT_RETRY)
        self.assertIn(
            "source config/nurec-scene0061-m8-ncore-eligible-time-window-preflight-attempt018.env",
            lines,
        )
        self.assertTrue(values["OUTPUT_DIR"].endswith("attempt_019"))


if __name__ == "__main__":
    unittest.main()
