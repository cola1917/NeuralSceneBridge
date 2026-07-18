# NuRec Integration Checklist

This document covers the remaining server-side path from the validated local
nuScenes dataset to the first NuRec reconstruction artifact.

## Current Local Status

- `scene-0061` is the first target scene (39 samples, 19.15 seconds).
- Scene blobs are uploaded to the server separately under
  `data/nuscenes-mini-scene-0061` (or full `data/nuscenes-mini`).
- The NVIDIA NCore nuScenes converter is vendored under `third_party/`.
- Reproducible converter images, environment validation, aux generation, and
  NuRec smoke-training scripts are present.
- Server GPU, NGC access, image pulls, NCore output, aux output, and NuRec
  artifacts have not yet been exercised.

## 1. Prepare Configuration

```bash
cp config/nurec-smoke.env.example config/nurec-smoke.env
```

Do not store `NGC_API_KEY` in the config file. Export it in the shell:

```bash
export NGC_API_KEY=<your-ngc-api-key>
```

The example uses `SHM_SIZE=32g`, which fits a 64 GB-class rental VM. Increase
it only after checking host RAM; the launcher rejects requests above 80% of
physical memory.

NuRec 26.04 has no structured `trainer.max_steps` field. Its schedules and
checkpoint cadence define total training iterations as
`trainer.max_epochs * dataset.n_samples_per_epoch`. The strict three-camera
1000-step run therefore uses `MAX_EPOCHS=1` and `SAMPLES_PER_EPOCH=1000`, then
checks `last.ckpt.global_step == 1000`.

## 2. Run Server Preflight

```bash
bash scripts/server_preflight.sh
```

This checks Docker access, Docker GPU access, GPU name/memory/driver, free disk,
NGC key presence, and expected Docker images.

## 3. Build And Validate The NCore Converter

```bash
bash scripts/setup_ncore_images.sh
```

## 4. Convert The Daytime Scene

```bash
bash scripts/run_ncore_nuscenes_converter.sh
```

For the closed-loop actor rebuild this wrapper uses
`CUBOID_SAMPLING=lidar-sweeps`. The vendored converter itself remains backwards
compatible (`keyframes` by default); the formal project wrapper opts into dense
interpolated tracks explicitly.

Inspect `outputs/ncore` and update `DATASET_PATH` in
`config/nurec-smoke.env` to the generated NCore `.json` manifest name.

## 5. Pull NVIDIA NRE Images

```bash
bash scripts/pull_nurec_images.sh
```

## 6. Generate Auxiliary Data

```bash
DATASET_PATH=<generated-manifest-name.json> \
  bash scripts/run_nurec_aux_data.sh
```

By default the aux script writes into the selected manifest's directory. A
nested path such as `scene-0061/scene-0061.json` therefore writes aux stores to
`outputs/ncore/scene-0061`, not to its parent. `OUTPUT_DIR` can override the
default, but NuRec requires the `.json`, `.zarr.itar`, and `.aux.*.zarr` files
to remain in the same mounted dataset directory.

## 7. Run A Single-GPU Smoke Reconstruction

The launcher mounts `CACHE_DIR` (default `.cache/nurec`) at `/home/.cache` in
the official container. Keep this directory between runs so pretrained model
weights are downloaded once rather than once per disposable container.

The formal example deliberately writes dense conversion and training outputs
to versioned roots ending in `dense_lidar_sweeps_v1` and `attempt_001`.
Do not point a dense attempt at the legacy `outputs/ncore` or
`outputs/nurec_formal_scene0061_6cam_40k` roots: those may contain a valid old
run whose artifacts would not prove that dense cuboids were used. The formal
launcher records one-second GPU/RAM/disk samples in
`launcher/resources.csv` and a peak/minimum summary in
`launcher/resources.summary.json`.

The example config defaults to three forward cameras and one epoch to reduce
the initial load on a 24-32 GB GPU:

```bash
bash scripts/run_nurec_train.sh
```

After the pipeline works, use all six nuScenes cameras in
`config/nurec-smoke.env`:

```text
camera_front,camera_front_left,camera_front_right,camera_back,camera_back_left,camera_back_right
```

## 8. Validate Training Artifacts

```bash
bash scripts/validate_nurec_artifacts.sh
```

When `REQUIRE_DYNAMIC_TRACKS=1`, validation also inspects the embedded
`sequence_tracks.json` and requires valid vehicle and pedestrian tracks. The
minimums are configurable with `EXPECTED_MIN_USDZ_TRACKS`,
`EXPECTED_MIN_USDZ_VEHICLES`, and `EXPECTED_MIN_USDZ_PEDESTRIANS`.

Formal multimodal runs set `REQUIRE_LIDAR_SUPERVISION=1`. The gate then proves
that the parsed configuration retains the expected training LiDAR, a positive
LiDAR ray count and batch ratio, a positive `loss.lidar.lambda_`, and the
requested LiDAR validation mode. The 26.04 six-camera recipe samples 2048
LiDAR rays alongside 6144 camera rays per batch. A log line such as
`call/n_frames_per_lidar: []` is not a dataset count: the 26.04 Gaussian
post-processing constructor passes an empty LiDAR list because that
camera-oriented post-processing has no per-LiDAR-frame parameters.

For a formal train-and-validation run, also set
`REQUIRE_LIDAR_VALIDATION_EVIDENCE=1`. This requires finite per-frame Chamfer
and ray-drop metrics with timestamps plus matched predicted/ground-truth PLY
pairs. NRE 26.04 has an audited exporter defect: `collect_metric()` resets its
`is_lidar` argument to false, placing LiDAR samples below `per_camera` and
leaving `per_lidar` empty. The gate rejects that layout by default. Set
`ALLOW_NRE_2604_LIDAR_GROUPING_BUG=1` only for the audited 26.04 image; the
acceptance output will retain the explicit
`nre_26_04_vendor_grouping_bug` classification rather than silently relabeling
the evidence.

A complete NuRec run contains:

```text
<RUN-ID>/artifacts/last.usdz
<RUN-ID>/config/parsed.yaml
<RUN-ID>/checkpoints/last.ckpt
```

## 9. Render A Trained Scene

Use the official NuRec container to render the trained neural scene. The
launcher defaults to the strict 1000-step output, renders every tenth frame
from `camera_front`, and writes a lightweight PNG preview without modifying the
training artifacts:

```bash
bash scripts/render_nurec_result.sh
```

Override the defaults with environment variables. For example, render all
three trained cameras at a denser frame interval:

```bash
RENDER_CAMERA_IDS=camera_front,camera_front_left,camera_front_right \
FRAME_STEP=5 \
IMAGE_SCALE=0.5 \
RENDER_DIR=outputs/nurec_1000step_preview_3cam \
  bash scripts/render_nurec_result.sh
```

The script fails if the output directory is already non-empty. Set
`ALLOW_NONEMPTY_OUTPUT=1` only when intentionally resuming into an existing
directory. Set `ARTIFACT_PATH` when the selected NuRec output contains more
than one run.

## Local Checks

On any Ubuntu machine with Python 3 available:

```bash
bash scripts/run_local_checks.sh
```

This checks shell syntax and validates the local nuScenes scene without
requiring NCore Python dependencies.
