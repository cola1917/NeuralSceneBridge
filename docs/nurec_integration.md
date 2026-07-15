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

By default the aux script writes into `outputs/ncore`, beside the manifest and
NCore stores. NuRec requires the `.json`, `.zarr.itar`, and `.aux.*.zarr` files
to be in the same mounted dataset directory.

## 7. Run A Single-GPU Smoke Reconstruction

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

A complete NuRec run contains:

```text
<RUN-ID>/artifacts/last.usdz
<RUN-ID>/config/parsed.yaml
<RUN-ID>/checkpoints/last.ckpt
```

## Local Checks

On any Ubuntu machine with Python 3 available:

```bash
bash scripts/run_local_checks.sh
```

This checks shell syntax and validates the local nuScenes scene without
requiring NCore Python dependencies.
