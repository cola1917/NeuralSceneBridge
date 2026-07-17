# NCore nuScenes Conversion

This project vendors the NVIDIA NCore nuScenes converter under
`third_party/ncore_converter`. The converter source is from `NVIDIA/ncore`
commit `540a8b5fa953c3bc7e31d2850fc6424aef4444ef`.

The local environment is for data conversion only:

```text
nuScenes raw data -> NCore V4 sequence JSON / store
```

NuRec training should still use NVIDIA's official NuRec containers.

Scene data is not committed. On the server, either place a full
`data/nuscenes-mini` download or extract `scene-0061` locally and upload that
folder:

```bash
python scripts/extract_nuscenes_scene.py --force
# then copy data/nuscenes-mini-scene-0061 to the server
```

## Files

- `data/nuscenes-mini-scene-0061/`: local/server single-scene subset for `scene-0061` (gitignored).
- `scripts/extract_nuscenes_scene.py`: build that subset from a full `nuscenes-mini` download.
- `requirements/ncore-converter.txt`: pinned pip dependencies for conversion.
- `environment/ncore-converter.yml`: Miniconda environment base.
- `docker/ncore-conda-base.Dockerfile`: local dependency layer with Ubuntu, Miniconda, and NCore deps.
- `docker/ncore-converter.Dockerfile`: runnable project image with the vendored converter.
- `scripts/setup_ncore_images.sh`: build both local Docker images on Ubuntu/Linux.
- `scripts/validate_ncore_env.py`: validate imports, CLI, and optional data.
- `scripts/run_ncore_nuscenes_converter.sh`: Ubuntu/Linux conversion wrapper.
- `scripts/pull_nurec_images.sh`: pull official NuRec and auxiliary data images.
- `scripts/run_nurec_aux_data.sh`: run the official auxiliary data container.
- `scripts/server_preflight.sh`: validate the Ubuntu server Docker/GPU setup.
- `scripts/run_nurec_train.sh`: launch a conservative single-GPU NuRec smoke run.
- `scripts/validate_nurec_artifacts.sh`: verify USDZ, config, and checkpoint outputs.

## Build Converter Images

The converter uses two local images:

```text
nsb/ncore-conda-base:2026-07-10  # dependency layer, built to avoid reinstalling Miniconda/Python deps every edit
nsb/ncore-converter:2026-07-17-dense-v1   # runnable converter image
```

The base image is only our local parent layer. It is not an NVIDIA NuRec image.
The conversion command uses `nsb/ncore-converter:2026-07-17-dense-v1`.

Bash:

```bash
bash scripts/setup_ncore_images.sh
```

The setup wrapper defaults to the NGC CUDA Ubuntu 22.04 base because Docker
Hub can be unreachable on rental networks. Both the base and Miniconda source
remain configurable:

```bash
UPSTREAM_BASE_IMAGE=ubuntu:22.04 \
MINICONDA_BASE_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda \
  bash scripts/setup_ncore_images.sh
```

This builds two images:

```text
nsb/ncore-conda-base:2026-07-10
nsb/ncore-converter:2026-07-17-dense-v1
```

## Validate Environment And Data

Before building the images, the local data can be checked without installing
NCore dependencies:

```bash
python scripts/validate_ncore_env.py \
  --converter-root third_party/ncore_converter \
  --skip-imports \
  --skip-cli-help \
  --nuscenes-root data/nuscenes-mini-scene-0061 \
  --scene-name scene-0061
```

After building the images, validate the full converter environment:

```bash
docker run --rm \
  -v "${PWD}:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace/third_party/ncore_converter \
  nsb/ncore-converter:2026-07-17-dense-v1 \
  python scripts/validate_ncore_env.py \
  --converter-root third_party/ncore_converter \
  --nuscenes-root data/nuscenes-mini-scene-0061 \
  --scene-name scene-0061
```

Expected local scene summary for `scene-0061`:

```text
samples: 39
camera_keyframes: 234
lidar_keyframes: 39
radar_keyframes: 195
```

## Convert scene-0061

Bash:

```bash
bash scripts/run_ncore_nuscenes_converter.sh
```

The project wrapper defaults `CUBOID_SAMPLING=lidar-sweeps` so dynamic cuboids
are interpolated at every LiDAR sweep. Set `CUBOID_SAMPLING=keyframes` only to
reproduce the source annotation cadence. Dense cuboids require rebuilding the
NCore/NuRec artifact; they do not repair an already-trained USDZ at runtime.

The wrapper can resolve either selector for manual debugging, but shared jobs
and the default path use the native nuScenes scene token. Set exactly one when
overriding the default:

```bash
SCENE_NAME=scene-1077 bash scripts/run_ncore_nuscenes_converter.sh

SCENE_TOKEN=d25718445d89453381c659b9c8734939 \
  bash scripts/run_ncore_nuscenes_converter.sh
```

NCore resolves both selectors to the same scene and records both
`nuscenes_scene_name` and `nuscenes_scene_token` in the sequence metadata.
Cross-project jobs must use `SCENE_TOKEN` as the canonical machine identifier;
the resolved scene name remains display metadata and the NCore sequence
directory name.

Equivalent converter command inside the container:

```bash
python -m tools.data_converter.nuscenes.main \
  --root-dir /workspace/data/nuscenes-mini-scene-0061 \
  --output-dir /workspace/outputs/ncore \
  nuscenes-v4 \
  --version v1.0-mini \
  --scene-token cc8c0bf57f984915a77078b10eb33198 \
  --cuboid-sampling lidar-sweeps \
  --store-type itar \
  --profile separate-sensors \
  --sequence-meta
```

The formal dense output is written under
`outputs/ncore_dense_lidar_sweeps_v1`. Keeping it separate from the legacy
`outputs/ncore` tree prevents a new conversion or acceptance check from
silently reusing keyframe-only stores.

## Generate NuRec Auxiliary Data

NuRec auxiliary data is generated by NVIDIA's official NGC tools image, not by
the local NCore converter image:

```bash
export NGC_API_KEY=<your-ngc-api-key>
bash scripts/pull_nurec_images.sh
```

After NCore conversion, pass the generated manifest or shard filename relative
to `outputs/ncore`:

```bash
DATASET_PATH=<generated-ncore-json-name> \
  bash scripts/run_nurec_aux_data.sh
```

For a monolithic shard, prefer `SHARD_FILE_PATTERN`:

```bash
SHARD_FILE_PATTERN=<generated-ncore-zarr-itar-name> \
  bash scripts/run_nurec_aux_data.sh
```

To limit cameras, pass space-separated NCore camera IDs:

```bash
DATASET_PATH=<generated-ncore-json-name> \
CAMERA_IDS="camera_front camera_left" \
  bash scripts/run_nurec_aux_data.sh
```

The aux output defaults to the directory containing the selected manifest or
shard. For example, `DATASET_PATH=scene-0061/scene-0061.json` writes to
`outputs/ncore/scene-0061`, while a manifest directly under `outputs/ncore`
writes there. Set `OUTPUT_DIR` explicitly to override this behavior. Keep an
explicit output beside the manifest because NuRec training expects the NCore
`.json`, `.zarr.itar`, and `.aux.*.zarr` files together.

## Run NuRec Smoke Training

Prepare the environment file and update `DATASET_PATH` to the generated NCore
manifest name:

```bash
cp config/nurec-smoke.env.example config/nurec-smoke.env
bash scripts/run_nurec_train.sh
```

The default smoke configuration uses the three forward cameras and one epoch.
See `docs/nurec_integration.md` for the complete server sequence.
