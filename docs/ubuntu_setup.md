# Ubuntu Setup

Recommended host OS for this project: **Ubuntu 22.04 LTS**.

Why 22.04:

- NVIDIA's NuRec containers only require Linux x86_64, Docker, NVIDIA Driver, and NVIDIA Container Toolkit; they do not require a specific Ubuntu release.
- Ubuntu 22.04 LTS is the safer default for CUDA, Docker, NVIDIA Container Toolkit, and long-running AV reconstruction workloads.
- The local NCore converter dependency layer also uses `ubuntu:22.04`.

Ubuntu 24.04 LTS is acceptable for a fresh workstation with recent drivers,
especially if using newer GPUs that need newer driver branches. For first pass
reproducibility, use 22.04 LTS unless the GPU/driver stack pushes you to 24.04.

## Required Host Runtime

Install:

- Docker Engine 23.0.1 or newer
- NVIDIA Driver compatible with your GPU
- NVIDIA Container Toolkit 1.13.5 or newer
- NGC CLI if you want to use NVIDIA's documented NGC workflow

Quick GPU runtime check:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Project preflight:

```bash
bash scripts/server_preflight.sh
```

## NuRec Images

NuRec uses two official NGC images:

```text
nvcr.io/nvidia/nre/nre-ga:26.04        # NuRec training / reconstruction
nvcr.io/nvidia/nre/nre-tools-ga:26.04  # auxiliary data generation
```

Pull both after setting `NGC_API_KEY`:

```bash
export NGC_API_KEY=<your-ngc-api-key>
bash scripts/pull_nurec_images.sh
```

## Local Converter Images

The local converter images are separate from the official NuRec images:

```bash
bash scripts/setup_ncore_images.sh
```

This builds:

```text
nsb/ncore-conda-base:2026-07-10  # local dependency layer
nsb/ncore-converter:2026-07-10   # runnable converter
```

Here "base image" means Docker parent image. In this repo it just caches Ubuntu,
Miniconda, and Python dependencies so the runnable converter image can rebuild
quickly when project files change. It is unrelated to NVIDIA's NuRec containers.

Then convert the first daytime scene:

```bash
bash scripts/run_ncore_nuscenes_converter.sh
```

After conversion, generate NuRec auxiliary data with the official tools image:

```bash
DATASET_PATH=<generated-ncore-json-name> \
  bash scripts/run_nurec_aux_data.sh
```

The auxiliary files are written beside the NCore files by default. Then launch
the one-epoch, three-camera smoke run:

```bash
cp config/nurec-smoke.env.example config/nurec-smoke.env
# Update DATASET_PATH in config/nurec-smoke.env.
bash scripts/run_nurec_train.sh
```
