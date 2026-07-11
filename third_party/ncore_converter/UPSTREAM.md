# NVIDIA NCore Converter Snapshot

This directory vendors only the pieces needed to run the NVIDIA NCore
nuScenes-to-NCore V4 converter from:

- Repository: https://github.com/NVIDIA/ncore
- Upstream commit: `540a8b5fa953c3bc7e31d2850fc6424aef4444ef`
- Retrieved on: 2026-07-10

Vendored paths:

- `tools/data_converter/cli.py`
- `tools/data_converter/structured_lidar_model.py`
- `tools/data_converter/README.md`
- `tools/data_converter/BUILD.bazel`
- `tools/data_converter/nuscenes/*`
- `LICENSE`
- `NCORE_README.md`

The Python runtime dependency is installed from PyPI as `nvidia-ncore==19.5.0`.
This project does not vendor the full NCore source tree or build it with Bazel.
