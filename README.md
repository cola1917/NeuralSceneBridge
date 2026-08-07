# NeuralSceneBridge

NeuralSceneBridge is a reconstruction bridge for downstream simulation. It
turns a recorded scene, sensor streams, and dynamic actor tracks into a pinned
NuRec/USDZ artifact, then serves that artifact through NVIDIA
`SensorsimService` for reproducible RGB/LiDAR replay and controlled edits.

The scene-0061 interview delivery is intentionally open-loop. It proves the
reconstruction can be replayed, a dynamic actor can be edited in the RGB path,
and camera pose requests can be probed. It does not by itself prove physical
RGB/LiDAR alignment, perception-grade reconstructed LiDAR, or CARLA
closed-loop behavior. The downstream ownership boundary is documented in
[`docs/downstream_simulation_handoff.md`](docs/downstream_simulation_handoff.md).

## Requirements

- Linux x86_64 with Docker and NVIDIA Container Toolkit
- NVIDIA GPU with enough free memory to load the canonical USDZ
- NVIDIA NRE image `nvcr.io/nvidia/nre/nre-ga:26.04`
- Python 3.10+ with `grpcio`, `protobuf`, `Pillow`, `opencv-python`, and
  `numpy`
- The NVIDIA NuRec Python protobuf package, normally installed under
  `/home/cwadmin/sim-env/data/CARLA_0.9.16/PythonAPI/examples/nvidia/nurec`

The renderer puts the active interpreter's site-packages ahead of ROS-exported
paths before loading the NuRec protobufs. This matters on hosts where
`PYTHONPATH` also exposes an older `/usr/lib/python3/dist-packages` protobuf.

The reconstruction artifact, checkpoint, dataset, raw frames, and MP4 files
are local runtime assets. They are deliberately not stored in Git. Configure
their paths in [demo/scene0061/manifest.json](demo/scene0061/manifest.json).
The checked-in manifest records the canonical USDZ identity and the target
track `c1958768d48640948f6053d04cffd35b`.

## Reconstruction To Simulation

The handoff is deliberately split into two products:

| Layer | This repository | Downstream consumer |
| --- | --- | --- |
| Reconstruction | USDZ artifact, actor tracks, sensor poses, and identity gates | `ClosedLoopBench` loads the pinned artifact |
| Renderer contract | `render_rgb` / `render_lidar`, logical windows, frames, and hashes | Observation adapter and agent runtime |
| Simulation loop | Fixed-trajectory replay and counterfactual probes | CARLA clock, ego control, collision state, and evaluation |

The current artifact is ready for renderer-level RGB studies and integration
plumbing. Its reconstructed LiDAR is kept as a diagnostic input because the
NRE 26.04 dynamic LiDAR path currently fails the downstream actor-aware check.
See [`docs/downstream_simulation_handoff.md`](docs/downstream_simulation_handoff.md)
for the exact boundary and evidence.

## Verify The Artifact

The manifest must point at the exact USDZ that was validated. Verify the
embedded actor inventory before starting a renderer:

```bash
python3 scripts/validate_nurec_usdz_tracks.py \
  outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/9aChcizbAsm4oDQKJMdBHM/artifacts/last.usdz \
  --min-total 223 --min-vehicles 23 --min-pedestrians 58
```

The command must report the target track in the required-track audit used by
the training run. Do not substitute an older smoke artifact merely because it
also has a `last.usdz` name.

## Start SensorsimService

Free GPU memory first. A running CARLA process or another renderer can consume
the entire device before NuRec starts. The complete command, including the
canonical artifact mount and port, is in
[docs/interview_demo.md](docs/interview_demo.md).

```bash
mkdir -p outputs/nurec_scene0061_demo_runtime_metrics
docker run -d --name nsb-nurec-demo-0061 \
  --shm-size=64g --gpus all --network host \
  -v "$PWD/outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/9aChcizbAsm4oDQKJMdBHM/artifacts:/scenes:ro" \
  -v "$PWD/outputs/nurec_scene0061_demo_runtime_metrics:/metrics" \
  nvcr.io/nvidia/nre/nre-ga:26.04 serve-grpc \
  --artifact-glob /scenes/last.usdz --host 127.0.0.1 --port 46443 \
  --health-port 46444 --test-scenes-are-valid --enable-editing-actors \
  --metrics-output-dir /metrics --enable-timing --timing-verbosity summary
```

Wait for the service log to say `Serving main gRPC on 127.0.0.1:46443`.

## Generate The Three Videos

Use a new output root for every capture attempt. The renderer writes frames
and JSONL evidence first, performs the V02/V03 probes, and encodes an MP4 only
when every requested frame is valid:

```bash
for case_id in V01 V02 V03; do
  python3 scripts/render_counterfactual_video.py \
    --manifest demo/scene0061/manifest.json \
    --artifact outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/9aChcizbAsm4oDQKJMdBHM/artifacts/last.usdz \
    --server-address 127.0.0.1:46443 \
    --case "$case_id" \
    --output-dir "outputs/nurec_scene0061_demo/$case_id"
done
```

The expected filenames are `original_replay.mp4`, `lead_vehicle_edit.mp4`,
and `camera_pose_sweep.mp4`. Each is a synchronized six-camera 3x2 grid in
ClosedLoopBench order (`2400x900`, `30 FPS`). Case definitions and bounded edit
parameters are in [demo/scene0061/cases](demo/scene0061/cases).

## Build And Validate The Report

The report generator binds every case to the manifest, case SHA-256, artifact
SHA-256, frame metadata, video file, and finite reconstruction metrics:

```bash
python3 scripts/generate_nurec_quality_report.py \
  --manifest demo/scene0061/manifest.json \
  --evidence-root outputs/nurec_scene0061_demo \
  --output demo/scene0061/quality_report.json

python3 scripts/generate_nurec_quality_report.py \
  --validate demo/scene0061/quality_report.json
```

The generator exits non-zero for `pending_capture` or `failed`. A report is
formally passing only when all three cases have zero dropped frames, complete
metadata, valid videos, passed probes, matching immutable identities, and the
finite Chamfer/ray-drop measurements.

## V04 RGB/LiDAR Diagnostic Video

V04 is a renderer-level multimodal visualization, not an alignment pass. The
canonical `v2b` script renders baseline and edited RGB/LiDAR responses on the
same logical windows. It does not capture an A/A repeat control, so the result
is an open-loop renderer diagnostic rather than an A/A-controlled comparison.
The view does not prove per-point actor ownership or rigid target registration.

```bash
python3 scripts/render_multimodal_alignment_video_v2b.py \
  --server-address 127.0.0.1:46443 \
  --output-dir outputs/nurec_scene0061_final/multimodal_20fps \
  --overwrite
```

The V04 timestamp fields describe logical-window pairing: RGB is sampled at
the window midpoint and LiDAR is referenced to the end of its spin. A metadata
value of `null` for physical timestamp alignment is intentional; it does not
mean that the two modalities were measured at exactly the same physical time.

## Output Layout

```text
outputs/nurec_scene0061_demo/<case-id>/
  frames/*.jpg                 # stitched 3x2 video frames (2400x900)
  camera_frames/<camera>/*.jpg # individual 800x450 camera frames
  frames.jsonl
  evidence.json
  pose_sweep_summary.json   # V03 numeric trajectory and extrema
  pose_sweep_summary.png    # V03 translation/yaw curves
  original_replay.mp4       # V01
  lead_vehicle_edit.mp4    # V02
  camera_pose_sweep.mp4     # V03; stitched grid includes pose readout

outputs/nurec_scene0061_final/multimodal_20fps/
  frames/*.jpg                 # final 1600x900 V04 frames
  rgb/{original,edited}.jpg
  lidar/{original,edited}.xyzi.bin
  frames.jsonl
  evidence.json
  V04_multimodal_alignment_20fps.mp4
```

The local final delivery also contains a retained previous V04 diagnostic
variant named `V04_multimodal_alignment_20fps_bak.mp4`. The main V04 video and
its evidence are playback artifacts, not a claim that reconstructed LiDAR is
ready for downstream perception.

The `outputs/` tree is ignored by Git. Do not add USDZ, checkpoint, dataset,
raw frames, or MP4 files to the repository.

## Tests And Limits

```bash
python3 -m unittest discover -s tests -p 'test*.py'
bash scripts/run_local_checks.sh
```

This demo does not claim CARLA closed-loop behavior, self-vehicle braking,
TTC, collision, route metrics, physical RGB/LiDAR world consistency, or
perception-grade reconstructed LiDAR. Training-time LiDAR supervision and a
non-empty `render_lidar` response are necessary checks, but neither is enough
to establish actor/world consistency. That gate must be completed by the
downstream ClosedLoopBench evaluation.
