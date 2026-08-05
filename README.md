# NeuralSceneBridge

NeuralSceneBridge is an independent NuRec/NRE neural-scene reconstruction and
editable-rendering demo. The interview deliverable is intentionally narrower
than an autonomous-driving stack: it uses NVIDIA `SensorsimService` gRPC
requests to replay a trained USDZ, edit one dynamic actor, and sweep the
camera pose. It does not start or control CARLA.

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

## V04 RGB/LiDAR Consistency Video

V04 is the final multimodal visualization. It renders baseline, A/A control,
and edited RGB/LiDAR responses on the same logical windows. The video keeps the
front-camera comparison on top and shows a fixed-scale LiDAR BEV difference
overlay below. A/A-controlled voxel and pixel differences are recorded in
`evidence.json`; the result does not claim per-point actor ownership or rigid
target registration.

```bash
python3 scripts/render_multimodal_alignment_video.py \
  --server-address 127.0.0.1:46443 \
  --output-dir outputs/nurec_scene0061_final/multimodal_20fps \
  --overwrite
```

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
  rgb/{original,repeat,edited}.jpg
  lidar/{original,repeat,edited}.xyzi.bin
  frames.jsonl
  evidence.json
  V04_multimodal_alignment_20fps.mp4
```

The `outputs/` tree is ignored by Git. Do not add USDZ, checkpoint, dataset,
raw frames, or MP4 files to the repository.

## Tests And Limits

```bash
python3 -m unittest discover -s tests -p 'test*.py'
bash scripts/run_local_checks.sh
```

This demo does not claim CARLA closed-loop behavior, self-vehicle braking,
TTC, collision, route metrics, or unverified RGB/LiDAR world consistency.
LiDAR evidence is optional and must be captured and validated separately before
it is mentioned in a quality claim.
