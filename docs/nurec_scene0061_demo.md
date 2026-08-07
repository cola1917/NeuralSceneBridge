# scene-0061 NuRec Interview Demo

This demo is the renderer-level handoff for the locked `scene-0061`
reconstruction. It proves that the pinned USDZ can be replayed, edited through
the dynamic-object RPC, and rendered under a bounded camera-pose sweep for
downstream simulation plumbing. It is open-loop and is not a CARLA
closed-loop acceptance result. The reconstructed LiDAR path currently remains
diagnostic; see [`docs/downstream_simulation_handoff.md`](downstream_simulation_handoff.md).

## Immutable Inputs

`demo/scene0061/manifest.json` is the source of truth. The canonical USDZ is
the artifact whose SHA-256 is
`69e2c36e31e113f9ad66968a0a0a4243c7989dae5582a91a43d150051eaf98b4`. The
renderer refuses a missing artifact, a changed size/hash, a missing target
track, or a target track that is not marked `DYNAMIC|CONTROLLABLE`.

The three checked-in cases are:

| Case | Evidence | Edit |
| --- | --- | --- |
| `V01` | `original_replay.mp4` | original USDZ trajectory |
| `V02` | `lead_vehicle_edit.mp4` | +0.5 m translation of the manifest target track |
| `V03` | `camera_pose_sweep.mp4` | bounded sensor-frame translation and yaw sweep |

## Start SensorsimService

The service must have enough free GPU memory to load the canonical USDZ. Stop
other GPU-heavy processes before starting it; the service otherwise fails
closed with CUDA out-of-memory and no capture evidence is produced.

The renderer also normalizes Python import precedence before loading the NuRec
protobuf package, so a ROS `PYTHONPATH` does not silently select an incompatible
system protobuf build.

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

Wait for `Serving main gRPC on 127.0.0.1:46443` before capturing. Keep the
container running for all three cases and remove it after capture:

```bash
docker rm -f nsb-nurec-demo-0061
```

## Capture The Cases

Each command writes an immutable evidence directory containing `frames/`,
`camera_frames/`, `frames.jsonl`, `evidence.json`, and the MP4. The six camera
cells use the ClosedLoopBench order and form a 3x2 `2400x900` grid at `30 FPS`.
For V03, the stitched frames and MP4 include a bottom pose readout with the
sensor-frame `dx/dy/dz`, yaw, sweep progress, and a progress rail; the raw
per-camera frames remain unmodified. V03 also writes
`pose_sweep_summary.json` and `pose_sweep_summary.png`.
The command refuses to encode a video when an RPC fails, a camera frame is
black, a frame is dropped, or a case probe does not pass.

RGB requests retain each logical camera/actor pose interval in their `PosePair`
metadata but sample the temporal field at the interval midpoint with a
one-microsecond wire window, matching the NuRec 26.04 replay path.

```bash
for case_id in V01 V02 V03; do
  python3 scripts/render_counterfactual_video.py \
    --case "$case_id" \
    --manifest demo/scene0061/manifest.json \
    --server-address 127.0.0.1:46443 \
    --artifact outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/9aChcizbAsm4oDQKJMdBHM/artifacts/last.usdz \
    --output-dir "outputs/nurec_scene0061_demo/$case_id"
done
```

Use `--probe-only` to verify the V02 A/A/B or V03 camera probes without
encoding a sequence. Never point a case at a different training run: the
manifest hash and artifact identity are part of every evidence record.

## V04 RGB/LiDAR Diagnostic Visualization

V04 is the final front-camera and LiDAR comparison view. The formal
`render_multimodal_alignment_video_v2b.py` path is the canonical public capture.
It renders baseline and edited requests on the same logical render windows.
Because it does not capture an A/A repeat, its difference overlay is an
open-loop visual diagnostic, not an A/A-controlled measurement.

```bash
python3 scripts/render_multimodal_alignment_video_v2b.py \
  --server-address 127.0.0.1:46443 \
  --output-dir outputs/nurec_scene0061_final/multimodal_20fps \
  --overwrite
```

The evidence distinguishes RGB response changes and LiDAR response changes,
but the metadata center and projected geometry are references rather than
per-point ownership labels. The result therefore documents counterfactual
renderer response, not a strict rigid or pointwise registration proof. RGB is
sampled at the logical window midpoint while LiDAR remains referenced to the
end of its spin; the evidence must not report that as a zero-microsecond
physical timestamp alignment.

The ClosedLoopBench investigation found that the NRE 26.04 dynamic LiDAR
renderer can omit vehicles at their true positions and emit fixed scattered
returns instead. V04 is therefore useful for exposing the failure, but it does
not make the reconstructed LiDAR suitable for perception-driven simulation.

## Downstream Simulation Boundary

`NeuralSceneBridge` owns reconstruction artifacts, identity gates, sensor RPC
requests, and open-loop evidence. `ClosedLoopBench` owns the synchronous CARLA
clock, ego/actor execution, observation delivery, and closed-loop evaluation.
Keep raw CARLA LiDAR and reconstructed NuRec LiDAR as separate provenance
routes when comparing them. Do not combine a successful RGB replay with the
current NuRec LiDAR and call the resulting stream physically aligned.

The current handoff status and next reconstruction gate are recorded in
[`docs/downstream_simulation_handoff.md`](downstream_simulation_handoff.md).

## Build The Quality Report

The report generator discovers `evidence.json` files below the supplied root,
checks their case/manifest/artifact hashes, verifies contiguous metadata and
zero dropped frames, and imports the canonical validation metrics from
`val/metrics.yaml` when available.

```bash
python3 scripts/generate_nurec_quality_report.py \
  --manifest demo/scene0061/manifest.json \
  --evidence-root outputs/nurec_scene0061_demo \
  --output demo/scene0061/quality_report.json
```

The output is `passed` only when USDZ and checkpoint identity, all three case
captures, and the finite Chamfer/ray-drop metrics pass. Missing captures are
reported as `pending_capture`; stale identities, dropped frames, malformed
metadata, and missing metrics are `failed`. Neither state has `pass: true`.

## Local Verification

```bash
python3 -m unittest discover -s tests -p 'test*.py'
```

The repository's `pytest` entry point may be unavailable on hosts that mix a
conda pytest launcher with the system `_pytest` package; the unittest command
does not require that optional integration.
