# Interview Demo Runbook

This runbook is the reproducible handoff for the scene-0061 NuRec
reconstruction demo. It covers NVIDIA NRE/SensorsimService rendering and the
renderer contract that a downstream simulator can consume. CARLA is neither
started nor used as a control loop in this repository; the downstream boundary
is described in [`docs/downstream_simulation_handoff.md`](downstream_simulation_handoff.md).

## 1. Check The Canonical Identity

Open `demo/scene0061/manifest.json` and use the recorded artifact URI. The
current canonical identity is:

```text
USDZ SHA-256: 69e2c36e31e113f9ad66968a0a0a4243c7989dae5582a91a43d150051eaf98b4
USDZ size:    1164047285 bytes
target track: c1958768d48640948f6053d04cffd35b
NuRec:        26.4.146 / c63f08a4
```

Validate the embedded actor arrays before rendering:

```bash
python3 scripts/validate_nurec_usdz_tracks.py \
  outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/9aChcizbAsm4oDQKJMdBHM/artifacts/last.usdz \
  --min-total 223 --min-vehicles 23 --min-pedestrians 58
```

The target track must exist and be one of the `DYNAMIC|CONTROLLABLE` tracks.
The renderer also rechecks the manifest SHA-256 and checkpoint identity before
opening the archive.

## 2. Start The Renderer Service

NuRec loads the complete neural scene into GPU memory. Ensure that no other
GPU-heavy process is occupying the device. Start the service with the artifact
directory mounted read-only:

```bash
mkdir -p outputs/nurec_scene0061_demo_runtime_metrics
docker rm -f nsb-nurec-demo-0061 >/dev/null 2>&1 || true
docker run -d --name nsb-nurec-demo-0061 \
  --shm-size=64g --gpus all --network host \
  -v "$PWD/outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/9aChcizbAsm4oDQKJMdBHM/artifacts:/scenes:ro" \
  -v "$PWD/outputs/nurec_scene0061_demo_runtime_metrics:/metrics" \
  nvcr.io/nvidia/nre/nre-ga:26.04 serve-grpc \
  --artifact-glob /scenes/last.usdz --host 127.0.0.1 --port 46443 \
  --health-port 46444 --test-scenes-are-valid --enable-editing-actors \
  --metrics-output-dir /metrics --enable-timing --timing-verbosity summary
```

Wait for:

```text
Serving main gRPC on 127.0.0.1:46443
```

If the container exits with `CUDA error: out of memory`, preserve its logs and
free GPU memory before retrying. Do not switch to a different `last.usdz` to
make the service start.

## 3. Run A Short Probe First

The formal cases use six cameras in the order
`camera_front_left`, `camera_front`, `camera_front_right`, `camera_back_left`,
`camera_back`, `camera_back_right`, stitched as a 3x2 `2400x900` grid at
`30 FPS`. For the first runtime check, request only a few sparse samples in a
fresh output directory:

```bash
python3 scripts/render_counterfactual_video.py \
  --manifest demo/scene0061/manifest.json \
  --artifact outputs/nurec_scene0061_renderable_lidar_v3_6cam_40k_formal_attempt_001/9aChcizbAsm4oDQKJMdBHM/artifacts/last.usdz \
  --server-address 127.0.0.1:46443 \
  --case V01 --start-timestamp 1532402927598150 \
  --end-timestamp 1532402930112460 --frame-step 4 \
  --output-dir outputs/nurec_scene0061_demo_probe/V01
```

For a probe-only V02 or V03 run, add `--probe-only`. V02 performs the fixed
A/A/B baseline-repeat/edit sequence. It requires exact request digest
repeatability, exact repeated RGB payload identity, a changed target pose and
RGB payload, and an unchanged non-target actor digest. V03 renders +x, +y and
+yaw probes and requires distinct successful RGB payloads.

## 4. Generate The Full Cases

After the short probe succeeds, use a new immutable output root:

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

The renderer writes stitched `frames/*.jpg`, per-camera
`camera_frames/<camera>/*.jpg`, and `frames.jsonl` before encoding. It records,
per frame, the timestamp, all six camera poses, dynamic-object digest, total
and per-camera RPC latency, decoded dimensions, RGB SHA-256, response status,
and any drop reason. A dropped, empty, black, malformed, or wrong-size camera
response prevents MP4 encoding.
For V03, the stitched frame/MP4 view also carries a readable sensor-frame pose
overlay (`dx/dy/dz`, yaw, and progress). The same trajectory is saved as
`pose_sweep_summary.json` plus `pose_sweep_summary.png`; raw per-camera frames
remain unmodified.

The three output names are fixed by the case files:

```text
V01 -> original_replay.mp4
V02 -> lead_vehicle_edit.mp4
V03 -> camera_pose_sweep.mp4
```

## 5. Build The Formal Report

The report generator reads the three `evidence.json` files, verifies the
manifest/case/artifact hashes, checks contiguous frame metadata and zero drops,
and imports the finite Chamfer and ray-drop measurements from the canonical
validation run:

```bash
python3 scripts/generate_nurec_quality_report.py \
  --manifest demo/scene0061/manifest.json \
  --evidence-root outputs/nurec_scene0061_demo \
  --output demo/scene0061/quality_report.json

python3 scripts/generate_nurec_quality_report.py \
  --validate demo/scene0061/quality_report.json
```

The command returns success only for a report whose `status` is `passed` and
whose `pass` field is true. `pending_capture`, missing metrics, null values,
stale hashes, incomplete probes, and failed case captures all return non-zero.
The report includes each case's frame/time range, drop count, image metrics,
latency p50/p95, V02 actor delta and repeatability fields, and V03 maximum pose
change. The report must never be edited by hand to change `pass`.

## 6. Shutdown And Preserve Evidence

```bash
docker rm -f nsb-nurec-demo-0061
```

Keep the output directory and service logs outside Git. The repository should
contain only source code, case JSON, manifest, report JSON, tests, and these
instructions. `outputs/`, `.cache/`, USDZ, checkpoints, datasets, raw frames,
and MP4 files are ignored or local-only.

## Scope Limits

- No CARLA process, ego control, braking, TTC, collision, or route metrics.
- No camera intrinsic, FOV, focal-length, or principal-point edits.
- V02 edits only `c1958768d48640948f6053d04cffd35b`.
- V03 edits only camera extrinsic pose within the checked-in bounds.
- The formal V04 script uses a live RGB/LiDAR A/A-controlled capture for
  counterfactual response comparison, but it does not claim per-point ownership
  or strict rigid world registration. The `v2`/`v2b` visual variants in the
  final playback directory do not include the A/A control.
- RGB/LiDAR pairing is by logical render window: RGB uses the window midpoint
  and LiDAR uses the end-of-spin reference. This is not physical timestamp
  alignment.
- The current NuRec LiDAR output is not perception-grade or closed-loop-ready;
  ClosedLoopBench's actor-aware LiDAR gate remains the acceptance boundary.
