# TensorRT export and FPS benchmark

Run from the repository root:

```powershell
.\.venv-training\Scripts\python.exe scripts\benchmark_tensorrt.py
```

The script exports `models/yolo26s-cooking-crops-v1.pt` to the adjacent `.engine`
using FP16, static 640 x 640 input, batch 1, and a 2 GiB builder workspace limit.
It reuses an existing engine; use `--rebuild` after changing the weights or input
size. Build engines on the deployment GPU/software environment.

Dependencies in `.venv-training`: `tensorrt-cu12>=10,<11`, `onnx>=1.17,<2`,
`onnxslim`, and `onnxruntime`, in addition to the training dependencies.

After warmup, each measurement runs unthrottled for 30 seconds:

- Model-only: a preprocessed tensor already on the GPU; includes Python backend
  dispatch and synchronization, excludes preprocessing and postprocessing.
- End-to-end prediction: cycles the 15 decoded cooking crops, including CPU
  preprocessing, transfer, inference, and postprocessing. Excludes camera,
  video decoding, rendering, storage, and VLM execution.

Results include aggregate FPS, mean latency, median and p95 latency, input
shape/dtype, and runtime versions in `outputs/tensorrt-benchmark/results.json`.
Use `--seconds 60` for a longer measurement or `--output PATH` to preserve runs.
This measures this configuration under the current GPU load; it does not prove
an absolute maximum across precisions, image sizes, or batch sizes. It does not
validate prediction accuracy. The source crops retain the baseline model's ROI.

## Concurrent app test at 30 FPS

```powershell
.\.venv-training\Scripts\python.exe -m scripts.evaluate_app_load --fps 30 --output outputs/app-30fps-evaluation-new
```

This reuses the frozen 35-image recipe evaluation while decoding the mushroom
video and running TensorRT in a separate thread. The default target is 30 FPS.
It drops missed slots rather than building a backlog. YOLO outputs do not feed
the VLM or recipe state; this is a coexistence test, not an integrated camera app.

On 2026-09-06, the final run processed 945 video frames in 31.51 seconds:
29.99 FPS, one missed slot, mean decode/crop/predict time 12.64 ms and p95
17.18 ms. A prior Event.wait pacing implementation reached only 28.76 FPS;
using Python's high-resolution sleep improved pacing. Both runs are preserved
under `outputs/app-30fps-evaluation` and `outputs/app-30fps-evaluation-refined`.

All 35 VLM responses were valid. Recipe recommendations remained 25/35 correct
(71.4%), visual matches 20/35 (57.1%), and premature recommendations 6, unchanged
from the baseline. Mean VLM request time in the final warm run was 0.892 seconds;
different warmup/cache/power conditions prevent attributing that latency change
to FPS. The first run included a 9.071-second initial VLM request.

Thirty FPS is a feasible detector target in this short run. There is no measured
accuracy improvement: YOLO is not integrated with recipe decisions, and no dense
temporal ground truth exists here to compare 10 versus 30 FPS event recall.
Camera capture, rendering, and long-duration thermal stability remain untested.
