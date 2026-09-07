"""Export the trained YOLO checkpoint and measure warm batch-one throughput.

Run with .venv-training/Scripts/python.exe scripts/benchmark_tensorrt.py
Requires torch, ultralytics, tensorrt-cu12, onnx, onnxslim, onnxruntime.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/ultralytics-config"))
os.environ["YOLO_AUTOINSTALL"] = "False"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/yolo26s-cooking-crops-v1.pt")
    parser.add_argument("--images", type=Path, default=ROOT / "outputs/local-model-check")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tensorrt-benchmark/results.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or args.warmup < 1:
        parser.error("seconds must be positive and warmup at least one")

    import cv2
    import numpy as np
    import torch
    import tensorrt as trt
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    paths = sorted(args.images.glob("*input.jpg"))
    if not paths:
        raise RuntimeError(f"No *input.jpg crops in {args.images}")
    images = [cv2.imread(str(p)) for p in paths]
    if any(im is None for im in images):
        raise RuntimeError("Could not decode an input crop")
    engine = args.weights.with_suffix(".engine")
    if args.rebuild or not engine.exists():
        exporter = YOLO(str(args.weights))
        engine = Path(exporter.export(format="engine", imgsz=args.imgsz,
                                      quantize=16, batch=1, dynamic=False,
                                      workspace=2, device=0))
        del exporter
        gc.collect()
        torch.cuda.empty_cache()

    model = YOLO(str(engine), task="detect")
    def predict(i):
        return model.predict(images[i % len(images)], imgsz=args.imgsz,
                             device=0, conf=0.25, verbose=False, rect=False)

    print("Warming full prediction pipeline", flush=True)
    for i in range(args.warmup):
        predict(i)
    backend = model.predictor.model
    tensor = model.predictor.preprocess([images[0]])

    def measure(fn):
        for i in range(args.warmup):
            fn(i)
        torch.cuda.synchronize()
        samples = []
        started = time.perf_counter()
        while time.perf_counter() - started < args.seconds:
            before = time.perf_counter()
            fn(len(samples))
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - before) * 1000)
        elapsed = time.perf_counter() - started
        return {"frames": len(samples), "seconds": elapsed,
                "fps": len(samples) / elapsed,
                "latency_ms_mean": float(np.mean(samples)),
                "latency_ms_p50": float(np.percentile(samples, 50)),
                "latency_ms_p95": float(np.percentile(samples, 95))}

    with torch.inference_mode():
        print("Measuring model-only throughput", flush=True)
        raw = measure(lambda i: backend(tensor))
        print("Measuring end-to-end prediction throughput", flush=True)
        full = measure(predict)
    result = {
        "weights": str(args.weights), "engine": str(engine),
        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "tensorrt": trt.__version__, "ultralytics": ultralytics.__version__,
        "input_shape": list(tensor.shape), "input_dtype": str(tensor.dtype),
        "batch": 1, "crop_count": len(images), "warmup": args.warmup,
        "model_only": raw, "end_to_end": full,
        "scope": "Unthrottled warm batch-one inference. Model-only uses a resident GPU tensor. "
                 "End-to-end cycles decoded crops including preprocessing, transfers, inference and postprocessing. "
                 "Excludes camera/decode, rendering, VLM and disk I/O. No accuracy validation. "
                 "Measured throughput is specific to current GPU load and power state, not a universal maximum."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
