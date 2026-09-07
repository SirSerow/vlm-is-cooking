"""Run the existing recipe benchmark alongside paced TensorRT video inference.

This tests coexistence, not YOLO/VLM fusion or temporal detection accuracy.
Run as a module using the training environment from the repository root.
"""
import argparse
import json
from pathlib import Path
import shutil
import threading
import time
import traceback

from scripts.evaluate_recipe import evaluate


def video_worker(args, stop, ready, stats):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    cap = None
    try:
        torch.set_num_threads(4)
        model = YOLO(str(args.engine), task="detect")
        cap = cv2.VideoCapture(str(args.video))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            raise RuntimeError("Cannot open source video")
        def infer(frame):
            model.predict(frame[330:1080, 500:1120], imgsz=640, rect=False,
                          device=0, conf=0.25, verbose=False)
            torch.cuda.synchronize()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot decode video")
        for _ in range(50):
            infer(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        latencies = []
        missed = 0
        started = time.perf_counter()
        deadline = started
        ready.set()
        while not stop.is_set():
            # Python's sleep uses a high-resolution timer on Windows; Event.wait
            # may oversleep enough to lose slots at a 33 ms cadence.
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            if stop.is_set():
                break
            before = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Video decode failed")
            infer(frame)
            after = time.perf_counter()
            latencies.append((after - before) * 1000)
            deadline += 1 / args.fps
            if after > deadline:
                skip = int((after - deadline) * args.fps) + 1
                missed += skip
                deadline += skip / args.fps
                for _ in range(skip):
                    if not cap.grab():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            stats.update(frames=len(latencies), elapsed_seconds=after-started,
                         achieved_fps=len(latencies)/(after-started), missed_slots=missed)
        stats.update(target_fps=args.fps, source_fps=source_fps,
                     mean_processing_ms=float(np.mean(latencies)),
                     p95_processing_ms=float(np.percentile(latencies, 95)),
                     max_processing_ms=max(latencies),
                     frames_over_budget=sum(t > 1000/args.fps for t in latencies))
    except Exception:
        stats["error"] = traceback.format_exc()
        ready.set()
    finally:
        if cap is not None:
            cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--engine", type=Path, default=Path("models/yolo26s-cooking-crops-v1.engine"))
    parser.add_argument("--video", type=Path, default=Path("S:/Projects/cooking-detection-project/datasets/video/007_roasting-mushrooms.mp4"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/recipe-evaluation/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("outputs/app-30fps-evaluation"))
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("fps must be positive")
    if args.output.resolve() == args.baseline.resolve():
        parser.error("output must differ from baseline")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((args.baseline / "results.json").read_text())
    samples = json.loads((args.baseline / "samples.json").read_text())
    for sample in samples:
        shutil.copy2(args.baseline / sample["image"], args.output / sample["image"])
    shutil.copy2(args.baseline / "samples.json", args.output / "samples.json")
    stop, ready = threading.Event(), threading.Event()
    stats = {}
    worker = threading.Thread(target=video_worker, args=(args, stop, ready, stats))
    worker.start()
    result = None
    try:
        if not ready.wait(120):
            raise RuntimeError("Video worker startup timed out")
        if "error" in stats:
            raise RuntimeError(stats["error"])
        print("30 FPS video worker ready; running recipe evaluation", flush=True)
        result = evaluate(baseline["config"], samples, args.output, baseline["model"])
    finally:
        stop.set()
        worker.join()
        comparison = {"video_load": stats,
                      "baseline": baseline["summary"],
                      "under_load": result["summary"] if result else None,
                      "scope": "Existing VLM recipe app with concurrent decoded-video TensorRT load. "
                               "YOLO predictions do not feed recipe decisions. Same frozen 35 images as baseline. "
                               "No camera/display test or temporal accuracy labels; differences do not establish an FPS accuracy effect."}
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2))
        print(json.dumps(comparison, indent=2), flush=True)
    if "error" in stats:
        raise RuntimeError(stats["error"])


if __name__ == "__main__":
    main()
