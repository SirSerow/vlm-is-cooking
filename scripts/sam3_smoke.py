"""Run real image inference on a pod; save candidates for visual inspection."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--class-name", default="pan")
    parser.add_argument("--classes", type=Path, default=Path(__file__).resolve().parents[1] / "config/kitchen_classes.v1.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    contract = json.loads(args.classes.read_text())
    entries = contract["classes"]
    if [entry["id"] for entry in entries] != list(range(len(entries))):
        parser.error("class IDs must be contiguous and ordered")
    entry = next((item for item in entries if item["name"] == args.class_name), None)
    if entry is None:
        parser.error("class-name must be in the annotation dictionary")
    for path in (args.image, args.checkpoint):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if args.output.exists():
        parser.error("Use a new output directory to avoid mixing runs")

    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check the pod GPU and NVIDIA driver.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This baseline requires a GPU supporting BF16 (Ampere or newer).")
    with Image.open(args.image) as source:
        frame = ImageOps.exif_transpose(source).convert("RGB")
    model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), load_from_HF=False, device="cuda")
    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(frame)
        result = processor.set_text_prompt(state=state, prompt=entry["prompt"])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    masks = result["masks"].detach().cpu().numpy()
    scores = result["scores"].detach().float().cpu().tolist()
    if len(masks) != len(scores):
        raise RuntimeError("SAM returned inconsistent mask/score counts")
    args.output.mkdir(parents=True)
    preview = frame.copy()
    draw = ImageDraw.Draw(preview)
    candidates = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.reshape(frame.height, frame.width)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        mask_name = f"mask_{index:03d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / mask_name)
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f'{entry["name"]} {score:.2f}', fill="red")
        candidates.append({"class_id": entry["id"], "bbox_xyxy": box, "score": score, "mask": mask_name})
    preview.save(args.output / "preview.jpg")
    with args.checkpoint.open("rb") as checkpoint:
        checkpoint_sha256 = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    metadata = {
        "review_status": "pending", "image": str(args.image.resolve()),
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "image_size": list(frame.size), "dictionary": contract["version"],
        "dictionary_sha256": hashlib.sha256(args.classes.read_bytes()).hexdigest(),
        "prompt": entry["prompt"], "threshold": args.threshold,
        "checkpoint_sha256": checkpoint_sha256,
        "sam3_revision": "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7",
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "inference_seconds": seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "candidates": candidates,
    }
    (args.output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Inference OK: {len(candidates)} candidates; inspect {args.output / 'preview.jpg'}")
    print("Zero candidates can be valid. This is a runtime check, not an accuracy test.")


if __name__ == "__main__":
    main()
