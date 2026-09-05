"""Resumable, frame-based SAM3 candidate annotation; all output needs review."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--classes", type=Path, default=Path("config/kitchen_classes.v1.json"))
    p.add_argument("--limit", type=int)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    a = p.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    torch.set_num_threads(2)
    from pycocotools import mask as mask_utils
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    contract = json.loads(a.classes.read_text())
    frames = json.loads((a.input / "manifest.json").read_text())
    if not 0 <= a.shard_index < a.shards:
        p.error("shard-index must be in [0, shards)")
    with a.checkpoint.open("rb") as f:
        checkpoint_hash = hashlib.file_digest(f, "sha256").hexdigest()
    identity = {"checkpoint_sha256": checkpoint_hash, "classes": contract,
                "manifest_sha256": hashlib.sha256((a.input / "manifest.json").read_bytes()).hexdigest(),
                "threshold": 0.5, "sam3_revision": "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7",
                "torch": torch.__version__, "review_status": "pending"}
    a.output.mkdir(parents=True, exist_ok=True)
    run_path = a.output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != identity:
        raise RuntimeError("Run configuration differs; use a new output directory")
    run_temp = a.output / f"run.{a.shard_index}.tmp"
    run_temp.write_text(json.dumps(identity, indent=2))
    run_temp.replace(run_path)
    model = build_sam3_image_model(checkpoint_path=str(a.checkpoint), load_from_HF=False, device="cuda")
    processor = Sam3Processor(model, confidence_threshold=0.5)
    start = time.monotonic()
    for index, frame in enumerate(frames[:a.limit] if a.limit else frames):
        if index % a.shards != a.shard_index:
            continue
        rel = Path(frame["image"])
        target = a.output / "annotations" / rel.parent.name / (rel.stem + ".json")
        if target.exists():
            continue
        image_path = a.input / rel
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != frame["image_sha256"]:
            raise RuntimeError(f"Image hash mismatch: {rel}")
        image = Image.open(image_path).convert("RGB")
        candidates, labels = [], []
        preview = image.copy()
        draw = ImageDraw.Draw(preview)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            for entry in contract["classes"]:
                processor.reset_all_prompts(state)
                result = processor.set_text_prompt(state=state, prompt=entry["prompt"])
                masks = result["masks"].detach().cpu().numpy()
                scores = result["scores"].detach().float().cpu().tolist()
                for mask, score in zip(masks, scores):
                    mask = mask.reshape(image.height, image.width)
                    yy, xx = np.nonzero(mask)
                    if len(xx) == 0:
                        continue
                    x1, y1, x2, y2 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
                    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("ascii")
                    candidates.append({"class_id": entry["id"], "class_name": entry["name"],
                        "prompt": entry["prompt"], "score": score, "bbox_xyxy": [x1,y1,x2,y2],
                        "segmentation": rle, "review_status": "pending"})
                    labels.append(f'{entry["id"]} {(x1+x2)/2/image.width:.8f} {(y1+y2)/2/image.height:.8f} {(x2-x1)/image.width:.8f} {(y2-y1)/image.height:.8f}')
                    draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
                    draw.text((x1,y1), f'{entry["name"]} {score:.2f}', fill="red", font_size=24, stroke_width=1, stroke_fill="white")
        for kind in ("annotations", "candidate_labels", "previews"):
            (a.output / kind / rel.parent.name).mkdir(parents=True, exist_ok=True)
        (a.output / "candidate_labels" / rel.parent.name / (rel.stem+".txt")).write_text("\n".join(labels) + ("\n" if labels else ""))
        preview.thumbnail((960, 540))
        preview.save(a.output / "previews" / rel.parent.name / (rel.stem+".jpg"), quality=80)
        data = {**frame, "width": image.width, "height": image.height,
                "review_status": "pending", "candidates": candidates}
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(target)
        if index % 10 == 0 or index == len(frames)-1:
            print(f"{index+1}/{len(frames)} frames; {time.monotonic()-start:.1f}s", flush=True)
    completed = len(list((a.output / "annotations").rglob("*.json")))
    print(f"Completed {completed}/{len(frames)}", flush=True)


if __name__ == "__main__":
    main()
