"""Build a YOLO dataset from approved annotations saved by the review dashboard."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/all-videos-1fps"
REVIEW_DB = ROOT / "outputs/review-tool/reviews.sqlite3"
ROI = [500, 330, 1120, 1080]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("crop", "full"), default="crop")
    parser.add_argument("--exclude-episodes", nargs="*", type=int, default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def episode_number(source_video):
    return int(Path(source_video).stem.split("_", 1)[0])


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output dataset is not empty: {output}")

    frames = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    classes = json.loads((ROOT / "config/kitchen_classes.v1.json").read_text(encoding="utf-8"))["classes"]
    class_names = [item["name"] for item in classes]
    excluded = set(args.exclude_episodes)

    with sqlite3.connect(REVIEW_DB) as database:
        rows = database.execute("SELECT frame, body FROM reviews ORDER BY frame").fetchall()
    approved = []
    for frame_index, body in rows:
        review = json.loads(body)
        frame = frames[frame_index]
        if review["status"] != "approved" or review["scope"] != args.scope:
            continue
        if episode_number(frame["source_video"]) in excluded:
            continue
        if review["image"] != frame["image"] or review["image_sha256"] != frame["image_sha256"]:
            raise ValueError(f"Review {frame_index} does not match the source manifest")
        approved.append((frame_index, frame, review))

    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter({name: 0 for name in class_names})
    provenance = []

    bounds = ROI if args.scope == "crop" else [0, 0, 1920, 1080]
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    for frame_index, frame, review in approved:
        source_image = SOURCE / frame["image"]
        if hashlib.sha256(source_image.read_bytes()).hexdigest() != frame["image_sha256"]:
            raise ValueError(f"Source image hash changed: {source_image}")
        name = f"{Path(frame['image']).parent.name}_{Path(frame['image']).stem}"
        with Image.open(source_image) as image:
            rendered = image.crop(bounds) if args.scope == "crop" else image.copy()
            rendered.convert("RGB").save(images_dir / f"{name}.jpg", quality=95)

        lines = []
        for box in review["boxes"]:
            class_id = box["class_id"]
            x1, y1, x2, y2 = box["bbox_xyxy"]
            if not left <= x1 < x2 <= right or not top <= y1 < y2 <= bottom:
                raise ValueError(f"Review {frame_index} has a box outside its {args.scope} scope")
            lines.append(
                f"{class_id} {((x1 + x2) / 2 - left) / width:.8f} "
                f"{((y1 + y2) / 2 - top) / height:.8f} "
                f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
            )
            counts[class_names[class_id]] += 1
        (labels_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        provenance.append({"frame_index": frame_index, **frame, "review": review})

    yaml = f"path: {output.as_posix()}\ntrain: images\nval: images\nnames:\n"
    yaml += "".join(f"  {item['id']}: {item['name']}\n" for item in classes)
    (output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    summary = {
        "images": len(provenance),
        "boxes": sum(counts.values()),
        "class_counts": dict(counts),
        "negative_images": sum(not item["review"]["boxes"] for item in provenance),
        "scope": args.scope,
        "excluded_episodes": sorted(excluded),
        "source": str(REVIEW_DB),
        "evaluation": "training-fit diagnostics only; no independent validation/test session",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
