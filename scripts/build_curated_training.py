"""Build the explicitly reviewed crop set; never alter the raw SAM3 annotations."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "outputs/all-videos-1fps"
    out = ROOT / "data/curated-crops-v1"
    review = json.loads((ROOT / "config/curated_crop_review.v1.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    selected = []
    for video in sorted({f["source_video"] for f in manifest}):
        frames = [f for f in manifest if f["source_video"] == video]
        selected.extend(frames[min(len(frames)-1, int(len(frames)*q))] for q in (0.15,0.5,0.85))
    assert len(selected) == len(review["labels_by_selected_index"]) == 45
    names = [c["name"] for c in json.loads((ROOT / "config/kitchen_classes.v1.json").read_text())["classes"]]
    counts = Counter({name: 0 for name in names})
    for folder in ("images", "labels", "review"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    records = []
    for i, item in enumerate(selected):
        path = source / item["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["image_sha256"]
        im = Image.open(path).convert("RGB").crop(review["crop_xyxy"])
        name = f"{i:02d}_{Path(item['source_video']).stem}_{Path(item['image']).stem}"
        im.save(out / "images" / f"{name}.jpg", quality=95)
        labels = review["labels_by_selected_index"][str(i)]
        lines = []
        preview = im.copy()
        draw = ImageDraw.Draw(preview)
        for cid,x1,y1,x2,y2 in labels:
            assert 0 <= cid < len(names) and 0 <= x1 < x2 <= im.width and 0 <= y1 < y2 <= im.height
            lines.append(f"{cid} {(x1+x2)/2/im.width:.8f} {(y1+y2)/2/im.height:.8f} {(x2-x1)/im.width:.8f} {(y2-y1)/im.height:.8f}")
            counts[names[cid]] += 1
            draw.rectangle([x1,y1,x2-1,y2-1], outline="lime", width=3)
            draw.text((x1+4,y1+4), names[cid], font_size=20, fill="black", stroke_width=2, stroke_fill="white")
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines)+("\n" if lines else ""))
        preview.save(out / "review" / f"{name}.jpg")
        records.append({**item,"output":name,"labels":labels,"crop_xyxy":review["crop_xyxy"],"review_status":review["status"]})
    # This is an in-sample diagnostic path, not a held-out validation split.
    yaml = f"path: {out.as_posix()}\ntrain: images\nval: images\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i,name in enumerate(names))
    (out / "dataset.yaml").write_text(yaml)
    (out / "provenance.json").write_text(json.dumps(records,indent=2))
    summary = {"images":45,"boxes":sum(counts.values()),"class_counts":dict(counts),
               "negative_images":sum(not r["labels"] for r in records),
               "evaluation":"training-fit diagnostics only; no independent validation/test session",
               "scope":"visually corrected cooking-area crop baseline; remaining full-frame labels unreviewed"}
    (out / "summary.json").write_text(json.dumps(summary,indent=2))
    for page in range(15):
        canvas = Image.new("RGB",(1860,780),"white")
        for j in range(3):
            i=page*3+j
            canvas.paste(Image.open(out / "review" / (records[i]["output"]+".jpg")),(j*620,30))
            ImageDraw.Draw(canvas).text((j*620+4,3), str(i),fill="black",font_size=20)
        canvas.save(out / "review" / f"sheet-{page+1:02d}.jpg")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
