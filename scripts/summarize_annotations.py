"""Validate a completed candidate run and write delivery summary and review index."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    from pycocotools import mask as mask_utils
    manifest = json.loads((a.input / "manifest.json").read_text())
    by_video, classes = Counter(), Counter()
    for entry in json.loads((a.output / "run.json").read_text())["classes"]["classes"]:
        classes[entry["name"]] = 0
    empty = 0
    links = []
    for frame in manifest:
        rel = Path(frame["image"])
        stem = Path(rel.parent.name) / rel.stem
        data = json.loads((a.output / "annotations" / stem.with_suffix(".json")).read_text())
        if data["image_sha256"] != frame["image_sha256"] or data["review_status"] != "pending":
            raise RuntimeError(f"Provenance/status mismatch: {stem}")
        label_path = a.output / "candidate_labels" / stem.with_suffix(".txt")
        labels = label_path.read_text().splitlines()
        if len(labels) != len(data["candidates"]):
            raise RuntimeError(f"Label count mismatch: {stem}")
        for candidate, label in zip(data["candidates"], labels):
            cid, *coords = label.split()
            assert int(cid) == candidate["class_id"] and len(coords) == 4
            assert all(0 <= float(c) <= 1 for c in coords)
            x1,y1,x2,y2 = candidate["bbox_xyxy"]
            assert 0 <= x1 < x2 <= data["width"] and 0 <= y1 < y2 <= data["height"]
            assert candidate["segmentation"]["size"] == [data["height"],data["width"]]
            mx, my, mw, mh = mask_utils.toBbox(candidate["segmentation"]).tolist()
            assert [mx, my, mx+mw, my+mh] == [x1,y1,x2,y2]
            expected = [(x1+x2)/2/data["width"], (y1+y2)/2/data["height"],
                        (x2-x1)/data["width"], (y2-y1)/data["height"]]
            assert all(abs(float(c)-e) < 1e-7 for c,e in zip(coords,expected))
            classes[candidate["class_name"]] += 1
        preview = Path("previews") / stem.with_suffix(".jpg")
        assert (a.output / preview).is_file()
        by_video[frame["source_video"]] += 1
        empty += not data["candidates"]
        links.append(f'<a href="{preview.as_posix()}">{html.escape(str(stem))}</a>')
    actual = len(list((a.output / "annotations").rglob("*.json")))
    assert actual == len(manifest), (actual, len(manifest))
    summary = {"frames": len(manifest), "videos": len(by_video),
               "frames_by_video": dict(by_video), "candidates_by_class": dict(classes),
               "total_candidates": sum(classes.values()), "empty_frames": empty,
               "review_status": "pending", "sampling": "first frame, then at least 1 second apart",
               "split": "unassigned; one cooking session; do not split by episode"}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (a.output / "review.html").write_text('<!doctype html><meta charset="utf-8"><title>SAM3 candidates</title>'
        '<h1>SAM3 candidate annotations — review required</h1><p>All 15 classes queried per frame. '
        'Check missing objects and conflicting classes before training.</p><ul>' +
        ''.join(f'<li>{link}</li>' for link in links) + '</ul>')
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
