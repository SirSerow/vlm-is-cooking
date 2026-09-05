"""Run the locally trained crop model on unused frames from the same session."""
import json
import os
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "outputs/ultralytics-config")


def main():
    from ultralytics import YOLO
    source = ROOT / "outputs/all-videos-1fps"
    output = ROOT / "outputs/local-model-check"
    output.mkdir(exist_ok=True)
    all_frames = json.loads((source / "manifest.json").read_text())
    training = json.loads((ROOT / "data/curated-crops-v1/provenance.json").read_text())
    used = {r["image"] for r in training}
    frames = []
    for video in sorted({f["source_video"] for f in all_frames}):
        candidates = [f for f in all_frames if f["source_video"] == video and f["image"] not in used]
        frames.append(candidates[int(len(candidates)*0.33)])
    model = YOLO(str(ROOT / "models/yolo26s-cooking-crops-v1.pt"))
    expected = [c["name"] for c in json.loads((ROOT / "config/kitchen_classes.v1.json").read_text())["classes"]]
    assert list(model.names.values()) == expected
    records = []
    for f in frames:
        crop = Image.open(source / f["image"]).convert("RGB").crop((500,330,1120,1080))
        name = Path(f["source_video"]).stem
        crop.save(output / f"{name}-input.jpg")
        result = model.predict(crop, imgsz=640, device=0, conf=0.25, verbose=False)[0]
        result.save(filename=str(output / f"{name}-prediction.jpg"))
        records.append({"source":f,"speed_ms":result.speed,"predictions":json.loads(result.to_json())})
    (output / "predictions.json").write_text(json.dumps(records,indent=2))
    print("Checked 15 unused frames from the SAME session; this is not an independent test.")


if __name__ == "__main__":
    main()
