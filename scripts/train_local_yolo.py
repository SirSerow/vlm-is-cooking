"""Train the corrected crop baseline locally; metrics are in-sample diagnostics."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "outputs/ultralytics-config")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ["YOLO_AUTOINSTALL"] = "False"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/curated-crops-v1")
    parser.add_argument("--run-name", default="yolo26s-curated-crops-v1")
    parser.add_argument("--output-model", type=Path, default=ROOT / "models/yolo26s-cooking-crops-v1.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from ultralytics import YOLO, settings
    settings.update({"sync":False,"wandb":False,"comet":False,"mlflow":False})
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU unavailable")
    project = ROOT / "outputs/local-training"
    project.mkdir(parents=True, exist_ok=True)
    pretrained = ROOT / "yolo26s.pt"
    model = YOLO(str(pretrained))
    model.train(data=str(args.dataset.resolve() / "dataset.yaml"),
                epochs=60, imgsz=640, batch=4, device=0, workers=0,
                optimizer="AdamW", lr0=0.001, lrf=0.01, freeze=10,
                amp=True, cache="ram", seed=42, deterministic=True,
                patience=0, val=False, save=True, save_period=20,
                mosaic=0.5, close_mosaic=10, mixup=0.0, degrees=10,
                translate=0.1, scale=0.3, fliplr=0.5, flipud=0.5,
                project=str(project), name=args.run_name, exist_ok=False)
    run = Path(model.trainer.save_dir)
    destination = args.output_model.resolve()
    destination.parent.mkdir(exist_ok=True)
    shutil.copy2(run / "weights/last.pt", destination)
    freeze = subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (run / "installed.freeze.txt").write_text(freeze)
    (run / "scope.json").write_text(json.dumps({
        "model":str(destination), "pretrained_model":str(pretrained),
        "dataset":str(args.dataset.resolve()), "epochs":60,
        "evaluation":"training data reused for mandatory framework diagnostics; no independent validation/test",
        "scope":json.loads((args.dataset.resolve() / "summary.json").read_text()),
        "crop_xyxy":[500,330,1120,1080], "source_resolution":[1920,1080],
        "deployment_status":"experimental; not validated for full-frame use"},indent=2))
    print("TRAINING_COMPLETE", destination, flush=True)


if __name__ == "__main__":
    main()
