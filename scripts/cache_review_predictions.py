"""Cache YOLO predictions for every SAM3 sample, with original-frame coordinates."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'outputs/ultralytics-config')
os.environ['YOLO_AUTOINSTALL'] = 'False'


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def main():
    from PIL import Image
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    source = ROOT / 'outputs/all-videos-1fps'
    state = ROOT / 'outputs/review-tool'
    dest = state / 'predictions'
    dest.mkdir(parents=True, exist_ok=True)
    frames = json.loads((source / 'manifest.json').read_text())
    weight = ROOT / 'models/yolo26s-cooking-crops-v1.pt'
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    model = YOLO(str(weight))
    done = 0
    started = time.time()
    def status(kind, error=None):
        atomic(state / 'inference-status.json', dict(state=kind, done=done, total=len(frames),
               elapsed_seconds=round(time.time()-started, 1), model_sha256=digest, error=error))
    status('running')
    try:
        for i, frame in enumerate(frames):
            output = dest / f'{i:05d}.json'
            if output.exists() and json.loads(output.read_text()).get('model_sha256') == digest:
                done += 1
                continue
            with Image.open(source / frame['image']) as im:
                crop = im.convert('RGB').crop((500, 330, 1120, 1080))
            result = model.predict(crop, imgsz=640, device=0 if torch.cuda.is_available() else 'cpu', conf=0.05, verbose=False)[0]
            boxes = []
            for b in json.loads(result.to_json()):
                xy = b['box']
                boxes.append(dict(class_id=b['class'], class_name=b['name'], score=b['confidence'],
                                  bbox_xyxy=[xy['x1']+500, xy['y1']+330, xy['x2']+500, xy['y2']+330]))
            atomic(output, dict(model_sha256=digest, image_sha256=frame['image_sha256'], crop_xyxy=[500,330,1120,1080],
                   confidence_floor=0.05, boxes=boxes, speed_ms=result.speed))
            done += 1
            if done % 25 == 0:
                status('running')
                print(f'{done}/{len(frames)}', flush=True)
        status('complete')
    except Exception as e:
        status('failed', str(e))
        raise


if __name__ == '__main__':
    main()
