"""Private annotation review app. Run with the local training environment."""
import argparse
import csv
import io
import json
import math
import mimetypes
import secrets
import sqlite3
from contextlib import contextmanager
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/all-videos-1fps'
STATE = ROOT / 'outputs/review-tool'
WEB = ROOT / 'review-app'
RUN = ROOT / 'outputs/local-training/yolo26s-curated-crops-v1'
MODEL = ROOT / 'models/yolo26s-cooking-crops-v1.pt'
ROI = [500, 330, 1120, 1080]
TOKEN = secrets.token_urlsafe(32)
FRAMES = json.loads((SOURCE / 'manifest.json').read_text())
CLASSES = json.loads((ROOT / 'config/kitchen_classes.v1.json').read_text())['classes']
TRAIN = {r['image']: r for r in json.loads((ROOT / 'data/curated-crops-v1/provenance.json').read_text())}
STATE.mkdir(parents=True, exist_ok=True)
DB = STATE / 'reviews.sqlite3'


@contextmanager
def connect():
    db = sqlite3.connect(DB, timeout=15)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


with connect() as db:
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS reviews (frame INTEGER PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, frame INTEGER, revision INTEGER, body TEXT)')


def get_review(index):
    with connect() as db:
        row = db.execute('SELECT body FROM reviews WHERE frame=?', (index,)).fetchone()
    return json.loads(row['body']) if row else None


def validate_review(body):
    index = body.get('frame')
    if type(index) is not int or not 0 <= index < len(FRAMES):
        raise ValueError('Invalid frame')
    if body.get('status') not in ('draft', 'approved', 'needs_work'):
        raise ValueError('Invalid review status')
    if body.get('scope') not in ('full', 'crop'):
        raise ValueError('Choose full-frame or cooking-area review scope')
    if type(body.get('revision')) is not int or body['revision'] < 0:
        raise ValueError('Invalid revision')
    boxes = body.get('boxes')
    if not isinstance(boxes, list) or len(boxes) > 500:
        raise ValueError('Invalid boxes')
    bounds = ROI if body['scope'] == 'crop' else [0, 0, 1920, 1080]
    cleaned = []
    for b in boxes:
        c, xy = b.get('class_id'), b.get('bbox_xyxy')
        if type(c) is not int or not 0 <= c < len(CLASSES):
            raise ValueError('Invalid class')
        if not isinstance(xy, list) or len(xy) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy):
            raise ValueError('Invalid box coordinates')
        if not bounds[0] <= xy[0] < xy[2] <= bounds[2] or not bounds[1] <= xy[1] < xy[3] <= bounds[3]:
            raise ValueError('Box outside review scope or zero area')
        cleaned.append({'class_id': c, 'bbox_xyxy': xy})
    note = body.get('note', '')
    if not isinstance(note, str) or len(note) > 10000:
        raise ValueError('Note too long')
    return {k: body[k] for k in ('frame', 'status', 'scope', 'revision')} | {'boxes': cleaned, 'note': note}


def save_review(body, reviewer='local reviewer'):
    record = validate_review(body)
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT revision FROM reviews WHERE frame=?', (record['frame'],)).fetchone()
        current = row['revision'] if row else 0
        if current != record['revision']:
            raise FileExistsError('This frame changed in another browser. Reload before saving.')
        record.update(revision=current + 1, updated_at=datetime.now(timezone.utc).isoformat(), reviewer=reviewer,
                      image=FRAMES[record['frame']]['image'], image_sha256=FRAMES[record['frame']]['image_sha256'])
        value = json.dumps(record)
        db.execute('INSERT OR REPLACE INTO reviews VALUES (?,?,?)', (record['frame'], record['revision'], value))
        db.execute('INSERT INTO history(frame,revision,body) VALUES (?,?,?)', (record['frame'], record['revision'], value))
    return record


def annotation(index):
    rel = Path(FRAMES[index]['image'])
    return json.loads((SOURCE / 'annotations' / rel.parent.name / (rel.stem + '.json')).read_text())


def prediction(index):
    path = STATE / 'predictions' / f'{index:05d}.json'
    return json.loads(path.read_text()) if path.exists() else None


def training():
    with (RUN / 'results.csv').open() as f:
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    counts = [0] * len(CLASSES)
    for t in TRAIN.values():
        for label in t['labels']:
            counts[label[0]] += 1
    return {'rows': rows, 'scope': json.loads((RUN / 'scope.json').read_text()), 'counts': counts,
            'config': (RUN / 'args.yaml').read_text(), 'frames': [i for i, f in enumerate(FRAMES) if f['image'] in TRAIN],
            'log': (ROOT / 'outputs/local-training.log').read_text(errors='replace')[-50000:]}


def export_reviews():
    with connect() as db:
        records = [json.loads(r['body']) for r in db.execute('SELECT body FROM reviews ORDER BY frame')]
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('reviews.json', json.dumps(records, indent=2))
        z.writestr('classes.json', json.dumps(CLASSES, indent=2))
        z.writestr('README.txt', 'All saved decisions are in reviews.json. Only approved frames have YOLO labels and images.\n'
                   'Crop reviews export cropped images and crop-normalized boxes; full reviews export original images.\n'
                   'Empty approved labels are explicit negatives. These are box corrections, not edited SAM masks.\n'
                   'No training run is started or changed by reviewing/exporting. Retain session grouping when splitting.\n')
        from PIL import Image
        for r in records:
            if r['status'] != 'approved':
                continue
            frame = FRAMES[r['frame']]
            name = Path(frame['image']).parent.name + '_' + Path(frame['image']).stem
            bounds = ROI if r['scope'] == 'crop' else [0, 0, 1920, 1080]
            x, y, right, bottom = bounds
            w, h = right-x, bottom-y
            lines = []
            for b in r['boxes']:
                x1, y1, x2, y2 = b['bbox_xyxy']
                lines.append(f"{b['class_id']} {((x1+x2)/2-x)/w:.8f} {((y1+y2)/2-y)/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
            z.writestr(f"{r['scope']}/labels/{name}.txt", '\n'.join(lines))
            if r['scope'] == 'full':
                z.write(SOURCE / frame['image'], f'full/images/{name}.jpg')
            else:
                with Image.open(SOURCE / frame['image']) as im:
                    buf = io.BytesIO()
                    im.crop(ROI).save(buf, format='JPEG', quality=95)
                    z.writestr(f'crop/images/{name}.jpg', buf.getvalue())
    return out.getvalue()


class Handler(BaseHTTPRequestHandler):
    def reply(self, data, content_type='application/json', status=200, filename=None):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode() if content_type == 'application/json' else data.encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            q = parse_qs(url.query)
            if url.path == '/api/bootstrap':
                with connect() as db:
                    reviews = {r['frame']: json.loads(r['body'])['status'] for r in db.execute('SELECT frame,body FROM reviews')}
                frames = [f | {'id': i, 'training': f['image'] in TRAIN, 'status': reviews.get(i, 'pending')}
                          for i, f in enumerate(FRAMES)]
                return self.reply({'frames': frames, 'classes': CLASSES, 'roi': ROI, 'token': TOKEN})
            if url.path == '/api/status':
                p = STATE / 'inference-status.json'
                return self.reply(json.loads(p.read_text()) if p.exists() else {'state': 'not_started', 'done': 0, 'total': len(FRAMES)})
            if url.path == '/api/training':
                return self.reply(training())
            if url.path == '/api/export':
                return self.reply(export_reviews(), 'application/zip', filename='kitchen-reviewed-labels.zip')
            if url.path in ('/api/frame', '/image', '/thumb'):
                index = int(q.get('id', ['-1'])[0])
                if not 0 <= index < len(FRAMES):
                    raise ValueError('Invalid frame')
                if url.path == '/api/frame':
                    return self.reply({'source': FRAMES[index], 'sam': annotation(index)['candidates'],
                                       'yolo': prediction(index), 'training': TRAIN.get(FRAMES[index]['image']), 'review': get_review(index)})
                path = SOURCE / FRAMES[index]['image']
                if url.path == '/thumb':
                    from PIL import Image
                    with Image.open(path) as im:
                        im.thumbnail((256, 144))
                        b = io.BytesIO()
                        im.save(b, format='JPEG', quality=75)
                        return self.reply(b.getvalue(), 'image/jpeg')
                return self.reply(path.read_bytes(), 'image/jpeg')
            if url.path == '/model':
                return self.reply(MODEL.read_bytes(), 'application/octet-stream', filename=MODEL.name)
            if url.path == '/training-config':
                return self.reply((RUN / 'args.yaml').read_bytes(), 'text/plain', filename='training-args.yaml')
            paths = {'/evaluation': 'evaluation.html', '/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            if url.path in paths:
                p = WEB / paths[url.path]
                return self.reply(p.read_bytes(), mimetypes.guess_type(p.name)[0] or 'text/plain')
            self.reply({'error': 'Not found'}, status=404)
        except (ValueError, KeyError) as e:
            self.reply({'error': str(e)}, status=400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            self.log_error('%s', e)
            self.reply({'error': 'Server error; see server log'}, status=500)

    def do_POST(self):
        if self.path != '/api/review':
            return self.reply({'error': 'Not found'}, status=404)
        if self.headers.get('X-Review-Token') != TOKEN:
            return self.reply({'error': 'Reload this page before saving'}, status=403)
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self.reply({'error': 'Cross-origin write refused'}, status=403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 256000:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            result = save_review(body, self.headers.get('Tailscale-User-Login', 'local reviewer'))
            self.reply(result)
        except FileExistsError as e:
            self.reply({'error': str(e)}, status=409)
        except (ValueError, TypeError, AttributeError) as e:
            self.reply({'error': str(e)}, status=400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Review app: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
