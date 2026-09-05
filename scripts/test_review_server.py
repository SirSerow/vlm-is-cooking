"""Focused persistence, export, coordinate and HTTP boundary tests; isolated DB."""
import copy
import io
import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image
import review_server as app


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_db = app.DB
        app.DB = Path(self.temp.name) / 'test.sqlite3'
        with app.connect() as db:
            db.execute('CREATE TABLE reviews(frame INTEGER PRIMARY KEY,revision INTEGER,body TEXT)')
            db.execute('CREATE TABLE history(id INTEGER PRIMARY KEY,frame INTEGER,revision INTEGER,body TEXT)')
        self.body = dict(frame=0, revision=0, status='approved', scope='crop', note='test',
                         boxes=[dict(class_id=0,bbox_xyxy=[500,330,1120,1080])])

    def tearDown(self):
        app.DB = self.old_db
        self.temp.cleanup()

    def test_revision_conflict_and_history(self):
        saved = app.save_review(self.body)
        self.assertEqual(saved['revision'],1)
        self.assertEqual(app.get_review(0)['image_sha256'],app.FRAMES[0]['image_sha256'])
        with self.assertRaises(FileExistsError):
            app.save_review(self.body)
        saved.update(status='needs_work',revision=1)
        app.save_review(saved)
        with app.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM history').fetchone()[0],2)

    def test_invalid_geometry_and_scope(self):
        for xy in ([0,0,1120,1080],[500,330,500,500],[500,330,float('nan'),1080],[500,330,1121,1080]):
            value=copy.deepcopy(self.body);value['boxes'][0]['bbox_xyxy']=xy
            with self.assertRaises(ValueError):app.save_review(value)
        value=copy.deepcopy(self.body);value['boxes'][0]['class_id']=15
        with self.assertRaises(ValueError):app.save_review(value)

    def test_export_pairs_crop_labels_and_images_and_keeps_negatives(self):
        app.save_review(self.body)
        app.save_review(dict(self.body,frame=1,scope='full',boxes=[]))
        app.save_review(dict(self.body,frame=2,status='draft'))
        with zipfile.ZipFile(io.BytesIO(app.export_reviews())) as z:
            labels=[x for x in z.namelist() if '/labels/' in x]
            self.assertEqual(len(labels),2)
            crop=next(x for x in labels if x.startswith('crop/'))
            self.assertEqual(z.read(crop).decode(),'0 0.50000000 0.50000000 1.00000000 1.00000000')
            im=Image.open(io.BytesIO(z.read(crop.replace('/labels/','/images/').replace('.txt','.jpg'))))
            self.assertEqual(im.size,(620,750))
            self.assertEqual(z.read(next(x for x in labels if x.startswith('full/'))),b'')

    def test_http_boundaries(self):
        server=app.ThreadingHTTPServer(('127.0.0.1',0),app.Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            for endpoint,status in [('/api/frame?id=-1',400),('/image?id=999999',400),('/../../.git/config',404)]:
                with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(url+endpoint)
                self.assertEqual(caught.exception.code,status)
            for headers in ({},{'X-Review-Token':app.TOKEN,'Origin':'https://untrusted.example'}):
                req=urllib.request.Request(url+'/api/review',data=json.dumps(self.body).encode(),headers=headers)
                with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(req)
                self.assertEqual(caught.exception.code,403)
            req=urllib.request.Request(url+'/api/review',data=json.dumps(self.body).encode(),headers={'X-Review-Token':app.TOKEN})
            self.assertEqual(json.load(urllib.request.urlopen(req))['revision'],1)
        finally:
            server.shutdown();server.server_close();worker.join()

    def test_all_prediction_cache_provenance_and_coordinates(self):
        for i,frame in enumerate(app.FRAMES):
            pred=app.prediction(i)
            self.assertIsNotNone(pred)
            self.assertEqual(pred['image_sha256'],frame['image_sha256'])
            self.assertEqual(pred['crop_xyxy'],app.ROI)
            for b in pred['boxes']:
                x,y,X,Y=b['bbox_xyxy']
                self.assertTrue(499.99<=x<X<=1120.01 and 329.99<=y<Y<=1080.01)


if __name__=='__main__':unittest.main()
