# SAM3 all-video candidate annotation

The run covers the 15 numbered episode videos under
`S:/Projects/cooking-detection-project/datasets/video`, sampled at the first frame
and then at least one second apart. It excludes `original_uncut.mp4`, the source
recording used to make the episodes. These are sampled-frame annotations, not
dense video tracking. The manifest has exact decoded source frame indices and
timestamps and SHA-256 hashes for source videos and JPEG inputs.

All 15 dictionary prompts run on every frame at a score threshold of 0.5. Boxes
are computed from visible mask pixels. Masks are stored as compressed COCO RLE
(`size` is height/width, ASCII `counts`). No cross-class suppression is applied:
conflicting labels remain visible for human review. Scores are model scores,
not calibrated correctness probabilities. Instance tracking IDs are not assigned.

Output folders:

- `annotations/<episode>/<frame>.json`: provenance, boxes, scores, masks and status.
- `candidate_labels/<episode>/<frame>.txt`: normalized YOLO detection candidates.
- `previews/<episode>/<frame>.jpg`: annotated visual previews, resized for review.
- `summary.json`: verified per-video frame counts and per-class candidate counts.
- `run.json`, `classes.json`, `manifest.json`: model and input provenance.
- `review.html`: links to every preview.

Original sampled JPEGs are included under `images` in the final local package;
the preparation copy is under `data/annotation-frames/images` in this project.
They retain full resolution; previews are not training images. Candidate label
paths use the same episode/frame names as those JPEGs.

All candidates have `review_status: pending`. Review entire frames for missed
objects, class confusion, missing handles and duplicate boxes before training.
Empty predictions are not automatically verified negative frames. No train/val
split or training YAML is generated: the clips belong to one cooking session
and must not leak across independent evaluation splits.

Reproduce preparation with `scripts/prepare_annotation_frames.py`. On the pod,
`scripts/run_annotation_batch.sh` runs four disjoint shards and validates coverage
before producing the output archive. Existing completed frames are resumed only
when the model, dictionary, threshold and manifest match.
