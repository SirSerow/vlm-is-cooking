# Local YOLO26s crop baseline

For step-by-step commands to prepare approved labels and train your own run, see
[Train it yourself](train-guide.md). This document describes the original baseline.

The initial full-frame SAM3 candidates are not reliable enough for direct
supervised training. Visual review found hob-as-plate false positives, pan/bowl/
plate confusion, duplicate bottle/jar labels, missing silicone lids, missing thin
tools, and scissors mistaken for knives. Automatic structural checks alone did
not detect these semantic errors.

## Reviewed data

Selected three frames per numbered video at approximately 15%, 50%, and 85% of
its duration. Reviewed all 45 cooking-area crops at source coordinates
`[500, 330, 1120, 1080]` (620 × 750 pixels). Explicit corrections are in
`config/curated_crop_review.v1.json`; build with `scripts/build_curated_training.py`.
This is assistant visual review, not independently verified human ground truth.

The dataset has 68 boxes and six negative crops. Class counts: pan 24, lid 6,
knife 8, cutting_board 6, bowl 2, plate 14, spatula 7, spoon 1. All 15 canonical
class IDs remain fixed; the other seven classes have no positive examples and
are unsupported by this initial model. Even the represented rare classes have
too few examples to establish reliability.

Boxes bound visible extents within the crop. When a lid completely covers the
pan body, only its visible handle is labelled as pan. Food, packages, cans,
hands, hob and scissors are outside this dictionary. Corrected labels apply
only to these crops; none are propagated to the 2,187 full-frame annotations.

Within this reviewed ROI sample, matching SAM candidates to the corrected boxes
at class-aware IoU ≥ 0.5 produced 46 matches, 27 unmatched/wrong candidates and
22 missing corrected objects. This describes the selected sample against the
assistant's review, not a population accuracy estimate.

## Training

Run `scripts/train_local_yolo.py` with `.venv-training/Scripts/python.exe`.
The isolated environment uses PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128 and
Ultralytics 8.4.140. The complete resolved versions are saved in the run folder.

YOLO26s COCO-pretrained weights, 60 epochs, 640-pixel input, batch 4, AdamW,
initial learning rate 0.001, frozen first 10 layers, mixed precision, seed 42.
The local device is an RTX 3060 Laptop GPU with 6 GB VRAM. Training artifacts go
to `outputs/local-training/yolo26s-curated-crops-v1`; the last checkpoint is
copied to `models/yolo26s-cooking-crops-v1.pt`.

All images remain in one training set because there is no independent cooking
session to hold out. The framework's mandatory diagnostic validation uses the
same images (`val: images`). Its mAP is training-fit information only, not
validation/test accuracy. Epoch validation is disabled; the planned final epoch
checkpoint is delivered rather than selecting a checkpoint by a claimed
held-out metric.

`scripts/check_local_model.py` checks class-ID agreement and saves predictions
on 15 unused frames from the same session. These are qualitative checks, not an
independent evaluation. Use the same crop coordinates when trying this model;
full-frame deployment is not validated.

Before deployment: expand corrected full-frame labels, include all target
classes, add independently recorded sessions, then evaluate per-class recall
and false positives. The original 47,667 candidates remain available for that
review process. This model is an experimental baseline.

## Completed run and visual check

The 60-epoch run completed successfully on 2026-09-05. The exported checkpoint
loaded successfully, retained the 15 canonical class names, and ran inference
on one unused frame from each of the 15 videos at confidence threshold 0.25.
All 15 prediction previews were visually inspected.

Several pans, plates, bowls and lids were detected, and the sampled empty hob
had no detections. Failures remain: knives and cutting boards were missed in
some preparation frames, spatulas produced duplicate boxes, a pan produced
duplicate boxes, and the opening-lid sample had no detections despite visible
objects. These checks do not establish generalization to another session.
The model is not ready for reliable automatic annotation.

Prediction images, three contact sheets and machine-readable predictions are
in `outputs/local-model-check`. Training logs, loss curves, configuration and
checkpoints are in `outputs/local-training/yolo26s-curated-crops-v1`.
