# SAM3 remote annotation environment

This prepares Linux x86-64 image inference for sampled video frames. It is an
environment and smoke-test baseline, not yet a batch annotator or review UI.
Source is pinned to `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`;
image inference explicitly uses the SAM3 checkpoint, not SAM3.1 multiplex video.

## Pod configuration

- NVIDIA Ampere or newer with BF16 support. Start with 24 GB VRAM for a
  single-image pilot; this is a provisional capacity target, not a measured minimum.
- Start with 32 GB host RAM and 50 GB free persistent storage, plus video space.
- Linux x86-64, Python 3.12, git, ffmpeg, and a driver compatible with CUDA 12.8.
- Keep the virtual environment, checkpoint cache, source data, and outputs on
  persistent storage. Do not assume the pod's root filesystem survives termination.
- Before renting a pod, request checkpoint access at
  [facebook/sam3](https://huggingface.co/facebook/sam3) and review its license.

The [upstream installation guide](https://github.com/facebookresearch/sam3#installation)
currently uses PyTorch 2.10 with CUDA 12.8. Our baseline pairs it with torchvision
0.25.0, disables compilation, and does not require optional FlashAttention 3.

## Install on an existing pod

Copy/clone this repository onto the pod, then from its root:

```bash
bash environments/sam3/bootstrap.sh
source /workspace/venvs/sam3/bin/activate
export HF_HOME=/workspace/cache/huggingface
hf auth login
hf download facebook/sam3 sam3.pt --local-dir /workspace/models/sam3
nvidia-smi
```

Use an interactive login or pod secret for authentication. Never place a token in
the Dockerfile, repository, image build arguments, or logs. For a custom install
location set `SAM3_ENV_DIR`; for an existing Python 3.12 executable set `PYTHON_BIN`.
Bootstrap does not install system packages: use a pod image with git, ffmpeg,
Python 3.12 and venv support, or build the image below.

## Optional container

Build from the repository root on a machine with Docker:

```bash
docker build -f environments/sam3/Dockerfile -t kitchen-sam3:baseline .
docker run --rm -it --gpus all --shm-size=8g \
  -v /your/persistent/storage:/workspace/data \
  -v /your/hf-cache:/workspace/cache/huggingface kitchen-sam3:baseline
```

Use the built image with the pod provider's custom-image workflow. Do not mount
over `/workspace` itself in this image: that hides the installed virtual environment.
The container includes code at `/opt/kitchen`. No service or public port is started.
The base image is version-tagged, not digest-pinned; transitive SAM dependencies
remain resolver-selected. Save `installed.freeze.txt` and the built image digest
after the successful pod pilot before reproducing a large annotation run.

## Real-image smoke test

Extract one representative frame from a video already copied to the pod:

```bash
mkdir -p /workspace/data/pilot
ffmpeg -n -ss 00:00:10 -i /workspace/data/kitchen.mp4 \
  -frames:v 1 /workspace/data/pilot/frame.jpg
python scripts/sam3_smoke.py /workspace/data/pilot/frame.jpg \
  --checkpoint /workspace/models/sam3/sam3.pt \
  --class-name pan --output /workspace/data/pilot/smoke-pan
```

For the container, download the checkpoint into `/workspace/data/models/sam3`
instead and pass that path so it persists on the example mount.

Inspect `preview.jpg`, mask PNGs, and `result.json`. A successful run proves model
loading and GPU inference; it does not prove annotation accuracy. Repeat on a
frame containing a knife and on a negative frame, using a new output directory.
Record inference time and peak allocated VRAM. This first timing includes warmup.

## Before batch annotation

1. Create a manifest assigning whole cooking sessions to train/val/test before
   sampling. Keep fragments of one session together. Store source hashes and
   original frame indices/timestamps, including variable-frame-rate videos.
2. Pilot roughly 100–200 diverse frames and compare candidates with complete
   human labels. Measure missing instances, wrong classes, and box quality per
   class, along with review time and GPU cost. This is a proposed pilot size.
3. Test prompt variants against the dictionary boundaries. The initial prompt
   per class is a starting point; e.g. `cup` may miss measuring cups and `pot lid`
   may miss other lids. Resolve duplicate/conflicting pan/pot/wok predictions.
4. Review whole frames, including empty predictions. Missing objects become
   false background during YOLO training. Standard YOLO detection labels have no
   generic ignore-region field: unresolved frames must be withheld or corrected.
5. Export only accepted frames after review. Keep raw masks/candidates separate
   from final YOLO labels; the smoke test intentionally writes no training labels.
6. Add resumable sampling, deduplication, multi-class inference, review and export
   tooling after the pilot validates this approach. Dense video propagation is
   optional later work and requires separate memory/quality testing.

The canonical class order is [kitchen_classes.v1.json](../../config/kitchen_classes.v1.json).
Mask-derived boxes bound visible pixels, including visible handles. Humans must
correct fragmented masks and missing extremities. Define truncation/occlusion
policy consistently before annotation; do not silently infer hidden object extent.

## Validation status

Local checks cover Python syntax, argument validation, shell syntax, and dictionary
agreement. Linux dependency installation, checkpoint loading, and GPU inference
must pass on the pod before calling this environment validated.
