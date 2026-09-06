# Prepare reviewed data and train YOLO yourself

These commands are for **PowerShell on this Windows machine**. Run them from
the project folder. The Runpod pod is not needed. This workflow trains object
**bounding boxes**, not SAM3 masks or cooking actions.

The recommended next experiment uses your approved dashboard corrections.
The old 45-image dataset is useful for checking the workflow, but repeating
that small dataset will not fix the accuracy problems.

## 1. Check the existing environment

```powershell
Set-Location 'S:\Projects\vlm-is-cooking'
$env:YOLO_CONFIG_DIR = "$PWD\outputs\ultralytics-config"
$env:YOLO_AUTOINSTALL = 'False'
& .\.venv-training\Scripts\python.exe -c "import torch, ultralytics; print('torch:', torch.__version__); print('ultralytics:', ultralytics.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

Expected existing environment: PyTorch `2.10.0+cu128`, Ultralytics `8.4.140`,
CUDA available, RTX 3060 Laptop GPU (6 GB). No virtual-environment activation
is needed because the commands use its executables explicitly.

If `.venv-training` is missing, the Git repository alone does not contain the
environment, weights, videos or datasets. On a replacement machine with Python
3.12 and a compatible NVIDIA driver, recreate the pinned environment with:

```powershell
py -3.12 -m venv .venv-training
& .\.venv-training\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
& .\.venv-training\Scripts\python.exe -m pip install ultralytics==8.4.140
& .\.venv-training\Scripts\python.exe -m pip check
```

Re-run the GPU check. Restore your local data separately. For the exact original
resolved dependencies, see `outputs/local-training/yolo26s-curated-crops-v1/installed.freeze.txt`.

## 2. Review and approve the data

Start the dashboard if necessary:

```powershell
powershell -File .\scripts\start_review.ps1
```

Open **http://127.0.0.1:8877/** in your browser, not `index.html` directly.

1. Choose a video and frame, then open **Correct labels**.
2. Choose **Cooking area only** for a crop detector, or **Full frame** for a
   detector intended to see the entire image. Use a consistent scope for this
   experiment. The crop is `[500,330,1120,1080]` in the 1920×1080 source.
3. Initialize from SAM3 or YOLO if helpful, then correct classes, remove extra
   boxes and draw missing boxes. The “Use visible” buttons respect the current
   confidence, class and view filters; imported boxes are only a starting point.
4. Label **every visible target object within the chosen scope**, including
   partially visible objects that you can identify. Apply one consistent policy
   for occluded objects; the original dataset labels visible extents only.
5. Select **Approved within scope** and click **Save review**. For a truly empty
   frame, approve with zero boxes. Do not approve a frame with missing labels as
   a negative example.
6. Click **Export reviews** and save the ZIP in your Downloads folder.

The ZIP includes images and YOLO labels only for approved frames. Draft and
needs-work decisions appear in `reviews.json` but are not exported as training
images. Original SAM3 data is preserved.

Prioritize knives, cutting boards, spatulas, tongs, spoons, and pans containing
food. Include different positions, occlusion, motion and empty scenes. Avoid
collecting hundreds of nearly identical adjacent frames. Video names are useful
context, but do not prove which objects are visible.

Keep the 15 class IDs fixed:

```text
0 pan            5 cutting_board   10 tongs
1 pot            6 bowl            11 cup
2 wok            7 plate           12 bottle
3 lid            8 spatula         13 jar
4 knife          9 spoon           14 sponge
```

## 3. Extract a new dataset snapshot

Edit `$reviewZip` if your browser saved a different filename. Keep these
PowerShell variables in the same terminal for the subsequent commands.

```powershell
$reviewZip = "$env:USERPROFILE\Downloads\kitchen-reviewed-labels.zip"
$reviewStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$reviewExport = "$PWD\data\review-export-$reviewStamp"
Expand-Archive -LiteralPath $reviewZip -DestinationPath $reviewExport

# Choose 'crop' OR 'full'. Do not use crop labels with full-frame images.
$reviewScope = 'crop'
$reviewDataset = Join-Path $reviewExport $reviewScope
if (!(Test-Path "$reviewDataset\images")) {
    throw "No approved images for scope '$reviewScope'. Approve and export frames first."
}
$reviewImages = @(Get-ChildItem "$reviewDataset\images" -Filter '*.jpg')
if ($reviewImages.Count -eq 0) { throw 'The approved image folder is empty.' }
foreach ($reviewImage in $reviewImages) {
    if (!(Test-Path "$reviewDataset\labels\$($reviewImage.BaseName).txt")) {
        throw "Missing label file for $($reviewImage.Name)"
    }
}
Write-Output "Dataset: $reviewDataset"
Write-Output "Approved images: $($reviewImages.Count)"
```

An empty `.txt` is valid for an explicitly reviewed negative. Each non-empty
line must have `class_id center_x center_y width height`, with coordinates
normalized to that image's dimensions, not pixel coordinates. The dashboard
export performs this conversion for the selected scope.

### Option A: first local fitting experiment, using the current single session

The following creates a runnable dataset configuration. **It deliberately uses
the same images for train and diagnostic validation. Its metrics are not an
accuracy estimate.** Use this only while collecting independent recordings.

```powershell
$reviewClasses = (Get-Content .\config\kitchen_classes.v1.json -Raw | ConvertFrom-Json).classes
$reviewYaml = @(
    "path: $($reviewDataset.Replace('\','/'))"
    'train: images'
    'val: images'
    'names:'
) + @($reviewClasses | ForEach-Object { "  $($_.id): $($_.name)" })
$reviewYaml | Set-Content "$reviewDataset\dataset.yaml" -Encoding utf8
$reviewValidation = 'False'
Get-Content "$reviewDataset\dataset.yaml"
```

Ultralytics can still perform final diagnostic validation even with `val=False`.
Do not interpret its mAP as held-out performance.

### Option B: training with independent validation (recommended when available)

Record and label additional cooking sessions with the same class dictionary and
scope. The current dashboard contains only the original session; new recordings
must first be annotated/exported or prepared in the same YOLO image/label format.
Assign **whole recording sessions** to train, validation and optionally test.
The 15 existing numbered videos are portions of one recording, not 15 independent
sessions. Do not randomly split adjacent frames between train and validation.

Arrange a new dataset as:

```text
data/my-reviewed-dataset/
  images/train/    # images from training sessions
  images/val/      # images from separate validation sessions
  images/test/     # optional, untouched final test sessions
  labels/train/   # matching filenames, with .txt extension
  labels/val/
  labels/test/
  dataset.yaml
```

Use unique image filenames across recordings. Copy each image and its matching
label together. Do not include duplicate copies of a frame in multiple splits.
Set the variables and write the YAML:

```powershell
$reviewDataset = "$PWD\data\my-reviewed-dataset"
$reviewClasses = (Get-Content .\config\kitchen_classes.v1.json -Raw | ConvertFrom-Json).classes
$reviewYaml = @(
    "path: $($reviewDataset.Replace('\','/'))"
    'train: images/train'
    'val: images/val'
    'names:'
) + @($reviewClasses | ForEach-Object { "  $($_.id): $($_.name)" })
$reviewYaml | Set-Content "$reviewDataset\dataset.yaml" -Encoding utf8
$reviewValidation = 'True'
```

If you have test images, add `test: images/test` as a top-level YAML line, before
`names:`. A class absent from validation cannot have its recall reliably assessed.

## 4. Train a new model

Run one of the data preparation options above first. Close other GPU-intensive
programs. Keep the terminal open while training. This uses the original COCO
pretrained weights, not the weak cooking checkpoint, as the starting point.

```powershell
$reviewRun = 'yolo26s-reviewed-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& .\.venv-training\Scripts\yolo.exe detect train `
    model=yolo26s.pt `
    "data=$reviewDataset\dataset.yaml" `
    epochs=60 imgsz=640 batch=4 device=0 workers=0 `
    optimizer=AdamW lr0=0.001 lrf=0.01 freeze=10 `
    amp=True cache=False seed=42 deterministic=True patience=0 `
    mosaic=0.5 close_mosaic=10 mixup=0.0 `
    degrees=10 translate=0.1 scale=0.3 fliplr=0.5 flipud=0.0 `
    "val=$reviewValidation" save=True save_period=20 plots=True `
    project=outputs/local-training "name=$reviewRun" exist_ok=False
if ($LASTEXITCODE -ne 0) { throw 'Training failed. Read the error above.' }
$reviewRunDir = "$PWD\outputs\local-training\$reviewRun"
Write-Output "Training output: $reviewRunDir"
```

PowerShell backticks must be the final character on their lines, with no trailing
spaces. These are starting settings for the local 6 GB GPU. Vertical flips are
disabled here because the camera orientation is fixed; the original baseline
used `flipud=0.5`. This is a new experiment, not a bit-for-bit baseline reproduction.
Try `batch=2` or `batch=1` if CUDA reports out-of-memory. A smaller `yolo26n.pt`
can be tried as a separate experiment. Missing pretrained weights may download
on first use. Changing scope to full frame requires new full-frame labels;
simply passing larger images does not repair crop labels.

Each run has a unique folder. This command **does not replace** the model used
by the dashboard. The older `scripts/train_local_yolo.py` is hard-coded to the
45-image baseline and copies its checkpoint over the existing dashboard model;
use the command above for your own experiments instead.

## 5. Inspect the results and run inference

Inside `$reviewRunDir`, inspect:

- `results.csv` and `results.png`: losses and available diagnostic metrics.
- `args.yaml`: the exact settings used.
- `train_batch*.jpg`: check that images and boxes align after augmentation.
- `weights/last.pt`: the final saved checkpoint.
- `weights/best.pt`: selected using validation fitness; meaningful selection
  requires an independent validation set.

For Option A, use `last.pt`; for Option B, start by evaluating `best.pt`:

```powershell
$reviewCheckpoint = "$reviewRunDir\weights\last.pt"
# With independent validation instead:
# $reviewCheckpoint = "$reviewRunDir\weights\best.pt"

# Set this to a folder of NEW images of the SAME scope as training.
$reviewPredictionInput = "$PWD\data\my-check-images"
& .\.venv-training\Scripts\yolo.exe detect predict `
    "model=$reviewCheckpoint" "source=$reviewPredictionInput" `
    imgsz=640 device=0 conf=0.25 save=True save_txt=True save_conf=True `
    project=outputs/local-predictions "name=$reviewRun" exist_ok=False
```

Create/populate `data/my-check-images` before running inference. For a crop
model, crop 1920×1080 source images to `[500,330,1120,1080]` first (620×750 output).
Use full images only for a model trained and evaluated for full-frame use.
Confidence values alone do not prove correctness: inspect missed objects,
wrong classes and duplicates, especially tools and food-filled pans.

With a genuine validation set, run:

```powershell
& .\.venv-training\Scripts\yolo.exe detect val `
    "model=$reviewCheckpoint" "data=$reviewDataset\dataset.yaml" `
    split=val imgsz=640 batch=4 device=0 workers=0
```

For an untouched final test set declared in YAML, change `split=val` to
`split=test`. Do not repeatedly tune settings using the final test set.

The current dashboard's training page and inference cache are pinned to the
original run/model. New CLI runs will not appear there automatically. Keep the
new checkpoints separate until their results have been reviewed.

## 6. Resume an interrupted run

```powershell
& .\.venv-training\Scripts\yolo.exe detect train `
    "model=$reviewRunDir\weights\last.pt" resume=True device=0
```

In a new terminal, set `$reviewRunDir` to the interrupted run's actual folder
first. Resume requires an unfinished, resumable checkpoint. It continues the
saved run's configuration; it is not the way to start training on changed data
or add epochs to a completed run. For changed data, start a new run without
`resume=True`.

## Optional: rebuild the original 45-image dataset

```powershell
& .\.venv-training\Scripts\python.exe .\scripts\build_curated_training.py
```

This reads `outputs/all-videos-1fps` and
`config/curated_crop_review.v1.json`, and rebuilds `data/curated-crops-v1`.
It does **not** read your dashboard approvals. It overwrites the generated
baseline files in that directory. To train it without replacing the dashboard
checkpoint, set `$reviewDataset` to `$PWD\data\curated-crops-v1`, set
`$reviewValidation = 'False'`, and use the new-run command in section 4.

Keep your exported review ZIPs, dataset snapshots and model checkpoints backed
up separately: `data/`, `outputs/` and weight files are excluded from Git.
