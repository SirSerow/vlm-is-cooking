# Local VLM (Windows + Ollama)

This environment runs the kitchen structured observer on local images. Ollama
owns the inference runtime; `.venv-vlm` is a separate Python 3.12 environment
using only the standard library, with no pip dependencies or separate CUDA toolkit.

The default is `qwen3-vl:2b-instruct` (Q4_K_M, approximately 1.9 GB download),
with a 4096-token context and 512-token output limit to fit the laptop's 6 GB
RTX 3060. Model information: https://ollama.com/library/qwen3-vl:2b-instruct.
The client uses Ollama's local structured-output API:
https://docs.ollama.com/capabilities/structured-outputs.

## Setup

Install and start [Ollama for Windows](https://ollama.com/download/windows).
From the repository root, with Python 3.10+ available:

```powershell
./environments/vlm/bootstrap.ps1
# Or select an existing interpreter explicitly:
./environments/vlm/bootstrap.ps1 -Python ./.venv-training/Scripts/python.exe
```

The setup checks the running server, pulls the model, and runs validation tests.
It does not modify the training environment. Model files use Ollama's existing
configured storage (normally `%USERPROFILE%/.ollama/models`). Download requires
internet; subsequent inference uses only `127.0.0.1:11434`.

## Image smoke test

```powershell
./.venv-vlm/Scripts/python.exe scripts/vlm_smoke.py outputs/local-model-check/002_cutting-mushrooms-input.jpg --output outputs/vlm-smoke/mushrooms.json
./.venv-vlm/Scripts/python.exe scripts/vlm_smoke.py outputs/local-model-check/015_empty-plate-input.jpg --output outputs/vlm-smoke/empty-plate.json
```

Use any local JPEG or PNG in place of these existing dataset crops. The default
queries ask for mushroom state and location; pass `--queries path/to/queries.json`
for another ingredient. Each query needs `id`, `entity`, `ask`, and an `allowed`
string list containing `unknown`. See `config/vlm_queries.v1.json`.

Each command runs twice, printing observations and wall time. The JSON report
records raw responses, timings, image hash, model digest, Ollama version, and
loaded-model GPU allocation. Invalid ids, missing/duplicate answers, illegal
values, bad confidence, inconsistent unknowns, and truncated generation fail
the command. Confidence is model-reported and is not calibrated.

```powershell
./.venv-vlm/Scripts/python.exe -m unittest discover -s scripts -p test_vlm_smoke.py -v
ollama ps
# Release the model's GPU allocation before YOLO training:
ollama stop qwen3-vl:2b-instruct
```

The model stays loaded for five minutes after a request. Repeat-image warm
timings may benefit from caching; measure changed frames before setting a
periodic camera cadence. These tests establish local inference and response
validation, not cooking accuracy, camera integration, or recipe completion.

## Verified on 2026-09-06

- Windows, Python 3.12.14, Ollama 0.33.3, NVIDIA driver 581.29,
  RTX 3060 Laptop GPU (6144 MiB).
- Model digest: `ea422f1e73652a95479954d8572d3c8c6022f628ce2d38a1a04aae1b7f2d5300`.
- First model load/request: 84.11 seconds. Final prompt: sliced-mushroom
  image 0.80/0.65 seconds; empty-plate image 1.06/0.74 seconds. These images
  had already been processed, so these are cached smoke timings, not a fresh-frame
  throughput benchmark or proof of the three-second camera target.
- Mushroom image: `sliced`, `plate`, both runs. Empty plate: `unknown` for both
  queries, both runs; conservative abstention, but it did not choose the more
  specific `not_visible`. Final responses passed validation with no anomalies.
- Ollama reported all 1,975,968,070 model bytes in VRAM. An observed total GPU
  allocation during initial inference was 4423 MiB including desktop applications;
  this is a snapshot, not a measured peak or a YOLO co-residency guarantee.
- Two unit tests passed, including nine invalid-observation cases. An initial
  negative-image response failed unknown-id consistency; adding id enums and
  explicit unknown/anomaly instructions resolved it for these smoke cases.
  Raw initial and final reports are under `outputs/vlm-smoke/` (gitignored).
