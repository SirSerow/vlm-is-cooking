# Kitchen assistant

A local kitchen assistant built around structured observations. The initial app
analyzes a still image with Ollama and merges the answers into timestamped kitchen
state. It uses Python 3.12+ and the standard library.

## Run

Start the existing Ollama environment as described in
[the VLM setup guide](environments/vlm/README.md), then run from this directory:

```powershell
./.venv-vlm/Scripts/python.exe -m kitchen_assistant outputs/local-model-check/002_cutting-mushrooms-input.jpg --queries config/vlm_queries.v1.json --output outputs/app/observation.json
```

Pass `--model` or `--url` to select another model or Ollama server. The image can
be any local JPEG or PNG. Each invocation observes once, prints JSON, and exits.
Optionally install with `python -m pip install -e .` to use the
`kitchen-assistant` command from another directory.

## Code layout

| Module | Responsibility |
| --- | --- |
| `kitchen_assistant/cli.py` | Arguments, image input, JSON output |
| `kitchen_assistant/queries.py` | Typed queries and query-file loading |
| `kitchen_assistant/observer.py` | Ollama requests and response validation |
| `kitchen_assistant/state.py` | Frames, observations, field evidence and freshness |
| `kitchen_assistant/assistant.py` | Coordinate an observation and state update |
| `kitchen_assistant/recipe.py` | Supervised recipe steps and next-step recommendations |
| `kitchen_assistant/cook.py` | Interactive image observations and step confirmations |

The observer answers questions; it never decides whether a recipe step is done.
State retains the frame ID, capture time, source and confidence for each field.
Unknown answers replace previous values; older observations cannot overwrite
newer evidence. Consumers can use `KitchenState.value` with an explicit freshness
window. Model confidence is uncalibrated.

The `Observer` protocol is the small seam for alternate model backends or test
doubles. Live camera/video input, YOLO fusion, periodic scheduling, recipe
compilation and automatic progression are not implemented yet. The first recipe
demo will use one relevant instance of each entity, as specified in the
[plan review](docs/plan-review.md).

Existing `scripts/` and `review-app/` remain the training, evaluation and annotation
tools. The app does not import those experiments or require their GPU libraries.

## Follow the video recipe

The [inferred recipe](recipes/creamy-chicken-mushrooms.json) follows the training
video's chicken, mushroom, onion and cream sequence. It is a reconstruction of
the visible workflow; amounts and cooking temperatures are not established.

```powershell
./.venv-vlm/Scripts/python.exe -m kitchen_assistant.cook recipes/creamy-chicken-mushrooms.json
```

Enter an image path to observe the current visual step, `confirm` to finish the
current step, or `quit`. Two matching fresh images let the app suggest the next
step. Confirmation is always explicit; the model cannot advance the recipe.
Use different recent images within 30 seconds. Missing or uncertain evidence
asks for confirmation. Preparation, heating and doneness use manual steps.

## Evaluate recommendations on the training video

See [the evaluation report](docs/recipe-evaluation.md) for the measured results
and known failures. This is a supervised same-session benchmark, with the current
step supplied to the app. It does not measure autonomous recognition of recipe
position or generalization to unseen cooking sessions.

Sample the source clips with the existing OpenCV/Pillow training environment:

```powershell
./.venv-training/Scripts/python.exe -m scripts.evaluate_recipe --video-root S:/Projects/cooking-detection-project/datasets/video --sample-only
```

Inspect the `review-*.jpg` sheets under `outputs/recipe-evaluation/benchmark`, then
run the existing local Ollama model against the frozen samples:

```powershell
./.venv-vlm/Scripts/python.exe -m scripts.evaluate_recipe --video-root S:/Projects/cooking-detection-project/datasets/video --reuse-samples
```

The tool records answers, recommendations, image hashes, per-frame errors and
latency in `results.json`. Invalid model outputs count as failures. Use `--output`
to preserve a separate run. No model training or model downloads are performed.

## Tests

```powershell
./.venv-vlm/Scripts/python.exe -m unittest discover -s tests -v
```
