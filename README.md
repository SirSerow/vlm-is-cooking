# Kitchen assistant

A local experiment comparing two ways to control a visual cooking workflow:

1. A VLM directly assesses whether the current recipe step is complete.
2. A trained YOLO detector uses newly appearing next-step requirements as a
   deterministic presence-based completion baseline.

Both paths share the same recipe schema, input readers, temporal state machine,
console/web outputs, and evaluation surface. They are intentionally not fused.

## Runtime architecture

The application loads a recipe JSON, opens a camera, web stream, or local video,
and starts each step in `waiting_for_requirements`. Once all required objects and
ingredients are visible in two recent frames, the instruction becomes active.
Two stable completion assessments advance the workflow automatically.

The next-step condition must not already be true throughout activation. This
transition guard prevents pre-staged objects from skipping the current step.
The YOLO recipe validator also rejects adjacent steps without a newly appearing
entity. See [the runtime design and commands](docs/two-path-runtime.md).

## Run

VLM path:

```powershell
python -m pip install -e ".[input]"
kitchen-cook recipes/two-path-demo.json --strategy vlm --source video `
  --input datasets/video/example.mp4 --analysis-fps 1 --output both
```

YOLO path:

```powershell
python -m pip install -e ".[yolo]"
kitchen-cook recipes/two-path-demo.json --strategy yolo --source camera `
  --input 0 --model models/yolo26s-cooking-crops-v1.engine `
  --roi 500,330,1120,1080 --analysis-fps 2 --output both
```

The web page is available at `http://127.0.0.1:8080` when web output is enabled.

## Code layout

| Module | Responsibility |
| --- | --- |
| `recipe.py` | Recipe schema and legacy supervised benchmark session |
| `workflow.py` | Readiness, completion voting, and automatic advancement |
| `completion.py` | Shared strategy interface and VLM/YOLO strategies |
| `sources.py` | Camera, web-stream, video, and latest-frame scheduling |
| `yolo.py` | Ultralytics/TensorRT detector adapter |
| `outputs.py` | Console and lightweight local web output |
| `runtime.py` | Strategy-independent application controller |
| `runtime_cli.py` | Runtime command-line entry point |

The earlier closed-query observer and supervised recipe evaluation remain
available as a frozen baseline. Training, annotation, and TensorRT benchmark
scripts remain separate from the runtime package.

## Tests

```powershell
python -m unittest discover -s tests -v
```

No model download or inference is performed by the unit tests.

Use `--event-log PATH` during video runs and `python -m
scripts.evaluate_transitions` to compare transition timing against reviewed
windows. The complete procedure is in the runtime design document.
