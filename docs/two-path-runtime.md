# Two-path cooking runtime

The runtime compares two independent completion strategies while sharing the
same recipe, frame source, workflow state machine, and outputs.

## Workflow

Every step starts in `waiting_for_requirements`. The application displays the
step name and missing objects/ingredients. Two recent, distinct frames containing
all required entities activate the step and reveal its description and next-step
preview.

While active, the selected strategy returns one structured completion assessment
per analyzed frame. Two accepted completion assessments are required to advance.
For non-final steps, the controller must first see `not_started` or `in_progress`;
this prevents objects staged before activation from immediately skipping a step.

## Strategies

`VlmCompletionStrategy` sends only the current step and frame to Ollama. The next
step is deliberately hidden. It returns visible required entities, status,
confidence, and short evidence.

`YoloPresenceStrategy` treats stable presence of the next step's complete
requirement set as evidence that the current step is complete. This is a
constrained baseline, not action recognition. `Recipe.validate_yolo_path()`
rejects adjacent steps when the following step has no newly appearing entity.

## Inputs and scheduling

- `CameraReader`: local camera index.
- `WebStreamReader`: a stream URL supported by OpenCV/FFmpeg.
- `VideoReader`: deterministic sampling using source-video timestamps.

Camera and web input run capture in a background thread. Inference consumes only
the newest frame at the configured cadence, so slow inference cannot build a
queue. Video input remains deterministic and does not use real-time pacing.

## Outputs

Console mode prints workflow changes and optional detections. Web mode serves a
local page, the latest JPEG at `/frame.jpg`, and structured state at `/state`.

## Commands

Install only the dependencies needed by the selected path:

```powershell
python -m pip install -e ".[input]"
python -m pip install -e ".[yolo]"
```

VLM with a video and both outputs:

```powershell
kitchen-cook recipes/two-path-demo.json --strategy vlm --source video `
  --input datasets/video/example.mp4 --analysis-fps 1 --output both
```

YOLO TensorRT with the cooking-area crop:

```powershell
kitchen-cook recipes/two-path-demo.json --strategy yolo --source camera `
  --input 0 --model models/yolo26s-cooking-crops-v1.engine `
  --roi 500,330,1120,1080 --analysis-fps 2 --output both
```

Use `--debug` for console detections and `--max-frames` for bounded development
runs. Model accuracy and taxonomy expansion are intentionally separate from this
runtime refactor.

## Transition evaluation

For repeatable comparison, run both strategies on the same video and save their
source-timestamped event logs:

```powershell
kitchen-cook recipes/two-path-demo.json --strategy vlm --source video `
  --input datasets/video/example.mp4 --event-log outputs/vlm-events.json
```

After replacing the example transition windows with reviewed labels, score a run:

```powershell
python -m scripts.evaluate_transitions outputs/vlm-events.json `
  config/transition_evaluation.example.json --output outputs/vlm-score.json
```

The scorer reports correct, premature, late, missed, unexpected, and duplicate
transitions, along with precision, recall, and mean absolute timing error.
