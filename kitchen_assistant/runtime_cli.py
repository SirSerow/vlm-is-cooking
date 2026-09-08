"""Run a recipe from camera, web stream, or video with VLM or YOLO completion."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .completion import VlmCompletionStrategy, YoloPresenceStrategy
from .outputs import CompositeOutput, ConsoleOutput, WebOutput
from .recipe import Recipe
from .runtime import CookingApplication
from .sources import CameraReader, FrameScheduler, VideoReader, WebStreamReader
from .workflow import WorkflowSession
from .yolo import UltralyticsYoloDetector


def _roi(value: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI must contain four comma-separated integers") from error
    if len(result) != 4:
        raise argparse.ArgumentTypeError("ROI must contain x1,y1,x2,y2")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--strategy", choices=("vlm", "yolo"), required=True)
    parser.add_argument("--source", choices=("camera", "video", "web"), required=True)
    parser.add_argument("--input", help="Camera index, local video path, or web stream URL")
    parser.add_argument("--analysis-fps", type=float, default=1.0)
    parser.add_argument("--output", choices=("console", "web", "both"), default="console")
    parser.add_argument("--model", help="Ollama model name or YOLO weights/engine path")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--detection-confidence", type=float, default=0.25)
    parser.add_argument("--roi", type=_roi)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--event-log", type=Path, help="Write workflow transitions as JSON")
    args = parser.parse_args()
    if args.analysis_fps <= 0:
        parser.error("analysis-fps must be positive")
    if not 0 <= args.confidence <= 1 or not 0 <= args.detection_confidence <= 1:
        parser.error("confidence values must be between zero and one")

    recipe = Recipe.load(args.recipe)
    if args.source == "camera":
        reader = CameraReader(int(args.input or 0))
    elif args.source == "video":
        if not args.input:
            parser.error("video source needs --input PATH")
        reader = VideoReader(Path(args.input), fps=args.analysis_fps)
    else:
        if not args.input:
            parser.error("web source needs --input URL")
        reader = WebStreamReader(args.input)

    if args.strategy == "vlm":
        strategy = VlmCompletionStrategy(
            model=args.model or "qwen3-vl:4b-instruct-q4_K_M",
            base_url=args.ollama_url,
        )
    else:
        recipe.validate_yolo_path()
        weights = Path(args.model or "models/yolo26s-cooking-crops-v1.engine")
        device: int | str = int(args.device) if args.device.isdigit() else args.device
        detector = UltralyticsYoloDetector(
            weights,
            device=device,
            confidence=args.detection_confidence,
            roi=args.roi,
        )
        strategy = YoloPresenceStrategy(detector, args.detection_confidence)

    outputs = []
    if args.output in {"console", "both"}:
        outputs.append(ConsoleOutput(debug=args.debug))
    if args.output in {"web", "both"}:
        outputs.append(WebOutput(args.host, args.port))
    application = CookingApplication(
        WorkflowSession(recipe, datetime.now(timezone.utc), confidence_threshold=args.confidence),
        FrameScheduler(reader, args.analysis_fps),
        strategy,
        CompositeOutput(tuple(outputs)),
    )
    try:
        application.run(max_frames=args.max_frames)
    except KeyboardInterrupt:
        pass
    finally:
        if args.event_log:
            args.event_log.parent.mkdir(parents=True, exist_ok=True)
            args.event_log.write_text(
                json.dumps(
                    {
                        "recipe": str(args.recipe),
                        "strategy": args.strategy,
                        "source": args.source,
                        "input": args.input,
                        "events": [asdict(event) for event in application.session.events],
                    },
                    indent=2,
                    default=lambda value: value.isoformat(),
                )
                + "\n",
                encoding="utf-8",
            )



if __name__ == "__main__":
    main()
