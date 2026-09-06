"""Manually observe a local image and emit a JSON snapshot."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .assistant import CookingAssistant
from .observer import OllamaObserver
from .queries import load_queries
from .state import Frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-vl:2b-instruct")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    # For still images, captured_at means the time this image entered the session.
    frame = Frame(uuid4().hex, datetime.now(timezone.utc), args.image.read_bytes())
    assistant = CookingAssistant(OllamaObserver(model=args.model, base_url=args.url))
    observation = assistant.observe(frame, queries)
    report = {"observation": asdict(observation), "state": asdict(assistant.state)}
    output = json.dumps(report, indent=2, default=lambda value: value.isoformat())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
