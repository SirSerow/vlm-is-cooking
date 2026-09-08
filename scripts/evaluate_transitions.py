"""Compare a kitchen-cook event log with labelled video transition windows."""

import argparse
import json
from pathlib import Path

from kitchen_assistant.evaluation import ExpectedTransition, score_transitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_log", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    events = json.loads(args.event_log.read_text(encoding="utf-8"))["events"]
    label_data = json.loads(args.labels.read_text(encoding="utf-8"))
    expected = tuple(ExpectedTransition(**row) for row in label_data["transitions"])
    result = score_transitions(events, expected)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

