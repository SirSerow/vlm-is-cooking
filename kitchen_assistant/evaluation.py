"""Score predicted recipe transitions against labelled video-time windows."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedTransition:
    step_id: str
    earliest_seconds: float
    target_seconds: float
    latest_seconds: float

    def __post_init__(self) -> None:
        if not 0 <= self.earliest_seconds <= self.target_seconds <= self.latest_seconds:
            raise ValueError("Transition times must satisfy 0 <= earliest <= target <= latest")


def score_transitions(events: list[dict], expected: tuple[ExpectedTransition, ...]) -> dict:
    predictions: dict[str, float] = {}
    duplicate_predictions = 0
    unavailable_timestamps = 0
    for event in events:
        if event.get("kind") != "completed":
            continue
        step_id = event.get("step_id")
        seconds = event.get("source_seconds")
        if not isinstance(step_id, str) or not isinstance(seconds, (int, float)):
            unavailable_timestamps += 1
            continue
        if step_id in predictions:
            duplicate_predictions += 1
            continue
        predictions[step_id] = float(seconds)

    rows = []
    expected_ids = {transition.step_id for transition in expected}
    for transition in expected:
        predicted = predictions.get(transition.step_id)
        if predicted is None:
            outcome, error = "missed", None
        elif predicted < transition.earliest_seconds:
            outcome, error = "premature", predicted - transition.target_seconds
        elif predicted > transition.latest_seconds:
            outcome, error = "late", predicted - transition.target_seconds
        else:
            outcome, error = "correct", predicted - transition.target_seconds
        rows.append(
            {
                "step_id": transition.step_id,
                "predicted_seconds": predicted,
                "target_seconds": transition.target_seconds,
                "timing_error_seconds": error,
                "outcome": outcome,
            }
        )
    unexpected = sorted(set(predictions) - expected_ids)
    correct = sum(row["outcome"] == "correct" for row in rows)
    prediction_count = len(predictions)
    timing_errors = [abs(row["timing_error_seconds"]) for row in rows if row["timing_error_seconds"] is not None]
    return {
        "summary": {
            "expected_transitions": len(expected),
            "predicted_transitions": prediction_count,
            "correct": correct,
            "premature": sum(row["outcome"] == "premature" for row in rows),
            "late": sum(row["outcome"] == "late" for row in rows),
            "missed": sum(row["outcome"] == "missed" for row in rows),
            "unexpected": len(unexpected),
            "precision": correct / prediction_count if prediction_count else 0.0,
            "recall": correct / len(expected) if expected else 0.0,
            "mean_absolute_timing_error_seconds": sum(timing_errors) / len(timing_errors) if timing_errors else None,
            "duplicate_predictions": duplicate_predictions,
            "unavailable_timestamps": unavailable_timestamps,
        },
        "transitions": rows,
        "unexpected_step_ids": unexpected,
    }

