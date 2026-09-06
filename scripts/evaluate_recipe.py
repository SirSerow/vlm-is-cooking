"""Sample training-session video and score supervised next-step recommendations.

Run as a module from the repository root: python -m scripts.evaluate_recipe.
OpenCV and Pillow are only needed for this offline evaluation tool.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kitchen_assistant.assistant import CookingAssistant
from kitchen_assistant.observer import OllamaObserver
from kitchen_assistant.recipe import Recipe, RecipeSession
from kitchen_assistant.state import Frame


def sample_frames(config: dict, video_root: Path, output: Path) -> list[dict]:
    import cv2
    from PIL import Image, ImageDraw

    episodes = json.loads((video_root / "episodes.summary.json").read_text(encoding="utf-8"))["episodes"]
    starts = {Path(ep["output"]).name: ep["start"] for ep in episodes}
    samples = []
    for case in config["cases"]:
        sheet = Image.new("RGB", (1250, 330), "white")
        draw = ImageDraw.Draw(sheet)
        column = 0
        for phase in ("before", "after"):
            source = case[phase]
            capture = cv2.VideoCapture(str(video_root / source["clip"]))
            try:
                for seconds in source["seconds"]:
                    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                    ok, pixels = capture.read()
                    if not ok:
                        raise RuntimeError(f"Cannot read {source['clip']} at {seconds}s")
                    picture = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).crop(config["crop"])
                    path = output / f"{case['step']}-{phase}-{seconds}.jpg"
                    picture.save(path, quality=90)
                    thumbnail = picture.copy()
                    thumbnail.thumbnail((245, 295))
                    sheet.paste(thumbnail, (column * 250, 30))
                    draw.text((column * 250 + 4, 5), f"{phase} {seconds}s", fill="black")
                    samples.append({
                        "step": case["step"], "phase": phase, "clip": source["clip"],
                        "clip_seconds": seconds, "video_seconds": starts[source["clip"]] + seconds,
                        "image": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "expected_match": phase == "after",
                    })
                    column += 1
            finally:
                capture.release()
        sheet.save(output / f"review-{case['step']}.jpg")
    (output / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    return samples


def evaluate(config: dict, samples: list[dict], output: Path, model: str) -> dict:
    recipe = Recipe.load(Path(config["recipe"]))
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    records = []
    for case in config["cases"]:
        index = next(i for i, step in enumerate(recipe.steps) if step.id == case["step"])
        session = RecipeSession(recipe, epoch, index=index)
        assistant = CookingAssistant(OllamaObserver(model=model), session=session)
        gold_votes = []
        case_samples = [sample for sample in samples if sample["step"] == case["step"]]
        for sample in case_samples:
            now = epoch + timedelta(seconds=sample["video_seconds"])
            image = (output / sample["image"]).read_bytes()
            if hashlib.sha256(image).hexdigest() != sample["sha256"]:
                raise ValueError(f"Sample changed after review: {sample['image']}")
            frame = Frame(sample["sha256"], now, image)
            gold_votes.append((now, sample["expected_match"]))
            recent = [match for timestamp, match in gold_votes[-3:] if now - timestamp <= session.max_age]
            expected_ready = recent[-1] and sum(recent) >= 2
            expected_step = recipe.steps[index + 1].id if expected_ready else case["step"]
            started = time.perf_counter()
            record = {**sample, "expected_ready": expected_ready, "expected_step": expected_step}
            try:
                observation = assistant.observe(frame, session.current_step.queries)
                recommendation = session.recommend(now)
                matches = all(
                    assistant.state.fields[key].value == value
                    and assistant.state.fields[key].confidence >= session.confidence_threshold
                    for key, value in session.current_step.expected_after.items()
                )
                record.update(
                    observation=asdict(observation), recommendation=asdict(recommendation),
                    predicted_match=matches, correct_match=matches == sample["expected_match"],
                    correct_recommendation=recommendation.suggested_step == expected_step,
                )
            except (ValueError, TypeError, KeyError, OSError) as error:
                record.update(error=f"{type(error).__name__}: {error}", correct_match=False, correct_recommendation=False)
            record["seconds"] = round(time.perf_counter() - started, 3)
            records.append(record)
            (output / "results.partial.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
            print(json.dumps({"step": case["step"], "phase": sample["phase"],
                              "seconds": record["seconds"], "correct": record["correct_recommendation"],
                              "result": record.get("recommendation", record.get("error"))}), flush=True)
    positive = [r for r in records if r["expected_match"]]
    negative = [r for r in records if not r["expected_match"]]
    summary = {
        "samples": len(records), "valid_responses": sum("error" not in r for r in records),
        "correct_visual_matches": sum(r["correct_match"] for r in records),
        "correct_next_step_recommendations": sum(r["correct_recommendation"] for r in records),
        "positive_matches": sum(r.get("predicted_match", False) for r in positive), "positive_samples": len(positive),
        "false_positive_matches": sum(r.get("predicted_match", False) for r in negative), "negative_samples": len(negative),
        "premature_next_steps": sum(r.get("recommendation", {}).get("status") == "ready" and not r["expected_ready"] for r in records),
        "always_hold_baseline_correct": sum(not r["expected_ready"] for r in records),
        "always_match_baseline_correct": len(positive),
        "successful_cases": sum(all(r["correct_recommendation"] for r in records if r["step"] == case["step"]) for case in config["cases"]),
        "cases": len(config["cases"]),
        "mean_request_seconds": round(sum(r["seconds"] for r in records) / len(records), 3),
    }
    sources = [Path(config["recipe"]), *Path("kitchen_assistant").glob("*.py")]
    result = {
        "model": model, "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "config": config, "summary": summary, "records": records,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/recipe_evaluation.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/recipe-evaluation/benchmark"))
    parser.add_argument("--model", default="qwen3-vl:2b-instruct")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--reuse-samples", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    samples = (json.loads((args.output / "samples.json").read_text(encoding="utf-8"))
               if args.reuse_samples else sample_frames(config, args.video_root, args.output))
    if not args.sample_only:
        evaluate(config, samples, args.output, args.model)


if __name__ == "__main__":
    main()
