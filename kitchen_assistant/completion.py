"""Interchangeable VLM and YOLO completion strategies."""

import base64
import json
import math
from dataclasses import dataclass
from typing import Protocol
from urllib.request import Request, urlopen

from .recipe import Step
from .state import Detection, Frame, StepAssessment, StrategyResult


class CompletionStrategy(Protocol):
    name: str

    def analyze(self, frame: Frame, step: Step, next_step: Step | None) -> StrategyResult: ...


class Detector(Protocol):
    def detect(self, frame: Frame) -> tuple[Detection, ...]: ...


def _entity_array_schema(entities: tuple[str, ...]) -> dict:
    schema: dict = {
        "type": "array",
        "uniqueItems": True,
        "maxItems": len(entities),
        "items": {"type": "string"},
    }
    if entities:
        schema["items"]["enum"] = list(entities)
    return schema


def completion_response_schema(step: Step) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["visible_objects", "visible_ingredients", "status", "confidence", "evidence"],
        "properties": {
            "visible_objects": _entity_array_schema(step.required_objects),
            "visible_ingredients": _entity_array_schema(step.required_ingredients),
            "status": {
                "type": "string",
                "enum": ["not_started", "in_progress", "complete", "uncertain"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        },
    }


def parse_completion(data: dict, step: Step) -> tuple[tuple[str, ...], tuple[str, ...], str, float, tuple[str, ...]]:
    expected = {"visible_objects", "visible_ingredients", "status", "confidence", "evidence"}
    if set(data) != expected:
        raise ValueError("Unexpected VLM completion fields")
    objects = data["visible_objects"]
    ingredients = data["visible_ingredients"]
    if not isinstance(objects, list) or len(objects) != len(set(objects)) or not set(objects) <= set(step.required_objects):
        raise ValueError("Invalid visible_objects")
    if not isinstance(ingredients, list) or len(ingredients) != len(set(ingredients)) or not set(ingredients) <= set(step.required_ingredients):
        raise ValueError("Invalid visible_ingredients")
    status = data["status"]
    if status not in {"not_started", "in_progress", "complete", "uncertain"}:
        raise ValueError("Invalid completion status")
    confidence = data["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid completion confidence")
    evidence = data["evidence"]
    if not isinstance(evidence, list) or len(evidence) > 4 or not all(isinstance(item, str) for item in evidence):
        raise ValueError("Invalid completion evidence")
    return tuple(objects), tuple(ingredients), status, float(confidence), tuple(evidence)


@dataclass(slots=True)
class VlmCompletionStrategy:
    model: str = "qwen3-vl:4b-instruct-q4_K_M"
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 300
    name: str = "vlm"

    def analyze(self, frame: Frame, step: Step, next_step: Step | None) -> StrategyResult:
        del next_step  # Deliberately hidden from the VLM to avoid answer leakage.
        schema = completion_response_schema(step)
        prompt = (
            "Judge only the current cooking step from this kitchen frame. "
            "Use complete only when the described action is visibly finished. "
            "Use uncertain when one image cannot establish completion. "
            "Report visible entities only from the supplied lists. JSON only. "
            + json.dumps(
                {
                    "step": {"name": step.name, "description": step.description},
                    "required_objects": step.required_objects,
                    "required_ingredients": step.required_ingredients,
                    "response_schema": schema,
                }
            )
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(frame.image).decode()],
                }
            ],
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512},
            "keep_alive": "5m",
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if result.get("done_reason") == "length":
            raise ValueError("Model response was truncated")
        data = json.loads(result["message"]["content"])
        objects, ingredients, status, confidence, evidence = parse_completion(data, step)
        detections = tuple(
            Detection(entity, confidence, kind=kind)
            for kind, entities in (("object", objects), ("ingredient", ingredients))
            for entity in entities
        )
        assessment = StepAssessment(
            step.id,
            status,
            confidence,
            frame.id,
            frame.captured_at,
            self.name,
            evidence,
            frame.source_seconds,
        )
        return StrategyResult(detections, assessment)


@dataclass(slots=True)
class YoloPresenceStrategy:
    detector: Detector
    confidence_threshold: float = 0.25
    name: str = "yolo"

    def analyze(self, frame: Frame, step: Step, next_step: Step | None) -> StrategyResult:
        raw_detections = tuple(
            detection for detection in self.detector.detect(frame)
            if detection.confidence >= self.confidence_threshold
        )
        ingredient_ids = set(step.required_ingredients)
        if next_step:
            ingredient_ids.update(next_step.required_ingredients)
        detections = tuple(
            Detection(
                detection.entity,
                detection.confidence,
                detection.bbox,
                "ingredient" if detection.entity in ingredient_ids else "object",
            )
            for detection in raw_detections
        )
        present = frozenset(detection.entity for detection in detections)
        target = next_step.required_entities if next_step is not None else step.required_entities
        complete = bool(target) and target <= present
        missing = sorted(target - present)
        assessment = StepAssessment(
            step.id,
            "complete" if complete else "in_progress",
            min((detection.confidence for detection in detections if detection.entity in target), default=0.0),
            frame.id,
            frame.captured_at,
            self.name,
            ("next requirements visible",) if complete else ("missing: " + ", ".join(missing),),
            frame.source_seconds,
        )
        return StrategyResult(detections, assessment)
