"""Frames, observations, detections, and completion assessments."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class Frame:
    id: str
    captured_at: datetime
    image: bytes
    source_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    id: str
    value: str
    confidence: float


@dataclass(frozen=True, slots=True)
class Observation:
    frame_id: str
    captured_at: datetime
    answers: tuple[Answer, ...]
    anomalies: tuple[str, ...] = ()
    source: str = "vlm"


@dataclass(frozen=True, slots=True)
class Evidence:
    value: str
    confidence: float
    frame_id: str
    captured_at: datetime
    source: str


@dataclass(slots=True)
class KitchenState:
    fields: dict[str, Evidence] = field(default_factory=dict)

    def update(self, observation: Observation) -> None:
        for answer in observation.answers:
            previous = self.fields.get(answer.id)
            if previous and previous.captured_at >= observation.captured_at:
                continue
            self.fields[answer.id] = Evidence(
                answer.value,
                answer.confidence,
                observation.frame_id,
                observation.captured_at,
                observation.source,
            )

    def value(self, query_id: str, *, now: datetime, max_age: timedelta) -> str:
        evidence = self.fields.get(query_id)
        if evidence is None or now - evidence.captured_at > max_age:
            return "unknown"
        return evidence.value


@dataclass(frozen=True, slots=True)
class Detection:
    entity: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None
    kind: str = "object"


@dataclass(frozen=True, slots=True)
class StepAssessment:
    step_id: str
    status: str
    confidence: float
    frame_id: str
    captured_at: datetime
    source: str
    evidence: tuple[str, ...] = ()
    source_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"not_started", "in_progress", "complete", "uncertain"}:
            raise ValueError(f"Unsupported completion status: {self.status}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Assessment confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class StrategyResult:
    detections: tuple[Detection, ...]
    assessment: StepAssessment

    @property
    def present_entities(self) -> frozenset[str]:
        return frozenset(detection.entity for detection in self.detections)
