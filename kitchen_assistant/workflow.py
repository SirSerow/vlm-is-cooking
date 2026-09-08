"""Automatic two-phase recipe workflow with temporal transition checks."""

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from .recipe import Recipe, Step
from .state import StepAssessment


class WorkflowPhase(StrEnum):
    WAITING_FOR_REQUIREMENTS = "waiting_for_requirements"
    ACTIVE = "active"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    kind: str
    step_id: str | None
    frame_id: str
    captured_at: datetime
    source_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class WorkflowView:
    recipe_title: str
    phase: str
    step_index: int
    step_id: str | None
    step_name: str | None
    description: str | None
    required_objects: tuple[str, ...]
    required_ingredients: tuple[str, ...]
    missing_objects: tuple[str, ...]
    missing_ingredients: tuple[str, ...]
    next_step_name: str | None
    next_required_objects: tuple[str, ...]
    next_required_ingredients: tuple[str, ...]
    assessment_status: str | None
    assessment_confidence: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class WorkflowSession:
    recipe: Recipe
    started_at: datetime
    index: int = 0
    confidence_threshold: float = 0.7
    max_age: timedelta = timedelta(seconds=30)
    required_matches: int = 2
    require_incomplete_transition: bool = True
    phase: WorkflowPhase = WorkflowPhase.WAITING_FOR_REQUIREMENTS
    events: list[WorkflowEvent] = field(default_factory=list)
    _presence_votes: deque[tuple[str, datetime, bool]] = field(default_factory=lambda: deque(maxlen=3))
    _completion_votes: deque[tuple[str, datetime, bool]] = field(default_factory=lambda: deque(maxlen=3))
    _seen_incomplete: bool = False
    _last_assessment: StepAssessment | None = None

    def __post_init__(self) -> None:
        if self.current_step is None:
            self.phase = WorkflowPhase.COMPLETE
        elif not self.current_step.required_entities:
            self.phase = WorkflowPhase.ACTIVE

    @property
    def current_step(self) -> Step | None:
        return self.recipe.steps[self.index] if self.index < len(self.recipe.steps) else None

    @property
    def next_step(self) -> Step | None:
        next_index = self.index + 1
        return self.recipe.steps[next_index] if next_index < len(self.recipe.steps) else None

    def observe_presence(
        self,
        present: frozenset[str],
        *,
        frame_id: str,
        captured_at: datetime,
        source_seconds: float | None = None,
    ) -> bool:
        """Record current-step readiness and return True only on activation."""
        step = self.current_step
        if step is None or self.phase != WorkflowPhase.WAITING_FOR_REQUIREMENTS:
            return False
        if captured_at < self.started_at or any(vote[0] == frame_id for vote in self._presence_votes):
            return False
        matches = step.required_entities <= present
        self._presence_votes.append((frame_id, captured_at, matches))
        votes = [match for _, at, match in self._presence_votes if captured_at - at <= self.max_age]
        if votes and votes[-1] and sum(votes) >= self.required_matches:
            self.phase = WorkflowPhase.ACTIVE
            self._completion_votes.clear()
            self._seen_incomplete = False
            self.events.append(WorkflowEvent("activated", step.id, frame_id, captured_at, source_seconds))
            return True
        return False

    def observe_completion(self, assessment: StepAssessment) -> bool:
        """Record completion evidence and automatically advance when stable."""
        step = self.current_step
        if step is None or self.phase != WorkflowPhase.ACTIVE or assessment.step_id != step.id:
            return False
        if assessment.captured_at < self.started_at:
            return False
        if any(vote[0] == assessment.frame_id for vote in self._completion_votes):
            return False
        self._last_assessment = assessment
        accepted = assessment.status == "complete" and assessment.confidence >= self.confidence_threshold
        guarded = self.require_incomplete_transition and self.next_step is not None
        if assessment.status in {"not_started", "in_progress"} and not self._seen_incomplete:
            self._seen_incomplete = True
            self._completion_votes.clear()
        if guarded and not self._seen_incomplete:
            return False
        self._completion_votes.append((assessment.frame_id, assessment.captured_at, accepted))
        votes = [
            match
            for _, at, match in self._completion_votes
            if assessment.captured_at - at <= self.max_age
        ]
        transition_seen = self._seen_incomplete or not guarded
        if votes and votes[-1] and sum(votes) >= self.required_matches and transition_seen:
            self._advance(assessment)
            return True
        return False

    def _advance(self, assessment: StepAssessment) -> None:
        completed = self.current_step
        self.events.append(
            WorkflowEvent(
                "completed",
                completed.id if completed else None,
                assessment.frame_id,
                assessment.captured_at,
                assessment.source_seconds,
            )
        )
        self.index += 1
        self.started_at = assessment.captured_at
        self._presence_votes.clear()
        self._completion_votes.clear()
        self._seen_incomplete = False
        self._last_assessment = None
        if self.current_step is None:
            self.phase = WorkflowPhase.COMPLETE
        elif self.current_step.required_entities:
            self.phase = WorkflowPhase.WAITING_FOR_REQUIREMENTS
        else:
            self.phase = WorkflowPhase.ACTIVE

    def view(self, present: frozenset[str] = frozenset()) -> WorkflowView:
        step = self.current_step
        following = self.next_step
        if step is None:
            return WorkflowView(
                self.recipe.title,
                self.phase,
                self.index,
                None,
                None,
                None,
                (),
                (),
                (),
                (),
                None,
                (),
                (),
                None,
                None,
            )
        missing_objects = tuple(entity for entity in step.required_objects if entity not in present)
        missing_ingredients = tuple(entity for entity in step.required_ingredients if entity not in present)
        assessment = self._last_assessment
        return WorkflowView(
            recipe_title=self.recipe.title,
            phase=self.phase,
            step_index=self.index,
            step_id=step.id,
            step_name=step.name,
            description=step.description if self.phase == WorkflowPhase.ACTIVE else None,
            required_objects=step.required_objects,
            required_ingredients=step.required_ingredients,
            missing_objects=missing_objects,
            missing_ingredients=missing_ingredients,
            next_step_name=following.name if self.phase == WorkflowPhase.ACTIVE and following else None,
            next_required_objects=following.required_objects if self.phase == WorkflowPhase.ACTIVE and following else (),
            next_required_ingredients=following.required_ingredients if self.phase == WorkflowPhase.ACTIVE and following else (),
            assessment_status=assessment.status if assessment else None,
            assessment_confidence=assessment.confidence if assessment else None,
        )
