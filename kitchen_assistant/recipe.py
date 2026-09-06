"""A small, supervised recipe flow built on fresh kitchen observations."""

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .queries import Query
from .state import KitchenState, Observation


@dataclass(frozen=True, slots=True)
class Step:
    id: str
    instruction: str
    verification: str
    expected_after: dict[str, str]
    queries: tuple[Query, ...]


@dataclass(frozen=True, slots=True)
class Recipe:
    title: str
    steps: tuple[Step, ...]

    @classmethod
    def load(cls, path: Path) -> "Recipe":
        data = json.loads(path.read_text(encoding="utf-8"))
        steps = tuple(
            Step(
                row["id"], row["instruction"], row["verification"],
                row.get("expected_after", {}),
                tuple(Query(q["id"], q["entity"], q["ask"], tuple(q["allowed"]))
                      for q in row.get("queries", [])),
            )
            for row in data["steps"]
        )
        if not steps or len({step.id for step in steps}) != len(steps):
            raise ValueError("A recipe needs steps with unique IDs")
        for step in steps:
            if step.verification not in {"vision", "user_confirmation"}:
                raise ValueError(f"Unsupported verification: {step.verification}")
            queries = {q.id: q for q in step.queries}
            if step.verification == "vision" and not step.expected_after:
                raise ValueError(f"Visual step {step.id} needs expected_after")
            for key, value in step.expected_after.items():
                if key not in queries or value == "unknown" or value not in queries[key].allowed:
                    raise ValueError(f"Expected value for {key} must belong to its query")
        return cls(data["title"], steps)


@dataclass(frozen=True, slots=True)
class Recommendation:
    current_step: str | None
    suggested_step: str | None
    instruction: str
    status: str
    reason: str


@dataclass(slots=True)
class RecipeSession:
    recipe: Recipe
    started_at: datetime
    index: int = 0
    confidence_threshold: float = 0.7
    max_age: timedelta = timedelta(seconds=30)
    _votes: deque[tuple[str, datetime, bool]] = field(default_factory=lambda: deque(maxlen=3))
    _last_observed_at: datetime | None = None
    _known: bool = False

    @property
    def current_step(self) -> Step | None:
        return self.recipe.steps[self.index] if self.index < len(self.recipe.steps) else None

    def observe(self, observation: Observation, state: KitchenState) -> None:
        step = self.current_step
        if step is None or step.verification != "vision":
            return
        if observation.captured_at < self.started_at:
            return
        if self._last_observed_at and observation.captured_at <= self._last_observed_at:
            return
        if any(frame_id == observation.frame_id for frame_id, _, _ in self._votes):
            return
        self._last_observed_at = observation.captured_at
        evidence = [state.fields.get(key) for key in step.expected_after]
        self._known = all(
            item is not None and item.frame_id == observation.frame_id
            and item.value != "unknown" and item.confidence >= self.confidence_threshold
            for item in evidence
        )
        matches = self._known and all(
            state.fields[key].value == value for key, value in step.expected_after.items()
        )
        self._votes.append((observation.frame_id, observation.captured_at, matches))

    def recommend(self, now: datetime) -> Recommendation:
        step = self.current_step
        if step is None:
            return Recommendation(None, None, "Recipe checklist complete.", "complete", "All steps were confirmed.")
        if step.verification == "user_confirmation":
            return Recommendation(step.id, step.id, step.instruction, "confirm", "This action needs your confirmation.")
        votes = [matches for _, captured_at, matches in self._votes if now - captured_at <= self.max_age]
        # Require the latest observation to agree as well as two of the last three.
        if votes and votes[-1] and sum(votes) >= 2:
            next_step = self.recipe.steps[self.index + 1] if self.index + 1 < len(self.recipe.steps) else None
            return Recommendation(
                step.id, next_step.id if next_step else None,
                next_step.instruction if next_step else "Confirm the final step.",
                "ready", "Two fresh frames match. Confirm the current step before continuing.",
            )
        if not votes or not self._known:
            return Recommendation(step.id, step.id, step.instruction, "confirm", "Visual evidence is missing, uncertain or stale.")
        return Recommendation(step.id, step.id, step.instruction, "observe", "Waiting for two matching fresh frames.")

    def confirm(self, now: datetime) -> None:
        """Called by the user; model answers never advance the recipe."""
        if self.current_step is not None:
            self.index += 1
            self.started_at = now
            self._votes.clear()
            self._last_observed_at = None
            self._known = False
