"""Recipe loading plus the preserved supervised recommendation session."""

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .queries import Query
from .state import KitchenState, Observation


def _identifiers(values: object, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{field_name} must be a list of non-empty identifiers")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class Step:
    id: str
    name: str
    description: str
    required_objects: tuple[str, ...] = ()
    required_ingredients: tuple[str, ...] = ()
    verification: str = "automatic"
    expected_after: dict[str, str] = field(default_factory=dict)
    queries: tuple[Query, ...] = ()

    @property
    def instruction(self) -> str:
        """Backward-compatible name used by the supervised benchmark."""
        return self.description

    @property
    def required_entities(self) -> frozenset[str]:
        return frozenset((*self.required_objects, *self.required_ingredients))


@dataclass(frozen=True, slots=True)
class Recipe:
    title: str
    steps: tuple[Step, ...]

    @classmethod
    def load(cls, path: Path) -> "Recipe":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("steps")
        if not isinstance(data.get("title"), str) or not isinstance(rows, list):
            raise ValueError("Recipe needs a title and a list of steps")
        steps = tuple(cls._load_step(row) for row in rows)
        if not steps or len({step.id for step in steps}) != len(steps):
            raise ValueError("A recipe needs steps with unique IDs")
        return cls(data["title"], steps)

    @staticmethod
    def _load_step(row: dict) -> Step:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Every recipe step needs a string id")
        description = row.get("description", row.get("instruction"))
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Step {row['id']} needs a description")
        name = row.get("name", row["id"].replace("_", " ").title())
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Step {row['id']} needs a name")
        queries = tuple(
            Query(query["id"], query["entity"], query["ask"], tuple(query["allowed"]))
            for query in row.get("queries", [])
        )
        verification = row.get("verification", "automatic")
        if verification not in {"automatic", "vision", "user_confirmation"}:
            raise ValueError(f"Unsupported verification: {verification}")
        expected_after = row.get("expected_after", {})
        query_by_id = {query.id: query for query in queries}
        if verification == "vision" and not expected_after:
            raise ValueError(f"Visual step {row['id']} needs expected_after")
        for key, value in expected_after.items():
            if key not in query_by_id or value == "unknown" or value not in query_by_id[key].allowed:
                raise ValueError(f"Expected value for {key} must belong to its query")
        required_objects = _identifiers(row.get("required_objects"), "required_objects")
        required_ingredients = _identifiers(row.get("required_ingredients"), "required_ingredients")
        overlap = set(required_objects) & set(required_ingredients)
        if overlap:
            raise ValueError(f"Step {row['id']} classifies entities as both object and ingredient: {sorted(overlap)}")
        return Step(
            id=row["id"],
            name=name,
            description=description,
            required_objects=required_objects,
            required_ingredients=required_ingredients,
            verification=verification,
            expected_after=expected_after,
            queries=queries,
        )

    def validate_yolo_path(self) -> None:
        """Reject adjacent steps that presence alone cannot distinguish."""
        for current, following in zip(self.steps, self.steps[1:]):
            if not following.required_entities:
                raise ValueError(f"YOLO path needs requirements for step {following.id}")
            markers = following.required_entities - current.required_entities
            if not markers:
                raise ValueError(
                    f"YOLO cannot distinguish {current.id} from {following.id}: "
                    "the following step needs a newly appearing entity"
                )


@dataclass(frozen=True, slots=True)
class Recommendation:
    current_step: str | None
    suggested_step: str | None
    instruction: str
    status: str
    reason: str


@dataclass(slots=True)
class RecipeSession:
    """Legacy supervised session retained for the frozen recipe benchmark."""

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
            item is not None
            and item.frame_id == observation.frame_id
            and item.value != "unknown"
            and item.confidence >= self.confidence_threshold
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
        if votes and votes[-1] and sum(votes) >= 2:
            following = self.recipe.steps[self.index + 1] if self.index + 1 < len(self.recipe.steps) else None
            return Recommendation(
                step.id,
                following.id if following else None,
                following.instruction if following else "Confirm the final step.",
                "ready",
                "Two fresh frames match. Confirm the current step before continuing.",
            )
        if not votes or not self._known:
            return Recommendation(step.id, step.id, step.instruction, "confirm", "Visual evidence is missing, uncertain or stale.")
        return Recommendation(step.id, step.id, step.instruction, "observe", "Waiting for two matching fresh frames.")

    def confirm(self, now: datetime) -> None:
        if self.current_step is not None:
            self.index += 1
            self.started_at = now
            self._votes.clear()
            self._last_observed_at = None
            self._known = False
