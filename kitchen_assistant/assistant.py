"""One observation at a time; recipe decisions belong outside perception."""

from dataclasses import dataclass, field
from typing import Protocol

from .queries import Query
from .recipe import RecipeSession
from .state import Frame, KitchenState, Observation


class Observer(Protocol):
    def observe(self, frame: Frame, queries: tuple[Query, ...]) -> Observation: ...


@dataclass(slots=True)
class CookingAssistant:
    observer: Observer
    state: KitchenState = field(default_factory=KitchenState)
    session: RecipeSession | None = None

    def observe(self, frame: Frame, queries: tuple[Query, ...]) -> Observation:
        observation = self.observer.observe(frame, queries)
        self.state.update(observation)
        if self.session is not None:
            self.session.observe(observation, self.state)
        return observation
