"""Shared application controller for both completion strategies."""

from dataclasses import dataclass

from .completion import CompletionStrategy
from .outputs import Output
from .sources import FrameScheduler
from .workflow import WorkflowPhase, WorkflowSession


@dataclass(slots=True)
class CookingApplication:
    session: WorkflowSession
    scheduler: FrameScheduler
    strategy: CompletionStrategy
    output: Output

    def run(self, *, max_frames: int | None = None) -> None:
        processed = 0
        self.output.start()
        self.output.publish(None, self.session.view())
        try:
            for frame in self.scheduler:
                step = self.session.current_step
                if step is None:
                    break
                was_active = self.session.phase == WorkflowPhase.ACTIVE
                result = self.strategy.analyze(frame, step, self.session.next_step)
                self.session.observe_presence(
                    result.present_entities,
                    frame_id=frame.id,
                    captured_at=frame.captured_at,
                    source_seconds=frame.source_seconds,
                )
                # Do not complete a step on the same frame that activated it.
                if was_active:
                    self.session.observe_completion(result.assessment)
                self.output.publish(frame, self.session.view(result.present_entities), result)
                processed += 1
                if self.session.phase == WorkflowPhase.COMPLETE:
                    break
                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            self.scheduler.close()
            self.output.close()
