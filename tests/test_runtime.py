import unittest
from datetime import datetime, timedelta, timezone

from kitchen_assistant.recipe import Recipe, Step
from kitchen_assistant.runtime import CookingApplication
from kitchen_assistant.sources import FrameScheduler
from kitchen_assistant.state import Detection, Frame, StepAssessment, StrategyResult
from kitchen_assistant.workflow import WorkflowSession


class FakeReader:
    live = False

    def __init__(self, frames):
        self.frames = iter(frames)
        self.closed = False

    def read(self):
        return next(self.frames, None)

    def close(self):
        self.closed = True


class FakeStrategy:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def analyze(self, frame, step, next_step):
        self.calls += 1
        statuses = ["in_progress", "in_progress", "in_progress", "complete", "complete"]
        assessment = StepAssessment(
            step.id,
            statuses[self.calls - 1],
            0.9,
            frame.id,
            frame.captured_at,
            self.name,
        )
        return StrategyResult((Detection("knife", 0.9),), assessment)


class RecordingOutput:
    def __init__(self):
        self.views = []
        self.closed = False

    def start(self):
        pass

    def publish(self, frame, view, result=None):
        self.views.append(view)

    def close(self):
        self.closed = True


class RuntimeTests(unittest.TestCase):
    def test_application_uses_one_shared_controller_and_closes_resources(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        frames = [Frame(str(i), now + timedelta(seconds=i), b"image") for i in range(5)]
        reader = FakeReader(frames)
        output = RecordingOutput()
        recipe = Recipe("Demo", (Step("cut", "Cut", "Cut it", ("knife",), ()), Step("done", "Done", "Done", (), ())))
        session = WorkflowSession(recipe, now)
        app = CookingApplication(session, FrameScheduler(reader), FakeStrategy(), output)
        app.run(max_frames=5)
        self.assertEqual(session.current_step.id, "done")
        self.assertTrue(reader.closed)
        self.assertTrue(output.closed)
        self.assertGreaterEqual(len(output.views), 2)


if __name__ == "__main__":
    unittest.main()

