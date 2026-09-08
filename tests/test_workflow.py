import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kitchen_assistant.recipe import Recipe, Step
from kitchen_assistant.state import StepAssessment
from kitchen_assistant.workflow import WorkflowPhase, WorkflowSession


def recipe() -> Recipe:
    return Recipe(
        "Test",
        (
            Step("cut", "Cut", "Cut the onion.", ("knife",), ("onion",)),
            Step("cook", "Cook", "Cook in the pan.", ("pan",), ("onion",)),
            Step("serve", "Serve", "Serve it.", ("plate",), ("dish",)),
        ),
    )


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.session = WorkflowSession(recipe(), self.now)

    def assess(self, seconds: int, status: str, confidence: float = 0.9) -> bool:
        step = self.session.current_step
        return self.session.observe_completion(
            StepAssessment(
                step.id,
                status,
                confidence,
                f"assessment-{seconds}",
                self.now + timedelta(seconds=seconds),
                "test",
            )
        )

    def test_requirements_activate_after_two_distinct_matching_frames(self):
        present = frozenset({"knife", "onion"})
        self.assertFalse(self.session.observe_presence(present, frame_id="one", captured_at=self.now))
        self.assertTrue(
            self.session.observe_presence(present, frame_id="two", captured_at=self.now + timedelta(seconds=1))
        )
        self.assertEqual(self.session.phase, WorkflowPhase.ACTIVE)
        self.assertEqual(self.session.view(present).description, "Cut the onion.")

    def test_missing_requirement_keeps_description_hidden(self):
        self.session.observe_presence(frozenset({"knife"}), frame_id="one", captured_at=self.now)
        view = self.session.view(frozenset({"knife"}))
        self.assertIsNone(view.description)
        self.assertEqual(view.missing_ingredients, ("onion",))

    def test_incomplete_then_two_completions_advance_automatically(self):
        self.session.phase = WorkflowPhase.ACTIVE
        self.assess(1, "in_progress")
        self.assertFalse(self.assess(2, "complete"))
        self.assertTrue(self.assess(3, "complete"))
        self.assertEqual(self.session.current_step.id, "cook")
        self.assertEqual(self.session.phase, WorkflowPhase.WAITING_FOR_REQUIREMENTS)

    def test_prestaged_completion_does_not_skip_step(self):
        self.session.phase = WorkflowPhase.ACTIVE
        self.assertFalse(self.assess(1, "complete"))
        self.assertFalse(self.assess(2, "complete"))
        self.assertEqual(self.session.current_step.id, "cut")
        self.assess(3, "in_progress")
        self.assertFalse(self.assess(4, "complete"))
        self.assertTrue(self.assess(5, "complete"))

    def test_low_confidence_and_duplicate_frames_do_not_advance(self):
        self.session.phase = WorkflowPhase.ACTIVE
        self.assess(1, "in_progress")
        self.assess(2, "complete", 0.2)
        step = self.session.current_step
        duplicate = StepAssessment(step.id, "complete", 0.9, "same", self.now + timedelta(seconds=3), "test")
        self.session.observe_completion(duplicate)
        self.session.observe_completion(duplicate)
        self.assertEqual(self.session.current_step.id, "cut")

    def test_new_recipe_schema_and_yolo_validation(self):
        document = {
            "title": "Demo",
            "steps": [
                {"id": "a", "name": "A", "description": "First", "required_objects": ["knife"], "required_ingredients": ["onion"]},
                {"id": "b", "name": "B", "description": "Second", "required_objects": ["pan"], "required_ingredients": ["onion"]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = Recipe.load(path)
        loaded.validate_yolo_path()
        self.assertEqual(loaded.steps[0].name, "A")
        self.assertEqual(loaded.steps[1].required_objects, ("pan",))

    def test_yolo_validation_rejects_indistinguishable_steps(self):
        ambiguous = Recipe(
            "Ambiguous",
            (
                Step("a", "A", "A", ("pan",), ()),
                Step("b", "B", "B", ("pan",), ()),
            ),
        )
        with self.assertRaisesRegex(ValueError, "cannot distinguish"):
            ambiguous.validate_yolo_path()


if __name__ == "__main__":
    unittest.main()

