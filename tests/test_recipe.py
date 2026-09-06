import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kitchen_assistant.recipe import Recipe, RecipeSession
from kitchen_assistant.state import Answer, KitchenState, Observation


RECIPE = Path(__file__).resolve().parents[1] / "recipes/creamy-chicken-mushrooms.json"


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.recipe = Recipe.load(RECIPE)
        self.session = RecipeSession(self.recipe, self.now, index=4)
        self.state = KitchenState()

    def observe(self, seconds, value="pan", frame_id=None, confidence=0.9):
        observation = Observation(
            frame_id or str(seconds), self.now + timedelta(seconds=seconds),
            (Answer("mushroom.location", value, confidence),),
        )
        self.state.update(observation)
        self.session.observe(observation, self.state)
        return self.session.recommend(observation.captured_at)

    def test_two_frames_suggest_next_step_without_advancing(self):
        self.assertEqual(self.observe(1).status, "observe")
        result = self.observe(4)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.suggested_step, "saute_mushrooms")
        self.assertEqual(self.session.current_step.id, "add_mushrooms")
        self.session.confirm(self.now + timedelta(seconds=5))
        self.assertEqual(self.session.current_step.id, "saute_mushrooms")
        self.assertEqual(self.session.recommend(self.now + timedelta(seconds=5)).status, "confirm")

    def test_duplicate_and_pre_entry_frames_do_not_count(self):
        self.observe(-1)
        self.observe(1, frame_id="same")
        self.assertNotEqual(self.observe(2, frame_id="same").status, "ready")

    def test_unknown_and_contradiction_cancel_ready_recommendation(self):
        for value in ("unknown", "plate"):
            with self.subTest(value=value):
                self.setUp()
                self.observe(1)
                self.assertEqual(self.observe(4).status, "ready")
                self.assertNotEqual(self.observe(7, value=value).status, "ready")

    def test_low_confidence_and_expired_votes_do_not_count(self):
        self.observe(1, confidence=0.2)
        self.assertNotEqual(self.observe(4).status, "ready")
        self.assertNotEqual(self.observe(40).status, "ready")
        self.assertEqual(self.session.recommend(self.now + timedelta(seconds=80)).status, "confirm")

    def test_previous_step_votes_are_cleared_on_confirmation(self):
        self.observe(1)
        self.observe(4)
        self.session.confirm(self.now + timedelta(seconds=5))
        self.session.confirm(self.now + timedelta(seconds=6))
        self.assertEqual(self.session.current_step.id, "add_onion")
        self.assertEqual(self.session.recommend(self.now + timedelta(seconds=6)).status, "confirm")

    def test_recipe_finishes_only_after_final_confirmation(self):
        self.session.index = len(self.recipe.steps) - 1
        self.assertEqual(self.session.recommend(self.now).status, "confirm")
        self.session.confirm(self.now)
        self.assertEqual(self.session.recommend(self.now).status, "complete")


if __name__ == "__main__":
    unittest.main()
