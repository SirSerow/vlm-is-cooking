import unittest
from datetime import datetime, timezone

from kitchen_assistant.completion import YoloPresenceStrategy, parse_completion
from kitchen_assistant.recipe import Step
from kitchen_assistant.state import Detection, Frame


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.frame = Frame("frame", self.now, b"image")
        self.current = Step("cut", "Cut", "Cut onion", ("knife",), ("onion",))
        self.following = Step("cook", "Cook", "Cook onion", ("pan",), ("onion",))

    def test_yolo_completes_when_all_next_requirements_are_visible(self):
        strategy = YoloPresenceStrategy(
            FakeDetector((Detection("pan", 0.9), Detection("onion", 0.8, kind="ingredient")))
        )
        result = strategy.analyze(self.frame, self.current, self.following)
        self.assertEqual(result.assessment.status, "complete")
        self.assertEqual(result.present_entities, frozenset({"pan", "onion"}))

    def test_yolo_waits_when_a_next_requirement_is_missing(self):
        strategy = YoloPresenceStrategy(FakeDetector((Detection("pan", 0.9),)))
        result = strategy.analyze(self.frame, self.current, self.following)
        self.assertEqual(result.assessment.status, "in_progress")
        self.assertIn("onion", result.assessment.evidence[0])

    def test_vlm_completion_parser_is_closed_to_recipe_entities(self):
        data = {
            "visible_objects": ["knife"],
            "visible_ingredients": ["onion"],
            "status": "in_progress",
            "confidence": 0.8,
            "evidence": ["onion is being cut"],
        }
        parsed = parse_completion(data, self.current)
        self.assertEqual(parsed[2], "in_progress")
        data["visible_objects"] = ["spoon"]
        with self.assertRaises(ValueError):
            parse_completion(data, self.current)


if __name__ == "__main__":
    unittest.main()

