import unittest

from kitchen_assistant.evaluation import ExpectedTransition, score_transitions


class EvaluationTests(unittest.TestCase):
    def test_scores_correct_premature_late_missed_and_unexpected(self):
        expected = (
            ExpectedTransition("a", 9, 10, 11),
            ExpectedTransition("b", 19, 20, 21),
            ExpectedTransition("c", 29, 30, 31),
            ExpectedTransition("d", 39, 40, 41),
        )
        events = [
            {"kind": "completed", "step_id": "a", "source_seconds": 10},
            {"kind": "completed", "step_id": "b", "source_seconds": 15},
            {"kind": "completed", "step_id": "c", "source_seconds": 35},
            {"kind": "completed", "step_id": "extra", "source_seconds": 50},
        ]
        result = score_transitions(events, expected)
        summary = result["summary"]
        self.assertEqual((summary["correct"], summary["premature"], summary["late"], summary["missed"]), (1, 1, 1, 1))
        self.assertEqual(result["unexpected_step_ids"], ["extra"])


if __name__ == "__main__":
    unittest.main()

