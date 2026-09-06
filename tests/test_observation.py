import io
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from kitchen_assistant.assistant import CookingAssistant
from kitchen_assistant.observer import OllamaObserver, parse_answers
from kitchen_assistant.queries import Query
from kitchen_assistant.state import Answer, Frame, KitchenState, Observation


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.queries = (Query("onion.state", "onion", "state", ("chopped", "unknown")),)
        self.data = {
            "answers": [{"id": "onion.state", "value": "chopped", "confidence": 0.9}],
            "unknown": [], "anomalies": [],
        }

    def test_ollama_observation_reaches_state(self):
        response = {"message": {"content": json.dumps(self.data)}, "done_reason": "stop"}
        with patch("kitchen_assistant.observer.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
            assistant = CookingAssistant(OllamaObserver())
            observation = assistant.observe(Frame("frame-1", self.now, b"image"), self.queries)

        payload = json.loads(request.call_args.args[0].data)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["images"], ["aW1hZ2U="])
        self.assertEqual(observation.answers[0].value, "chopped")
        evidence = assistant.state.fields["onion.state"]
        self.assertEqual((evidence.frame_id, evidence.captured_at), ("frame-1", self.now))

    def test_invalid_model_result_does_not_update_state(self):
        self.data["answers"][0]["value"] = "step_completed"
        response = {"message": {"content": json.dumps(self.data)}}
        assistant = CookingAssistant(OllamaObserver())
        with patch("kitchen_assistant.observer.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaises(ValueError):
                assistant.observe(Frame("frame-1", self.now, b"image"), self.queries)
        self.assertEqual(assistant.state.fields, {})

    def test_duplicate_answers_and_invalid_confidence_are_rejected(self):
        self.data["answers"] *= 2
        with self.assertRaises(ValueError):
            parse_answers(self.data, self.queries)
        self.data["answers"] = self.data["answers"][:1]
        for confidence in (True, float("nan"), -0.1, 1.1):
            self.data["answers"][0]["confidence"] = confidence
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                parse_answers(self.data, self.queries)

    def test_unknown_replaces_previous_evidence_and_old_frames_are_ignored(self):
        state = KitchenState()
        old = Observation("old", self.now, (Answer("onion.state", "chopped", 0.9),))
        new = Observation("new", self.now + timedelta(seconds=1), (Answer("onion.state", "unknown", 0.1),))
        for observation in (old, new, old):
            state.update(observation)
        self.assertEqual(state.fields["onion.state"].value, "unknown")
        self.assertEqual(state.fields["onion.state"].frame_id, "new")

    def test_expired_and_missing_fields_are_unknown(self):
        state = KitchenState()
        state.update(Observation("frame-1", self.now, (Answer("onion.state", "chopped", 0.9),)))
        self.assertEqual(state.value("onion.state", now=self.now, max_age=timedelta(seconds=3)), "chopped")
        self.assertEqual(state.value("onion.state", now=self.now + timedelta(seconds=4), max_age=timedelta(seconds=3)), "unknown")
        self.assertEqual(state.value("pan.present", now=self.now, max_age=timedelta(seconds=3)), "unknown")


if __name__ == "__main__":
    unittest.main()
