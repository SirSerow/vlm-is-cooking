"""Verify invalid observations cannot pass the VLM boundary."""
import copy
import unittest
from vlm_smoke import validate


class ValidationTests(unittest.TestCase):
    def test_valid_and_unknown(self):
        validate(self.result(), self.queries())
        result = self.result()
        result["answers"][0]["value"] = "unknown"
        result["unknown"] = ["mushroom.state"]
        validate(result, self.queries())

    @staticmethod
    def queries():
        return [{"id": "mushroom.state", "allowed": ["sliced", "unknown"]}]

    @staticmethod
    def result():
        return {"answers": [{"id": "mushroom.state", "value": "sliced", "confidence": 0.9}],
                "unknown": [], "anomalies": []}

    def test_rejects_invalid_observations(self):
        bad = []
        for field, value in [("id", "step_completed"), ("value", "cooked"),
                             ("confidence", True), ("confidence", float("nan")), ("confidence", 1.1)]:
            result = self.result()
            result["answers"][0][field] = value
            bad.append(result)
        result = self.result()
        result["answers"] *= 2
        bad.append(result)
        result = self.result()
        result["answers"] = []
        bad.append(result)
        result = self.result()
        result["unknown"] = ["mushroom.state"]
        bad.append(result)
        result = self.result()
        result["step_completed"] = True
        bad.append(result)
        for result in bad:
            with self.subTest(result=result), self.assertRaises(ValueError):
                validate(copy.deepcopy(result), self.queries())


if __name__ == "__main__":
    unittest.main()
