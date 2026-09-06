"""Ollama transport and validation of untrusted model answers."""

import base64
import json
import math
from dataclasses import asdict, dataclass
from urllib.request import Request, urlopen

from .queries import Query
from .state import Answer, Frame, Observation


def response_schema(queries: tuple[Query, ...]) -> dict:
    ids = [query.id for query in queries]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answers", "unknown", "anomalies"],
        "properties": {
            "answers": {
                "type": "array", "minItems": len(queries), "maxItems": len(queries),
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "value", "confidence"],
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "value": {"type": "string", "enum": sorted({v for q in queries for v in q.allowed})},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "unknown": {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": ids}},
            "anomalies": {"type": "array", "items": {"type": "string"}},
        },
    }


def parse_answers(data: dict, queries: tuple[Query, ...]) -> tuple[Answer, ...]:
    if set(data) != {"answers", "unknown", "anomalies"}:
        raise ValueError("Unexpected observation fields")
    expected = {query.id: query for query in queries}
    answers = tuple(Answer(**row) for row in data["answers"])
    if len(answers) != len(expected) or {a.id for a in answers} != set(expected):
        raise ValueError("Expected exactly one answer per query")
    for answer in answers:
        if answer.value not in expected[answer.id].allowed:
            raise ValueError(f"Invalid value for {answer.id}: {answer.value}")
        if (type(answer.confidence) not in (int, float)
                or not math.isfinite(answer.confidence) or not 0 <= answer.confidence <= 1):
            raise ValueError(f"Invalid confidence for {answer.id}")
    if sorted(data["unknown"]) != sorted(a.id for a in answers if a.value == "unknown"):
        raise ValueError("Unknown IDs must agree with answers")
    if not isinstance(data["anomalies"], list) or not all(isinstance(a, str) for a in data["anomalies"]):
        raise ValueError("Anomalies must be a list of strings")
    return answers


@dataclass(slots=True)
class OllamaObserver:
    model: str = "qwen3-vl:2b-instruct"
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 300

    def observe(self, frame: Frame, queries: tuple[Query, ...]) -> Observation:
        schema = response_schema(queries)
        prompt = (
            "Observe this kitchen image. Answer every query exactly once using its allowed values. "
            "If the ingredient is absent use not_visible when allowed. If uncertain use unknown. "
            "The unknown array must contain exactly the full query ids whose answer value is unknown; "
            "if none, use []. Never put entity names in unknown. Do not infer recipe completion. "
            "Anomalies means visible spills or unusual hazards, not absent ingredients or commentary; "
            "normally use []. JSON only. Queries: "
            + json.dumps([asdict(query) for query in queries])
            + " Response schema: " + json.dumps(schema)
        )
        payload = {
            "model": self.model, "stream": False, "format": schema,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(frame.image).decode()]}],
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512},
            "keep_alive": "5m",
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if result.get("done_reason") == "length":
            raise ValueError("Model response was truncated")
        data = json.loads(result["message"]["content"])
        return Observation(frame.id, frame.captured_at, parse_answers(data, queries), tuple(data["anomalies"]))
