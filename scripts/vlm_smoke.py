"""Local Ollama image inference with closed-vocabulary response validation (stdlib only)."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:11434"


def api(route, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(URL + route, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate(result, queries):
    if set(result) != {"answers", "unknown", "anomalies"}:
        raise ValueError("Unexpected response keys")
    expected = {q["id"]: q for q in queries}
    seen = set()
    for answer in result["answers"]:
        if set(answer) != {"id", "value", "confidence"}:
            raise ValueError("Unexpected answer keys")
        key = answer["id"]
        if key not in expected or key in seen:
            raise ValueError("Unknown or duplicate query id")
        seen.add(key)
        if answer["value"] not in expected[key]["allowed"]:
            raise ValueError("Answer outside allowed vocabulary")
        confidence = answer["confidence"]
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
    if seen != set(expected):
        raise ValueError("Missing query answers")
    unknown = [a["id"] for a in result["answers"] if a["value"] == "unknown"]
    if not isinstance(result["unknown"], list) or sorted(result["unknown"]) != sorted(unknown):
        raise ValueError("Unknown list disagrees with answers")
    if not isinstance(result["anomalies"], list) or not all(isinstance(a, str) for a in result["anomalies"]):
        raise ValueError("Invalid anomalies")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--queries", type=Path, default=ROOT / "config/vlm_queries.v1.json")
    parser.add_argument("--model", default="qwen3-vl:2b-instruct")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/vlm-smoke/result.json")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    queries = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]
    if not queries or len({q["id"] for q in queries}) != len(queries):
        parser.error("Queries must have unique ids and cannot be empty")
    for q in queries:
        if not isinstance(q["allowed"], list) or "unknown" not in q["allowed"] or not all(isinstance(v, str) for v in q["allowed"]):
            parser.error("Each query needs a string vocabulary including unknown")
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "answers": {"type": "array", "minItems": len(queries), "maxItems": len(queries), "items": {
            "type": "object", "additionalProperties": False, "properties": {
                "id": {"type": "string", "enum": [q["id"] for q in queries]},
                "value": {"type": "string", "enum": sorted({v for q in queries for v in q["allowed"]})},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["id", "value", "confidence"]}},
        "unknown": {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": [q["id"] for q in queries]}},
        "anomalies": {"type": "array", "items": {"type": "string"}}},
        "required": ["answers", "unknown", "anomalies"]}
    image_bytes = args.image.read_bytes()
    prompt = ("Observe this kitchen image. Answer every query exactly once using its allowed values. "
              "If the ingredient is absent use not_visible when allowed. If uncertain use unknown. "
              "The unknown array must contain exactly the full query ids whose answer value is unknown; "
              "if none, use []. Never put entity names in unknown. Do not infer recipe completion. "
              "Anomalies means visible spills or unusual hazards, not absent ingredients or commentary; "
              "normally use []. JSON only. Queries: " + json.dumps(queries)
              + " Response schema: " + json.dumps(schema))
    report = {"image": str(args.image.resolve()), "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
              "model": args.model, "ollama": api("/api/version"), "models": api("/api/tags"),
              "queries": queries, "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.repeat):
        start = time.perf_counter()
        response = api("/api/chat", {"model": args.model, "stream": False, "format": schema,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(image_bytes).decode()]}],
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512}, "keep_alive": "5m"})
        run = {"wall_seconds": time.perf_counter() - start, "response": response, "valid": False}
        report["runs"].append(run)
        try:
            if response.get("done_reason") == "length":
                raise ValueError("Generation truncated")
            result = json.loads(response["message"]["content"])
            validate(result, queries)
            run["valid"] = True
        finally:
            report["loaded_models"] = api("/api/ps")
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"run": index + 1, "seconds": round(run["wall_seconds"], 2), "result": result}), flush=True)


if __name__ == "__main__":
    main()
