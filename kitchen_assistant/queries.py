"""The closed questions that perception can answer."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Query:
    id: str
    entity: str
    ask: str
    allowed: tuple[str, ...]


def load_queries(path: Path) -> tuple[Query, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = tuple(
        Query(row["id"], row["entity"], row["ask"], tuple(row["allowed"]))
        for row in data["queries"]
    )
    if not queries or len({query.id for query in queries}) != len(queries):
        raise ValueError("Queries must have unique IDs and cannot be empty")
    for query in queries:
        if query.id != f"{query.entity}.{query.ask}":
            raise ValueError("Query IDs must follow entity.attribute")
        if "unknown" not in query.allowed or not all(isinstance(v, str) for v in query.allowed):
            raise ValueError("Query values must be strings and include unknown")
    return queries
