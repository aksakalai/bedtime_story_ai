from __future__ import annotations

import json
from typing import Any, TypeVar

from .schemas import SchemaError

T = TypeVar("T")


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    if not raw_text or not raw_text.strip():
        raise SchemaError("Model returned an empty response.")

    text = raw_text.strip()
    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise SchemaError("Could not extract a JSON object from model output.")


def parse_json_response(raw_text: str, parser: type[T]) -> T:
    payload = extract_json_payload(raw_text)
    if not hasattr(parser, "from_dict"):
        raise TypeError("Parser must define from_dict.")
    return parser.from_dict(payload)  # type: ignore[return-value]
