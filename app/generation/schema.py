import json
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Answer:
    answer: str
    source_chunk_ids: list[str] = field(default_factory=list)
    supported: bool = False

    @staticmethod
    def unsupported(answer: str) -> "Answer":
        return Answer(answer=answer, source_chunk_ids=[], supported=False)


def parse_answer(raw: str) -> Optional[Answer]:
    cleaned = _clean_response(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned = _fix_trailing_commas(cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None
    if "answer" not in data or "source_chunk_ids" not in data or "supported" not in data:
        return None

    answer = str(data["answer"])
    chunk_ids = list(data["source_chunk_ids"])
    supported = data["supported"]
    if isinstance(supported, str):
        supported = supported.lower() == "true"
    elif not isinstance(supported, bool):
        return None

    return Answer(answer=answer, source_chunk_ids=chunk_ids, supported=supported)


def _clean_response(raw: str) -> str:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    brace = raw.find("{")
    if brace != -1:
        raw = raw[brace:]
    depth = 0
    end = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end != -1:
        raw = raw[:end]
    return raw


def _fix_trailing_commas(text: str) -> str:
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*\]", "]", text)
    return text
