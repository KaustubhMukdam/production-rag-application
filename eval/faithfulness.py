"""
Eval scorers — Phase 4.

Three Ollama-judge scorers:
  FaithfulnessScorer     — unchanged from Phase 3
  AnswerRelevancyScorer  — NEW: did the answer address the question?
  ContextPrecisionScorer — NEW: what fraction of retrieved chunks were useful?

All scorers use the shared _call_ollama / _parse_json helpers.
"""
import json
import re
import requests

OLLAMA_JUDGE_MODEL = "llama3.2:3b"

_SYSTEM_JSON = "You are a precise JSON generator. Output ONLY valid JSON arrays. Never explain."


def _call_ollama(prompt: str, model: str = OLLAMA_JUDGE_MODEL) -> str:
    for attempt in range(3):
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_JSON},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return (resp.json()["message"]["content"] or "").strip()
        except Exception:
            if attempt < 2:
                import time
                time.sleep(1)
    return ""


def _find_top_level_list(text: str) -> str | None:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def _parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    matched = _find_top_level_list(text)
    if matched:
        text = matched
    text = re.sub(r"//.*", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import ast
        try:
            text = text.replace("true", "True").replace("false", "False").replace("null", "None")
            return ast.literal_eval(text)
        except Exception:
            print(f"Failed to parse JSON: {repr(text)}")
            return []


# ---------------------------------------------------------------------------
# FaithfulnessScorer (unchanged)
# ---------------------------------------------------------------------------

class FaithfulnessScorer:
    def _extract_claims(self, answer: str) -> list[str]:
        prompt = (
            "Return ONLY a JSON array of strings, each string is one factual claim "
            "from the text below.\n\n"
            f"Text: {answer}\n\n"
            "JSON:"
        )
        result = _call_ollama(prompt)
        parsed = _parse_json(result)
        if parsed:
            return parsed
        return [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 10]

    def _verify_claims(self, claims: list[str], contexts: list[str]) -> list[bool]:
        ctx = "\n\n".join(contexts)
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        prompt = (
            "Return ONLY a JSON array of booleans: true if the claim is supported "
            "by the context, false if not.\n\n"
            f"Context:\n{ctx}\n\n"
            f"Claims:\n{numbered}\n\n"
            "Example: [true, false, true]\n\n"
            "JSON:"
        )
        result = _call_ollama(prompt)
        parsed = _parse_json(result)
        if parsed:
            return parsed
        verdicts = []
        for i in range(len(claims)):
            m = re.search(
                rf'{i+1}[.)].*?(\bTrue\b|\bFalse\b|\bSupported\b|\bUnsupported\b|\bYes\b|\bNo\b)',
                result, re.DOTALL | re.IGNORECASE,
            )
            if m:
                verdicts.append(m.group(1).lower() in ("true", "supported", "yes"))
            else:
                return []
        return verdicts

    def score(self, answer: str, contexts: list[str]) -> float:
        if not answer.strip():
            return 0.0
        claims = self._extract_claims(answer)
        if not claims:
            return 0.0
        verdicts = self._verify_claims(claims, contexts)
        if not verdicts:
            return 0.0
        return sum(1 for v in verdicts if v) / len(verdicts)


# ---------------------------------------------------------------------------
# AnswerRelevancyScorer (new)
# ---------------------------------------------------------------------------

class AnswerRelevancyScorer:
    """Score how well the answer addresses the question (0–1).

    Prompts the Ollama judge to rate relevance.  Falls back to 0.0 on any
    parse failure so the eval run never crashes.
    """

    def score(self, question: str, answer: str) -> float:
        """Return a relevancy score in [0.0, 1.0].

        Args:
            question: The original user question.
            answer:   The generated answer to evaluate.

        Returns:
            Float in [0.0, 1.0].  0.0 if the answer is empty or the judge
            fails to return parseable output.
        """
        if not answer.strip():
            return 0.0

        prompt = (
            "Rate how well the answer responds to the question on a scale of 0 to 1.\n"
            'Return ONLY a JSON object: {"score": <float between 0 and 1>}\n\n'
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "JSON:"
        )
        raw = _call_ollama(prompt)
        try:
            data = json.loads(raw)
            score = float(data.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# ContextPrecisionScorer (new)
# ---------------------------------------------------------------------------

class ContextPrecisionScorer:
    """Score what fraction of retrieved chunks were useful for answering.

    Simplified Ragas Context Precision: no ground-truth required.
    The Ollama judge decides per-chunk relevance.
    """

    def score(self, question: str, contexts: list[str]) -> float:
        """Return the fraction of contexts judged relevant in [0.0, 1.0].

        Args:
            question: The original user question.
            contexts: List of retrieved chunk texts.

        Returns:
            Float in [0.0, 1.0].  0.0 if contexts is empty or judge fails.
        """
        if not contexts:
            return 0.0

        # Truncate each chunk to 200 chars to keep the prompt manageable
        numbered = "\n".join(f"{i+1}. {c[:200]}" for i, c in enumerate(contexts))
        prompt = (
            "For each numbered chunk below, output true if it is useful for answering "
            "the question, false if not.\n"
            "Return ONLY a JSON array of booleans, one per chunk.\n\n"
            f"Question: {question}\n\n"
            f"Chunks:\n{numbered}\n\n"
            f"Example for {len(contexts)} chunk(s): "
            f"[{', '.join(['true'] * len(contexts))}]\n\n"
            "JSON:"
        )
        raw = _call_ollama(prompt)
        parsed = _parse_json(raw)

        if not parsed or not isinstance(parsed, list):
            return 0.0

        # Guard against the judge returning more verdicts than chunks
        verdicts = parsed[:len(contexts)]
        if not verdicts:
            return 0.0

        return sum(1 for v in verdicts if v) / len(contexts)
