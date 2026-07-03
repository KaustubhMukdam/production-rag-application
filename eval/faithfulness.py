import json
import re
import requests

OLLAMA_JUDGE_MODEL = "llama3.2:3b"


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
                return text[start : i + 1]
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


class FaithfulnessScorer:
    def _call_ollama(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": OLLAMA_JUDGE_MODEL, "prompt": prompt, "temperature": 0, "stream": False},
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()["response"] or ""
            except Exception:
                if attempt < 2:
                    import time
                    time.sleep(1)
        return ""

    def _call(self, prompt: str) -> str:
        return self._call_ollama(prompt)

    def _extract_claims(self, answer: str) -> list[str]:
        prompt = (
            "Return ONLY a JSON array of strings. No other text.\n\n"
            f"Decompose: {answer}"
        )
        result = self._call(prompt)
        parsed = _parse_json(result)
        if parsed:
            return parsed
        return [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 10]

    def _verify_claims(self, claims: list[str], contexts: list[str]) -> list[bool]:
        ctx = "\n\n".join(contexts)
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        prompt = (
            f"Context:\n{ctx}\n\n"
            f"Claims:\n{numbered}\n\n"
            "Return ONLY a JSON array of booleans: true if supported by context, false if not. "
            "Example: [true, false, true]"
        )
        result = self._call(prompt)
        parsed = _parse_json(result)
        if parsed:
            return parsed
        # fallback: extract verdict per claim from English text
        verdicts = []
        for i in range(len(claims)):
            m = re.search(rf'{i+1}[.)].*?(\bTrue\b|\bFalse\b|\bSupported\b|\bUnsupported\b|\bYes\b|\bNo\b)', result, re.DOTALL | re.IGNORECASE)
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
