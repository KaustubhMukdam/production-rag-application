import json
import os
import re
import requests

JUDGE_MODEL = "llama-3.1-8b-instant"


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
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self._last_call = 0.0

    def _call(self, prompt: str) -> str:
        import time
        elapsed = time.time() - self._last_call
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        for attempt in range(3):
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
                )
                if resp.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.exceptions.HTTPError:
                time.sleep(2 ** (attempt + 1))
            except Exception as e:
                print(f"Exception on Groq API call: {e}")
                return ""
            finally:
                self._last_call = time.time()
        return ""

    def _extract_claims(self, answer: str) -> list[str]:
        prompt = (
            "Decompose the following answer into atomic claims (one fact per claim). "
            "Return a JSON list of strings. Do not explain.\n\n"
            f"Answer: {answer}\n\nClaims:"
        )
        result = self._call(prompt)
        return _parse_json(result)

    def _verify_claims(self, claims: list[str], contexts: list[str]) -> list[bool]:
        ctx = "\n\n".join(contexts)
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        prompt = (
            f"Context:\n{ctx}\n\n"
            f"Claims:\n{numbered}\n\n"
            "For each claim, determine if it is directly supported by the context. "
            "Return a JSON array of booleans: true if supported, false if not. "
            "Example: [true, false, true]\n\n"
            "Result:"
        )
        result = self._call(prompt)
        return _parse_json(result)

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
