import json
import re
from typing import Any, Dict, Optional

import requests


THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class InferenceError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        timeout_seconds: float,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key or "EMPTY"
        self.session = requests.Session()

    def healthcheck(self) -> None:
        health_base = self.api_base[:-3] if self.api_base.endswith("/v1") else self.api_base
        response = self.session.get(
            f"{health_base}/health",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def create_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not enable_thinking:
            # vLLM's OpenAI-compatible server expects `chat_template_kwargs` at
            # the top level; it ignores OpenAI-python's `extra_body` wrapper.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        response = self.session.post(
            f"{self.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"unexpected response payload: {payload}") from exc


# ---------------------------------------------------------------------------
# Robust JSON helpers
# ---------------------------------------------------------------------------

# LaTeX commands like \frac, \det, \Delta — the model often emits a single
# backslash, which is illegal in JSON (e.g. \f is the form-feed escape).
_LATEX_BSLASH_RE = re.compile(r"(?<!\\)\\([a-zA-Z]+)")


def _maybe_fix_latex_backslashes(text: str) -> str:
    """Escape lone backslashes that precede letters (LaTeX commands)."""
    return _LATEX_BSLASH_RE.sub(lambda m: "\\\\" + m.group(1), text)


# Lines that start a JSON string value may contain unescaped double-quotes
# inside Greek-guillemet quotations:  «"Foo": bar»
# Strategy: after the first ":"  (the JSON key-value separator), scan for
# «...» spans and backslash-escape every " found inside them.
_GUILLEMET_SPAN_RE = re.compile(r"\u00ab.*?\u00bb", re.DOTALL)


def _escape_quotes_in_guillemets(text: str) -> str:
    """Backslash-escape every double-quote inside \u00ab...\u00bb spans."""

    def _fix_span(m: re.Match) -> str:
        return m.group(0).replace('"', '\\"')

    return _GUILLEMET_SPAN_RE.sub(_fix_span, text)


def _robust_parse_json_object(json_text: str) -> Dict[str, Any]:
    """Try to parse *json_text* as a JSON object, applying fix-ups on failure."""
    # Strategy 1 – raw parse
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2 – escape LaTeX backslashes
    try:
        return json.loads(_maybe_fix_latex_backslashes(json_text))
    except json.JSONDecodeError:
        pass

    # Strategy 3 – fix unescaped quotes inside «...», then LaTeX, then parse
    try:
        fixed = _escape_quotes_in_guillemets(json_text)
        fixed = _maybe_fix_latex_backslashes(fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 4 – combine both in opposite order
    try:
        fixed = _maybe_fix_latex_backslashes(json_text)
        fixed = _escape_quotes_in_guillemets(fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise InferenceError(f"failed to parse model JSON after fix-ups: {json_text[:500]}")


def parse_qa_json(raw_content: str) -> Dict[str, Any]:
    cleaned = THINK_TAG_PATTERN.sub("", raw_content).strip()
    match = JSON_OBJECT_PATTERN.search(cleaned)
    if not match:
        raise InferenceError(f"model did not return a JSON object: {cleaned[:500]}")

    parsed = _robust_parse_json_object(match.group(0))

    question = parsed.get("question")
    answer = parsed.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise InferenceError(f"missing question in model output: {str(parsed)[:200]}")
    if not isinstance(answer, str) or not answer.strip():
        raise InferenceError(f"missing answer in model output: {str(parsed)[:200]}")

    return {
        "question": question.strip(),
        "answer": answer.strip(),
    }
