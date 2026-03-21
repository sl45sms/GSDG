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


def parse_qa_json(raw_content: str) -> Dict[str, Any]:
    cleaned = THINK_TAG_PATTERN.sub("", raw_content).strip()
    match = JSON_OBJECT_PATTERN.search(cleaned)
    if not match:
        raise InferenceError(f"model did not return a JSON object: {cleaned}")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InferenceError(f"failed to parse model JSON: {cleaned}") from exc

    question = parsed.get("question")
    answer = parsed.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise InferenceError(f"missing question in model output: {parsed}")
    if not isinstance(answer, str) or not answer.strip():
        raise InferenceError(f"missing answer in model output: {parsed}")

    return {
        "question": question.strip(),
        "answer": answer.strip(),
    }
