"""Async OpenAI-compatible client built on aiohttp.

Provides the same logical interface as `openai_client.OpenAICompatibleClient`
but every network call is non-blocking, so a single process can keep many
requests in flight concurrently.
"""

import logging
from typing import Any, Dict, Optional

import aiohttp

from gsdg.openai_client import InferenceError, parse_qa_json  # pure helpers, no I/O

LOGGER = logging.getLogger("gsdg.async_client")


class AsyncOpenAICompatibleClient:
    """Non-blocking client for an OpenAI-compatible vLLM endpoint."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        api_base: str,
        model: str,
        timeout_seconds: float,
        api_key: Optional[str] = None,
    ) -> None:
        self._session = session
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._auth_headers = {
            "Authorization": f"Bearer {api_key or 'EMPTY'}",
            "Content-Type": "application/json",
        }

    # -- healthcheck --------------------------------------------------------

    async def healthcheck(self) -> None:
        health_base = (
            self.api_base[:-3] if self.api_base.endswith("/v1") else self.api_base
        )
        url = f"{health_base}/health"
        async with self._session.get(url, timeout=self.timeout) as resp:
            resp.raise_for_status()

    # -- chat completion ----------------------------------------------------

    async def create_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        url = f"{self.api_base}/chat/completions"
        async with self._session.post(
            url,
            headers=self._auth_headers,
            json=payload,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"unexpected response payload: {body}") from exc
