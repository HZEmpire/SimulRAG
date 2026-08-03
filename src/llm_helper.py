"""One asynchronous LLM interface for all supported providers."""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any

from dotenv import load_dotenv

from .data_io import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


class LLMError(RuntimeError):
    pass


def parse_json_response(text: str) -> Any:
    """Extract the first JSON object or array from a model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    starts = sorted(
        index for marker in ("{", "[") if (index := cleaned.find(marker)) >= 0
    )
    for start in starts:
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Model did not return valid JSON: {text[:240]!r}")


class LLMClient:
    """Provider-neutral async text and JSON generation with bounded retries."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-5-nano",
        *,
        max_retries: int = 3,
        concurrency: int = 8,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(concurrency)
        self._client: Any = None
        self._openai_accepts_temperature: bool | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            from openai import AsyncOpenAI

            key = os.getenv("OPENAI_API_KEY")
            self._validate_key("OPENAI_API_KEY", key)
            self._client = AsyncOpenAI(api_key=key, timeout=120.0)
        elif self.provider in {"anthropic", "claude"}:
            from anthropic import AsyncAnthropic

            key = os.getenv("ANTHROPIC_API_KEY")
            self._validate_key("ANTHROPIC_API_KEY", key)
            self._client = AsyncAnthropic(api_key=key, timeout=120.0)
        elif self.provider in {"gemini", "google"}:
            from google import genai

            key = os.getenv("GEMINI_API_KEY") or os.getenv("Google_API_KEY")
            self._validate_key("GEMINI_API_KEY", key)
            self._client = genai.Client(api_key=key)
        else:
            raise ValueError(
                f"Unsupported provider {self.provider!r}; use openai, anthropic, or gemini"
            )
        return self._client

    @staticmethod
    def _validate_key(name: str, value: str | None) -> None:
        if not value:
            raise LLMError(f"{name} is not set")
        normalized = value.strip().strip('"').strip("'").upper()
        placeholders = ("ADD_", "YOUR_", "REPLACE_", "<", "...")
        if normalized.startswith(placeholders) or "YOUR_API_KEY" in normalized:
            raise LLMError(f"{name} still contains a placeholder value")

    async def _openai_complete(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        if self._openai_accepts_temperature is not False:
            kwargs["temperature"] = temperature
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["reasoning"] = {"effort": "minimal"}
        try:
            response = await client.responses.create(**kwargs)
        except Exception as exc:
            # Some reasoning models only support their default sampling settings.
            if "temperature" not in str(exc).lower() or "temperature" not in kwargs:
                raise
            kwargs.pop("temperature")
            self._openai_accepts_temperature = False
            response = await client.responses.create(**kwargs)
        else:
            if "temperature" in kwargs:
                self._openai_accepts_temperature = True
        text = getattr(response, "output_text", None)
        if not text:
            pieces: list[str] = []
            for output in getattr(response, "output", []):
                for content in getattr(output, "content", []):
                    value = getattr(content, "text", None)
                    if value:
                        pieces.append(value)
            text = "\n".join(pieces)
        if not text:
            raise LLMError("OpenAI returned an empty response")
        return text.strip()

    async def _anthropic_complete(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
        if not text:
            raise LLMError("Anthropic returned an empty response")
        return text.strip()

    async def _gemini_complete(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        from google.genai import types

        client = self._get_client()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        if not response.text:
            raise LLMError("Gemini returned an empty response")
        return response.text.strip()

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> str:
        last_error: Exception | None = None
        attempts = 0
        for attempt in range(self.max_retries):
            attempts = attempt + 1
            try:
                async with self.semaphore:
                    if self.provider == "openai":
                        return await self._openai_complete(
                            prompt, temperature, max_tokens
                        )
                    if self.provider in {"anthropic", "claude"}:
                        return await self._anthropic_complete(
                            prompt, temperature, max_tokens
                        )
                    return await self._gemini_complete(prompt, temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001 - provider SDK errors vary.
                last_error = exc
                if not self._is_retryable(exc):
                    break
                if attempt + 1 < self.max_retries:
                    delay = min(12.0, (2**attempt) + random.random())
                    print(
                        f"LLM call failed ({attempt + 1}/{self.max_retries}): "
                        f"{exc}; retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
        raise LLMError(
            f"LLM call failed after {attempts} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, LLMError) and any(
            marker in str(exc).lower()
            for marker in ("api_key", "api key", "placeholder", "unsupported provider")
        ):
            return False
        status_code = getattr(exc, "status_code", None)
        if status_code is not None and status_code not in {408, 409, 429}:
            return int(status_code) >= 500
        return True

    async def complete_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> Any:
        last_error: Exception | None = None
        current_prompt = prompt
        for attempt in range(self.max_retries):
            text = await self.complete(
                current_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                return parse_json_response(text)
            except ValueError as exc:
                last_error = exc
                current_prompt = (
                    prompt
                    + "\n\nYour previous response was not valid JSON. Return only the "
                    "requested JSON structure with no Markdown or commentary."
                )
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        raise LLMError(f"Could not parse JSON response: {last_error}") from last_error
