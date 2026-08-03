"""Cached and user-supplied simulator adapters."""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import httpx

from .llm_helper import LLMClient, parse_json_response
from .prompts import tool_call_prompt


class SimulatorError(RuntimeError):
    pass


class CachedSimulator:
    """Replay benchmark-generation evidence without invoking a simulator."""

    def __init__(self, evidence: dict[str, dict[str, str]]) -> None:
        self.evidence = evidence

    async def query(
        self,
        question: str,
        selected_claims: list[str],
        llm: LLMClient,
        handbook: str,
        temperature: float,
        max_retries: int,
    ) -> dict[str, Any]:
        del selected_claims, llm, handbook, temperature, max_retries
        try:
            item = self.evidence[question]
        except KeyError as exc:
            raise SimulatorError(
                "No cached simulator evidence for this question"
            ) from exc
        return {
            "source": "built-in-cache",
            "derived_quantitative_question": item["derived_quantitative_question"],
            "derived_quantitative_answer": item["derived_quantitative_answer"],
        }


class CustomSimulator:
    """Invoke a local JSON-over-stdio command or an HTTP JSON endpoint."""

    def __init__(self, target: str, timeout: float = 180.0) -> None:
        self.target = target
        self.timeout = timeout

    def _command(self) -> list[str]:
        path = Path(self.target).expanduser()
        if path.exists():
            path = path.resolve()
            if path.suffix == ".py":
                return [sys.executable, str(path)]
            return [str(path)]
        command = shlex.split(self.target)
        if not command:
            raise SimulatorError("Simulator command is empty")
        return command

    async def _run_subprocess(self, payload: dict[str, Any]) -> Any:
        command = self._command()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(
                    (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                ),
                timeout=self.timeout,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise SimulatorError(
                f"Simulator exceeded {self.timeout:.0f}s timeout"
            ) from exc
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise SimulatorError(
                f"Simulator exited with status {process.returncode}: {message}"
            )
        text = stdout.decode("utf-8", errors="replace").strip()
        if not text:
            raise SimulatorError("Simulator returned empty stdout")
        try:
            return parse_json_response(text)
        except ValueError:
            return text

    async def _run_http(self, payload: dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.target, json=payload)
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return response.text

    async def query(
        self,
        question: str,
        selected_claims: list[str],
        llm: LLMClient,
        handbook: str,
        temperature: float,
        max_retries: int,
    ) -> dict[str, Any]:
        base_prompt = tool_call_prompt(question, selected_claims, handbook)
        prompt = base_prompt
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                tool_call = await llm.complete_json(
                    prompt, temperature=temperature, max_tokens=800
                )
                if not isinstance(tool_call, dict):
                    raise SimulatorError("Tool call must be a JSON object")
                tool = tool_call.get("tool")
                arguments = tool_call.get("arguments")
                if not isinstance(tool, str) or not isinstance(arguments, dict):
                    raise SimulatorError(
                        "Tool call requires string 'tool' and object 'arguments'"
                    )
                if self.target.startswith(("http://", "https://")):
                    output = await self._run_http(tool_call)
                    source = "custom-http"
                else:
                    output = await self._run_subprocess(tool_call)
                    source = "custom-subprocess"
                return {
                    "source": source,
                    "tool_call": tool_call,
                    "simulator_output": output,
                }
            except Exception as exc:  # noqa: BLE001 - retry validation and execution errors.
                last_error = exc
                if attempt + 1 < max_retries:
                    prompt = (
                        base_prompt
                        + f"\n\nThe previous call failed validation or execution: {exc}. "
                        "Correct the function name, parameters, types, units, or ranges and try again."
                    )
                    await asyncio.sleep(1.0 + attempt)
        raise SimulatorError(
            f"Simulator call failed after {max_retries} attempts: {last_error}"
        ) from last_error
