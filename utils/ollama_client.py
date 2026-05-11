"""
utils/ollama_client.py -- Wrapper around the Ollama HTTP API.

BUG FIX #4: generate_json() now tries json.loads() directly first before
falling back to regex extraction. The original regex r'{.*}' with
re.DOTALL could match partial nested objects incorrectly.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import ollama

from config import config

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when Ollama API fails after retries."""


class OllamaClient:
    """
    Thin, retry-safe wrapper around the Ollama Python SDK.
    All agents import this module-level singleton.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self._client = ollama.Client(host=self.base_url)

    # ── Core generation ───────────────────────────────────────────────────

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """Send prompt to Ollama and return response text. Retries on error."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat(
                    model=model,
                    messages=messages,
                    options={"temperature": temperature, "num_predict": 4096},
                )
                return response["message"]["content"].strip()
            except Exception as exc:
                logger.warning(
                    "Ollama attempt %d/%d failed (%s): %s",
                    attempt, max_retries, model, exc,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                else:
                    raise OllamaError(
                        f"Ollama failed after {max_retries} attempts: {exc}"
                    ) from exc
        return ""

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: str = "",
        fallback: Any = None,
    ) -> Any:
        """
        Like generate() but returns parsed JSON.

        BUG FIX #4: Strategy:
          1. Try json.loads() on the raw response directly (handles clean responses).
          2. Strip markdown fences and retry json.loads().
          3. Regex-extract the outermost {...} or [...] and try json.loads().
          4. Return fallback if all attempts fail.

        The original single-pass regex r'{.*}' with re.DOTALL matched
        the FIRST '{' to the LAST '}', which could grab partial nested
        objects and produce invalid JSON.
        """
        full_prompt = (
            prompt + "\n\nIMPORTANT: Return ONLY valid JSON. "
            "No markdown fences, no explanations, no preamble."
        )
        raw = self.generate(
            model=model, prompt=full_prompt, system=system, temperature=0.05
        )

        # Attempt 1: direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt 2: strip markdown fences, retry
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Attempt 3: find outermost JSON object or array
        # Use json.JSONDecoder to find the first complete JSON value
        decoder = json.JSONDecoder()
        for start_char, close_char in [("{", "}"), ("[", "]")]:
            idx = cleaned.find(start_char)
            if idx == -1:
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned, idx)
                return obj
            except json.JSONDecodeError:
                continue

        logger.warning(
            "generate_json: all parse attempts failed. Raw: %s", raw[:200]
        )
        return fallback

    # ── Convenience shortcuts ─────────────────────────────────────────────

    def ask_analysis(self, prompt: str, system: str = "") -> str:
        return self.generate(model=config.MODEL_ANALYSIS, prompt=prompt,
                             system=system)

    def ask_translation(self, prompt: str) -> str:
        return self.generate(model=config.MODEL_TRANSLATION, prompt=prompt,
                             temperature=0.05)

    def ask_repair(self, prompt: str) -> str:
        return self.generate(model=config.MODEL_REPAIR, prompt=prompt,
                             temperature=0.02)

    def ask_json(self, prompt: str, model_key: str = "analysis") -> Any:
        model_map = {
            "analysis":    config.MODEL_ANALYSIS,
            "translation": config.MODEL_TRANSLATION,
            "repair":      config.MODEL_REPAIR,
            "testgen":     config.MODEL_TESTGEN,
        }
        model = model_map.get(model_key, config.MODEL_ANALYSIS)
        return self.generate_json(model=model, prompt=prompt, fallback={})


# Module-level singleton used by all agents
ollama_client = OllamaClient()