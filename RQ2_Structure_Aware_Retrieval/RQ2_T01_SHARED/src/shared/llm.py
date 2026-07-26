"""LLaMA 3.1 8B via Ollama wrapper."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import ollama

from .utils import setup_logger

logger = setup_logger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient:
    """Thin Ollama wrapper with JSON helpers and token-count logging."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_ctx: int | None = 16384,
    ) -> None:
        self._model = model
        self._client = ollama.Client(host=base_url)
        self._options: dict = {"temperature": temperature}
        if num_ctx is not None:
            # Ollama's default num_ctx is 4096 — silently truncates long
            # prompts from the START (preserving the instruction at the
            # end). T05 PageIndex chapter-selection prompts are 6-9k
            # tokens and were losing the entire chapter list pre-fix,
            # which masked the LLM's actual navigation skill behind
            # 100% `no_candidates` exits on Code Civil. Llama 3.1's
            # trained context is 128k; 16k holds every observed T05
            # prompt with headroom.
            self._options["num_ctx"] = num_ctx

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate a completion and return text + usage stats."""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        response = self._client.chat(
            model=self._model,
            messages=messages,
            options=self._options,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        text = response["message"]["content"]
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        logger.debug(
            "LLM %s | in=%d out=%d | %.0f ms",
            self._model, input_tokens, output_tokens, latency_ms,
        )
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
        """Generate a completion and parse it as JSON.

        Strips markdown fences if present; retries once on parse failure.
        """
        for attempt in range(2):
            response = self.generate(prompt, system_prompt)
            text = response.text.strip()

            fence_match = _JSON_FENCE_RE.search(text)
            if fence_match:
                text = fence_match.group(1).strip()

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt == 0:
                    logger.warning("JSON parse failed on attempt 1, retrying")
                    continue
                logger.error("JSON parse failed after retry. Raw text: %s", text[:200])
                raise
        raise RuntimeError("unreachable")
