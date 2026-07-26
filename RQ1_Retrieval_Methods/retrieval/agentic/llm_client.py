"""
Ollama LLM client — shared by CRAG (Tier 4.1) and ReAct (Tier 4.2).

Classes
-------
OllamaClient
    Thin wrapper around the Ollama REST API.
        generate()        — POST /api/generate (non-streaming) → (text, latency_ms)
        chat_with_tools() — POST /api/chat with native function-calling tools
                            (Llama 3.1 native tool format) → (parsed_dict, latency_ms)
    Raises ConnectionError on server unreachable; ValueError on bad response.
"""

from __future__ import annotations

import time

import requests


class OllamaClient:
    """
    Thin wrapper around the Ollama REST API.

    Parameters
    ----------
    model      : Ollama model tag (default: llama3.1:8b)
    base_url   : Ollama server base URL (default: http://localhost:11434)

    Usage
    -----
    client = OllamaClient()
    text, latency_ms = client.generate("Quelle est la capitale de la Belgique ?")
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ):
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ── Public interface ─────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: int = 1800,
    ) -> tuple[str, float]:
        """
        Generate a completion for *prompt*.

        Parameters
        ----------
        prompt      : full prompt string
        temperature : sampling temperature (0.0 = greedy/deterministic)
        max_tokens  : maximum tokens to generate
        timeout     : HTTP read timeout in seconds (default 1800 — covers cold
                      model-load on first call; subsequent GPU calls are ~1-2 s)

        Returns
        -------
        text       : generated text (stripped of leading/trailing whitespace)
        latency_ms : wall-clock time for the HTTP round-trip (time.perf_counter())

        Raises
        ------
        ConnectionError : if Ollama server is unreachable
        ValueError      : if the server returns an unexpected response
        """
        payload = {
            "model":      self.model,
            "prompt":     prompt,
            "stream":     False,
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        t0 = time.perf_counter()
        try:
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Ollama server unreachable at {self.base_url}. "
                "Run 'ollama serve' before starting experiments."
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if response.status_code != 200:
            raise ValueError(
                f"Ollama /api/generate returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        data = response.json()
        if "response" not in data:
            raise ValueError(f"Unexpected Ollama response format: {data}")

        return data["response"].strip(), latency_ms

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: int = 1800,
    ) -> tuple[dict, float]:
        """
        Issue a chat request with native function-calling tools.

        Wraps Ollama's POST /api/chat (Llama 3.1 / 3.2 native tool format).
        Returns the *raw* assistant message dict so the caller can inspect
        both `content` (free-text) and `tool_calls` (structured).

        Parameters
        ----------
        messages    : list of {"role": "system"|"user"|"assistant", "content": str}
        tools       : list of tool schema dicts:
                        [{"type": "function",
                          "function": {"name": ..., "description": ...,
                                       "parameters": {...JSON Schema...}}}]
        temperature : 0.0 = greedy
        max_tokens  : num_predict equivalent (128 default — enough for one
                      Pensée + Action turn)
        timeout     : HTTP read timeout in seconds

        Returns
        -------
        message    : assistant message dict, e.g.
                       {"role": "assistant",
                        "content": "...",
                        "tool_calls": [
                            {"function": {"name": "search",
                                          "arguments": {"query": "..."}}}
                        ]}
                     `tool_calls` may be absent or empty if the model emitted
                     plain text — caller falls back to its regex parser.
        latency_ms : wall-clock time for the HTTP round-trip
        """
        payload = {
            "model":      self.model,
            "messages":   messages,
            "tools":      tools,
            "stream":     False,
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        t0 = time.perf_counter()
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Ollama server unreachable at {self.base_url}."
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if response.status_code != 200:
            raise ValueError(
                f"Ollama /api/chat returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        data = response.json()
        if "message" not in data:
            raise ValueError(f"Unexpected Ollama /api/chat response: {data}")

        return data["message"], latency_ms

    def preflight(self) -> float:
        """
        Verify the server is running and the model is available.
        Returns the latency of the tags endpoint in ms.
        Raises ConnectionError if the server is unreachable.
        """
        t0 = time.perf_counter()
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Ollama server unreachable at {self.base_url}."
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        models = [m["name"] for m in resp.json().get("models", [])]
        # Accept both "llama3.1:8b" and "llama3.1:8b-instruct-..." tags
        if not any(self.model.split(":")[0] in m for m in models):
            raise ValueError(
                f"Model '{self.model}' not found in Ollama. "
                f"Available: {models}. Run 'ollama pull {self.model}'."
            )
        return latency_ms
