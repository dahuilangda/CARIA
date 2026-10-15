"""LLM client for the extraction harness.

Talks to any OpenAI-compatible /v1/chat/completions endpoint, including
vision-capable models. Configuration is read from a `.env` file when present
(searched in the repository root and this directory) and from process
environment variables, which take precedence.

Configuration keys
------------------
EXTRACTION_BASE_URL   e.g. http://host:port/v1
EXTRACTION_API_KEY    bearer token for the endpoint
EXTRACTION_MODEL      declared model identifier
EXTRACTION_TIMEOUT    per-request timeout in seconds (default 180)

Credentials and endpoints live only in `.env` or the process environment;
nothing is embedded in source files, argv or log output. The defaults are
placeholders: the first run without configuration fails with a clear message.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
ENV_FILES = (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env")

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = ""

_CONTENT_PART_TEXT = {"type": "text", "text": ""}
_CONTENT_PART_IMAGE = {"type": "image_url", "image_url": {"url": ""}}


def load_env() -> dict[str, str]:
    """Parse the first `.env` file found and return its KEY=VALUE pairs.

    Existing process environment variables always win over file entries.
    """
    values: dict[str, str] = {}
    for env_file in ENV_FILES:
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values.setdefault(key.strip(), value.strip().strip("'\""))
        break
    return {k: v for k, v in values.items() if not os.environ.get(k)}


def setting(name: str, default: str | None = None) -> str | None:
    """Read a configuration value from the environment, then `.env` files."""
    return os.environ.get(name) or load_env().get(name) or default


@dataclass(frozen=True)
class Completion:
    """One finished model response."""

    text: str
    finish_reason: str
    model: str


def text_message(content: str) -> dict[str, Any]:
    part = dict(_CONTENT_PART_TEXT)
    part["text"] = content
    return {"role": "user", "content": [part]}


def image_message(prompt: str, image_data_url: str) -> dict[str, Any]:
    """Build a user message that pairs a question with one data-URL image."""
    image_part = dict(_CONTENT_PART_IMAGE)
    image_part["image_url"]["url"] = image_data_url
    text_part = dict(_CONTENT_PART_TEXT)
    text_part["text"] = prompt
    return {"role": "user", "content": [text_part, image_part]}


class ExtractionClient:
    """Thin, retrying client for one OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or setting("EXTRACTION_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or setting("EXTRACTION_API_KEY") or ""
        self.model = model or setting("EXTRACTION_MODEL") or DEFAULT_MODEL
        self.timeout = float(timeout or setting("EXTRACTION_TIMEOUT") or 180)
        if not self.api_key:
            raise ValueError(
                "EXTRACTION_API_KEY is not configured; add it to `.env` or the environment"
            )

    def complete(
        self,
        system: str,
        user: str | dict[str, Any],
        max_tokens: int,
    ) -> Completion:
        """Send one chat request and return the finished assistant text."""
        user_content = user if isinstance(user, dict) else user
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
        }
        body = self._send(payload, "extraction")
        choice = (body.get("choices") or [{}])[0]
        finish = choice.get("finish_reason") or "stop"
        text = (choice.get("message") or {}).get("content") or ""
        if finish != "stop":
            raise ValueError(f"Model response stopped early ({finish}); raise --max-tokens")
        if not text.strip():
            raise ValueError("Model returned an empty response")
        return Completion(text=text, finish_reason=finish, model=body.get("model", self.model))

    def _send(self, payload: dict, label: str) -> dict:
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read()[:200].decode(errors="replace")
                if error.code not in RETRYABLE_STATUS or attempt == RETRY_ATTEMPTS:
                    raise ValueError(f"{label} endpoint returned HTTP {error.code}: {detail}") from error
                last_error = error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                if attempt == RETRY_ATTEMPTS:
                    raise ValueError(f"{label} endpoint unreachable") from error
                last_error = error
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise ValueError(f"{label} endpoint unreachable") from last_error
