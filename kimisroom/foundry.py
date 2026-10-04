"""Minimal client for the Azure Foundry resource (Azure credit): OpenAI-compatible v1 chat
completions for every deployment (GPT, DeepSeek, Kimi, Grok). The user's Entra identity has no
data-plane role on the resource, so the resource key is read through their `az` login at
runtime and kept in memory only: never written to disk, logged or printed.

    from kimisroom.foundry import chat
    r = chat("gpt-6.1-sol", [{"role": "user", "content": "hi"}], max_completion_tokens=50)
"""

from __future__ import annotations

import json
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request

ENDPOINT = "https://polysem-foundry-jk.cognitiveservices.azure.com/openai/v1/chat/completions"
_lock = threading.Lock()
_key: list[str] = []


def _auth() -> dict:
    with _lock:
        if not _key:
            _key.append(subprocess.run(
                ["az", "cognitiveservices", "account", "keys", "list", "-g", "polysem-foundry", "-n", "polysem-foundry-jk",
                 "--query", "key1", "-o", "tsv"], check=True, capture_output=True, text=True).stdout.strip())
        return {"api-key": _key[0]}


def chat(deployment: str, messages: list[dict], retries: int = 8, timeout: float = 300, **params) -> dict:
    """One chat completion; retries 429/5xx and network errors with jittered backoff."""
    body = json.dumps({"model": deployment, "messages": messages, **params}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
            **_auth(), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code not in (408, 429, 500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(f"{deployment}: HTTP {e.code}: {detail}") from None
            wait = float(e.headers.get("retry-after") or 0) or min(60, 2 ** attempt)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = min(60, 2 ** attempt)
        time.sleep(wait * (1 + 0.25 * random.random()))
    raise AssertionError("unreachable")


def text(response: dict) -> str:
    return (response["choices"][0]["message"].get("content") or "").strip()
