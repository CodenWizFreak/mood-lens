"""
llm_client.py — Native Groq client wrapper for MoodLens.

Exposes the same interface the rest of the codebase already uses:
    from llm_client import GroqClient as Groq
    client = Groq()
    resp   = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", ...}, {"role": "user", ...}],
        temperature=0.8,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,   # or True
        stop=None,
    )
    text  = resp.choices[0].message.content          # non-stream
    for chunk in resp:                               # stream
        delta = chunk.choices[0].delta.content

Auth:
    GROQ_API_KEY environment variable (get yours free at console.groq.com)

Required dep:
    pip install groq>=0.9.0
"""

from __future__ import annotations

import os
from typing import Any, Optional

try:
    from groq import Groq as _GroqSDK
except ImportError as e:
    raise ImportError(
        "groq package not installed. Run: pip install groq"
    ) from e


class GroqClient:
    """
    Thin wrapper around the official Groq SDK.

    Auth resolution order:
        1. explicit `api_key=` kwarg
        2. env GROQ_API_KEY
    """

    def __init__(self, api_key: Optional[str] = None, **_unused: Any):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "No Groq API key found. Set GROQ_API_KEY in your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )
        self._client = _GroqSDK(api_key=key)
        self.chat = self._client.chat
