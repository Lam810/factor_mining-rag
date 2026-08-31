#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenRouter chat client.

The published version imported ``langchain_openrouter``, a package that does not
exist on PyPI, so this module could never be imported.  It now talks to the
OpenRouter REST API directly through ``requests``: one dependency instead of the
whole LangChain stack, and no invented package name.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You answer strictly from the supplied document excerpts. "
    "If the excerpts do not contain the answer, say so instead of guessing. "
    "When you use a number from a table, name the table's column header."
)


class LLMModel:
    """Minimal OpenRouter chat-completions client."""

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model_name: str = OPENROUTER_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: int = 120,
    ) -> None:
        if not api_key:
            raise ValueError(
                "No OpenRouter API key. Set the OPENROUTER_API_KEY environment "
                "variable; do not write the key into a source file."
            )
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.2),
                "max_tokens": kwargs.get("max_tokens", 2048),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._post(messages, **kwargs)

    def rag_generate(
        self,
        query: str,
        context: Sequence[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        excerpts = "\n\n".join(f"[Excerpt {i + 1}]\n{c}" for i, c in enumerate(context))
        prompt = (
            f"Document excerpts:\n\n{excerpts}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the excerpts above."
        )
        return self.generate(prompt, system_prompt or DEFAULT_SYSTEM_PROMPT, **kwargs)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)


_llm_model_instance: Optional[LLMModel] = None


def get_llm_model() -> LLMModel:
    global _llm_model_instance
    if _llm_model_instance is None:
        _llm_model_instance = LLMModel()
    return _llm_model_instance
