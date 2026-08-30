#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generation model, served through OpenRouter.

Earlier revisions of this file imported ``langchain_openrouter``, which is not
a published package -- ``pip install`` for this project could never succeed.
OpenRouter's chat endpoint is wire-compatible with the OpenAI API, so this
version talks to it with the official ``openai`` SDK pointed at OpenRouter's
``base_url``, which is the integration OpenRouter's own docs recommend and
needs no extra dependency beyond a client this project already needs.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from typing import List, Optional

from openai import OpenAI

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


class LLMModel:
    """Chat completion model served through OpenRouter."""

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model_name: str = OPENROUTER_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
    ):
        """
        Args:
            api_key: OpenRouter API key.
            model_name: OpenRouter model slug, e.g. ``"deepseek/deepseek-chat"``.
            base_url: OpenRouter's OpenAI-compatible endpoint.

        Raises:
            ValueError: If no API key is available.
        """
        if not api_key:
            raise ValueError(
                "An OpenRouter API key is required: set the OPENROUTER_API_KEY "
                "environment variable or pass api_key= explicitly."
            )

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a completion for a single prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            The model's reply text.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def rag_generate(self, query: str, context: List[str], system_prompt: Optional[str] = None) -> str:
        """Generate an answer to ``query`` grounded in retrieved ``context``.

        Args:
            query: User query.
            context: Retrieved chunk texts, most relevant first.
            system_prompt: Optional system prompt.

        Returns:
            The model's reply text.
        """
        context_text = "\n\n".join(f"Document {i + 1}:\n{ctx}" for i, ctx in enumerate(context))
        rag_prompt = (
            "Answer the user's question using only the reference documents below. "
            "If the answer is not contained in them, say so explicitly instead of guessing.\n\n"
            f"Reference documents:\n{context_text}\n\n"
            f"Question:\n{query}"
        )
        return self.generate(rag_prompt, system_prompt)

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


_llm_model_instance = None


def get_llm_model() -> LLMModel:
    """Return the process-wide :class:`LLMModel` singleton."""
    global _llm_model_instance
    if _llm_model_instance is None:
        _llm_model_instance = LLMModel()
    return _llm_model_instance
