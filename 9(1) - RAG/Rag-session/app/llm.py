"""Solar Pro3 LLM utility for RAG answer generation.

Uses Upstage Solar API (OpenAI-compatible) with solar-pro3 model.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
MODEL = os.getenv("SOLAR_MODEL", "solar-pro3")

NO_RAG_PROMPT = "Answer the following question concisely.\n\nQuestion: {question}"

RAG_PROMPT = """\
Answer the question based ONLY on the provided context.
If the answer is not found in the context, reply exactly: "The provided context does not contain this information."
Do NOT use any outside knowledge.

Context:
{context}

Question: {question}"""


def _get_api_key() -> str:
    """Get the first available Upstage API key."""
    key = os.getenv("UPSTAGE_API_KEY1") or os.getenv("UPSTAGE_API_KEY", "")
    return key.strip()


def generate(question: str, context: str | None = None) -> str:
    """Generate an answer using Solar LLM.

    Args:
        question: The user question.
        context: Retrieved context for RAG. None for no-RAG generation.

    Returns:
        str: The model's answer text.

    Hints:
        - Use _get_api_key() and OpenAI(api_key=..., base_url=BASE_URL)
        - If context is provided, use RAG_PROMPT; otherwise use NO_RAG_PROMPT
        - Use client.chat.completions.create(model=MODEL, messages=[...])
        - Set temperature=0 for deterministic output, max_tokens=1024
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if context is not None and not isinstance(context, str):
        raise TypeError("context must be a string or None.")

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "Upstage API key is missing. Set UPSTAGE_API_KEY in the .env file."
        )

    if context is None:
        prompt = NO_RAG_PROMPT.format(question=question.strip())
    else:
        prompt = RAG_PROMPT.format(
            question=question.strip(),
            context=context.strip(),
        )

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    if not response.choices:
        raise RuntimeError("Solar API returned no completion choices.")
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Solar API returned an empty answer.")
    return answer.strip()
