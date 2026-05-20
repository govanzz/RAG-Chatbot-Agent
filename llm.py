"""Ollama client and grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from config import (
    AUTO_SELECT_INSTALLED_MODEL,
    DEFAULT_OLLAMA_MODEL,
    FALLBACK_OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_PREFERENCE,
    OLLAMA_TIMEOUT,
    REQUESTED_DEFAULT_MODEL,
)
from rag import format_context, has_sufficient_context

INSUFFICIENT_CONTEXT_ANSWER = (
    "I don't have enough information in the retrieved Mandai website content "
    "to answer that confidently."
)

SYSTEM_PROMPT = """You are a Mandai website knowledge assistant.
Answer only using the retrieved context.
Do not invent facts.
If unsure, say the retrieved content is insufficient.
Keep answers concise and helpful.
Include citations using page titles and URLs."""


class OllamaError(RuntimeError):
    pass


class OllamaConnectionError(OllamaError):
    pass


class OllamaModelNotFoundError(OllamaError):
    pass


@dataclass
class LLMResult:
    answer: str
    model: str


def _ollama_url(path: str) -> str:
    return f"{OLLAMA_BASE_URL.rstrip('/')}/{path.lstrip('/')}"


def get_installed_models() -> list[str]:
    try:
        response = requests.get(_ollama_url("/api/tags"), timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaConnectionError(
            "Could not connect to Ollama. Start it with: ollama serve"
        ) from exc

    payload = response.json()
    return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]


def select_ollama_model(
    requested_model: str | None = None,
    installed_models: list[str] | None = None,
) -> str:
    if requested_model:
        return requested_model

    if not AUTO_SELECT_INSTALLED_MODEL:
        return DEFAULT_OLLAMA_MODEL

    try:
        installed = installed_models if installed_models is not None else get_installed_models()
    except OllamaConnectionError:
        return DEFAULT_OLLAMA_MODEL

    for candidate in OLLAMA_MODEL_PREFERENCE:
        if candidate in installed:
            return candidate
    return DEFAULT_OLLAMA_MODEL


def _model_missing_message(model: str) -> str:
    if model == REQUESTED_DEFAULT_MODEL:
        return f"Model '{model}' is missing. Install it with: ollama pull llama3.1:8b"
    return (
        f"Model '{model}' is missing. Install it with: ollama pull {model}. "
        "For the default model, run: ollama pull llama3.1:8b"
    )


def _build_user_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    return f"""Retrieved Mandai website context:

{format_context(chunks)}

User question:
{question}

Write the answer using only the retrieved context. If the context does not
directly answer the question, reply exactly:
{INSUFFICIENT_CONTEXT_ANSWER}
"""


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    model: str | None = None,
) -> LLMResult:
    selected_model = model or select_ollama_model()

    if not has_sufficient_context(chunks):
        return LLMResult(answer=INSUFFICIENT_CONTEXT_ANSWER, model=selected_model)

    payload = {
        "model": selected_model,
        "system": SYSTEM_PROMPT,
        "prompt": _build_user_prompt(question, chunks),
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 4096,
        },
    }

    try:
        response = requests.post(
            _ollama_url("/api/generate"),
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise OllamaConnectionError(
            "Ollama is not running or is unreachable. Start it with: ollama serve"
        ) from exc

    if response.status_code >= 400:
        message = response.text
        if "not found" in message.lower() or "pull" in message.lower():
            if selected_model != FALLBACK_OLLAMA_MODEL:
                try:
                    installed = get_installed_models()
                except OllamaConnectionError:
                    installed = []
                if FALLBACK_OLLAMA_MODEL in installed:
                    return generate_answer(question, chunks, model=FALLBACK_OLLAMA_MODEL)
            raise OllamaModelNotFoundError(_model_missing_message(selected_model))
        raise OllamaError(f"Ollama returned an error: {message}")

    data = response.json()
    answer = data.get("response", "").strip()
    if not answer:
        answer = INSUFFICIENT_CONTEXT_ANSWER
    return LLMResult(answer=answer, model=selected_model)
