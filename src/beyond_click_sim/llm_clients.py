from __future__ import annotations

from typing import Any


def _load_dotenv() -> None:
    from dotenv import load_dotenv

    load_dotenv()


def _openai_client_class() -> type[Any]:
    from openai import OpenAI

    return OpenAI


def openai_client() -> Any:
    """Return the default OpenAI client configured from environment variables."""

    _load_dotenv()
    return _openai_client_class()()


def vllm_client(base_url: str, api_key: str = "EMPTY") -> Any:
    """Return an OpenAI-compatible client for a vLLM endpoint."""

    _load_dotenv()
    return _openai_client_class()(base_url=base_url, api_key=api_key)


def ollama_client(
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> Any:
    """Return an OpenAI-compatible client for an Ollama endpoint."""

    _load_dotenv()
    return _openai_client_class()(base_url=base_url, api_key=api_key)
