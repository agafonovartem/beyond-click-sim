from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
import httpx
from openai import OpenAI


OLLAMA_LOCAL_BASE_URL = "http://localhost:11434/v1"
VLLM_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_LOCAL_8001_BASE_URL = "http://127.0.0.1:8001/v1"
VLLM_LOCAL_8002_BASE_URL = "http://127.0.0.1:8002/v1"
OPENAI_VK_PROXY_BASE_URL = "https://ai-proxy.vk.team/v1"
OPENAI_VK_PROXY_API_KEY_ENV = "OPENAI_VK_PROXY_API_KEY"


def openai_client() -> Any:
    """Return the default OpenAI client configured from environment variables."""

    load_dotenv()
    return OpenAI()


def openai_vk_proxy_client(
    base_url: str = OPENAI_VK_PROXY_BASE_URL,
    api_key_env: str = OPENAI_VK_PROXY_API_KEY_ENV,
    trust_env: bool = False,
) -> Any:
    """Return an OpenAI-compatible client for the VK AI proxy endpoint."""

    load_dotenv()
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    http_client = httpx.Client(trust_env=trust_env)
    return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)


def vllm_client(base_url: str, api_key: str = "EMPTY") -> Any:
    """Return an OpenAI-compatible client for a vLLM endpoint."""

    return OpenAI(base_url=base_url, api_key=api_key)


def ollama_client(
    base_url: str = OLLAMA_LOCAL_BASE_URL,
    api_key: str = "ollama",
) -> Any:
    """Return an OpenAI-compatible client for an Ollama endpoint."""

    return OpenAI(base_url=base_url, api_key=api_key)


def make_llm_client(client_name: str) -> Any:
    """Return one of the fixed LLM clients used by experiment runners."""

    if client_name == "ollama_local":
        return ollama_client()
    if client_name == "vllm_local":
        return vllm_client(base_url=VLLM_LOCAL_BASE_URL)
    if client_name == "vllm_local_8001":
        return vllm_client(base_url=VLLM_LOCAL_8001_BASE_URL)
    if client_name == "vllm_local_8002":
        return vllm_client(base_url=VLLM_LOCAL_8002_BASE_URL)
    if client_name == "openai":
        return openai_client()
    if client_name == "openai_vk_proxy":
        return openai_vk_proxy_client()
    raise ValueError(f"Unsupported LLM client: {client_name!r}")
