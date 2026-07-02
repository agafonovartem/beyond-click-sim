from __future__ import annotations

import pytest

from beyond_click_sim import llm_clients


class FakeOpenAI:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)


class FakeHTTPXClient:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)


def patch_client_dependencies(monkeypatch) -> list[bool]:
    dotenv_calls: list[bool] = []
    FakeOpenAI.calls = []
    FakeHTTPXClient.calls = []
    monkeypatch.setattr(llm_clients, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_clients.httpx, "Client", FakeHTTPXClient)
    monkeypatch.setattr(llm_clients, "load_dotenv", lambda: dotenv_calls.append(True))
    return dotenv_calls


def test_openai_client_uses_default_openai_constructor(monkeypatch) -> None:
    dotenv_calls = patch_client_dependencies(monkeypatch)

    client = llm_clients.openai_client()

    assert isinstance(client, FakeOpenAI)
    assert dotenv_calls == [True]
    assert FakeOpenAI.calls == [{}]


def test_openai_vk_proxy_client_uses_vk_proxy_env_key(monkeypatch) -> None:
    dotenv_calls = patch_client_dependencies(monkeypatch)
    monkeypatch.setenv("OPENAI_VK_PROXY_API_KEY", "test-vk-proxy-key")

    client = llm_clients.openai_vk_proxy_client()

    assert isinstance(client, FakeOpenAI)
    assert dotenv_calls == [True]
    assert FakeHTTPXClient.calls == [{"trust_env": False}]
    assert isinstance(FakeOpenAI.calls[0]["http_client"], FakeHTTPXClient)
    assert FakeOpenAI.calls == [
        {
            "base_url": "https://ai-proxy.vk.team/v1",
            "api_key": "test-vk-proxy-key",
            "http_client": FakeOpenAI.calls[0]["http_client"],
        }
    ]


def test_openai_vk_proxy_client_requires_env_key(monkeypatch) -> None:
    dotenv_calls = patch_client_dependencies(monkeypatch)
    monkeypatch.delenv("OPENAI_VK_PROXY_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_VK_PROXY_API_KEY is not set"):
        llm_clients.openai_vk_proxy_client()

    assert dotenv_calls == [True]
    assert FakeHTTPXClient.calls == []
    assert FakeOpenAI.calls == []


def test_vllm_client_passes_base_url_and_api_key(monkeypatch) -> None:
    dotenv_calls = patch_client_dependencies(monkeypatch)

    llm_clients.vllm_client("http://127.0.0.1:8000/v1")

    assert dotenv_calls == []
    assert FakeOpenAI.calls == [
        {"base_url": "http://127.0.0.1:8000/v1", "api_key": "EMPTY"}
    ]


def test_ollama_client_uses_openai_compatible_defaults(monkeypatch) -> None:
    dotenv_calls = patch_client_dependencies(monkeypatch)

    llm_clients.ollama_client()

    assert dotenv_calls == []
    assert FakeOpenAI.calls == [
        {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}
    ]


def test_make_llm_client_uses_fixed_named_clients(monkeypatch) -> None:
    dotenv_calls = patch_client_dependencies(monkeypatch)
    monkeypatch.setenv("OPENAI_VK_PROXY_API_KEY", "test-vk-proxy-key")

    llm_clients.make_llm_client("ollama_local")
    llm_clients.make_llm_client("vllm_local")
    llm_clients.make_llm_client("vllm_local_8001")
    llm_clients.make_llm_client("vllm_local_8002")
    llm_clients.make_llm_client("openai_vk_proxy")

    assert dotenv_calls == [True]
    assert FakeHTTPXClient.calls == [{"trust_env": False}]
    assert isinstance(FakeOpenAI.calls[4]["http_client"], FakeHTTPXClient)
    assert FakeOpenAI.calls == [
        {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
        {"base_url": "http://127.0.0.1:8000/v1", "api_key": "EMPTY"},
        {"base_url": "http://127.0.0.1:8001/v1", "api_key": "EMPTY"},
        {"base_url": "http://127.0.0.1:8002/v1", "api_key": "EMPTY"},
        {
            "base_url": "https://ai-proxy.vk.team/v1",
            "api_key": "test-vk-proxy-key",
            "http_client": FakeOpenAI.calls[4]["http_client"],
        },
    ]
