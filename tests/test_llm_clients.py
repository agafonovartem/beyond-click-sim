from __future__ import annotations

from beyond_click_sim import llm_clients


class FakeOpenAI:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)


def patch_client_dependencies(monkeypatch) -> list[bool]:
    dotenv_calls: list[bool] = []
    FakeOpenAI.calls = []
    monkeypatch.setattr(llm_clients, "_openai_client_class", lambda: FakeOpenAI)
    monkeypatch.setattr(llm_clients, "_load_dotenv", lambda: dotenv_calls.append(True))
    return dotenv_calls


def test_openai_client_uses_default_openai_constructor(monkeypatch) -> None:
    dotenv_calls = patch_client_dependencies(monkeypatch)

    client = llm_clients.openai_client()

    assert isinstance(client, FakeOpenAI)
    assert dotenv_calls == [True]
    assert FakeOpenAI.calls == [{}]


def test_vllm_client_passes_base_url_and_api_key(monkeypatch) -> None:
    dotenv_calls = patch_client_dependencies(monkeypatch)

    llm_clients.vllm_client("http://127.0.0.1:8000/v1")

    assert dotenv_calls == [True]
    assert FakeOpenAI.calls == [
        {"base_url": "http://127.0.0.1:8000/v1", "api_key": "EMPTY"}
    ]


def test_ollama_client_uses_openai_compatible_defaults(monkeypatch) -> None:
    dotenv_calls = patch_client_dependencies(monkeypatch)

    llm_clients.ollama_client()

    assert dotenv_calls == [True]
    assert FakeOpenAI.calls == [
        {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}
    ]
