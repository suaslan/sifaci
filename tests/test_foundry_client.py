from __future__ import annotations

from types import SimpleNamespace

from config import CHAT_MAX_TOKENS, CHAT_TEMPERATURE
from src import foundry_client


class _FakeChatModel:
    def __init__(self):
        self.client = SimpleNamespace(
            settings=SimpleNamespace(temperature=None, max_tokens=None)
        )
        self.unloaded = False

    def download(self, callback):
        callback(100.0)

    def load(self):
        return None

    def get_chat_client(self):
        return self.client

    def unload(self):
        self.unloaded = True


def test_chat_model_uses_central_generation_settings(monkeypatch):
    model = _FakeChatModel()
    manager = SimpleNamespace(
        catalog=SimpleNamespace(get_model=lambda _alias: model)
    )
    monkeypatch.setattr(foundry_client, "get_foundry_manager", lambda: manager)
    monkeypatch.setattr(foundry_client, "_prepare_execution_providers", lambda _: None)
    monkeypatch.setattr(foundry_client, "_chat_model", None)
    monkeypatch.setattr(foundry_client, "_chat_client", None)

    client = foundry_client._get_chat_client()

    assert client.settings.temperature == CHAT_TEMPERATURE
    assert client.settings.max_tokens == CHAT_MAX_TOKENS
    foundry_client.close_chat_model()
    assert model.unloaded is True
