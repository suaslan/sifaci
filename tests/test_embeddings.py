from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import embeddings


class FakeEmbeddingClient:
    def generate_embedding(self, text):
        assert text == "tek metin"
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1, 2.5, -3])])

    def generate_embeddings(self, texts):
        assert texts == ["ilk", "ikinci"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[1, 0]),
                SimpleNamespace(embedding=[0, 1]),
            ]
        )


def test_generate_embedding_returns_float_vector(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_embedding_client", FakeEmbeddingClient)

    assert embeddings.generate_embedding(" tek metin ") == [1.0, 2.5, -3.0]


def test_generate_embeddings_uses_one_batch_call(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_embedding_client", FakeEmbeddingClient)

    assert embeddings.generate_embeddings(["ilk", "ikinci"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def test_generate_embeddings_does_not_load_model_for_empty_input(monkeypatch):
    def fail_if_called():
        raise AssertionError("client should not be loaded")

    monkeypatch.setattr(embeddings, "_get_embedding_client", fail_if_called)

    assert embeddings.generate_embeddings([]) == []


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        embeddings.generate_embedding("  ")


def test_batch_rejects_one_string():
    with pytest.raises(TypeError, match="sequence of strings"):
        embeddings.generate_embeddings("tek metin")
