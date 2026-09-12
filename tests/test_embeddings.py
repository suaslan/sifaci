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


def test_iter_embedding_batches_loads_once_and_uses_32_item_batches(monkeypatch):
    load_calls = 0
    batch_sizes: list[int] = []

    class BatchClient:
        def generate_embeddings(self, texts):
            batch_sizes.append(len(texts))
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[index, 1]) for index, _ in enumerate(texts)]
            )

    def get_client():
        nonlocal load_calls
        load_calls += 1
        return BatchClient()

    monkeypatch.setattr(embeddings, "_get_embedding_client", get_client)
    batches = list(
        embeddings.iter_embedding_batches(
            [f"chunk-{index}" for index in range(65)],
            batch_size=32,
        )
    )

    assert load_calls == 1
    assert batch_sizes == [32, 32, 1]
    assert [offset for offset, _ in batches] == [0, 32, 64]
    assert sum(len(vectors) for _, vectors in batches) == 65


def test_iter_embedding_batches_accepts_64_and_rejects_other_sizes():
    assert list(
        embeddings.iter_embedding_batches(
            ["a", "b"],
            batch_size=64,
            embedding_function=lambda texts: [[1.0] for _ in texts],
        )
    ) == [(0, [[1.0], [1.0]])]

    with pytest.raises(ValueError, match="32 or 64"):
        list(
            embeddings.iter_embedding_batches(
                ["a"],
                batch_size=16,
                embedding_function=lambda texts: [[1.0] for _ in texts],
            )
        )
