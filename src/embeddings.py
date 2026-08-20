"""Generate text embeddings locally with Microsoft Foundry Local."""

from __future__ import annotations

import atexit
import math
import threading
from collections.abc import Sequence
from typing import Any

from config import EMBEDDING_MODEL_NAME
from src.foundry_runtime import FoundryRuntimeError, get_foundry_manager


class EmbeddingError(RuntimeError):
    """Raised when the local embedding runtime cannot return a valid vector."""


_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_model: Any | None = None
_embedding_client: Any | None = None


def _get_embedding_client() -> Any:
    """Load the configured model once and reuse its client for later calls."""

    global _model, _embedding_client
    if _embedding_client is not None:
        return _embedding_client

    with _load_lock:
        if _embedding_client is not None:
            return _embedding_client

        model = None
        try:
            manager = get_foundry_manager()
            model = manager.catalog.get_model(EMBEDDING_MODEL_NAME)
            if model is None:
                raise EmbeddingError(
                    f"'{EMBEDDING_MODEL_NAME}' embedding modeli Foundry Local "
                    "kataloğunda bulunamadı."
                )

            model.download()
            model.load()
            client = model.get_embedding_client()
            if client is None:
                raise EmbeddingError(
                    f"'{EMBEDDING_MODEL_NAME}' bir embedding istemcisi oluşturmadı."
                )
        except EmbeddingError:
            raise
        except FoundryRuntimeError as error:
            raise EmbeddingError(str(error)) from error
        except Exception as error:
            if model is not None:
                try:
                    model.unload()
                except Exception:
                    pass
            raise EmbeddingError(
                f"'{EMBEDDING_MODEL_NAME}' yerel embedding modeli hazırlanamadı: "
                f"{error}"
            ) from error

        _model = model
        _embedding_client = client
        return client


def generate_embedding(text: str) -> list[float]:
    """Return one numerical embedding vector for non-empty ``text``."""

    clean_text = _validate_text(text)
    client = _get_embedding_client()
    try:
        with _inference_lock:
            response = client.generate_embedding(clean_text)
        return _extract_vectors(response, expected_count=1)[0]
    except EmbeddingError:
        raise
    except Exception as error:
        raise EmbeddingError(f"Embedding üretilemedi: {error}") from error


def generate_embeddings(texts: Sequence[str]) -> list[list[float]]:
    """Return embeddings for ``texts`` in the same order using one batch call."""

    if isinstance(texts, (str, bytes, bytearray)):
        raise TypeError("texts must be a sequence of strings, not one string")
    clean_texts = [_validate_text(text, index=index) for index, text in enumerate(texts)]
    if not clean_texts:
        return []

    client = _get_embedding_client()
    try:
        with _inference_lock:
            response = client.generate_embeddings(clean_texts)
        return _extract_vectors(response, expected_count=len(clean_texts))
    except EmbeddingError:
        raise
    except Exception as error:
        raise EmbeddingError(f"Embedding grubu üretilemedi: {error}") from error


def close_embedding_model() -> None:
    """Unload the cached model; normally called automatically at process exit."""

    global _model, _embedding_client
    with _load_lock:
        model = _model
        _embedding_client = None
        _model = None
        if model is not None:
            try:
                model.unload()
            except Exception:
                # Interpreter shutdown must not fail because native resources
                # have already been released by the Foundry runtime.
                pass


def _validate_text(text: str, *, index: int | None = None) -> str:
    if not isinstance(text, str):
        location = f"texts[{index}]" if index is not None else "text"
        raise TypeError(f"{location} must be a string")
    clean_text = text.strip()
    if not clean_text:
        location = f"texts[{index}]" if index is not None else "text"
        raise ValueError(f"{location} cannot be empty")
    return clean_text


def _extract_vectors(response: Any, *, expected_count: int) -> list[list[float]]:
    data = getattr(response, "data", None)
    if data is None:
        raise EmbeddingError("Foundry Local yanıtında 'data' alanı bulunamadı.")

    items = list(data)
    if len(items) != expected_count:
        raise EmbeddingError(
            "Foundry Local beklenmeyen sayıda embedding döndürdü: "
            f"beklenen={expected_count}, alınan={len(items)}."
        )

    vectors: list[list[float]] = []
    dimension: int | None = None
    for item in items:
        raw_vector = getattr(item, "embedding", None)
        if raw_vector is None:
            raise EmbeddingError("Foundry Local embedding verisi boş döndü.")
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError) as error:
            raise EmbeddingError("Embedding yalnızca sayısal değerler içermelidir.") from error
        if not vector or not all(math.isfinite(value) for value in vector):
            raise EmbeddingError("Embedding boş veya sonlu olmayan bir değer içeriyor.")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise EmbeddingError("Batch içindeki embedding boyutları birbiriyle uyuşmuyor.")
        vectors.append(vector)
    return vectors


atexit.register(close_embedding_model)
