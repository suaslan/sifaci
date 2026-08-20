"""Reusable Microsoft Foundry Local chat-model client."""

from __future__ import annotations

import atexit
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from config import (
    CHAT_MAX_TOKENS,
    CHAT_TEMPERATURE,
    FOUNDRY_MODEL_ALIAS,
    FOUNDRY_PREPARE_EXECUTION_PROVIDERS,
    FOUNDRY_TEST_PROMPT,
)
from src.foundry_runtime import FoundryRuntimeError, get_foundry_manager


class FoundryLocalError(RuntimeError):
    """Raised when Foundry Local cannot produce a valid chat response."""


_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_chat_model: Any | None = None
_chat_client: Any | None = None


def _show_model_download_progress(progress: float) -> None:
    print(f"\rModel indiriliyor: %{progress:.1f}", end="", flush=True)


def _prepare_execution_providers(manager: Any) -> None:
    if not FOUNDRY_PREPARE_EXECUTION_PROVIDERS:
        return
    current_provider = ""

    def show_progress(provider_name: str, progress: float) -> None:
        nonlocal current_provider
        current_provider = provider_name
        print(
            f"\rYürütme sağlayıcısı hazırlanıyor: {provider_name} %{progress:.1f}",
            end="",
            flush=True,
        )

    manager.download_and_register_eps(progress_callback=show_progress)
    if current_provider:
        print()


def _get_chat_client() -> Any:
    """Load the configured chat model once and reuse it across RAG requests."""

    global _chat_model, _chat_client
    if _chat_client is not None:
        return _chat_client

    with _load_lock:
        if _chat_client is not None:
            return _chat_client

        model = None
        try:
            manager = get_foundry_manager()
            _prepare_execution_providers(manager)
            model = manager.catalog.get_model(FOUNDRY_MODEL_ALIAS)
            if model is None:
                raise FoundryLocalError(
                    f"'{FOUNDRY_MODEL_ALIAS}' chat modeli Foundry Local "
                    "kataloğunda bulunamadı."
                )

            model.download(_show_model_download_progress)
            print()
            model.load()
            client = model.get_chat_client()
            if client is None:
                raise FoundryLocalError(
                    f"'{FOUNDRY_MODEL_ALIAS}' bir chat istemcisi oluşturmadı."
                )
            client.settings.temperature = CHAT_TEMPERATURE
            client.settings.max_tokens = CHAT_MAX_TOKENS
        except FoundryLocalError:
            raise
        except FoundryRuntimeError as error:
            raise FoundryLocalError(str(error)) from error
        except Exception as error:
            if model is not None:
                try:
                    model.unload()
                except Exception:
                    pass
            raise FoundryLocalError(
                f"'{FOUNDRY_MODEL_ALIAS}' yerel chat modeli hazırlanamadı: {error}"
            ) from error

        _chat_model = model
        _chat_client = client
        return client


def complete_chat(messages: Sequence[Mapping[str, str]]) -> str:
    """Return one non-streaming local chat completion for validated messages."""

    prepared_messages: list[dict[str, str]] = []
    allowed_roles = {"system", "user", "assistant"}
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in allowed_roles:
            raise ValueError(f"messages[{index}] has an unsupported role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}] content cannot be empty")
        prepared_messages.append({"role": role, "content": content.strip()})
    if not prepared_messages:
        raise ValueError("messages cannot be empty")

    client = _get_chat_client()
    try:
        with _inference_lock:
            response = client.complete_chat(prepared_messages)
        choices = getattr(response, "choices", None)
        if not choices:
            raise FoundryLocalError("Yerel chat modeli boş bir cevap döndürdü.")
        message = getattr(choices[0], "message", None)
        answer = getattr(message, "content", None)
        if not isinstance(answer, str) or not answer.strip():
            raise FoundryLocalError("Yerel chat modelinin cevabında metin bulunamadı.")
        return answer.strip()
    except FoundryLocalError:
        raise
    except Exception as error:
        raise FoundryLocalError(f"Yerel chat cevabı üretilemedi: {error}") from error


def close_chat_model() -> None:
    """Unload the cached model; normally called automatically at process exit."""

    global _chat_model, _chat_client
    with _load_lock:
        model = _chat_model
        _chat_client = None
        _chat_model = None
        if model is not None:
            try:
                model.unload()
            except Exception:
                pass


def test_model() -> str:
    """Send a small smoke-test prompt to the configured local chat model."""

    answer = complete_chat([{"role": "user", "content": FOUNDRY_TEST_PROMPT}])
    print(f"Model cevabı: {answer}")
    return answer


atexit.register(close_chat_model)


if __name__ == "__main__":
    try:
        test_model()
    except FoundryLocalError as error:
        print(f"Hata: {error}")
        raise SystemExit(1) from error
