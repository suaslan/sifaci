"""Shared initialization for the Microsoft Foundry Local SDK."""

from __future__ import annotations

import threading
from typing import Any

from config import (
    FOUNDRY_APP_DATA_DIR,
    FOUNDRY_APP_NAME,
    FOUNDRY_LOGS_DIR,
    FOUNDRY_MODEL_CACHE_DIR,
    FOUNDRY_PREPARE_EXECUTION_PROVIDERS,
)


class FoundryRuntimeError(RuntimeError):
    """Raised when the Foundry Local runtime cannot be initialized."""


_manager_lock = threading.Lock()
_provider_lock = threading.Lock()
_providers_prepared = False


def get_foundry_manager() -> Any:
    """Return the process-wide Foundry Local manager, initializing it once."""

    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except (ImportError, OSError) as error:
        raise FoundryRuntimeError(
            "Foundry Local Python SDK yüklenemedi. Önce "
            "'python -m pip install -r requirements.txt' komutunu çalıştırın."
        ) from error

    with _manager_lock:
        try:
            manager = FoundryLocalManager.instance
        except Exception:
            manager = None
        if manager is not None:
            return manager

        try:
            FoundryLocalManager.initialize(
                Configuration(
                    app_name=FOUNDRY_APP_NAME,
                    app_data_dir=str(FOUNDRY_APP_DATA_DIR),
                    model_cache_dir=str(FOUNDRY_MODEL_CACHE_DIR),
                    logs_dir=str(FOUNDRY_LOGS_DIR),
                )
            )
            manager = FoundryLocalManager.instance
            if manager is None:
                raise FoundryRuntimeError(
                    "Foundry Local başlatıldı ancak manager örneği oluşturulmadı."
                )
            return manager
        except FoundryRuntimeError:
            raise
        except Exception as error:
            try:
                manager = FoundryLocalManager.instance
            except Exception:
                manager = None
            if manager is not None:
                return manager
            raise FoundryRuntimeError(
                f"Foundry Local çalışma zamanı başlatılamadı: {error}"
            ) from error


def prepare_execution_providers(manager: Any | None = None) -> None:
    """Download/register compatible acceleration providers once per process."""

    global _providers_prepared
    if not FOUNDRY_PREPARE_EXECUTION_PROVIDERS or _providers_prepared:
        return
    with _provider_lock:
        if _providers_prepared:
            return
        active_manager = manager if manager is not None else get_foundry_manager()
        current_provider = ""

        def show_progress(provider_name: str, progress: float) -> None:
            nonlocal current_provider
            current_provider = provider_name
            print(
                f"\rYürütme sağlayıcısı hazırlanıyor: {provider_name} %{progress:.1f}",
                end="",
                flush=True,
            )

        active_manager.download_and_register_eps(progress_callback=show_progress)
        if current_provider:
            print()
        _providers_prepared = True


def select_model_variant(model: Any, device_type: str) -> Any:
    """Select a cached-first model variant for an explicit device preference.

    Foundry Local may prefer a cached GPU variant even when its execution
    provider is unavailable in the installed runtime.  CPU is therefore a
    useful deterministic choice for long-running ingestion jobs.
    """

    preference = device_type.strip().upper()
    if preference == "AUTO":
        return model

    matching: list[Any] = []
    for variant in getattr(model, "variants", ()):
        runtime = getattr(getattr(variant, "info", None), "runtime", None)
        raw_device = getattr(runtime, "device_type", "")
        variant_device = str(getattr(raw_device, "value", raw_device)).upper()
        if variant_device == preference:
            matching.append(variant)

    if not matching:
        available = sorted(
            {
                str(
                    getattr(
                        getattr(getattr(item, "info", None), "runtime", None),
                        "device_type",
                        "UNKNOWN",
                    )
                )
                for item in getattr(model, "variants", ())
            }
        )
        raise FoundryRuntimeError(
            f"'{getattr(model, 'alias', 'model')}' için {preference} varyantı yok. "
            f"Kullanılabilir aygıtlar: {', '.join(available) or 'NONE'}"
        )

    matching.sort(key=lambda variant: not bool(getattr(variant, "is_cached", False)))
    model.select_variant(matching[0])
    return model
