"""Shared initialization for the Microsoft Foundry Local SDK."""

from __future__ import annotations

import threading
from typing import Any

from config import FOUNDRY_APP_NAME


class FoundryRuntimeError(RuntimeError):
    """Raised when the Foundry Local runtime cannot be initialized."""


_manager_lock = threading.Lock()


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
                Configuration(app_name=FOUNDRY_APP_NAME)
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
