from __future__ import annotations

import sys
from types import SimpleNamespace

from src.foundry_runtime import get_foundry_manager


def test_manager_is_initialized_when_sdk_instance_is_none(monkeypatch):
    created_manager = object()

    class FakeManager:
        instance = None

        @classmethod
        def initialize(cls, configuration):
            assert configuration.app_name == "sifaci"
            cls.instance = created_manager

    class FakeConfiguration:
        def __init__(self, *, app_name):
            self.app_name = app_name

    fake_sdk = SimpleNamespace(
        Configuration=FakeConfiguration,
        FoundryLocalManager=FakeManager,
    )
    monkeypatch.setitem(sys.modules, "foundry_local_sdk", fake_sdk)

    assert get_foundry_manager() is created_manager
