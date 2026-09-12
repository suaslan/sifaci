from __future__ import annotations

import sys
from types import SimpleNamespace

from src.foundry_runtime import get_foundry_manager, select_model_variant


def test_manager_is_initialized_when_sdk_instance_is_none(monkeypatch):
    created_manager = object()

    class FakeManager:
        instance = None

        @classmethod
        def initialize(cls, configuration):
            assert configuration.app_name == "sifaci"
            assert configuration.model_cache_dir
            cls.instance = created_manager

    class FakeConfiguration:
        def __init__(self, *, app_name, app_data_dir, model_cache_dir, logs_dir):
            self.app_name = app_name
            self.app_data_dir = app_data_dir
            self.model_cache_dir = model_cache_dir
            self.logs_dir = logs_dir

    fake_sdk = SimpleNamespace(
        Configuration=FakeConfiguration,
        FoundryLocalManager=FakeManager,
    )
    monkeypatch.setitem(sys.modules, "foundry_local_sdk", fake_sdk)

    assert get_foundry_manager() is created_manager


def test_select_model_variant_prefers_cached_requested_device():
    cpu_uncached = SimpleNamespace(
        id="embedding-cpu:1",
        info=SimpleNamespace(runtime=SimpleNamespace(device_type="CPU")),
        is_cached=False,
    )
    cpu_cached = SimpleNamespace(
        id="embedding-cpu:2",
        info=SimpleNamespace(runtime=SimpleNamespace(device_type="CPU")),
        is_cached=True,
    )
    gpu_cached = SimpleNamespace(
        id="embedding-gpu:1",
        info=SimpleNamespace(runtime=SimpleNamespace(device_type="GPU")),
        is_cached=True,
    )

    class FakeModel:
        alias = "embedding"
        variants = [gpu_cached, cpu_uncached, cpu_cached]
        selected = None

        def select_variant(self, variant):
            self.selected = variant

    model = FakeModel()
    assert select_model_variant(model, "CPU") is model
    assert model.selected is cpu_cached
