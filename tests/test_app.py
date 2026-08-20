from __future__ import annotations

import app

from app import split_rag_response
from src.rag import DISCLAIMER


def test_split_rag_response_extracts_sources_and_disclaimer():
    response = (
        "Kaynağa dayalı cevap.\n\n"
        "Kaynaklar: TİTCK KÜB, TİTCK KT\n\n"
        f"{DISCLAIMER}"
    )

    answer, sources, disclaimer = split_rag_response(response)

    assert answer == "Kaynağa dayalı cevap."
    assert sources == "TİTCK KÜB, TİTCK KT"
    assert disclaimer == DISCLAIMER


def test_split_rag_response_has_safe_fallback():
    answer, sources, disclaimer = split_rag_response("Düz cevap")

    assert answer == "Düz cevap"
    assert sources == "Belirtilmemiş"
    assert disclaimer == DISCLAIMER


class _DebugStreamlit:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name in {"chat_message", "expander"}:
                return self
            return None

        return method


def _assistant_message():
    return {
        "role": "assistant",
        "content": "Yanıt",
        "sources": "Kaynak",
        "disclaimer": DISCLAIMER,
        "debug": {
            "query_embedding_created": True,
            "query_embedding_dimension": 2,
            "query_embedding_preview": [1.0, 0.0],
            "top_chunks": [
                {
                    "medicine_name": "ALFA",
                    "chunk_type": "warnings",
                    "chunk_text": "Uyarı",
                    "similarity_score": 0.9,
                }
            ],
            "retrieved_context": "[KAYNAK 1]",
            "model_called": True,
        },
    }


def test_debug_details_render_only_when_enabled(monkeypatch):
    debug_st = _DebugStreamlit()
    monkeypatch.setattr(app, "DEBUG", True)

    app._render_assistant_message(debug_st, _assistant_message())

    assert any(name == "expander" for name, _, _ in debug_st.calls)
    assert any(name == "dataframe" for name, _, _ in debug_st.calls)
    assert any(name == "code" for name, _, _ in debug_st.calls)

    production_st = _DebugStreamlit()
    monkeypatch.setattr(app, "DEBUG", False)

    app._render_assistant_message(production_st, _assistant_message())

    assert not any(name == "expander" for name, _, _ in production_st.calls)
    assert not any(name == "dataframe" for name, _, _ in production_st.calls)
