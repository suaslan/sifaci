from __future__ import annotations

from fastapi.testclient import TestClient

import api
from src.rag import DISCLAIMER


def _ready_database_stats():
    return {"medicine_count": 1, "chunk_count": 1}


def _test_client(monkeypatch):
    """Create an API client without depending on the developer's local database."""

    monkeypatch.setattr(api, "initialize_database", lambda: None)
    monkeypatch.setattr(api, "require_database_content", _ready_database_stats)
    return TestClient(api.app)


def test_answer_endpoint_returns_structured_rag_response(monkeypatch):
    monkeypatch.setattr(
        api,
        "answer_query",
        lambda _question, **_kwargs: (
            "Kayıtlı cevap.\n\n"
            "Kaynaklar: TİTCK KÜB, TİTCK KT\n\n"
            f"{DISCLAIMER}"
        ),
    )
    with _test_client(monkeypatch) as client:
        response = client.post("/answer", json={"question": "Yan etkileri nelerdir?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Kayıtlı cevap.",
        "sources": ["TİTCK KÜB", "TİTCK KT"],
        "disclaimer": DISCLAIMER,
        "suggestions": [],
    }


def test_answer_endpoint_rejects_blank_question(monkeypatch):
    with _test_client(monkeypatch) as client:
        response = client.post("/answer", json={"question": "   "})

    assert response.status_code == 422


def test_answer_endpoint_hides_internal_errors(monkeypatch):
    def fail(_question: str) -> str:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api, "answer_query", fail)

    with _test_client(monkeypatch) as client:
        response = client.post("/answer", json={"question": "Soru"})

    assert response.status_code == 503
    assert "secret internal detail" not in response.text


def test_health_endpoint_returns_database_and_model_status(monkeypatch):
    monkeypatch.setattr(api, "get_database_stats", _ready_database_stats)
    with _test_client(monkeypatch) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["medicine_count"], int)
    assert isinstance(payload["chunk_count"], int)
    assert payload["llm"]
    assert payload["embedding_model"]
    assert "medicine_names" not in payload


def test_missing_answer_never_lists_catalog_medicines(monkeypatch):
    monkeypatch.setattr(
        api,
        "answer_query",
        lambda _question, **_kwargs: (
            "İlgili bilgi belgelerde bulunamadı.\n\n"
            f"Kaynaklar: Bulunamadı\n\n{DISCLAIMER}"
        ),
    )
    with _test_client(monkeypatch) as client:
        response = client.post("/answer", json={"question": "Bilinmeyen ilaç"})

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "available_medicines" not in response.json()


def test_catalog_suggestions_are_unique_and_capped_at_five(monkeypatch):
    def not_found(_question, *, debug_trace):
        debug_trace["similar_medicines"] = [
            "ALFA",
            "ALFA",
            "ALFA PLUS",
            "ALFA FORTE",
            "ALFA DUO",
            "ALFA SR",
            "ALFA EXTRA",
        ]
        return f"Bu ilaç ürün kataloğunda bulunamadı.\n\nKaynaklar: Bulunamadı\n\n{DISCLAIMER}"

    monkeypatch.setattr(api, "answer_query", not_found)
    with _test_client(monkeypatch) as client:
        response = client.post("/answer", json={"question": "Alfaa"})

    assert response.status_code == 200
    assert response.json()["suggestions"] == [
        "ALFA",
        "ALFA PLUS",
        "ALFA FORTE",
        "ALFA DUO",
        "ALFA SR",
    ]
