"""End-to-end retrieval/RAG scenarios with deterministic local test vectors.

``SCENARIOS`` is the executable specification requested for each test: every
entry explicitly declares its input, expected retrieval, and expected answer
behavior. Foundry Local is replaced only at the model boundary; SQLite,
cosine ranking, source formatting, and RAG safety controls run normally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from config import MISSING_INFORMATION_RESPONSE
from src.database import insert_chunk, insert_medicine
from src.rag import DISCLAIMER, answer_query
from src.retrieval import get_top_chunks


@dataclass(frozen=True)
class Scenario:
    input: str
    expected_retrieval: str
    expected_answer_behavior: str


SCENARIOS = {
    "known_side_effect": Scenario(
        input="ALFA ilacının yan etkileri nelerdir?",
        expected_retrieval="ALFA side_effects chunk'ı ilk sırada döner.",
        expected_answer_behavior="Kayıtlı baş ağrısı ve bulantı bilgisi aktarılır.",
    ),
    "frequency_present": Scenario(
        input="ALFA kullanım sıklığı nedir?",
        expected_retrieval="ALFA frequency chunk'ı döner.",
        expected_answer_behavior="Kaynakta bulunan genel kullanım sıklığı aktarılır.",
    ),
    "frequency_missing": Scenario(
        input="BETA kullanım sıklığı nedir?",
        expected_retrieval="BETA için yalnızca indications chunk'ı bulunur.",
        expected_answer_behavior="Sıklık uydurulmaz; bilgi bulunamadı yanıtı verilir.",
    ),
    "unknown_medicine": Scenario(
        input="GAMMA ilacının yan etkileri nelerdir?",
        expected_retrieval="Eşik üstünde hiçbir chunk bulunmaz.",
        expected_answer_behavior="Model çağrılmaz ve bilgi bulunamadı denir.",
    ),
    "personal_dose": Scenario(
        input="Ben 70 kiloyum, ALFA'dan kaç tane almalıyım?",
        expected_retrieval="ALFA'nın ilgili kayıtlı chunk'ı bulunur.",
        expected_answer_behavior="Model çağrılmadan kişisel doz hesabı reddedilir.",
    ),
    "similar_names": Scenario(
        input="ALFA PLUS ilacının yan etkileri nelerdir?",
        expected_retrieval="ALFA PLUS sonucu ALFA sonucundan önce gelir.",
        expected_answer_behavior="Yanıt ALFA PLUS kaynağına dayandırılır.",
    ),
    "low_similarity": Scenario(
        input="Uzak ve alakasız bir soru",
        expected_retrieval="Tüm similarity skorları eşik altında kalır.",
        expected_answer_behavior="Boş retrieval güvenilir sonuç yok olarak ele alınır.",
    ),
    "serious_side_effect": Scenario(
        input="ALFA ilacının ciddi yan etkileri nelerdir?",
        expected_retrieval="ALFA serious_side_effects chunk'ı döner.",
        expected_answer_behavior="Ciddi etki ve acil değerlendirme uyarısı açıkça gösterilir.",
    ),
    "sources_at_end": Scenario(
        input="ALFA ilacının yan etkileri nelerdir?",
        expected_retrieval="ALFA yan etki kaynağı adıyla birlikte döner.",
        expected_answer_behavior="Kaynak adı yanıt metninden sonra, uyarıdan önce gösterilir.",
    ),
    "unsupported_generation": Scenario(
        input="ALFA ilacının yan etkileri nelerdir?",
        expected_retrieval="ALFA yan etki chunk'ı modele bağlam olarak verilir.",
        expected_answer_behavior="Kaynakta olmayan 50 mg önerisi backend tarafından reddedilir.",
    ),
}


QUERY_VECTORS = {
    SCENARIOS["known_side_effect"].input: [1, 0, 0, 0, 0, 0],
    SCENARIOS["frequency_present"].input: [0, 1, 0, 0, 0, 0],
    SCENARIOS["frequency_missing"].input: [0, 0, 0, 0, 1, 0],
    SCENARIOS["unknown_medicine"].input: [0, 0, 0, 0, 0, 1],
    SCENARIOS["personal_dose"].input: [0, 1, 0, 0, 0, 0],
    SCENARIOS["similar_names"].input: [0, 0, 0, 1, 0, 0],
    SCENARIOS["low_similarity"].input: [0, 0, 0, 0, 0, 1],
    SCENARIOS["serious_side_effect"].input: [0, 0, 1, 0, 0, 0],
}


@pytest.fixture()
def scenario_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "scenario_medicines.db"

    alfa_id = insert_medicine(
        {
            "medicine_name": "ALFA 10 mg tablet",
            "source_name": "TİTCK ALFA KT",
            "source_reference": "ALFA-KT-1",
        },
        database_path=database_path,
    )
    insert_chunk(
        alfa_id,
        "Baş ağrısı ve bulantı görülebilir.",
        "side_effects",
        [1, 0, 0, 0, 0, 0],
        database_path=database_path,
    )
    insert_chunk(
        alfa_id,
        "Kayıtlı genel kullanım sıklığı: günde iki kez.",
        "frequency",
        [0, 1, 0, 0, 0, 0],
        database_path=database_path,
    )
    insert_chunk(
        alfa_id,
        "Solunum güçlüğü ciddi bir yan etkidir; derhal acil değerlendirme gerekir.",
        "serious_side_effects",
        [0, 0, 1, 0, 0, 0],
        database_path=database_path,
    )

    alfa_plus_id = insert_medicine(
        {
            "medicine_name": "ALFA PLUS 20 mg tablet",
            "source_name": "TİTCK ALFA PLUS KT",
            "source_reference": "ALFA-PLUS-KT-1",
        },
        database_path=database_path,
    )
    insert_chunk(
        alfa_plus_id,
        "Uyku hali görülebilir.",
        "side_effects",
        [0, 0, 0, 1, 0, 0],
        database_path=database_path,
    )

    beta_id = insert_medicine(
        {
            "medicine_name": "BETA 5 mg tablet",
            "source_name": "TİTCK BETA KT",
            "source_reference": "BETA-KT-1",
        },
        database_path=database_path,
    )
    insert_chunk(
        beta_id,
        "BETA için yalnızca kayıtlı endikasyon bilgisi vardır.",
        "indications",
        [0, 0, 0, 0, 1, 0],
        database_path=database_path,
    )
    return database_path


def _retriever(database_path: Path):
    def retrieve(query: str, top_k: int = 5):
        return get_top_chunks(
            query,
            top_k=top_k,
            database_path=database_path,
            embedding_function=lambda text: QUERY_VECTORS[text],
        )

    return retrieve


def _grounded_chat(expected_context: str, answer: str):
    def chat(messages):
        assert expected_context in messages[1]["content"]
        return answer

    return chat


def _chat_must_not_run(_messages):
    raise AssertionError("Bu senaryoda chat modeli çağrılmamalıdır")


def test_known_medicine_side_effect(scenario_database):
    scenario = SCENARIOS["known_side_effect"]
    retrieval = _retriever(scenario_database)

    chunks = retrieval(scenario.input)
    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=_grounded_chat(
            "Baş ağrısı ve bulantı görülebilir.",
            "Kayıtlı yan etkiler baş ağrısı ve bulantıdır.",
        ),
    )

    assert chunks[0]["medicine_name"] == "ALFA 10 mg tablet"
    assert chunks[0]["chunk_type"] == "side_effects"
    assert "baş ağrısı ve bulantı" in answer.casefold()


def test_frequency_present(scenario_database):
    scenario = SCENARIOS["frequency_present"]
    retrieval = _retriever(scenario_database)

    chunks = retrieval(scenario.input)
    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=_grounded_chat("Kategori: frequency", "Günde iki kez kullanılır."),
    )

    assert chunks[0]["chunk_type"] == "frequency"
    assert "Günde iki kez" in answer


def test_frequency_missing(scenario_database):
    scenario = SCENARIOS["frequency_missing"]
    retrieval = _retriever(scenario_database)

    chunks = retrieval(scenario.input)
    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=_grounded_chat("Kategori: indications", MISSING_INFORMATION_RESPONSE),
    )

    assert chunks[0]["medicine_name"] == "BETA 5 mg tablet"
    assert all(chunk["chunk_type"] != "frequency" for chunk in chunks)
    assert answer.startswith(MISSING_INFORMATION_RESPONSE)


def test_unknown_medicine(scenario_database):
    scenario = SCENARIOS["unknown_medicine"]
    retrieval = _retriever(scenario_database)

    assert retrieval(scenario.input) == []
    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=_chat_must_not_run,
    )

    assert answer.startswith(MISSING_INFORMATION_RESPONSE)
    assert "Kaynaklar: Bulunamadı" in answer


def test_personalized_dose_request(scenario_database):
    scenario = SCENARIOS["personal_dose"]
    retrieval = _retriever(scenario_database)

    assert retrieval(scenario.input)[0]["medicine_name"] == "ALFA 10 mg tablet"
    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=_chat_must_not_run,
    )

    assert "Kişiye özel doz" in answer
    assert "doktorunuza veya eczacınıza danışın" in answer


def test_similar_medicine_names(scenario_database):
    scenario = SCENARIOS["similar_names"]
    retrieval = _retriever(scenario_database)

    chunks = retrieval(scenario.input)
    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=_grounded_chat("Uyku hali görülebilir.", "Uyku hali görülebilir."),
    )

    assert chunks[0]["medicine_name"] == "ALFA PLUS 20 mg tablet"
    assert "Kaynaklar: TİTCK ALFA PLUS KT" in answer


def test_low_similarity_score(scenario_database):
    scenario = SCENARIOS["low_similarity"]
    retrieval = _retriever(scenario_database)

    chunks = retrieval(scenario.input)
    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=_chat_must_not_run,
    )

    assert chunks == []
    assert answer.startswith(MISSING_INFORMATION_RESPONSE)


def test_serious_side_effect(scenario_database):
    scenario = SCENARIOS["serious_side_effect"]
    retrieval = _retriever(scenario_database)

    chunks = retrieval(scenario.input)
    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=_grounded_chat(
            "Kategori: serious_side_effects",
            "Solunum güçlüğü ciddi bir etkidir; derhal acil değerlendirme gerekir.",
        ),
    )

    assert chunks[0]["chunk_type"] == "serious_side_effects"
    assert "ciddi" in answer.casefold()
    assert "acil" in answer.casefold()


def test_source_information_is_shown_at_answer_end(scenario_database):
    scenario = SCENARIOS["sources_at_end"]
    retrieval = _retriever(scenario_database)

    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=lambda _: "Kayıtlı yan etki cevabı.",
    )

    source_position = answer.rindex("Kaynaklar: TİTCK ALFA KT")
    disclaimer_position = answer.rindex(DISCLAIMER)
    assert source_position > answer.index("Kayıtlı yan etki cevabı.")
    assert disclaimer_position > source_position
    assert answer.endswith(DISCLAIMER)


def test_model_cannot_add_information_outside_context(scenario_database):
    scenario = SCENARIOS["unsupported_generation"]
    retrieval = _retriever(scenario_database)

    answer = answer_query(
        scenario.input,
        retrieval_function=retrieval,
        chat_function=lambda _: "Bu ilaçtan günde 50 mg kullanılmalıdır.",
    )

    assert answer.startswith(MISSING_INFORMATION_RESPONSE)
    assert "50 mg" not in answer
