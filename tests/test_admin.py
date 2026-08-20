from __future__ import annotations

import pytest

from src.admin import save_medicine_from_form, validate_medicine_form
from src.database import get_chunks, get_medicine_by_name


def test_validate_requires_only_medicine_name():
    with pytest.raises(ValueError, match="İlaç adı zorunludur"):
        validate_medicine_form({"medicine_name": "   "})

    record = validate_medicine_form({"medicine_name": " Test ilaç "})

    assert record["medicine_name"] == "Test ilaç"
    assert record["active_ingredient"] is None
    assert record["source_reference"] is None


def test_save_form_keeps_blanks_and_embeds_nonempty_fields(tmp_path):
    database_path = tmp_path / "medicines.db"
    values = {
        "medicine_name": "FORM TEST 10 mg",
        "active_ingredient": "Test maddesi",
        "indications": "Kayıtlı endikasyon metni.",
        "usage_information": "",
        "dosage_information": "",
        "frequency_information": "",
        "route_of_administration": "",
        "common_side_effects": "Kayıtlı yaygın yan etki.",
        "serious_side_effects": "",
        "warnings": "",
        "contraindications": "",
        "interactions": "",
        "source_name": "Resmî kaynak",
        "source_reference": "REF-1",
    }

    medicine_id, chunk_count = save_medicine_from_form(
        values,
        database_path=database_path,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
    )
    medicine = get_medicine_by_name("FORM TEST 10 mg", database_path=database_path)
    chunks = get_chunks(medicine_id, database_path=database_path)

    assert medicine is not None
    assert medicine["usage_information"] is None
    assert medicine["dosage_information"] is None
    assert medicine["source_name"] == "Resmî kaynak"
    assert chunk_count == 3
    assert len(chunks) == 3
    assert all(chunk["embedding"] == [0.1, 0.2, 0.3] for chunk in chunks)


def test_name_only_record_is_allowed(tmp_path):
    database_path = tmp_path / "medicines.db"

    medicine_id, chunk_count = save_medicine_from_form(
        {"medicine_name": "Yalnızca isim"},
        database_path=database_path,
        embedding_function=lambda _: (_ for _ in ()).throw(
            AssertionError("embedding must not be called")
        ),
    )

    assert medicine_id > 0
    assert chunk_count == 0
    assert get_chunks(medicine_id, database_path=database_path) == []
