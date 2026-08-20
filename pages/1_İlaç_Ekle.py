"""Administrator page for adding medicine records."""

from __future__ import annotations

import logging

import streamlit as st

from src.admin import save_medicine_from_form


LOGGER = logging.getLogger(__name__)


def main() -> None:
    st.set_page_config(
        page_title="İlaç Ekle",
        page_icon="➕",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("İlaç Ekle")
    st.caption("Yönetici ilaç veri girişi ve embedding oluşturma paneli")
    st.warning(
        "Yalnızca doğrulanmış ürün bilgilerini girin. Boş alanlar boş saklanır; "
        "uygulama eksik tıbbi bilgileri tamamlamaz."
    )

    with st.form(
        "add_medicine_form",
        clear_on_submit=False,
        enter_to_submit=False,
        border=True,
    ):
        st.subheader("Temel bilgiler")
        identity_left, identity_right = st.columns(2)
        with identity_left:
            medicine_name = st.text_input(
                "İlaç adı *",
                max_chars=300,
                placeholder="Örn. Ürün adı, doz gücü ve farmasötik form",
            )
            active_ingredient = st.text_area(
                "Etken madde",
                height=100,
                placeholder="Boş bırakılabilir",
            )
        with identity_right:
            source_name = st.text_input(
                "Kaynak adı",
                max_chars=500,
                placeholder="Örn. TİTCK KÜB veya TİTCK KT",
            )
            source_reference = st.text_area(
                "Kaynak referansı",
                height=100,
                placeholder="Resmî belge bağlantısı, belge numarası veya onay tarihi",
            )

        st.subheader("Kullanım bilgileri")
        usage_left, usage_right = st.columns(2)
        with usage_left:
            indications = st.text_area("Endikasyon", height=140)
            usage_information = st.text_area("Kullanım bilgisi", height=140)
            route_of_administration = st.text_area("Uygulama yolu", height=110)
        with usage_right:
            dosage_information = st.text_area("Doz bilgisi", height=140)
            frequency_information = st.text_area("Kullanım sıklığı", height=140)
            interactions = st.text_area("Etkileşimler", height=110)

        st.subheader("Güvenlik bilgileri")
        safety_left, safety_right = st.columns(2)
        with safety_left:
            common_side_effects = st.text_area("Yaygın yan etkiler", height=140)
            warnings = st.text_area("Uyarılar", height=140)
            contraindications = st.text_area("Kontrendikasyonlar", height=140)
        with safety_right:
            serious_side_effects = st.text_area("Ciddi yan etkiler", height=140)
            st.info(
                "Ciddi yan etki ve acil değerlendirme ifadelerini resmî kaynaktaki "
                "anlamı değiştirmeden girin."
            )

        submitted = st.form_submit_button(
            "Kaydet ve embedding oluştur",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    form_values = {
        "medicine_name": medicine_name,
        "active_ingredient": active_ingredient,
        "indications": indications,
        "usage_information": usage_information,
        "dosage_information": dosage_information,
        "frequency_information": frequency_information,
        "route_of_administration": route_of_administration,
        "common_side_effects": common_side_effects,
        "serious_side_effects": serious_side_effects,
        "warnings": warnings,
        "contraindications": contraindications,
        "interactions": interactions,
        "source_name": source_name,
        "source_reference": source_reference,
    }

    try:
        with st.spinner("İlaç kaydediliyor, metin parçalanıyor ve embedding üretiliyor..."):
            medicine_id, chunk_count = save_medicine_from_form(form_values)
    except (TypeError, ValueError) as error:
        st.error(str(error))
        return
    except Exception as error:
        LOGGER.exception("Administrator medicine save failed")
        st.error(
            "Kayıt tamamlanamadı. Veritabanını ve Foundry Local embedding "
            f"modelini kontrol edin. Teknik ayrıntı: {error}"
        )
        return

    st.success(
        f"İlaç kaydedildi. Kayıt numarası: {medicine_id} · Oluşturulan chunk: "
        f"{chunk_count}"
    )
    if chunk_count == 0:
        st.info(
            "Bilgi alanlarının tamamı boş olduğu için embedding oluşturulacak "
            "bir metin parçası bulunmadı. İlaç kaydı boş alanlarla saklandı."
        )


if __name__ == "__main__":
    main()
