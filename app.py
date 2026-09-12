"""Streamlit interface for the local medicine information assistant."""

from __future__ import annotations

import logging
from typing import Any

from config import (
    APP_DESCRIPTION,
    APP_TITLE,
    DEBUG,
    EMBEDDING_MODEL_NAME,
    FOUNDRY_MODEL_ALIAS,
    MAX_QUESTION_CHARS,
)
from src.database import get_database_stats, initialize_database, require_database_content
from src.rag import DISCLAIMER, answer_query
from src.response_format import split_rag_response


LOGGER = logging.getLogger(__name__)
def _render_assistant_message(st: Any, message: dict[str, Any]) -> None:
    with st.chat_message("assistant", avatar="💊"):
        st.markdown(message["content"])
        st.markdown("##### Kullanılan kaynaklar")
        if message["sources"] == "Bulunamadı":
            st.caption("Güvenilir bir kaynak eşleşmesi bulunamadı.")
        else:
            st.write(message["sources"])
        st.caption(message["disclaimer"])

        if DEBUG and message.get("debug"):
            _render_debug_trace(st, message["debug"])


def _render_debug_trace(st: Any, trace: dict[str, Any]) -> None:
    """Render development-only retrieval details in a collapsed panel."""

    with st.expander("Debug ayrıntıları", expanded=False):
        embedding_status = (
            "Evet" if trace.get("query_embedding_created") else "Hayır"
        )
        st.markdown(f"**Query embedding oluşturuldu mu?** {embedding_status}")
        if trace.get("query_embedding_created"):
            st.write(
                "Boyut:",
                trace.get("query_embedding_dimension"),
                "· İlk değerler:",
                trace.get("query_embedding_preview", []),
            )

        st.markdown("**Bulunan top chunk'lar ve similarity skorları**")
        top_chunks = trace.get("top_chunks", [])
        if top_chunks:
            rows = [
                {
                    "medicine_name": chunk.get("medicine_name"),
                    "chunk_type": chunk.get("chunk_type"),
                    "similarity_score": round(
                        float(chunk.get("similarity_score", 0.0)), 6
                    ),
                    "chunk_text": chunk.get("chunk_text"),
                }
                for chunk in top_chunks
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("Similarity eşiğini geçen chunk bulunamadı.")

        st.markdown("**Modele gönderilen retrieved context**")
        st.code(
            trace.get("retrieved_context") or "Retrieved context oluşturulmadı.",
            language="text",
        )
        st.write("Model çağrıldı:", "Evet" if trace.get("model_called") else "Hayır")
        if trace.get("model_call_block_reason"):
            st.caption(trace["model_call_block_reason"])
        if trace.get("response_safety_action"):
            st.warning(trace["response_safety_action"])


def _render_sidebar(st: Any) -> None:
    stats = get_database_stats()
    with st.sidebar:
        st.header("Sistem bilgileri")
        first_column, second_column = st.columns(2)
        first_column.metric("İlaç sayısı", stats["medicine_count"])
        second_column.metric("Chunk sayısı", stats["chunk_count"])

        st.markdown("**Kullanılan LLM**")
        st.code(FOUNDRY_MODEL_ALIAS, language="text")
        st.markdown("**Embedding modeli**")
        st.code(EMBEDDING_MODEL_NAME, language="text")

        if DEBUG:
            st.warning("DEBUG modu açık · Retrieval ayrıntıları gösteriliyor.")

        st.divider()
        st.warning(
            "Bu asistan kişisel reçete, tanı, doz hesabı veya tedavi önerisi vermez."
        )
        if st.button("Sohbeti temizle", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💊",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    initialize_database()
    try:
        require_database_content()
    except RuntimeError as error:
        st.error(str(error))
        st.code("python -m src.ingestion", language="powershell")
        st.stop()

    st.markdown(
        """
        <style>
        .block-container {max-width: 900px; padding-top: 2.5rem;}
        [data-testid="stChatMessage"] {border-radius: 0.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _render_sidebar(st)

    st.title(APP_TITLE)
    st.caption(APP_DESCRIPTION)
    st.info(
        "Bu sistem yalnızca kayıtlı ürün bilgilerini görüntüler; kişisel reçete "
        "veya tedavi önerisi vermez. Acil bir durumda sağlık hizmetine başvurun."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            "Örneğin: *X ilacının yan etkileri nelerdir?*"
        )

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            _render_assistant_message(st, message)

    user_question = st.chat_input(
        "Bir ilaç hakkında sorunuzu yazın...",
        max_chars=MAX_QUESTION_CHARS,
    )
    if not user_question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_question)

    with st.spinner("Kayıtlı kaynaklar taranıyor ve yanıt hazırlanıyor..."):
        debug_trace: dict[str, Any] | None = {} if DEBUG else None
        try:
            raw_response = answer_query(user_question, debug_trace=debug_trace)
            answer, sources, disclaimer = split_rag_response(raw_response)
        except Exception:
            LOGGER.exception("Medicine assistant could not generate a response")
            answer = (
                "Yanıt şu anda oluşturulamadı. Foundry Local modellerinin "
                "hazır olduğunu ve veritabanında embedding bulunduğunu kontrol edin."
            )
            sources = "Bulunamadı"
            disclaimer = DISCLAIMER

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "disclaimer": disclaimer,
    }
    if DEBUG and debug_trace is not None:
        assistant_message["debug"] = debug_trace
    st.session_state.messages.append(assistant_message)
    _render_assistant_message(st, assistant_message)


if __name__ == "__main__":
    main()
