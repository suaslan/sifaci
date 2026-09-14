"""HTTP API exposing the local, safety-constrained Şifacı RAG pipeline."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import (
    API_HOST,
    API_PORT,
    APP_NAME,
    DATABASE_PATH,
    EMBEDDING_MODEL_NAME,
    FOUNDRY_MODEL_ALIAS,
    MAX_QUESTION_CHARS,
)
from src.database import (
    get_readiness_stats,
    initialize_database,
    require_database_content,
)
from src.rag import answer_query
from src.response_format import parse_source_names, split_rag_response


LOGGER = logging.getLogger(__name__)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class AnswerResponse(BaseModel):
    answer: str
    medicine: str | None = None
    sources: list[str]
    disclaimer: str
    suggestions: list[str] = Field(default_factory=list, max_length=5)


class HealthResponse(BaseModel):
    status: str
    catalog_medicine_count: int
    ready_medicine_count: int
    chunk_count: int
    embedded_chunk_count: int
    llm: str
    embedding_model: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    stats = require_database_content()
    LOGGER.info(
        "Medicine database ready: path=%s medicines=%s chunks=%s",
        DATABASE_PATH,
        stats["medicine_count"],
        stats["chunk_count"],
    )
    yield


app = FastAPI(
    title=f"{APP_NAME} API",
    description="Yerel ilaç bilgi asistanı RAG API'si.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    stats = get_readiness_stats()
    return HealthResponse(
        status="ok",
        catalog_medicine_count=stats["catalog_medicine_count"],
        ready_medicine_count=stats["ready_medicine_count"],
        chunk_count=stats["chunk_count"],
        embedded_chunk_count=stats["embedded_chunk_count"],
        llm=f"Microsoft Foundry Local: {FOUNDRY_MODEL_ALIAS}",
        embedding_model=EMBEDDING_MODEL_NAME,
    )


@app.post("/answer", response_model=AnswerResponse, response_model_exclude_none=True)
def answer_medicine_question(payload: AnswerRequest) -> AnswerResponse:
    return _answer(payload.question)


@app.post("/api/chat", response_model=AnswerResponse, response_model_exclude_none=True)
def chat_medicine_question(payload: ChatRequest) -> AnswerResponse:
    """Public RAG endpoint used by the frontend chat flow."""

    return _answer(payload.message)


def _answer(raw_question: str) -> AnswerResponse:
    question = raw_question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Soru boş olamaz.")

    try:
        debug_trace: dict[str, object] = {}
        raw_response = answer_query(question, debug_trace=debug_trace)
    except Exception as error:
        LOGGER.exception("Şifacı API could not generate an answer")
        raise HTTPException(
            status_code=503,
            detail=(
                "Yanıt oluşturulamadı. Foundry Local kayıtlarını ve "
                "ilaç bilgi tabanını kontrol edin."
            ),
        ) from error

    answer, source_names, disclaimer = split_rag_response(raw_response)
    parsed_sources = parse_source_names(source_names)
    raw_suggestions = debug_trace.get("similar_medicines")
    suggestions = (
        list(
            dict.fromkeys(
                str(item).strip()
                for item in raw_suggestions
                if str(item).strip()
            )
        )[:5]
        if isinstance(raw_suggestions, list)
        else []
    )
    top_chunks = debug_trace.get("top_chunks")
    medicine = None
    if isinstance(top_chunks, list) and top_chunks:
        first_chunk = top_chunks[0]
        if isinstance(first_chunk, dict):
            medicine = str(first_chunk.get("medicine_name") or "").strip() or None
    if medicine is None:
        detected = debug_trace.get("detected_medicine")
        medicine = str(detected).strip() if detected else None
    return AnswerResponse(
        answer=answer,
        medicine=medicine,
        sources=parsed_sources,
        disclaimer=disclaimer,
        suggestions=suggestions,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
