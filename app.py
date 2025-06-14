#!/usr/bin/env python3
"""
app.py – FastAPI que expone /qa y permite peticiones desde la extensión
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# función que genera el prompt y llama a Groq
from ExtractorVideos import answer_question

# ──────────────────────────────  FastAPI  ──────────────────────────────
app = FastAPI(
    title="YouTube-QA API",
    version="0.1.0",
    docs_url=None,           # Desactiva Swagger UI (/docs)
    redoc_url=None,          # Desactiva ReDoc (/redoc)
    openapi_url=None,        # Desactiva el JSON de la spec (/openapi.json)
)

# ──────────────────────────────  CORS  ────────────────────────────────
allow_origins = [
    "http://localhost:*",
    "chrome-extension://*",          # durante desarrollo
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────  Esquemas  ────────────────────────────
class QARequest(BaseModel):
    video: str
    question: str

class QAResponse(BaseModel):
    answer: str

# ──────────────────────────────  Endpoint  ────────────────────────────
@app.post("/qa", response_model=QAResponse)
async def qa_endpoint(req: QARequest) -> QAResponse:
    """
    Devuelve la respuesta del LLM a la pregunta `req.question`
    basada en la transcripción del vídeo `req.video`.
    """
    try:
        answer = await run_in_threadpool(
            answer_question,
            req.video,
            req.question,
        )
        return QAResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
