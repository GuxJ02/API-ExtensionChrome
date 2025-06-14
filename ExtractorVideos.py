#!/usr/bin/env python3
"""
ExtractorVideos.py  –  Lógica de chunks + llamada a Groq.
"""

import os
import re
from typing import List, Dict
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from groq import Groq

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ No se encontró la variable GROQ_API_KEY en el entorno.")
client = Groq(api_key=API_KEY)


def extract_video_id(arg: str) -> str:
    m = re.search(r"(?:v=|be/)([\w-]{11})", arg)
    return m.group(1) if m else arg


def seconds_to_timestamp(sec: float) -> str:
    total_ms = int(sec * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def get_timestamped_chunks(
    video_id: str,
    languages=("es", "en"),
    max_seconds: int = 30,
    max_chars: int = 500
) -> List[Dict[str, str]]:
    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except (TranscriptsDisabled, NoTranscriptFound):
        raise RuntimeError("No hay subtítulos disponibles en español/inglés.")
    except VideoUnavailable:
        raise RuntimeError("El vídeo no está disponible.")
    except Exception as e:
        raise RuntimeError(f"Error al obtener transcripción: {e}")

    segs = [
        {"start": s["start"],
         "end":   s["start"] + s["duration"],
         "text":  s["text"].strip()}
        for s in raw
    ]

    chunks = []
    curr = None
    for seg in segs:
        if curr is None:
            curr = seg.copy()
            continue
        dur    = seg["end"] - curr["start"]
        length = len(curr["text"]) + 1 + len(seg["text"])
        if dur > max_seconds or length > max_chars:
            chunks.append(curr)
            curr = seg.copy()
        else:
            curr["end"]   = seg["end"]
            curr["text"] += " " + seg["text"]
    if curr:
        chunks.append(curr)

    result = []
    for ch in chunks:
        ts = f"[{seconds_to_timestamp(ch['start'])}–{seconds_to_timestamp(ch['end'])}]"
        result.append({"ts_range": ts, "text": ch["text"]})
    return result


def build_qa_prompt(chunks: List[Dict[str, str]], question: str) -> str:
    lines = "\n".join(f"{c['ts_range']} {c['text']}" for c in chunks)
    return f"""
Vas a recibir la transcripción segmentada de un vídeo de YouTube, en bloques de texto con su intervalo de tiempo:
{lines}

A continuación, un usuario hará una pregunta concreta sobre el contenido del vídeo.
Tu tarea es:
  • Leer los bloques con sus timestamps.
  • Identificar las partes relevantes.
  • Responder con claridad, precisión y de forma simple, citando los timestamps cuando aporte valor.
  • En caso de citar un timestamp hazlo refiriéndote al minuto y segundo por ejemplo "en el minuto 7 segundo 24 se menciona que..".
    Nota: los subtítulos automáticos pueden contener faltas de ortografía
    o confundir fonemas (p.ej. v/b). Considera coincidencias aproximadas 
    y responde aunque la palabra aparezca con una variación mínima.
Pregunta del usuario:
{question}

Respuesta:
""".strip()


def query_groq(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )
    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""
    return answer.strip()


def answer_question(video_arg: str, question: str) -> str:
    video_id = extract_video_id(video_arg)
    chunks   = get_timestamped_chunks(video_id)
    prompt   = build_qa_prompt(chunks, question)
    return query_groq(prompt)
