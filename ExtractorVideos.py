#!/usr/bin/env python3
"""
ExtractorVideos.py  –  Lógica de chunks + llamada a Groq,
con fallback a yt-dlp + webvtt-py preservando timestamps y chunking.
"""

import os
import re
import logging
from typing import List, Dict
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from yt_dlp import YoutubeDL
import requests
from io import StringIO
import webvtt
from groq import Groq

# ── Configuración básica ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ No se encontró la variable GROQ_API_KEY en el entorno.")
client = Groq(api_key=API_KEY)

# ── Helpers de tiempo y chunking ────────────────────────────────────────────────
def extract_video_id(arg: str) -> str:
    m = re.search(r"(?:v=|be/)([\w-]{11})", arg)
    return m.group(1) if m else arg

def seconds_to_timestamp(sec: float) -> str:
    total_ms = int(sec * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def parse_segments(
    raw: List[Dict[str, float]],
    max_seconds: int = 30,
    max_chars:   int = 500,
) -> List[Dict[str, str]]:
    """
    Agrupa raw=[{"start","end","text"}] en chunks según duración y longitud.
    Devuelve [{"ts_range","text"},…].
    """
    chunks, curr = [], None
    for seg in raw:
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

# ── Extracción principal con fallback ──────────────────────────────────────────
def get_timestamped_chunks(
    video_id: str,
    languages=("es","en")
) -> List[Dict[str, str]]:
    """
    Intenta con youtube-transcript-api; si falla, usa fallback yt-dlp + webvtt.
    Devuelve lista de chunks con ts_range y text.
    """
    try:
        logging.info("Intentando youtube-transcript-api")
        raw = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        segs = [
            {"start": s["start"], "end": s["start"] + s["duration"], "text": s["text"].strip()}
            for s in raw
        ]
        return parse_segments(segs)
    except (TranscriptsDisabled, NoTranscriptFound):
        raise RuntimeError("No hay subtítulos disponibles en español/inglés.")
    except VideoUnavailable:
        raise RuntimeError("El vídeo no está disponible.")
    except Exception as e:
        logging.warning(f"youtube-transcript-api falló ({type(e).__name__}): {e}")
        logging.info("Ha fallado youtube-transcript-api, usando fallback con yt-dlp + webvtt")
        return get_timestamped_chunks_yt_dlp(video_id)

# ── Fallback con yt-dlp + webvtt ──────────────────────────────────────────────
def get_timestamped_chunks_yt_dlp(
    video_id: str,
    languages: tuple = ("es","en"),
    max_seconds: int = 30,
    max_chars:   int = 500
) -> List[Dict[str, str]]:
    """
    Extrae subtítulos con yt-dlp + webvtt, luego chunking igual que get_timestamped_chunks.
    """
    logging.info("FALLBACK: extrayendo con yt-dlp + webvtt")
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(languages),
        "subtitlesformat": "vtt",
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    # Obtén la URL del archivo VTT
    subs = info.get("requested_subtitles") or info.get("automatic_captions") or info.get("subtitles") or {}
    vtt_url = None
    for lang in languages:
        if lang in subs:
            entry = subs[lang]
            if isinstance(entry, list):
                entry = entry[0]
            vtt_url = entry.get("url")
            break
    if not vtt_url:
        raise RuntimeError("No hay subtítulos VTT disponibles en yt-dlp")

    # Descarga y parsea el VTT
    resp = requests.get(vtt_url)
    resp.raise_for_status()
    buffer = StringIO(resp.text)
    try:
        cues = list(webvtt.read_buffer(buffer))
    except Exception as e:
        raise RuntimeError(f"Error al parsear VTT: {e}")

    # Convierte cues a raw segments
    raw = []
    for c in cues:
        h, m, s = c.start.split(":")
        start = int(h)*3600 + int(m)*60 + float(s)
        h, m, s = c.end.split(":")
        end = int(h)*3600 + int(m)*60 + float(s)
        raw.append({
            "start": start,
            "end":   end,
            "text":  c.text.replace("\n"," ").strip()
        })
    return parse_segments(raw, max_seconds=max_seconds, max_chars=max_chars)

# ── Construcción del prompt y llamada a Groq ─────────────────────────────────
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

    # Opción A: usar YouTubeTranscriptApi
    chunks = get_timestamped_chunks(video_id)

    # Opción B: usar yt-dlp + webvtt (descomenta para probar solo este)
    #chunks = get_timestamped_chunks_yt_dlp(video_id)

    prompt = build_qa_prompt(chunks, question)
    return query_groq(prompt)
