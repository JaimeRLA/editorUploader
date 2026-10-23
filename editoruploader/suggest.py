"""Propuesta automática de la configuración de un canal según su temática (LLM local)."""
from __future__ import annotations

import re

from .captions import FONTS
from .channels import CATEGORIES, LANGUAGES
from .config import Settings
from .highlights import Ollama
from .music import MOODS

SCHEMA = {
    "type": "object",
    "properties": {
        "focus": {"type": "string"},
        "avoid": {"type": "string"},
        "music_moods": {"type": "array", "items": {"type": "string", "enum": list(MOODS)}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "searches": {"type": "array", "items": {"type": "string"}},
        "font": {"type": "string", "enum": list(FONTS)},
        "highlight_color": {"type": "string"},
        "category_id": {"type": "string", "enum": list(CATEGORIES)},
    },
    "required": ["focus", "avoid", "music_moods", "tags", "searches", "font", "highlight_color", "category_id"],
}

SYSTEM = """Eres un estratega de canales de YouTube Shorts. Configuras canales que publican clips
verticales de 60-90 segundos sacados de vídeos largos (entrevistas, podcasts, charlas, documentales,
tutoriales…) con licencia Creative Commons. Responde solo con el JSON pedido."""


def suggest(settings: Settings, *, name: str, niche: str, language: str) -> dict:
    """Propone focus, avoid, ambientes musicales, etiquetas, búsquedas, fuente, color y categoría.

    Lanza LLMError si Ollama no está disponible; quien llama decide seguir sin propuesta.
    """
    llm = Ollama(settings.ollama_url, settings.ollama_model)
    llm.check()
    lang = LANGUAGES.get(language, language or "el idioma del canal")
    moods = "; ".join(f"{k} = {v}" for k, v in MOODS.items())
    categories = "; ".join(f"{k} = {v}" for k, v in CATEGORIES.items())
    prompt = f"""Canal: «{name}». Temática: {niche}. Idioma del canal: {lang}.
Propón su configuración:
- focus: qué momentos de un vídeo largo funcionan mejor como Short en esta temática (una frase concreta).
- avoid: qué momentos descartar (una frase; p. ej. publicidad, saludos, momentos que necesitan contexto).
- music_moods: de 1 a 3 ambientes de música de fondo que encajan con la temática ({moods}).
- tags: de 4 a 8 etiquetas de YouTube en {lang}, sin «#».
- searches: de 2 a 4 búsquedas de YouTube en {lang} para encontrar vídeos largos de esta temática
  (entrevistas, podcasts, conferencias, documentales…).
- font: la tipografía de subtítulos que mejor encaja con el tono del canal.
- highlight_color: color hexadecimal (#RRGGBB) para resaltar la palabra que suena; muy legible sobre vídeo.
- category_id: categoría de YouTube ({categories})."""
    data = llm.chat(SYSTEM, prompt, SCHEMA, num_ctx=4096)
    llm.unload()
    color = data.get("highlight_color", "")
    return {
        "focus": str(data.get("focus", "")).strip(),
        "avoid": str(data.get("avoid", "")).strip(),
        "music_moods": [m for m in dict.fromkeys(data.get("music_moods", [])) if m in MOODS][:3],
        "tags": [t.strip().lstrip("#") for t in data.get("tags", []) if t.strip()][:8],
        "searches": [q.strip() for q in data.get("searches", []) if q.strip()][:4],
        "font": data.get("font") if data.get("font") in FONTS else "Poppins ExtraBold",
        "highlight_color": color.upper() if re.fullmatch(r"#[0-9A-Fa-f]{6}", color or "") else "#FFE600",
        "category_id": data.get("category_id") if data.get("category_id") in CATEGORIES else "22",
    }

