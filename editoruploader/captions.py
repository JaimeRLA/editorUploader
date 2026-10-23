"""Subtítulos ASS estilo Shorts: pocas palabras a la vez y la palabra que suena resaltada."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pysubs2

WIDTH, HEIGHT = 1080, 1920
BLACK = pysubs2.Color(0, 0, 0)

_GF = "https://github.com/google/fonts/raw/main/"
# Familia (como la nombra libass) → (fichero en tools/fonts, descarga). Todas con licencia libre.
FONTS = {
    "Poppins ExtraBold": ("Poppins-ExtraBold.ttf", _GF + "ofl/poppins/Poppins-ExtraBold.ttf"),
    "Anton": ("Anton-Regular.ttf", _GF + "ofl/anton/Anton-Regular.ttf"),
    "Bebas Neue": ("BebasNeue-Regular.ttf", _GF + "ofl/bebasneue/BebasNeue-Regular.ttf"),
    "Archivo Black": ("ArchivoBlack-Regular.ttf", _GF + "ofl/archivoblack/ArchivoBlack-Regular.ttf"),
    "Luckiest Guy": ("LuckiestGuy-Regular.ttf", _GF + "apache/luckiestguy/LuckiestGuy-Regular.ttf"),
    "Bangers": ("Bangers-Regular.ttf", _GF + "ofl/bangers/Bangers-Regular.ttf"),
    "Permanent Marker": ("PermanentMarker-Regular.ttf", _GF + "apache/permanentmarker/PermanentMarker-Regular.ttf"),
}


@dataclass
class CaptionStyle:
    font: str = "Poppins ExtraBold"
    font_size: int = 100
    text_color: str = "#FFFFFF"
    highlight_color: str = "#FFE600"
    outline_color: str = "#000000"
    uppercase: bool = True
    words_per_line: int = 3
    position: float = 0.68  # altura relativa (0 arriba, 1 abajo)
    hook_seconds: float = 3.0
    keyword_color: str = "#00E5FF"  # palabras clave del guion

    @property
    def max_chars(self) -> int:
        # Con letra más grande caben menos caracteres por línea en 1080 px de ancho.
        return max(8, round(18 * 100 / max(self.font_size, 40)))


def _color(hex_color: str, alpha: int = 0) -> pysubs2.Color:
    value = hex_color.lstrip("#")
    return pysubs2.Color(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), alpha)


def _chunks(words: list[dict], max_words: int, max_chars: int) -> list[list[dict]]:
    """Agrupa palabras en bloques cortos, cortando en pausas y signos de puntuación."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for i, word in enumerate(words):
        current.append(word)
        text = " ".join(w["w"] for w in current)
        nxt = words[i + 1] if i + 1 < len(words) else None
        pause = nxt is not None and nxt["s"] - word["e"] > 0.35
        too_long = nxt is not None and len(text) + 1 + len(nxt["w"]) > max_chars
        if len(current) >= max_words or pause or too_long or re.search(r"[.,;:?!…]$", word["w"]):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _clean(word: str, upper: bool) -> str:
    word = word.strip(",.;:")  # la puntuación final queda fea en subtítulos grandes
    return word.upper() if upper else word


def build(
    words: list[dict],
    clip_start: float,
    clip_end: float,
    dest: Path,
    *,
    hook: str = "",
    style: CaptionStyle | None = None,
    keywords: list[str] | None = None,
) -> Path:
    """Crea el .ass del clip; `words` lleva tiempos absolutos del vídeo original.

    La palabra que suena va en el color de resaltado; las palabras clave del guion, en su propio
    color y algo más grandes, para que el ojo se vaya a lo importante.
    """
    style = style or CaptionStyle()
    text_color, highlight = _color(style.text_color), _color(style.highlight_color)
    keyword_color = _color(style.keyword_color)
    key_tokens = {_norm(t) for k in keywords or [] for t in k.split() if len(_norm(t)) >= 3}
    outline = max(4, round(style.font_size * 0.08))
    subs = pysubs2.SSAFile()
    subs.info.update({"PlayResX": str(WIDTH), "PlayResY": str(HEIGHT), "WrapStyle": "2"})
    subs.styles["Caption"] = pysubs2.SSAStyle(
        fontname=style.font, fontsize=style.font_size, primarycolor=text_color,
        outlinecolor=_color(style.outline_color), backcolor=pysubs2.Color(0, 0, 0, 120),
        bold=False, outline=outline, shadow=outline // 2, alignment=pysubs2.Alignment.MIDDLE_CENTER,
        marginl=60, marginr=60, marginv=0,
    )
    subs.styles["Hook"] = pysubs2.SSAStyle(
        fontname=style.font, fontsize=round(style.font_size * 0.78), primarycolor=BLACK,
        outlinecolor=pysubs2.Color(255, 255, 255), backcolor=pysubs2.Color(255, 255, 255),
        borderstyle=3, outline=18, shadow=0, alignment=pysubs2.Alignment.TOP_CENTER,
        marginl=90, marginr=90, marginv=230,
    )

    clip_words = [
        {"w": w["w"], "s": max(w["s"], clip_start) - clip_start, "e": min(w["e"], clip_end) - clip_start}
        for w in words
        if w["e"] > clip_start and w["s"] < clip_end
    ]
    x, y = WIDTH // 2, round(HEIGHT * style.position)
    chunks = _chunks(clip_words, style.words_per_line, style.max_chars)
    for n, chunk in enumerate(chunks):
        chunk_start = chunk[0]["s"]
        # Nunca debe pisar al bloque siguiente (dos líneas superpuestas son ilegibles).
        limit = chunks[n + 1][0]["s"] if n + 1 < len(chunks) else clip_end - clip_start
        for i, word in enumerate(chunk):
            start = word["s"] if i else chunk_start
            end = chunk[i + 1]["s"] if i + 1 < len(chunk) else min(word["e"] + 0.15, limit)
            parts = []
            for j, other in enumerate(chunk):
                text = _escape(_clean(other["w"], style.uppercase))
                is_key = _norm(other["w"]) in key_tokens
                if j == i:
                    grow = "\\fscx118\\fscy118" if is_key else ""
                    parts.append(f"{{\\c{_ass_color(highlight)}{grow}}}{text}{{\\c{_ass_color(text_color)}\\fscx100\\fscy100}}")
                elif is_key:
                    parts.append(f"{{\\c{_ass_color(keyword_color)}\\fscx110\\fscy110}}{text}"
                                 f"{{\\c{_ass_color(text_color)}\\fscx100\\fscy100}}")
                else:
                    parts.append(text)
            # Cada bloque nuevo entra deslizándose un poco hacia arriba.
            place = f"\\move({x},{y + 22},{x},{y},0,90)" if i == 0 else f"\\pos({x},{y})"
            line = f"{{{place}}}" + " ".join(parts)
            subs.append(pysubs2.SSAEvent(start=_ms(start), end=_ms(max(end, start + 0.05)), text=line, style="Caption"))

    if hook:
        subs.append(pysubs2.SSAEvent(
            start=0, end=_ms(min(style.hook_seconds, clip_end - clip_start)),
            # Entra con un pequeño rebote de escala y sale con fundido.
            text="{\\fad(0,200)\\fscx40\\fscy40\\t(0,160,\\fscx108\\fscy108)\\t(160,260,\\fscx100\\fscy100)}"
                 + _wrap(_escape(hook), round(23 * 78 / max(style.font_size * 0.78, 30))), style="Hook",
        ))
    dest.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(dest))
    return dest


def _wrap(text: str, max_chars: int, max_lines: int = 3) -> str:
    """Parte el texto en líneas (\\N) de como mucho `max_chars`; el estilo no parte solo."""
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= max_chars:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;:") + "…"
    return "\\N".join(lines)


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def _ass_color(color: pysubs2.Color) -> str:
    return f"&H{color.b:02X}{color.g:02X}{color.r:02X}&"


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))
