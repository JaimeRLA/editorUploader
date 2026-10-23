"""Elige y guioniza los Shorts con un LLM local (Ollama) y el mapa de calor de YouTube.

1. El transcript se divide en frases numeradas y en ventanas solapadas de unos minutos.
2. Por cada ventana, el LLM propone Shorts como HISTORIAS (por número de frase, no por segundos:
   los modelos pequeños se inventan los tiempos): un gancho inicial (la frase más fuerte, que
   luego vuelve a aparecer en su contexto), de 1 a 4 tramos en orden que cuentan la historia
   saltándose el relleno, y las palabras clave. Los puntúa con una rúbrica fija.
3. Los mejores candidatos de todo el vídeo se comparan entre sí en una segunda llamada,
   porque las notas de ventanas distintas no son comparables.
4. Si YouTube publica el mapa de calor («lo más repetido»), se mezcla con la nota del LLM.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import requests

from .music import MOODS
from .timeline import Words, duration, speech_pieces
from .utils import log

WINDOW_SECONDS = 10 * 60
WINDOW_OVERLAP = 90
SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
WEAK_STARTS = {"y", "pero", "entonces", "pues", "o", "and", "but", "so", "because", "porque", "que"}

RUBRIC = """Eres un editor experto en YouTube Shorts, TikTok y Reels virales.
Te paso un fragmento de la transcripción de un vídeo, en líneas numeradas «[n] mm:ss texto».
Elige los mejores momentos para convertirlos en Shorts verticales. Un buen Short:
- GANCHO: la primera frase capta la atención por sí sola en 1-3 segundos (pregunta intrigante,
  afirmación polémica o sorprendente, dato impactante, inicio de una historia, emoción fuerte).
- AUTOCONTENIDO: se entiende sin haber visto el resto del vídeo. No empieza con «y», «pero»,
  «entonces» ni hace referencia a algo dicho antes que el espectador no conoce.
- REMATE: termina en un cierre satisfactorio (conclusión, chiste, revelación, consejo, frase
  memorable), nunca a mitad de una idea.
- INTENSIDAD: humor, sorpresa, confesión, conflicto, opinión fuerte, historia personal o un
  consejo práctico muy concreto.
Puntúa cada criterio de 0 a 10 con honestidad: un 9-10 es excepcional. Si en el fragmento no hay
nada que merezca un Short, devuelve la lista vacía."""

STORY = """
Monta cada Short como una HISTORIA con planteamiento, tensión y desenlace:
- cold_open_line: la frase MÁS impactante (revelación, giro, frase polémica o graciosa). Se pone al
  principio del Short como gancho y luego vuelve a sonar en su contexto. -1 si no hay ninguna buena.
- segments: de 1 a 4 tramos de líneas consecutivas, en orden cronológico, que juntos cuentan la
  historia completa. Salta las líneas de relleno, digresiones y repeticiones que haya entre tramos,
  pero que se entienda todo sin ellas. El último tramo termina en el remate.
- keywords: de 2 a 5 palabras del propio texto que conviene resaltar en los subtítulos (las que
  llevan el peso de la historia: nombres, cifras, palabras emocionales)."""

SIMPLE = """
- segments: un único tramo de líneas consecutivas (start_line y end_line). cold_open_line: -1.
- keywords: de 2 a 5 palabras del propio texto que conviene resaltar en los subtítulos."""

LANGUAGE_NAMES = {"es": "español", "en": "inglés", "pt": "portugués", "fr": "francés", "it": "italiano", "de": "alemán"}


@dataclass
class Profile:
    """Lo que el canal de destino pide al LLM: temática, enfoque e idioma de los títulos."""

    channel: str = ""
    niche: str = ""
    focus: str = ""
    avoid: str = ""
    language: str = ""  # idioma de títulos y descripciones; vacío = el del vídeo
    moods: list[str] = field(default_factory=lambda: list(MOODS))
    min_duration: float = 60
    max_duration: float = 90
    story: bool = True  # gancho inicial + varios tramos; si no, un solo tramo seguido
    jump_cuts: bool = True  # la duración se mide ya sin silencios

    def key(self) -> str:
        """Huella del perfil: si cambia, el análisis cacheado deja de valer."""
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    def system_prompt(self) -> str:
        lines = [RUBRIC + (STORY if self.story else SIMPLE)]
        if self.niche or self.channel:
            lines.append(
                f"\nLos Shorts son para el canal «{self.channel}», especializado en: {self.niche}.\n"
                "Prioriza los momentos que encajan con esa temática y que interesarían a su audiencia; "
                "si un momento no tiene nada que ver con la temática, no lo propongas."
            )
        if self.focus:
            lines.append(f"Busca especialmente: {self.focus}.")
        if self.avoid:
            lines.append(f"Evita: {self.avoid}.")
        return "\n".join(lines)

    def title_language(self) -> str:
        if not self.language:
            return "en el idioma del vídeo"
        return f"en {LANGUAGE_NAMES.get(self.language, self.language)}"


def _window_schema(moods: list[str]) -> dict:
    schema = json.loads(json.dumps(WINDOW_SCHEMA))
    schema["properties"]["clips"]["items"]["properties"]["mood"]["enum"] = moods
    return schema


WINDOW_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cold_open_line": {"type": "integer"},
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
                            "required": ["start_line", "end_line"],
                        },
                    },
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "description": {"type": "string"},
                    "hook_score": {"type": "integer"},
                    "standalone_score": {"type": "integer"},
                    "payoff_score": {"type": "integer"},
                    "intensity_score": {"type": "integer"},
                    "reason": {"type": "string"},
                    "mood": {"type": "string", "enum": list(MOODS)},
                },
                "required": [
                    "cold_open_line", "segments", "keywords", "title", "hook", "description",
                    "hook_score", "standalone_score", "payoff_score", "intensity_score", "reason", "mood",
                ],
            },
        }
    },
    "required": ["clips"],
}

RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "score": {"type": "integer"}},
                "required": ["id", "score"],
            },
        }
    },
    "required": ["ranking"],
}


class LLMError(RuntimeError):
    pass


@dataclass
class Sentence:
    start: float
    end: float
    text: str


@dataclass
class Candidate:
    start: float
    end: float
    title: str
    hook: str
    description: str
    reason: str
    window_score: float  # 0-100, nota de la rúbrica en su ventana
    mood: str = "chill"  # ambiente para la música de fondo (ver music.MOODS)
    segments: list[list[float]] = field(default_factory=list)  # tramos [inicio, fin] del original, en orden
    cold_open: list[float] | None = None  # frase gancho [inicio, fin] que abre el Short
    keywords: list[str] = field(default_factory=list)
    duration: float = 0.0  # duración del Short ya montado (sin silencios)
    refined: bool = False  # ya pasó por la reedición de guion (refine_story)
    rank_score: float | None = None  # 0-100, comparación global
    heat_score: float | None = None  # 0-100, percentil en el mapa de calor
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Ollama:
    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model
        self.think_supported = True

    def check(self) -> None:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"No puedo conectar con Ollama en {self.url} ({exc}). ¿Está arrancado?") from exc
        names = {m.get("name") for m in resp.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise LLMError(f"Ollama no tiene el modelo {self.model}. Descárgalo con: ollama pull {self.model}")

    def chat(self, system: str, user: str, schema: dict, num_ctx: int = 12288) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.2, "num_ctx": num_ctx},
            "keep_alive": "5m",
        }
        if self.think_supported:
            body["think"] = False  # con salida JSON estricta, «pensar» solo añade minutos
        for attempt in range(3):
            try:
                resp = requests.post(f"{self.url}/api/chat", json=body, timeout=600)
            except requests.RequestException as exc:
                if attempt == 2:
                    raise LLMError(f"Ollama no responde: {exc}") from exc
                continue
            if resp.status_code == 400 and "think" in resp.text and "think" in body:
                self.think_supported = False
                body.pop("think")
                continue
            if resp.status_code >= 400:
                raise LLMError(f"Ollama → HTTP {resp.status_code}: {resp.text[:300]}")
            content = resp.json().get("message", {}).get("content", "")
            try:
                return json.loads(content)
            except ValueError:
                continue  # JSON roto: se repite la petición
        raise LLMError("Ollama devolvió JSON no válido tres veces")

    def unload(self) -> None:
        """Saca el modelo de la VRAM (el render la necesita para NVENC)."""
        try:
            requests.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass


def build_sentences(words: list[dict], max_words: int = 30, gap: float = 0.8) -> list[Sentence]:
    sentences: list[Sentence] = []
    current: list[dict] = []
    for i, word in enumerate(words):
        current.append(word)
        next_gap = words[i + 1]["s"] - word["e"] if i + 1 < len(words) else 99
        if word["w"].endswith(SENTENCE_END) or next_gap > gap or len(current) >= max_words:
            sentences.append(Sentence(current[0]["s"], current[-1]["e"], " ".join(w["w"] for w in current)))
            current = []
    return sentences


def _windows(sentences: list[Sentence]) -> list[tuple[int, int]]:
    """Rangos [i, j) de frases de ~WINDOW_SECONDS, solapados WINDOW_OVERLAP segundos."""
    windows, start = [], 0
    while start < len(sentences):
        end = start
        while end < len(sentences) and sentences[end].end - sentences[start].start < WINDOW_SECONDS:
            end += 1
        end = max(end, start + 1)
        windows.append((start, end))
        if end >= len(sentences):
            break
        next_start = end
        while next_start > start + 1 and sentences[end - 1].end - sentences[next_start - 1].start < WINDOW_OVERLAP:
            next_start -= 1
        start = next_start
    return windows


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _story(clip: dict, sentences: list[Sentence], words: Words, a: int, b: int,
           profile: Profile) -> tuple[list[list[int]], int | None, float] | None:
    """Valida los tramos propuestos y los ajusta para que el Short montado dure lo pedido.

    Devuelve (tramos [[línea_ini, línea_fin], …], línea del gancho o None, duración montada).
    """
    min_d, max_d = profile.min_duration, profile.max_duration
    segs = []
    for seg in clip.get("segments") or []:
        try:
            i, j = int(seg["start_line"]), int(seg["end_line"])
        except (KeyError, TypeError, ValueError):
            continue
        i, j = sorted((i, j))
        i, j = max(a, i), min(b - 1, j)
        if i <= j:
            segs.append([i, j])
    if not segs:
        return None
    segs.sort()
    if not profile.story:
        segs = [[segs[0][0], segs[-1][1]]]
    merged = [segs[0]]
    for i, j in segs[1:]:
        if i <= merged[-1][1] + 1:  # tramos pegados o solapados: uno solo
            merged[-1][1] = max(merged[-1][1], j)
        else:
            merged.append([i, j])
    segs = merged[:4]
    # Si empieza con una muletilla de continuación («pero…», «y…») se adelanta una frase.
    first = sentences[segs[0][0]].text.split(" ", 1)[0].strip(",.¿¡").lower()
    if first in WEAK_STARTS and segs[0][0] > 0:
        segs[0][0] -= 1

    cold = None
    line = clip.get("cold_open_line", -1)
    if profile.story and isinstance(line, int) and a <= line < b and line != segs[0][0]:
        sentence = sentences[line]
        if 1.2 <= sentence.end - sentence.start <= 9:
            cold = line

    def span(i: int, j: int) -> float:
        return duration(speech_pieces(words, sentences[i].start, sentences[j].end, profile.jump_cuts))

    def total() -> float:
        d = sum(span(i, j) for i, j in segs)
        return d + (span(cold, cold) if cold is not None else 0)

    while total() < min_d and segs[-1][1] + 1 < len(sentences):
        segs[-1][1] += 1
    while total() > max_d:
        if segs[-1][1] > segs[-1][0]:
            segs[-1][1] -= 1
        elif len(segs) > 1:
            segs.pop()
        elif cold is not None:
            cold = None
        else:
            break
    final = total()
    if final < min_d * 0.97 or final > max_d * 1.05:
        return None
    return segs, cold, final


def _heat_percentile(heatmap: list[dict], start: float, end: float) -> float | None:
    if not heatmap:
        return None
    total = weight = 0.0
    for bucket in heatmap:
        overlap = min(end, bucket["end_time"]) - max(start, bucket["start_time"])
        if overlap > 0:
            total += bucket["value"] * overlap
            weight += overlap
    if not weight:
        return None
    mean = total / weight
    values = [b["value"] for b in heatmap]
    return 100 * sum(v <= mean for v in values) / len(values)


def _heat(heatmap: list[dict], segments: list[list[float]]) -> float | None:
    parts = [(_heat_percentile(heatmap, s, e), e - s) for s, e in segments]
    parts = [(h, w) for h, w in parts if h is not None]
    if not parts:
        return None
    return sum(h * w for h, w in parts) / sum(w for _, w in parts)


def _overlap_ratio(a: Candidate, b: Candidate) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    return max(0.0, inter) / min(a.end - a.start, b.end - b.start)


def find_highlights(
    words: list[dict],
    llm: Ollama,
    *,
    video_title: str,
    heatmap: list[dict] | None = None,
    profile: Profile | None = None,
    per_window: int = 4,
    rerank_top: int = 40,
) -> list[Candidate]:
    profile = profile or Profile()
    min_duration, max_duration = profile.min_duration, profile.max_duration
    system, schema = profile.system_prompt(), _window_schema(profile.moods)
    timed = Words(words)
    sentences = build_sentences(words)
    if not sentences:
        return []
    windows = _windows(sentences)
    log(f"  buscando momentos con {llm.model}: {len(windows)} bloque/s de ~{WINDOW_SECONDS // 60} min…")

    candidates: list[Candidate] = []
    for n, (a, b) in enumerate(windows, 1):
        lines = "\n".join(f"[{k}] {_mmss(sentences[k].start)} {sentences[k].text}" for k in range(a, b))
        prompt = (
            f"Título del vídeo: {video_title}\n"
            f"Cada Short, sumando sus tramos, debe durar entre {min_duration:.0f} y {max_duration:.0f} segundos.\n"
            f"Propón como máximo {per_window} fragmentos, sin solaparse.\n"
            f"title: título para YouTube {profile.title_language()}, máx. 60 caracteres, atractivo pero "
            "fiel al contenido, sin hashtags. hook: texto corto (máx. 7 palabras, mismo idioma que el "
            "título) para mostrar en pantalla al empezar; que despierte curiosidad sin repetir el título. "
            "description: 1-2 frases y 3 hashtags, en el mismo idioma.\n"
            "mood: ambiente de la música de fondo que mejor acompaña al fragmento: "
            + "; ".join(f"{k} = {MOODS[k]}" for k in profile.moods)
            + ".\n\n"
            f"{lines}"
        )
        try:
            data = llm.chat(system, prompt, schema)
        except LLMError as exc:
            log(f"  aviso: bloque {n}/{len(windows)} sin respuesta del LLM: {exc}")
            continue
        for clip in data.get("clips", [])[:per_window]:
            story = _story(clip, sentences, timed, a, b, profile)
            if not story:
                continue
            segs, cold, edited = story
            scores = [clip.get(k, 0) for k in ("hook_score", "standalone_score", "payoff_score", "intensity_score")]
            hook, standalone, payoff, intensity = (max(0, min(10, int(s))) for s in scores)
            window_score = 10 * (0.35 * hook + 0.25 * standalone + 0.25 * payoff + 0.15 * intensity)
            keywords = [str(k).strip() for k in clip.get("keywords") or [] if str(k).strip()][:5]
            candidates.append(Candidate(
                start=sentences[segs[0][0]].start,
                end=sentences[segs[-1][1]].end,
                segments=[[sentences[i].start, sentences[j].end] for i, j in segs],
                cold_open=[sentences[cold].start, sentences[cold].end] if cold is not None else None,
                keywords=keywords,
                duration=round(edited, 1),
                title=clip.get("title", "").strip(),
                hook=clip.get("hook", "").strip(),
                description=clip.get("description", "").strip(),
                reason=clip.get("reason", "").strip(),
                mood=clip.get("mood") if clip.get("mood") in profile.moods else profile.moods[0],
                window_score=round(window_score, 1),
            ))
        log(f"    bloque {n}/{len(windows)}: {len(candidates)} candidato/s en total")

    candidates = _dedupe(sorted(candidates, key=lambda c: c.window_score, reverse=True))
    candidates = candidates[:rerank_top]  # el resto no cabría en la comparación global
    _rerank(candidates, sentences, llm, profile)
    for cand in candidates:
        cand.heat_score = _heat(heatmap or [], cand.segments)
        llm_score = cand.window_score if cand.rank_score is None else (cand.window_score + cand.rank_score) / 2
        cand.score = round(llm_score if cand.heat_score is None else 0.7 * llm_score + 0.3 * cand.heat_score, 1)
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    kept: list[Candidate] = []
    for cand in candidates:
        if all(_overlap_ratio(cand, other) < 0.3 for other in kept):
            kept.append(cand)
    return kept


def _rerank(candidates: list[Candidate], sentences: list[Sentence], llm: Ollama, profile: Profile) -> None:
    """Segunda pasada: compara los mejores candidatos entre sí con la misma vara."""
    if len(candidates) < 2:
        return
    blocks = []
    for k, cand in enumerate(candidates):
        parts = []
        if cand.cold_open:
            parts.append("[gancho] " + " ".join(s.text for s in sentences if abs(s.start - cand.cold_open[0]) < 0.01))
        for seg_start, seg_end in cand.segments:
            parts.append(" ".join(s.text for s in sentences if s.start >= seg_start - 0.01 and s.end <= seg_end + 0.01))
        words = " … ".join(parts).split()
        limit = 120 if len(candidates) <= 20 else 80
        text = " ".join(words[:limit]) + (" […]" if len(words) > limit else "")
        blocks.append(f"#{k} ({cand.duration:.0f} s) «{cand.title}»\n{text}")
    audience = f" para la audiencia del canal ({profile.niche})" if profile.niche else ""
    prompt = (
        "Estos son los candidatos a Short de un mismo vídeo. Compáralos entre sí y da a cada uno una "
        f"nota de 0 a 100 según su potencial viral como Short{audience} (gancho inicial, se entiende "
        "solo, remate, intensidad). Usa todo el rango: el mejor cerca de 90-100 y el peor claramente "
        "más bajo.\n\n"
        + "\n\n".join(blocks)
    )
    try:
        data = llm.chat(profile.system_prompt(), prompt, RANK_SCHEMA, num_ctx=16384)
    except LLMError as exc:
        log(f"  aviso: no pude comparar los candidatos: {exc}")
        return
    for item in data.get("ranking", []):
        k = item.get("id")
        if isinstance(k, int) and 0 <= k < len(candidates):
            candidates[k].rank_score = float(max(0, min(100, item.get("score", 0))))


SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
        "cut_lines": {"type": "array", "items": {"type": "integer"}},
        "cold_open_line": {"type": "integer"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hook": {"type": "string"},
    },
    "required": ["start_line", "end_line", "cut_lines", "cold_open_line", "keywords", "hook"],
}

MAX_GAP = 20  # segundos máximos de relleno que se quitan entre dos tramos
MAX_CUT_SHARE = 0.35  # parte máxima del fragmento que se puede recortar

SCRIPT_SYSTEM = """Eres un editor profesional de Shorts virales que cuenta historias con cortes.
Recibes la transcripción numerada de un trozo de vídeo; las líneas con * son el fragmento elegido.
Reedítalo para que enganche y se entienda solo:
1. start_line / end_line: dónde empieza y acaba la historia. Ajústalo si hace falta: que empiece
   planteando algo interesante y acabe en el remate (conclusión, chiste, revelación), nunca a mitad.
2. cut_lines: líneas DENTRO de ese rango que sobran: relleno, repeticiones, muletillas, digresiones,
   preguntas del entrevistador que no aportan. Quitarlas debe dejar la historia más tensa sin que
   deje de entenderse. Normalmente sobran entre el 10 % y el 35 % de las líneas.
3. cold_open_line: la frase más fuerte de la historia (el clímax: revelación, giro, dato increíble,
   frase graciosa o polémica), que se pondrá AL PRINCIPIO como gancho y luego volverá a sonar en su
   sitio. Tiene que ser de la mitad o del final de la historia, NUNCA la primera línea. -1 si no hay.
4. keywords: 2 a 5 palabras del texto que llevan el peso (nombres, cifras, palabras emocionales).
5. hook: texto en pantalla de máx. 7 palabras que despierte curiosidad."""


def refine_story(cand: Candidate, words: list[dict], llm: Ollama, profile: Profile, context: float = 75) -> bool:
    """Segunda pasada solo para los Shorts elegidos: el LLM los reedita como historia.

    Decide el rango, las frases que sobran (se convierten en cortes entre tramos) y el gancho.
    Si la propuesta es válida y dura lo pedido, sustituye tramos, gancho y palabras clave.
    """
    if not profile.story:
        return False
    sentences = build_sentences(words)
    a = next((k for k, s in enumerate(sentences) if s.end >= cand.start - context), 0)
    b = next((k for k, s in enumerate(sentences) if s.start > cand.end + context), len(sentences))
    chosen = {k for k in range(a, b) for seg in cand.segments
              if sentences[k].start >= seg[0] - 0.01 and sentences[k].end <= seg[1] + 0.01}
    lines = "\n".join(f"{'*' if k in chosen else ' '}[{k}] {sentences[k].text}" for k in range(a, b))
    prompt = (
        f"Canal: {profile.channel} ({profile.niche}). Título del Short: {cand.title}.\n"
        f"El Short montado (sin las líneas cortadas) debe durar entre {profile.min_duration:.0f} y "
        f"{profile.max_duration:.0f} segundos. Hook {profile.title_language()}.\n\n{lines}"
    )
    try:
        data = llm.chat(SCRIPT_SYSTEM, prompt, SCRIPT_SCHEMA)
    except LLMError as exc:
        log(f"  aviso: sin guion para «{cand.title}»: {exc}")
        return False
    try:
        start, end = sorted((int(data["start_line"]), int(data["end_line"])))
    except (KeyError, TypeError, ValueError):
        return False
    start, end = max(a, start), min(b - 1, end)
    if end - start < 2:
        return False
    # Nunca se corta la primera ni la última línea, ni más del 40 % del total.
    cuts = sorted({int(k) for k in data.get("cut_lines") or [] if isinstance(k, int) and start < k < end})
    cuts = cuts[: int((end - start + 1) * 0.4)]
    segments, current = [], None
    for k in range(start, end + 1):
        if k in cuts:
            current = None
            continue
        if current is None:
            current = {"start_line": k, "end_line": k}
            segments.append(current)
        else:
            current["end_line"] = k
    proposal = {"segments": segments, "cold_open_line": data.get("cold_open_line", -1)}
    story = _story(proposal, sentences, Words(words), a, b, profile)
    if not story:
        return False
    segs, cold, edited = story
    new = [[sentences[i].start, sentences[j].end] for i, j in segs]
    # Los cortes deben quitar relleno, no saltar a otro tema: huecos cortos y poco recorte en total.
    gaps = [b2[0] - b1[1] for b1, b2 in zip(new, new[1:])]
    span = new[-1][1] - new[0][0]
    if any(g > MAX_GAP for g in gaps) or sum(gaps) > span * MAX_CUT_SHARE:
        return False
    cand.segments = new
    cand.cold_open = [sentences[cold].start, sentences[cold].end] if cold is not None else None
    cand.start, cand.end = cand.segments[0][0], cand.segments[-1][1]
    cand.duration = round(edited, 1)
    keywords = [str(k).strip() for k in data.get("keywords") or [] if str(k).strip()][:5]
    cand.keywords = keywords or cand.keywords
    if str(data.get("hook", "")).strip():
        cand.hook = str(data["hook"]).strip()
    return True
