"""Capa viral: encuentra los vídeos largos que más están funcionando en la temática de un canal.

1. Búsquedas: las del canal más las que propone el LLM a partir de la temática (se renuevan
   cada día), ordenadas por visitas y limitadas al periodo elegido (semana, mes, año).
   Si el canal tiene canales fuente, también se miran sus vídeos recientes.
2. De los candidatos con más visitas se leen sus métricas completas y se puntúan por:
   - velocidad: visitas por día desde que se publicó,
   - «outlier»: visitas frente a la mediana de los últimos vídeos de su canal (vídeos que rinden
     muy por encima de lo normal en ese canal, la señal más fiable de que el tema engancha),
   - participación: (me gusta + comentarios) por visita.
3. Se descartan los ya usados, los de otro idioma, fuera de duración o excluidos, y se deja
   un informe con el ranking en channels/<id>/viral.txt.
"""
from __future__ import annotations

import base64
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import quote, quote_plus

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .channels import LANGUAGES, Channel
from .config import Settings
from .highlights import LLMError, Ollama
from .media import is_creative_commons, spoken_languages, ytdlp_params
from .sources import SourceError, SourceVideo, _channel_videos_tab, _flat, youtube_video_id
from .utils import log

PERIODS = {"week": 3, "month": 4, "year": 5, "any": None}
SEARCH_RESULTS = 20  # resultados por búsqueda
DETAILED = 20  # candidatos de los que se leen las métricas completas
WORKERS = 4  # consultas a YouTube en paralelo (más dispara su control anti-bots)
MIN_RELEVANCE = 7


class _Silent:
    """yt-dlp escribe los errores aunque esté en modo silencioso; aquí se ignoran (el vídeo se salta)."""

    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def search_params(period: str, duration: int | None, creative_commons: bool) -> str:
    """Parámetro «sp» de la búsqueda de YouTube: orden por visitas + filtros.

    duration: 2 = más de 20 min, 3 = de 4 a 20 min, None = cualquiera.
    """
    filters = b""
    if PERIODS.get(period):
        filters += bytes([0x08, PERIODS[period]])
    filters += bytes([0x10, 0x01])  # solo vídeos (no canales ni listas)
    if duration:
        filters += bytes([0x18, duration])
    if creative_commons:
        filters += bytes([0x30, 0x01])
    message = bytes([0x08, 0x03]) + bytes([0x12, len(filters)]) + filters  # 0x08 0x03 = por visitas
    return quote(base64.b64encode(message).decode(), safe="")


@dataclass
class Viral:
    url: str
    title: str
    channel: str
    duration: float
    views: int
    days: int
    channel_url: str
    likes: int
    comments: int
    license: str
    spoken: list[str]
    score: float = 0.0

    @property
    def velocity(self) -> float:
        return self.views / max(self.days, 1)

    relevance: int = 0  # 0-10: cuánto encaja con la temática del canal (lo decide el LLM)
    channel_median: float | None = None  # visitas medianas de los últimos vídeos de su canal

    @property
    def outlier(self) -> float | None:
        # Sin datos del canal no se puede saber; cuenta como neutro.
        return self.views / self.channel_median if self.channel_median else None

    @property
    def engagement(self) -> float:
        return (self.likes + 2 * self.comments) / max(self.views, 1)


RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "relevance": {"type": "integer"}},
                "required": ["id", "relevance"],
            },
        }
    },
    "required": ["ratings"],
}


def _rate_relevance(settings: Settings, channel: Channel, items: list["Viral"]) -> None:
    """El LLM puntúa de 0 a 10 cuánto encaja cada vídeo con la temática del canal."""
    llm = Ollama(settings.ollama_url, settings.ollama_model)
    lines = "\n".join(f"#{k} {v.title} — {v.channel}" for k, v in enumerate(items))
    prompt = (
        f"Canal de YouTube Shorts «{channel.name}». Temática: {channel.niche}. "
        f"Busca: {channel.clips.focus or '-'}.\n"
        "Para cada vídeo, puntúa de 0 a 10 cuánto encaja su contenido con la temática del canal "
        "(10 = es exactamente de la temática; 5 = relacionado de lejos; 0 = nada que ver). "
        "Sé estricto: que sea popular no importa aquí.\n\n" + lines
    )
    try:
        data = llm.chat("Eres un editor que decide qué vídeos encajan en un canal temático.", prompt,
                        RELEVANCE_SCHEMA, num_ctx=8192)
        llm.unload()
    except LLMError as exc:
        log(f"  aviso: no pude comprobar la temática con el LLM ({exc}); se aceptan todos")
        for item in items:
            item.relevance = 10
        return
    for rating in data.get("ratings", []):
        k = rating.get("id")
        if isinstance(k, int) and 0 <= k < len(items):
            items[k].relevance = max(0, min(10, int(rating.get("relevance", 0))))


QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}


def trend_queries(settings: Settings, channel: Channel) -> list[str]:
    """Búsquedas propuestas por el LLM para la temática (una vez al día; se guardan en trends.json)."""
    cache = channel.dir / "trends.json"
    today = date.today().isoformat()
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("date") == today and data.get("niche") == channel.niche:
            return data["queries"]
    lang = LANGUAGES.get(channel.language, channel.language or "inglés")
    llm = Ollama(settings.ollama_url, settings.ollama_model)
    prompt = (
        f"Canal de YouTube Shorts «{channel.name}». Temática: {channel.niche}. "
        f"Busca: {channel.clips.focus or '-'}.\n"
        f"Propón 8 búsquedas de YouTube en {lang} para encontrar los vídeos LARGOS (más de 10 minutos) "
        "más virales ahora mismo en esta temática: podcasts y entrevistas con invitados famosos, debates, "
        "documentales, reseñas, retos, historias reales… Usa temas, personas y formatos concretos que "
        "estén de moda; nada genérico. Cada búsqueda de 2 a 6 palabras."
    )
    try:
        llm.check()
        queries = [q.strip() for q in llm.chat("Eres un analista de tendencias de YouTube.", prompt,
                                                 QUERY_SCHEMA, num_ctx=4096).get("queries", []) if q.strip()][:8]
        llm.unload()
    except LLMError as exc:
        log(f"  aviso: sin búsquedas de tendencia del LLM ({exc})")
        queries = []
    cache.write_text(json.dumps({"date": today, "niche": channel.niche, "queries": queries},
                                ensure_ascii=False, indent=2), encoding="utf-8")
    return queries


def _search(query: str, period: str, duration: int | None, cc: bool) -> list[SourceVideo]:
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}&sp={search_params(period, duration, cc)}"
    try:
        return _flat(url, SEARCH_RESULTS)
    except SourceError as exc:
        log(f"  ✗ búsqueda «{query}»: {exc}")
        return []


def _details(video: SourceVideo, settings: Settings) -> Viral | None:
    try:
        with YoutubeDL(ytdlp_params(settings) | {"skip_download": True, "logger": _Silent()}) as ydl:
            raw = ydl.extract_info(video.url, download=False, process=False)
    except DownloadError:
        return None
    uploaded = raw.get("upload_date")
    days = (date.today() - datetime.strptime(uploaded, "%Y%m%d").date()).days if uploaded else 365
    return Viral(
        url=video.url, title=raw.get("title") or video.title, channel=raw.get("channel") or "",
        duration=float(raw.get("duration") or video.duration or 0), views=int(raw.get("view_count") or 0),
        days=max(days, 1), channel_url=raw.get("channel_url") or "",
        likes=int(raw.get("like_count") or 0), comments=int(raw.get("comment_count") or 0),
        license=raw.get("license") or "", spoken=spoken_languages(raw),
    )


def _channel_median(channel_url: str) -> float | None:
    """Mediana de visitas de los últimos vídeos largos de un canal."""
    if not channel_url:
        return None
    try:
        recent = _flat(_channel_videos_tab(channel_url), 15)
    except SourceError:
        return None
    views = sorted(v.views for v in recent if v.views)
    return float(views[len(views) // 2]) if len(views) >= 3 else None


def _percentiles(values: list[float]) -> list[float]:
    order = sorted(values)
    return [order.index(v) / max(len(order) - 1, 1) for v in values]


_STOP = {
    "the", "and", "with", "this", "that", "what", "when", "your", "from", "have", "into", "about", "just",
    "explained", "video", "full", "episode", "new", "how", "why", "los", "las", "una", "para", "con", "por",
    "como", "que", "del", "sobre", "este", "esta",
}


def _keywords(title: str) -> set[str]:
    words = {w.strip(".,:;!?¡¿'\"()[]|-–—").lower() for w in title.split()}
    return {w for w in words if len(w) >= 4 and w not in _STOP and not w.isdigit()}


def _diverse(ranked: list["Viral"], want: int, recent_titles: list[str]) -> list["Viral"]:
    """Evita elegir varios vídeos del mismo tema (o del que ya se usó hace poco)."""
    taken = [_keywords(t) for t in recent_titles]
    picked = []
    for item in ranked:
        words = _keywords(item.title)
        if any(len(words & other) >= 2 for other in taken):
            continue
        picked.append(item)
        taken.append(words)
        if len(picked) >= want:
            break
    return picked


def find_viral(settings: Settings, channel: Channel, *, want: int, known: set[str],
               recent_titles: list[str] | None = None) -> list[SourceVideo]:
    src = channel.sources
    min_s, max_s = src.min_minutes * 60, src.max_minutes * 60
    queries = list(dict.fromkeys(src.searches + trend_queries(settings, channel)))
    # La búsqueda de YouTube solo filtra «más de 20 min» o «de 4 a 20»: se piden las dos si hace falta.
    durations = [2] if src.min_minutes >= 20 else [2, 3]
    log(f"  analizando lo viral de «{channel.niche}» ({src.period}): {len(queries)} búsqueda/s…")

    jobs = [(q, src.period, d, src.only_creative_commons) for q in queries for d in durations]
    pool: dict[str, SourceVideo] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for found in executor.map(lambda job: _search(*job), jobs):
            for video in found:
                pool.setdefault(video.url, video)
        for url in src.urls:  # canales fuente: sus vídeos recientes que destacan sobre el resto
            try:
                recent = _flat(_channel_videos_tab(url), 30)
            except SourceError as exc:
                log(f"  ✗ fuente {url}: {exc}")
                continue
            for video in recent:
                pool.setdefault(video.url, video)

    banned = [word.lower() for word in src.exclude]
    candidates = [
        v for v in pool.values()
        if v.url not in known and youtube_video_id(v.url)
        and (not v.duration or min_s <= v.duration <= max_s)
        and not any(word in v.title.lower() for word in banned)
    ]
    # Solo se leen a fondo los que más visitas tienen en la lista rápida.
    candidates.sort(key=lambda v: v.views or 0, reverse=True)
    candidates = candidates[:DETAILED]
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        details = [d for d in executor.map(lambda v: _details(v, settings), candidates) if d and d.views]
        channel_urls = sorted({d.channel_url for d in details if d.channel_url})
        medians = dict(zip(channel_urls, executor.map(_channel_median, channel_urls)))
    for item in details:
        item.channel_median = medians.get(item.channel_url)

    valid = []
    for item in details:
        text = f"{item.title} {item.channel}".lower()
        if any(word in text for word in banned):
            continue
        if src.only_creative_commons and not is_creative_commons(item.license):
            continue
        if src.languages and item.spoken and not set(item.spoken) & set(src.languages):
            continue
        if item.duration and not min_s <= item.duration <= max_s:
            continue
        valid.append(item)
    if not valid:
        return []

    _rate_relevance(settings, channel, valid)
    velocity = _percentiles([math.log10(v.velocity + 1) for v in valid])
    known_outliers = [v.outlier for v in valid if v.outlier is not None]
    outlier_pct = dict(zip(map(id, [v for v in valid if v.outlier is not None]), _percentiles(known_outliers)))
    engagement = _percentiles([v.engagement for v in valid])
    for item, a, c in zip(valid, velocity, engagement):
        b = outlier_pct.get(id(item), 0.5)
        virality = 0.45 * a + 0.35 * b + 0.20 * c
        # Lo viral de otra temática no sirve: la nota mezcla viralidad y encaje con el canal.
        item.score = round(100 * (0.7 * virality + 0.3 * item.relevance / 10), 1)
    valid.sort(key=lambda v: (v.relevance >= MIN_RELEVANCE, v.score), reverse=True)
    _report(channel, valid, queries)
    valid = [v for v in valid if v.relevance >= MIN_RELEVANCE]
    log(f"  {len(valid)} vídeo/s virales válidos; ranking en {channel.dir / 'viral.txt'}")
    return [SourceVideo(v.url, v.title, v.duration) for v in _diverse(valid, want, recent_titles or [])]


def _report(channel: Channel, ranked: list[Viral], queries: list[str]) -> None:
    lines = [
        f"Lo más viral para «{channel.name}» ({channel.niche}) — {date.today().isoformat()}",
        f"Búsquedas: {'; '.join(queries)}",
        "",
        f"nota = 70 % viralidad (visitas/día, visitas frente a suscriptores, participación) + 30 % encaje "
        f"con la temática; se descartan los de encaje < {MIN_RELEVANCE}/10.",
        "",
        "nota  tema   visitas   vis/día  ×canal  partic.  min  título — canal   (×canal = visitas / mediana de su canal)",
    ]
    for v in ranked:
        outlier = f"{v.outlier:>7.1f}x" if v.outlier is not None else "      ?"
        flag = " " if v.relevance >= MIN_RELEVANCE else "✗"
        lines.append(
            f"{v.score:>4.0f} {flag}{v.relevance:>2}/10 {v.views:>10,} {v.velocity:>9,.0f} {outlier} "
            f"{100 * v.engagement:>6.1f}% {v.duration / 60:>4.0f}  {v.title} — {v.channel}\n{'':>52}{v.url}"
        )
    (channel.dir / "viral.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
