"""Obtención de vídeos de internet: URLs sueltas, playlists, canales y búsquedas.

`discover` busca, en las fuentes configuradas de un canal, vídeos nuevos que aún no haya
procesado; `collect` resuelve URLs que se le pasan a mano.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote_plus, urlparse

import requests
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .config import TOOLS
from .utils import log

# Por debajo de esto no hay material que recortar; por encima, demasiado proceso.
MIN_SOURCE_SECONDS = 90
MAX_SOURCE_SECONDS = 5 * 3600

DIRECT_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
_YOUTUBE_ID = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/live/|/embed/)([\w-]{11})")
_CHANNEL_ROOT = re.compile(r"^/(@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+)/?$")

# Filtros de la búsqueda de YouTube (parámetro «sp»).
SEARCH_CC = "EgIwAQ%3D%3D"  # licencia Creative Commons
SEARCH_CC_LONG = "EgQYAjAB"  # Creative Commons y más de 20 minutos
SEARCH_LONG = "EgIYAg%3D%3D"  # más de 20 minutos


class SourceError(RuntimeError):
    pass


class Skip(SourceError):
    """El vídeo no sirve para este canal; se marca como descartado y no se vuelve a intentar."""


class NotReusable(Skip):
    """El vídeo no tiene una licencia que permita reutilizarlo."""


class WrongLanguage(Skip):
    """En el vídeo se habla un idioma que el canal no admite."""


class _Silent:
    """yt-dlp escribe los errores aunque esté en modo silencioso; quien llama ya los gestiona."""

    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def _base_opts() -> dict:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "logger": _Silent()}
    if (TOOLS / "deno").exists():
        opts["js_runtimes"] = {"deno": {"path": str(TOOLS / "deno")}}
    return opts


@dataclass
class SourceVideo:
    url: str
    title: str
    duration: float | None = None
    views: int | None = None


def is_youtube(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("youtu.be", "youtube.com") or host.endswith(".youtube.com")


def youtube_video_id(url: str) -> str | None:
    if not is_youtube(url):
        return None
    match = _YOUTUBE_ID.search(url)
    return match.group(1) if match else None


def is_direct_file(url: str) -> bool:
    return PurePosixPath(urlparse(url).path).suffix.lower() in DIRECT_EXTENSIONS


def search_url(query: str, creative_commons: bool, long_only: bool) -> str:
    """URL de búsqueda de YouTube con los filtros de licencia y duración."""
    if creative_commons and long_only:
        sp = SEARCH_CC_LONG
    elif creative_commons:
        sp = SEARCH_CC
    elif long_only:
        sp = SEARCH_LONG
    else:
        sp = ""
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    return f"{url}&sp={sp}" if sp else url


def collect(urls: list[str], limit: int = 3) -> list[SourceVideo]:
    """Convierte las URLs indicadas a mano en una lista de vídeos concretos."""
    videos: list[SourceVideo] = []
    for url in urls:
        try:
            videos += _expand(url, limit)
        except SourceError as exc:
            log(f"✗ {exc}")
    return _filter(videos, MIN_SOURCE_SECONDS, MAX_SOURCE_SECONDS, verbose=True)


def discover(
    urls: list[str],
    searches: list[str],
    *,
    want: int,
    known: set[str],
    min_seconds: float,
    max_seconds: float,
    creative_commons: bool,
) -> list[SourceVideo]:
    """Hasta `want` vídeos nuevos (no incluidos en `known`) de las fuentes de un canal.

    Se reparten por turnos entre las fuentes para que ninguna acapare todas las ejecuciones.
    La licencia se comprueba después, al descargar (la lista rápida de YouTube no la trae).
    """
    lookback = max(15, want * 5)
    long_only = min_seconds >= 20 * 60
    pools: list[list[SourceVideo]] = []
    for url in urls:
        try:
            pools.append(_expand(url, lookback))
        except SourceError as exc:
            log(f"  ✗ fuente {url}: {exc}")
    for query in searches:
        try:
            pools.append(_flat(search_url(query, creative_commons, long_only), lookback))
        except SourceError as exc:
            log(f"  ✗ búsqueda «{query}»: {exc}")

    picked: list[SourceVideo] = []
    seen = set(known)
    while len(picked) < want and any(pools):
        for pool in pools:
            while pool:
                video = pool.pop(0)
                if video.url in seen:
                    continue
                seen.add(video.url)
                if _filter([video], min_seconds, max_seconds):
                    picked.append(video)
                    break
            if len(picked) >= want:
                break
    return picked


def _filter(videos: list[SourceVideo], min_seconds: float, max_seconds: float, verbose: bool = False) -> list[SourceVideo]:
    unique: list[SourceVideo] = []
    seen: set[str] = set()
    for video in videos:
        if video.url in seen:
            continue
        seen.add(video.url)
        if video.duration and not min_seconds <= video.duration <= max_seconds:
            if verbose:
                log(f"– Omito «{video.title}»: dura {video.duration / 60:.1f} min")
            continue
        unique.append(video)
    return unique


def _expand(url: str, limit: int) -> list[SourceVideo]:
    video_id = youtube_video_id(url)
    if video_id:
        canonical = f"https://www.youtube.com/watch?v={video_id}"
        return [SourceVideo(canonical, _youtube_title(canonical))]
    if is_direct_file(url):
        return [SourceVideo(url, PurePosixPath(urlparse(url).path).stem)]
    if is_youtube(url):
        url = _channel_videos_tab(url)
    # Playlists, canales, búsquedas y vídeos de otras webs: lo resuelve yt-dlp.
    return _flat(url, limit)


def _channel_videos_tab(url: str) -> str:
    """youtube.com/@canal → youtube.com/@canal/videos (sin Shorts ni directos)."""
    parsed = urlparse(url)
    if _CHANNEL_ROOT.match(parsed.path):
        return parsed._replace(path=parsed.path.rstrip("/") + "/videos").geturl()
    return url


def _youtube_title(url: str) -> str:
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed", params={"url": url, "format": "json"}, timeout=15
        )
        resp.raise_for_status()
        return resp.json().get("title") or url
    except (requests.RequestException, ValueError):
        return url


def _flat(url: str, limit: int) -> list[SourceVideo]:
    opts = _base_opts() | {
        "extract_flat": "in_playlist",
        "playlistend": limit,
        "ignore_no_formats_error": True,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise SourceError(f"No pude leer {url}: {exc}") from exc

    entries = info.get("entries")
    if entries is None:  # era un vídeo suelto
        return [SourceVideo(info.get("webpage_url") or url, info.get("title") or url, info.get("duration"))]

    videos = []
    for entry in entries:
        if not entry or entry.get("ie_key") == "YoutubeTab":
            continue
        entry_url = entry.get("webpage_url") or entry.get("url")
        if not entry_url:
            continue
        video_id = youtube_video_id(entry_url)
        if video_id:
            entry_url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append(SourceVideo(entry_url, entry.get("title") or entry_url, entry.get("duration"),
                                  entry.get("view_count")))
    return videos[:limit]
