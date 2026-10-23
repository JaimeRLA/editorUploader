"""Descarga de vídeos (yt-dlp) y utilidades de ffmpeg/ffprobe."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .config import Settings
from .sources import NotReusable, SourceError
from .utils import log

# 1080p basta para un Short de 1080x1920 recortado de un 16:9.
FORMAT = "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float


def ytdlp_params(settings: Settings) -> dict:
    """Opciones comunes de yt-dlp: deno para YouTube y el ffmpeg de tools/."""
    params: dict = {"quiet": True, "no_warnings": True, "noprogress": True, "ffmpeg_location": settings.ffmpeg}
    if Path(settings.deno).exists():
        params["js_runtimes"] = {"deno": {"path": settings.deno}}
    return params


def is_creative_commons(license_text: str | None) -> bool:
    return bool(license_text) and "creative commons" in license_text.lower()


def _check_reusable(meta: dict, url: str, require_cc: bool) -> None:
    if require_cc and not is_creative_commons(meta.get("license")):
        raise NotReusable(
            f"«{meta.get('title') or url}» no tiene licencia Creative Commons; no se usa "
            "(only_creative_commons = true en la configuración del canal)"
        )


def download(url: str, folder: Path, settings: Settings, require_cc: bool = False,
             language: str | None = None) -> tuple[Path, dict]:
    """Descarga el vídeo (si no estaba ya) y devuelve el fichero y sus metadatos.

    Con `require_cc`, la licencia se comprueba ANTES de descargar y se rechaza todo lo que no
    sea Creative Commons. Con `language`, si el vídeo tiene varias pistas de audio (doblajes),
    se descarga la de ese idioma. Los metadatos incluyen el autor (para citarlo) y el «mapa de
    calor» de YouTube (las partes más repetidas), que sirve para elegir los mejores fragmentos.
    """
    folder.mkdir(parents=True, exist_ok=True)
    meta_file = folder / "info.json"
    existing = [p for p in folder.glob("source.*") if p.suffix not in (".part", ".ytdl", ".json")]
    if existing and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if "license" not in meta:  # descargado con una versión anterior: se completa
            meta.update({k: v for k, v in identity(url, settings).items() if k not in meta or not meta[k]})
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        _check_reusable(meta, url, require_cc)
        return existing[0], meta

    fmt = f"bv*[height<=1080]+ba[language^={language}]/{FORMAT}" if language else FORMAT
    params = ytdlp_params(settings) | {
        "format": fmt,
        "merge_output_format": "mkv",
        "outtmpl": str(folder / "source.%(ext)s"),
        "noplaylist": True,
    }
    try:
        with YoutubeDL(params) as ydl:
            raw = ydl.extract_info(url, download=False, process=False)
            _check_reusable(_meta(raw), url, require_cc)
            log("  descargando el vídeo…")
            info = ydl.process_ie_result(raw, download=True)
            path = Path(ydl.prepare_filename(info)).with_suffix(".mkv")
    except DownloadError as exc:
        raise SourceError(f"No pude descargar {url}: {exc}") from exc
    if not path.exists():
        candidates = [p for p in folder.glob("source.*") if p.suffix not in (".part", ".ytdl", ".json")]
        if not candidates:
            raise SourceError(f"yt-dlp no dejó ningún fichero para {url}")
        path = candidates[0]

    meta = _meta(info)
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, meta


def spoken_languages(info: dict) -> list[str]:
    """Idiomas que se hablan en el vídeo según YouTube (pistas de subtítulos automáticos «-orig»)."""
    langs = {key.split("-")[0] for key in (info.get("automatic_captions") or {}) if key.endswith("-orig")}
    if info.get("language"):
        langs.add(info["language"].split("-")[0])
    return sorted(langs)


def _meta(info: dict) -> dict:
    return {
        "spoken": spoken_languages(info),
        "title": info.get("title"),
        "duration": info.get("duration"),
        "language": info.get("language"),
        "heatmap": info.get("heatmap") or [],
        "chapters": info.get("chapters") or [],
        "channel": info.get("channel") or info.get("uploader") or "",
        "channel_url": info.get("channel_url") or info.get("uploader_url") or "",
        "license": info.get("license") or "",
        "webpage_url": info.get("webpage_url") or "",
    }


def identity(url: str, settings: Settings) -> dict:
    """Autor y licencia de un vídeo sin descargarlo."""
    try:
        with YoutubeDL(ytdlp_params(settings) | {"skip_download": True}) as ydl:
            raw = ydl.extract_info(url, download=False, process=False)
    except DownloadError as exc:
        raise SourceError(f"No pude leer {url}: {exc}") from exc
    meta = _meta(raw)
    return {k: meta[k] for k in ("title", "channel", "channel_url", "license", "language", "spoken", "webpage_url")}


def probe(path: Path, settings: Settings) -> VideoInfo:
    cmd = [
        settings.ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL).stdout)
    stream = data["streams"][0]
    num, _, den = stream.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration=float(data["format"].get("duration") or 0),
    )


def duration(path: Path, settings: Settings) -> float:
    cmd = [settings.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


_nvdec_cache: dict[str, bool] = {}


def nvdec_ok(path: Path, settings: Settings) -> bool:
    """¿Puede la GPU NVIDIA decodificar este vídeo? Se prueba con un segundo del fichero."""
    key = str(path)
    if key not in _nvdec_cache:
        try:
            run_ffmpeg(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-t", "1", "-i", key,
                        "-vf", "hwdownload,format=nv12", "-f", "null", "-"], settings)
            _nvdec_cache[key] = True
        except RuntimeError:
            _nvdec_cache[key] = False
    return _nvdec_cache[key]


def run_ffmpeg(args: list[str], settings: Settings) -> None:
    cmd = [settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg falló: {result.stderr.strip()[-1500:]}")


_encoder_cache: dict[str, str] = {}


def pick_encoder(settings: Settings) -> str:
    """h264_nvenc si la GPU NVIDIA lo permite; si no, libx264."""
    if settings.video_encoder != "auto":
        return settings.video_encoder
    if "auto" not in _encoder_cache:
        try:
            run_ffmpeg(
                ["-f", "lavfi", "-i", "color=black:s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"],
                settings,
            )
            _encoder_cache["auto"] = "h264_nvenc"
        except RuntimeError:
            _encoder_cache["auto"] = "libx264"
    return _encoder_cache["auto"]


def encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr",
                "-cq", "20", "-b:v", "0", "-maxrate", "16M", "-bufsize", "32M", "-profile:v", "high"]
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-profile:v", "high"]
    return ["-c:v", encoder]
