"""Configuración global (.env): herramientas, modelos y rutas comunes a todos los canales.

Lo específico de cada canal (temática, fuentes, edición, subida) está en channels/<id>/channel.toml.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _path(name: str, default: str) -> Path:
    path = Path(_env(name, default)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _binary(env_name: str, bundled: Path, command: str) -> str:
    """Ejecutable indicado en .env, el de tools/ o el del sistema (por ese orden)."""
    if _env(env_name):
        return str(_path(env_name, ""))
    if bundled.exists():
        return str(bundled)
    return shutil.which(command) or command


@dataclass(frozen=True)
class Settings:
    client_secrets: Path  # cliente OAuth de Google Cloud, compartido por todos los canales
    channels_dir: Path
    output_dir: Path
    work_dir: Path
    music_dir: Path
    fonts_dir: Path
    face_model: Path
    ffmpeg: str
    ffprobe: str
    deno: str
    whisper_model: str
    whisper_device: str
    ollama_url: str
    ollama_model: str
    video_encoder: str


def load_settings() -> Settings:
    ffmpeg_dir = TOOLS / "ffmpeg" / "bin"
    return Settings(
        client_secrets=_path("YOUTUBE_CLIENT_SECRETS", "credentials/client_secret.json"),
        channels_dir=_path("CHANNELS_DIR", "channels"),
        output_dir=_path("OUTPUT_DIR", "output"),
        work_dir=_path("WORK_DIR", "work"),
        music_dir=_path("MUSIC_DIR", "music"),
        fonts_dir=_path("FONTS_DIR", "tools/fonts"),
        face_model=_path("FACE_MODEL", "tools/models/face_landmarker.task"),
        ffmpeg=_binary("FFMPEG", ffmpeg_dir / "ffmpeg", "ffmpeg"),
        ffprobe=_binary("FFPROBE", ffmpeg_dir / "ffprobe", "ffprobe"),
        deno=_binary("DENO", TOOLS / "deno", "deno"),
        whisper_model=_env("WHISPER_MODEL", "large-v3-turbo"),
        whisper_device=_env("WHISPER_DEVICE", "auto"),
        ollama_url=_env("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL", "qwen3:8b"),
        video_encoder=_env("VIDEO_ENCODER", "auto"),
    )
