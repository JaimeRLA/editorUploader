"""`python -m editoruploader setup`: descarga en tools/ y music/ todo lo necesario (gratis).

No hace falta sudo: ffmpeg y deno son binarios estáticos, y las librerías OpenGL que
MediaPipe exige para arrancar se extraen de los paquetes de Ubuntu sin instalarlos.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests

from . import music
from .captions import FONTS
from .config import TOOLS, Settings
from .utils import log

# ffmpeg estable: la rama «master» exige drivers NVIDIA más nuevos para NVENC.
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"
DENO_URL = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
GL_PACKAGES = ("libgles2", "libglvnd0", "libegl1")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def _ffmpeg() -> None:
    target = TOOLS / "ffmpeg"
    if (target / "bin" / "ffmpeg").exists():
        log("✓ ffmpeg ya está")
        return
    log("↓ ffmpeg (build estático con libass y NVENC)…")
    with tempfile.TemporaryDirectory(dir=TOOLS) as tmp:
        archive = _download(FFMPEG_URL, Path(tmp) / "ffmpeg.tar.xz")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        folder = next(p for p in Path(tmp).iterdir() if p.is_dir())
        shutil.move(str(folder), target)


def _deno() -> None:
    target = TOOLS / "deno"
    if target.exists():
        log("✓ deno ya está")
        return
    log("↓ deno (lo usa yt-dlp para YouTube)…")
    with tempfile.TemporaryDirectory(dir=TOOLS) as tmp:
        archive = _download(DENO_URL, Path(tmp) / "deno.zip")
        with zipfile.ZipFile(archive) as zf:
            zf.extract("deno", TOOLS)
    target.chmod(0o755)


def _gl_libs() -> None:
    try:
        ctypes.CDLL("libGLESv2.so.2")
        log("✓ OpenGL ES ya está en el sistema")
        return
    except OSError:
        pass
    lib_dir = TOOLS / "lib"
    if (lib_dir / "libGLESv2.so.2").exists():
        log("✓ librerías OpenGL ES ya están en tools/lib")
        return
    if not shutil.which("apt-get") or not shutil.which("dpkg-deb"):
        log("✗ Falta libGLESv2. Instálala con el gestor de paquetes (en Ubuntu: sudo apt install libgles2)")
        return
    log("↓ librerías OpenGL ES para MediaPipe (sin instalar nada en el sistema)…")
    lib_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TOOLS) as tmp:
        subprocess.run(["apt-get", "download", *GL_PACKAGES], cwd=tmp, check=True, capture_output=True)
        root = Path(tmp) / "root"
        for deb in Path(tmp).glob("*.deb"):
            subprocess.run(["dpkg-deb", "-x", str(deb), str(root)], check=True)
        for lib in (root / "usr" / "lib" / "x86_64-linux-gnu").glob("*.so*"):
            shutil.copy2(lib, lib_dir / lib.name, follow_symlinks=False)


def _music(settings: Settings) -> None:
    missing = music.pack_downloads(settings.music_dir)
    for mood in music.MOODS:
        (settings.music_dir / mood).mkdir(parents=True, exist_ok=True)
    if not missing:
        log("✓ pack de música ya está")
        return
    log(f"↓ pack de música gratuita de Kevin MacLeod (CC BY 4.0): {len(missing)} temas…")
    credits = {}
    for url, dest, credit in missing:
        try:
            _download(url, dest)
        except requests.RequestException as exc:
            log(f"  aviso: no pude bajar {dest.name}: {exc}")
            continue
        credits[str(dest.relative_to(settings.music_dir))] = credit
    music.save_credits(settings.music_dir, credits)


def _check_gpu() -> None:
    from .transcribe import _preload_cuda_libs

    _preload_cuda_libs()
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
    except (ImportError, RuntimeError, OSError):
        count = 0
    log("✓ GPU NVIDIA disponible para Whisper" if count else "· Sin GPU para Whisper: irá por CPU (más lento)")


def _check_ollama(settings: Settings) -> None:
    try:
        resp = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
        names = {m.get("name") for m in resp.json().get("models", [])}
    except (requests.RequestException, ValueError):
        log(f"✗ Ollama no responde en {settings.ollama_url}. Instálalo desde https://ollama.com/download")
        return
    model = settings.ollama_model
    if model in names or f"{model}:latest" in names:
        log(f"✓ Ollama con el modelo {model}")
    else:
        log(f"✗ Ollama no tiene {model}. Descárgalo con: ollama pull {model}")


def setup(settings: Settings) -> None:
    TOOLS.mkdir(exist_ok=True)
    _ffmpeg()
    _deno()
    if settings.face_model.exists():
        log("✓ modelo de caras de MediaPipe ya está")
    else:
        log("↓ modelo de caras de MediaPipe…")
        _download(FACE_MODEL_URL, settings.face_model)
    for family, (filename, url) in FONTS.items():
        if not (settings.fonts_dir / filename).exists():
            log(f"↓ fuente {family}…")
            _download(url, settings.fonts_dir / filename)
    _music(settings)
    _gl_libs()
    _check_gpu()
    _check_ollama(settings)
