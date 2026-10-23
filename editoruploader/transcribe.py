"""Transcripción local con faster-whisper (marcas de tiempo por palabra)."""
from __future__ import annotations

import ctypes
import gc
import glob
import json
import site
import subprocess
from pathlib import Path

import numpy as np

from .utils import log


def _preload_cuda_libs() -> None:
    """Carga cuBLAS/cuDNN de los paquetes pip nvidia-* para que CTranslate2 los encuentre."""
    for base in site.getsitepackages():
        for pattern in ("cublas/lib/libcublasLt.so.*", "cublas/lib/libcublas.so.*", "cudnn/lib/libcudnn*.so.*"):
            for lib in sorted(glob.glob(f"{base}/nvidia/{pattern}")):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _load_model(model_name: str, device: str):
    from faster_whisper import WhisperModel

    if device in ("auto", "cuda"):
        _preload_cuda_libs()
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                # float16: int8_float16 (el «auto») falla con cuBLAS en las RTX 50.
                return WhisperModel(model_name, device="cuda", compute_type="float16"), "cuda"
        except (RuntimeError, ValueError, OSError) as exc:
            if device == "cuda":
                raise
            log(f"  aviso: no pude usar la GPU para Whisper ({exc}); uso la CPU")
    return WhisperModel(model_name, device="cpu", compute_type="int8"), "cpu"


SAMPLE_RATE = 16000
CHUNK_SECONDS = 600  # tramos de 10 min: memoria constante aunque el vídeo dure horas
CUT_WINDOW = 30  # el corte se busca en el silencio más claro de ±30 s alrededor del límite


def transcribe(
    media: Path, cache: Path, model_name: str, device: str, language: str | None, ffmpeg: str = "ffmpeg"
) -> dict:
    """Devuelve {"language": "es", "words": [{"w", "s", "e"}, ...]} y lo guarda en `cache`.

    El audio se procesa por tramos cortados en silencios; el avance se guarda tras cada
    tramo en `<cache>.partial`, así que un corte a mitad de un vídeo largo no se pierde.
    """
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    partial = cache.with_name(cache.name + ".partial")
    progress = {"language": language, "words": [], "next": 0.0}
    if partial.exists():
        progress = json.loads(partial.read_text(encoding="utf-8"))
        log(f"  retomo la transcripción en el minuto {progress['next'] / 60:.0f}")

    model, used = _load_model(model_name, device)
    log(f"  transcribiendo con Whisper {model_name} ({used})…")
    try:
        while True:
            offset = progress["next"]
            audio = load_audio(media, ffmpeg, offset, CHUNK_SECONDS + CUT_WINDOW)
            if len(audio) < SAMPLE_RATE // 2:
                break
            # ffmpeg puede devolver unas muestras menos de las pedidas: solo es el final si falta bastante.
            last = len(audio) < (CHUNK_SECONDS + CUT_WINDOW - 1) * SAMPLE_RATE
            cut = len(audio) if last else _quietest_point(audio, CHUNK_SECONDS)
            found, words = _run(model, audio[:cut], progress["language"])
            # El idioma del primer tramo se fija para el resto (evita saltos de idioma).
            progress["language"] = progress["language"] or found
            progress["words"] += [
                {"w": w["w"], "s": round(w["s"] + offset, 2), "e": round(w["e"] + offset, 2)} for w in words
            ]
            progress["next"] = offset + cut / SAMPLE_RATE
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(progress, ensure_ascii=False), encoding="utf-8")
            log(f"    {_hhmm(progress['next'])} transcrito ({len(progress['words'])} palabras)")
            if last:
                break
    except RuntimeError as exc:
        if used != "cuda":
            raise
        log(f"  aviso: Whisper falló en la GPU ({exc}); reintento en la CPU")
        used = None
    finally:
        # Libera la VRAM para el LLM que viene después.
        del model
        gc.collect()
    if used is None:
        return transcribe(media, cache, model_name, "cpu", language, ffmpeg)

    result = {"language": progress["language"], "words": progress["words"]}
    cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    partial.unlink(missing_ok=True)
    log(f"  {len(result['words'])} palabras ({result['language']})")
    return result


def load_audio(media: Path, ffmpeg: str, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Audio mono a 16 kHz en float32, decodificado con ffmpeg (más robusto que PyAV)."""
    cmd = [ffmpeg, "-v", "error", "-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(media), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True, check=True, stdin=subprocess.DEVNULL)
    return np.frombuffer(result.stdout, dtype=np.float32)


def _quietest_point(audio: np.ndarray, around: float) -> int:
    """Muestra con menos energía (ventanas de 100 ms) a ±CUT_WINDOW s de `around`."""
    frame = SAMPLE_RATE // 10
    lo = int((around - CUT_WINDOW) * SAMPLE_RATE)
    hi = min(len(audio), int((around + CUT_WINDOW) * SAMPLE_RATE))
    region = audio[lo:hi]
    frames = region[: len(region) // frame * frame].reshape(-1, frame)
    quietest = int(np.argmin(np.sqrt((frames**2).mean(axis=1))))
    return lo + quietest * frame + frame // 2


def _hhmm(seconds: float) -> str:
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


BATCH_SIZE = 16  # trozos de voz que Whisper procesa a la vez: ~3,5x más rápido en GPU


def _run(model, audio: np.ndarray, language: str | None) -> tuple[str, list[dict]]:
    from faster_whisper import BatchedInferencePipeline

    if model.model.device == "cuda":
        # Por lotes: corta la voz con VAD y transcribe varios trozos a la vez.
        segments, info = BatchedInferencePipeline(model=model).transcribe(
            audio, language=language, word_timestamps=True, batch_size=BATCH_SIZE, beam_size=5,
        )
    else:
        segments, info = model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=5,
            # Evita los bucles de texto repetido típicos de Whisper en vídeos largos.
            condition_on_previous_text=False,
        )
    words = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                words.append({"w": text, "s": round(word.start, 2), "e": round(word.end, 2)})
    return info.language, words
