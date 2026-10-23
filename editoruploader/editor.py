"""Montaje del Short: una historia con cortes, cámara virtual, zooms, transiciones, subtítulos y música.

Partiendo del guion del LLM (gancho + tramos):
1. Cada tramo se trocea quitando silencios (cortes de salto, ver timeline.py).
2. La cámara virtual (reframe.py) decide qué parte del vídeo horizontal se ve: el resultado
   ocupa SIEMPRE la pantalla vertical entera.
3. Cada fotograma se compone en Python: recorte con zoom (golpe de zoom en los cortes,
   acercamiento lento continuo, impacto en las transiciones), destello en los cambios de
   parte y barra de progreso. Se envía a ffmpeg ya en YUV 4:2:0.
4. El audio se monta con los mismos cortes (con micro-fundidos para que no chasquee), se le
   añade un «whoosh» en las transiciones y la música con ducking, y ffmpeg quema los subtítulos.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import captions, music, reframe
from .channels import EditStyle
from .config import Settings
from .highlights import Candidate
from .media import VideoInfo, encoder_args, pick_encoder
from .timeline import Words, speech_pieces

OUT_W, OUT_H, FPS = 1080, 1920, 30
SR = 48000
SAMPLES_PER_FRAME = SR // FPS
PUNCH = 1.12  # zoom de golpe en los cortes de salto (y en el gancho inicial)
PUSH = 0.05  # acercamiento lento: +5 % …
PUSH_SECONDS = 5.0  # … a lo largo de 5 s, y vuelve a empezar en cada corte
IMPACT = 0.14  # zoom de impacto al entrar en una parte nueva
IMPACT_SECONDS = 0.28
FLASH = (0.75, 0.45, 0.2)  # destello blanco en los primeros fotogramas de una parte nueva
MIN_HOLD = 2.2  # segundos mínimos con el mismo zoom antes de alternar (si no, marea)
BAR_HEIGHT = 12


@dataclass
class Piece:
    """Trozo continuo del original que entra en el montaje."""
    src_start: float
    src_end: float
    group: int  # parte del guion (0 = gancho si lo hay); se decodifica seguida
    zoom: float = 1.0
    transition: bool = False  # destello + whoosh + zoom de impacto al empezar
    out_start: float = 0.0

    @property
    def frames(self) -> int:
        return round((self.src_end - self.src_start) * FPS)


@dataclass
class Montage:
    pieces: list[Piece] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return sum(p.frames for p in self.pieces)

    @property
    def duration(self) -> float:
        return self.frames / FPS

    def groups(self) -> list[list[Piece]]:
        out: dict[int, list[Piece]] = {}
        for piece in self.pieces:
            out.setdefault(piece.group, []).append(piece)
        return [out[g] for g in sorted(out)]


def build_montage(cand: Candidate, words: Words, edit: EditStyle) -> Montage:
    """Convierte el guion (gancho + tramos) en trozos con sus zooms y transiciones."""
    parts: list[tuple[str, list[float]]] = []
    if edit.story and cand.cold_open:
        parts.append(("cold", cand.cold_open))
    parts += [("story", seg) for seg in (cand.segments or [[cand.start, cand.end]])]
    montage = Montage()
    for group, (kind, (start, end)) in enumerate(parts):
        zoom = PUNCH if kind == "cold" else 1.0
        held = 0.0
        for k, (a, b) in enumerate(speech_pieces(words, start, end, edit.jump_cuts)):
            if k and kind == "story" and held >= MIN_HOLD:
                zoom, held = (1.0 if zoom > 1.0 else PUNCH), 0.0  # golpe de zoom alterno en cada corte
            montage.pieces.append(Piece(
                src_start=a, src_end=b, group=group,
                zoom=zoom if edit.zoom_effects else 1.0,
                transition=edit.transitions and k == 0 and group > 0,
            ))
            held += b - a
    # Todo en la rejilla de fotogramas: así vídeo y audio cortan exactamente en el mismo sitio.
    for group in montage.groups():
        origin = group[0].src_start
        for piece in group:
            piece.src_start = origin + round((piece.src_start - origin) * FPS) / FPS
            piece.src_end = piece.src_start + max(1, round((piece.src_end - piece.src_start) * FPS)) / FPS
    out = 0.0
    for piece in montage.pieces:
        piece.out_start = out
        out += piece.frames / FPS
    return montage


def remap_words(words: Words, montage: Montage) -> list[dict]:
    """Palabras con tiempos del Short montado (para los subtítulos)."""
    out = []
    for piece in montage.pieces:
        for word in words.between(piece.src_start, piece.src_end):
            start, end = max(word["s"], piece.src_start), min(word["e"], piece.src_end)
            if end - start >= 0.04:
                shift = piece.out_start - piece.src_start
                out.append({"w": word["w"], "s": start + shift, "e": end + shift})
    return out


def _decode(src: Path, start: float, duration: float, cam: reframe.Camera, info: VideoInfo,
            ffmpeg: str, hw: bool):
    """Fotogramas RGB a FPS de la zona con imagen (sin bandas negras), a resolución completa."""
    crop = f",crop={cam.box[0]}:{cam.box[1]}:{cam.box[2]}:{cam.box[3]}" if cam.box else ""
    if hw:
        decode, vf = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"], f"hwdownload,format=nv12,fps={FPS}{crop}"
    else:
        decode, vf = [], f"fps={FPS}{crop}"
    cmd = [ffmpeg, "-v", "error", *decode, "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    size = cam.width * cam.height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(cam.height, cam.width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def _compose(frame: np.ndarray, cam: reframe.Camera, t_src: float, zoom: float) -> np.ndarray:
    """Recorte vertical con zoom alrededor del sujeto, escalado a 1080x1920 (subpíxel, sin saltos)."""
    h, w = frame.shape[:2]
    base_w = min(w, h * 9 / 16)  # el mayor recorte 9:16 que cabe
    cx, cy, cam_zoom = cam.at(t_src)
    crop_w = base_w / (zoom * cam_zoom)
    crop_h = crop_w * 16 / 9
    x = min(max(cx * w - crop_w / 2, 0), w - crop_w)
    # Con personas, la cara queda en el tercio superior; si no, el sujeto centrado.
    y = min(max(cy * h - crop_h * (0.40 if cam.faces else 0.5), 0), h - crop_h)
    scale = OUT_W / crop_w
    matrix = np.float32([[scale, 0, -x * scale], [0, scale, -y * scale]])
    return cv2.warpAffine(frame, matrix, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _bar_color(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _audio(src: Path, start: float, duration: float, ffmpeg: str) -> np.ndarray:
    cmd = [ffmpeg, "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
           "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    data = subprocess.run(cmd, capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(data, np.float32).reshape(-1, 2)


def _whoosh() -> np.ndarray:
    """«Whoosh» sintético: ruido filtrado con un barrido de frecuencia que sube y baja."""
    n = int(SR * 0.5)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    t = np.linspace(0, 1, n)
    cutoff = 250 + 7000 * np.sin(np.pi * t) ** 2
    alpha = 1 - np.exp(-2 * np.pi * cutoff / SR)
    out = np.empty(n)
    acc = 0.0
    for i in range(n):  # paso bajo de un polo con corte variable
        acc += alpha[i] * (noise[i] - acc)
        out[i] = acc
    out *= np.sin(np.pi * t) ** 1.5
    out *= 0.35 / (np.abs(out).max() + 1e-9)
    return np.repeat(out[:, None], 2, axis=1).astype(np.float32)


_WHOOSH: np.ndarray | None = None


def build_audio(src: Path, montage: Montage, ffmpeg: str, transitions: bool) -> tuple[np.ndarray, np.ndarray]:
    """Voz montada con los mismos cortes que el vídeo y pista de efectos (whoosh)."""
    global _WHOOSH
    total = montage.frames * SAMPLES_PER_FRAME
    voice = np.zeros((total, 2), np.float32)
    fade = int(SR * 0.006)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
    for group in montage.groups():
        origin = group[0].src_start
        audio = _audio(src, origin, group[-1].src_end - origin + 0.2, ffmpeg)
        for piece in group:
            first = round((piece.src_start - origin) * SR)
            count = piece.frames * SAMPLES_PER_FRAME
            chunk = audio[first: first + count].copy()
            if len(chunk) < count:
                chunk = np.pad(chunk, ((0, count - len(chunk)), (0, 0)))
            chunk[:fade] *= ramp  # micro-fundidos: sin chasquidos en los cortes
            chunk[-fade:] *= ramp[::-1]
            at = round(piece.out_start * FPS) * SAMPLES_PER_FRAME
            voice[at: at + count] = chunk
    sfx = np.zeros_like(voice)
    if transitions:
        _WHOOSH = _WHOOSH if _WHOOSH is not None else _whoosh()
        for piece in montage.pieces:
            if piece.transition:
                at = max(0, round((piece.out_start - 0.3) * SR))
                end = min(total, at + len(_WHOOSH))
                sfx[at:end] += _WHOOSH[: end - at]
    return voice, sfx


def make_short(
    *, src: Path, info: VideoInfo, cand: Candidate, words: Words, edit: EditStyle, settings: Settings,
    dest: Path, work: Path, hw: bool, track: tuple[Path, str] | None, seed: str,
) -> Path:
    """Monta y renderiza el Short completo en `dest`."""
    montage = build_montage(cand, words, edit)
    if not montage.pieces:
        raise RuntimeError("el guion no tiene ningún trozo con voz")
    stem = work / dest.stem
    work.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cameras = _cameras(src, info, montage, edit, settings, hw)

    # Audio: voz montada + efectos; la música se mezcla en ffmpeg.
    voice, sfx = build_audio(src, montage, settings.ffmpeg, edit.transitions)
    voice_file, sfx_file = stem.with_suffix(".voice.f32"), stem.with_suffix(".sfx.f32")
    voice.tofile(voice_file)
    sfx.tofile(sfx_file)

    # Subtítulos sobre la línea de tiempo montada.
    subs = None
    if edit.captions or (edit.hook and cand.hook):
        style = captions.CaptionStyle(
            font=edit.font, font_size=edit.font_size, text_color=edit.text_color,
            highlight_color=edit.highlight_color, outline_color=edit.outline_color,
            uppercase=edit.uppercase, words_per_line=edit.words_per_line, position=edit.position,
            hook_seconds=edit.hook_seconds, keyword_color=edit.keyword_color,
        )
        subs = captions.build(
            remap_words(words, montage) if edit.captions else [], 0, montage.duration,
            stem.with_suffix(".ass"), hook=cand.hook if edit.hook else "", style=style, keywords=cand.keywords,
        )

    duration = montage.duration
    encoder = pick_encoder(settings)
    video_filter = "[0:v]"
    if subs:
        video_filter += f"ass='{_filter_path(subs)}':fontsdir='{_filter_path(settings.fonts_dir)}',"
    video_filter += "format=yuv420p[v]"
    inputs = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",
              "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(voice_file),
              "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(sfx_file)]
    voice_chain = "[1:a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"
    if track:
        offset = music.start_offset(seed, _duration(track[0], settings), duration)
        inputs += ["-stream_loop", "-1", "-ss", f"{offset:.2f}", "-t", f"{duration:.3f}", "-i", str(track[0])]
        fade_out = max(0.0, duration - 1.5)
        audio_filter = (
            f"{voice_chain},asplit=2[voice][key];"
            f"[3:a]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I={edit.music_volume:g}:TP=-6:LRA=7,"
            f"afade=t=in:d=0.6,afade=t=out:st={fade_out:.2f}:d=1.5[bed];"
            # Ducking: la música baja sola mientras alguien habla y sube en las pausas.
            "[bed][key]sidechaincompress=threshold=0.02:ratio=5:attack=20:release=450[duck];"
            "[voice][duck][2:a]amix=inputs=3:duration=first:normalize=0,alimiter=limit=0.89[a]"
        )
    else:
        audio_filter = f"{voice_chain}[voice];[voice][2:a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.89[a]"
    tmp = dest.with_name(dest.stem + ".part.mp4")
    cmd = [settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
           "-filter_complex", f"{video_filter};{audio_filter}", "-map", "[v]", "-map", "[a]",
           *encoder_args(encoder), "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.3f}",
           "-movflags", "+faststart", str(tmp)]
    encoder_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _render_frames(src, info, montage, cameras, edit, settings, hw, encoder_proc.stdin)
        encoder_proc.stdin.close()
    except BrokenPipeError:
        pass
    err = encoder_proc.stderr.read().decode(errors="replace")
    if encoder_proc.wait() != 0:
        raise RuntimeError(f"ffmpeg falló al montar: {err.strip()[-800:]}")
    tmp.replace(dest)
    for leftover in (voice_file, sfx_file):
        leftover.unlink(missing_ok=True)
    return dest


def _cameras(src: Path, info: VideoInfo, montage: Montage, edit: EditStyle, settings: Settings,
             hw: bool) -> list[reframe.Camera]:
    """Cámara de cada parte del guion. Las zonas del original que se solapan o están cerca (el
    gancho suele estar dentro de un tramo) se analizan una sola vez."""
    spans = sorted((g[0].src_start, g[-1].src_end) for g in montage.groups())
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    analysed = [reframe.camera(src, a, b - a, info.width, info.height, settings.ffmpeg, settings.face_model,
                               hw, edit.layout) for a, b in merged]
    out = []
    for group in montage.groups():
        start = group[0].src_start
        out.append(next(cam for (a, b), cam in zip(merged, analysed) if a <= start <= b))
    return out


def _render_frames(src: Path, info: VideoInfo, montage: Montage, cameras: list[reframe.Camera],
                   edit: EditStyle, settings: Settings, hw: bool, sink) -> None:
    total = montage.frames
    bar = _bar_color(edit.highlight_color)
    written = 0
    last = None
    for group, cam in zip(montage.groups(), cameras):
        origin = group[0].src_start
        frames = _decode(src, origin, group[-1].src_end - origin + 0.5, cam, info, settings.ffmpeg, hw)
        index = 0
        for piece in group:
            first = round((piece.src_start - origin) * FPS)
            for _ in range(first - index):  # salta lo que el montaje ha quitado
                last = next(frames, last)
                index += 1
            for k in range(piece.frames):
                frame = next(frames, None)
                index += 1
                if frame is None:  # fin del fichero: se repite el último fotograma
                    frame = last
                if frame is None:
                    raise RuntimeError("no se pudo decodificar el vídeo")
                last = frame
                t_in = k / FPS
                zoom = piece.zoom
                if edit.zoom_effects:
                    zoom *= 1 + PUSH * min(t_in, PUSH_SECONDS) / PUSH_SECONDS
                    if piece.transition and t_in < IMPACT_SECONDS:
                        ease = 1 - t_in / IMPACT_SECONDS
                        zoom *= 1 + IMPACT * ease * ease
                out = _compose(frame, cam, piece.src_start + t_in, zoom)
                if piece.transition and k < len(FLASH):
                    out = cv2.addWeighted(out, 1 - FLASH[k], np.full_like(out, 255), FLASH[k], 0)
                if edit.progress_bar:
                    out[:BAR_HEIGHT, : max(1, round(OUT_W * (written + 1) / total))] = bar
                sink.write(cv2.cvtColor(out, cv2.COLOR_RGB2YUV_I420).tobytes())
                written += 1


def _filter_path(path: Path) -> str:
    """Ruta escapada para usarla dentro de un filtro de ffmpeg."""
    return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _duration(path: Path, settings: Settings) -> float:
    cmd = [settings.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0
