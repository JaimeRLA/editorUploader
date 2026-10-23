"""Cámara virtual vertical: decide qué parte del vídeo horizontal se ve en cada momento.

Se analizan ~5 fotogramas por segundo de cada tramo:
- Caras (MediaPipe Face Landmarker). Cada cara se sigue entre fotogramas y su «actividad» es
  cuánto se mueve su mandíbula (blendshape jawOpen): se encuadra a quien habla.
- Si no hay caras: prominencia visual (spectral residual de OpenCV) y movimiento entre
  fotogramas, para apuntar a lo importante (un coche, un experimento, el texto de una diapositiva).
- Cambios de plano (histograma de color): la cámara corta con el montaje original en vez de
  barrer de un plano a otro.
La cámara imita a un operador: quieta mientras el sujeto está en una zona muerta, paneo corto
si se desplaza, corte seco si cambia de persona o de plano. El resultado siempre llena la
pantalla vertical; nunca se muestra el vídeo horizontal entero con bandas.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import TOOLS

ANALYSIS_FPS = 5
ANALYSIS_WIDTH = 960
DEAD_ZONE = 0.035  # fracción del ancho que el sujeto puede moverse sin que la cámara le siga
CUT_DISTANCE = 0.12  # a partir de este salto se corta en vez de panear
PAN_SECONDS = 0.5
SWITCH_SECONDS = 0.8  # tiempo que otro hablante debe dominar para cambiar de plano
SCENE_CUT = 0.55  # correlación de histogramas por debajo de la cual hay cambio de plano
FACE_SHARE = 0.085  # ancho de cara (fracción del fotograma) que se considera un buen plano medio
MAX_SMART_ZOOM = 1.35  # cuánto se puede acercar en planos generales sin perder nitidez


@dataclass
class Camera:
    """Trayectoria de la cámara sobre un tramo del original.

    `keys` son (segundos desde `start`, centro x 0-1, centro y 0-1, zoom) del sujeto dentro de la
    zona con imagen (`box`, sin bandas negras). El zoom de la cámara (>1 en planos generales con
    caras pequeñas) se multiplica por los efectos del montador.
    """
    start: float
    width: int  # tamaño de la zona con imagen
    height: int
    box: tuple[int, int, int, int] | None = None  # (w, h, x, y) en el original; None = fotograma entero
    keys: list[tuple[float, float, float, float]] = field(default_factory=lambda: [(0.0, 0.5, 0.45, 1.0)])
    faces: bool = False  # si se ha encuadrado sobre todo a personas

    def at(self, t: float) -> tuple[float, float, float]:
        """Centro (x, y) 0-1 del sujeto y zoom de la cámara en el segundo `t` del original."""
        rel = t - self.start
        keys = self.keys
        if rel <= keys[0][0]:
            return keys[0][1:]
        for k0, k1 in zip(keys, keys[1:]):
            if k0[0] <= rel <= k1[0]:
                if k1[0] - k0[0] < 1e-6:
                    return k1[1:]
                a = (rel - k0[0]) / (k1[0] - k0[0])
                a = a * a * (3 - 2 * a)  # paneo suave (ease in-out)
                return tuple(v0 + (v1 - v0) * a for v0, v1 in zip(k0[1:], k1[1:]))
        return keys[-1][1:]


_gl_loaded = False


def _load_gl() -> None:
    """MediaPipe enlaza libGLESv2 aunque use la CPU; en WSL sin GPU OpenGL se cargan las de tools/lib."""
    global _gl_loaded
    if _gl_loaded:
        return
    for name in ("libGLdispatch.so.0", "libEGL.so.1", "libGLESv2.so.2"):
        lib = TOOLS / "lib" / name
        if lib.exists():
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    _gl_loaded = True


def _frames(src: Path, start: float, duration: float, src_w: int, src_h: int, ffmpeg: str, hw: bool):
    """Fotogramas RGB reducidos a ANALYSIS_WIDTH, a ANALYSIS_FPS por segundo.

    Con `hw`, NVDEC decodifica y reduce en la GPU y solo baja a la CPU el fotograma pequeño.
    """
    width = min(ANALYSIS_WIDTH, src_w)
    height = round(src_h * width / src_w / 2) * 2
    if hw:
        decode = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf = f"scale_cuda={width}:{height},hwdownload,format=nv12,fps={ANALYSIS_FPS}"
    else:
        decode, vf = [], f"fps={ANALYSIS_FPS},scale={width}:{height}"
    cmd = [
        ffmpeg, "-v", "error", *decode, "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def _letterbox(row_levels: list[np.ndarray], src_w: int, src_h: int) -> tuple[int, int, int, int] | None:
    """Bandas negras horizontales a partir del brillo de cada fila en los fotogramas analizados.

    Devuelve (w, h, x, y) de la zona con imagen en píxeles del original, o None si no hay bandas.
    """
    if not row_levels:
        return None
    levels = np.median(np.stack(row_levels), axis=0)  # por fila: brillo típico a lo largo del tramo
    rows = len(levels)
    dark = levels < 28
    top = int(np.argmin(dark)) if not dark.all() else 0
    bottom = int(np.argmin(dark[::-1])) if not dark.all() else 0
    height = rows - top - bottom
    # Solo bandas claras; las demás suelen ser fondos oscuros, no bandas.
    if height > rows * 0.94 or height < rows * 0.5:
        return None
    scale = src_h / rows
    # Unos píxeles hacia dentro: la medida es sobre el fotograma reducido y no debe asomar negro.
    y = round(top * scale / 2) * 2 + 4
    h = round(height * scale / 2) * 2 - 8
    return src_w, min(h, src_h - y), 0, y


_quiet_lock = threading.Lock()
_quiet_users = 0
_saved_stderr: int | None = None


@contextmanager
def _quiet_stderr():
    """MediaPipe (C++) y EGL escriben avisos inofensivos directamente en el descriptor 2.

    Con contador, porque varios clips se analizan a la vez en hilos distintos.
    """
    global _quiet_users, _saved_stderr
    with _quiet_lock:
        if _quiet_users == 0:
            sys.stderr.flush()
            _saved_stderr = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        _quiet_users += 1
    try:
        yield
    finally:
        with _quiet_lock:
            _quiet_users -= 1
            if _quiet_users == 0 and _saved_stderr is not None:
                os.dup2(_saved_stderr, 2)
                os.close(_saved_stderr)
                _saved_stderr = None


@dataclass
class _Sample:
    faces: list[tuple[float, float, float, float]]  # (centro x, ancho, jawOpen, centro y), 0-1
    salient: tuple[float, float]  # punto más llamativo/en movimiento (x, y), 0-1
    cut: bool  # empieza un plano nuevo
    rows: np.ndarray | None  # brillo por fila (bandas negras)


def _saliency_point(gray: np.ndarray, prev: np.ndarray | None, detector) -> tuple[float, float]:
    """Centro de lo más llamativo de la imagen, combinando prominencia y movimiento."""
    ok, sal = detector.computeSaliency(gray)
    sal = sal.astype(np.float32) if ok else np.zeros_like(gray, dtype=np.float32)
    sal /= sal.max() + 1e-6
    if prev is not None:
        motion = cv2.GaussianBlur(cv2.absdiff(gray, prev).astype(np.float32), (9, 9), 0)
        motion /= motion.max() + 1e-6
        weight = 0.55 * sal + 0.45 * motion
    else:
        weight = sal
    h, w = weight.shape
    # Ligera preferencia por el centro: los bordes rara vez son lo importante.
    ys, xs = np.mgrid[0:h, 0:w]
    bias = np.exp(-(((xs / w - 0.5) / 0.38) ** 2) - (((ys / h - 0.5) / 0.45) ** 2))
    weight = (weight * bias) ** 2
    total = weight.sum()
    if total < 1e-6:
        return 0.5, 0.45
    return float((weight * xs).sum() / total / w), float((weight * ys).sum() / total / h)


def _analyze(src: Path, start: float, duration: float, src_w: int, src_h: int, ffmpeg: str, model: Path,
             hw: bool, faces: bool) -> list[_Sample]:
    with _quiet_stderr():
        samples = _scan(src, start, duration, src_w, src_h, ffmpeg, model, hw, faces)
    if not samples and hw:  # NVDEC no pudo con este fichero: se repite por CPU
        with _quiet_stderr():
            samples = _scan(src, start, duration, src_w, src_h, ffmpeg, model, False, faces)
    return samples


def _scan(src, start, duration, src_w, src_h, ffmpeg, model, hw, faces) -> list[_Sample]:
    landmarker = None
    if faces:
        _load_gl()
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        ))
    detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    samples: list[_Sample] = []
    prev_gray, prev_hist = None, None
    try:
        for n, frame in enumerate(_frames(src, start, duration, src_w, src_h, ffmpeg, hw)):
            small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_RGB2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < SCENE_CUT
            found = []
            if landmarker is not None:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
                result = landmarker.detect_for_video(image, int(n * 1000 / ANALYSIS_FPS))
                for landmarks, shapes in zip(result.face_landmarks, result.face_blendshapes or []):
                    xs = [p.x for p in landmarks]
                    ys = [p.y for p in landmarks]
                    jaw = next((c.score for c in shapes if c.category_name == "jawOpen"), 0.0)
                    found.append(((min(xs) + max(xs)) / 2, max(xs) - min(xs), jaw, (min(ys) + max(ys)) / 2))
            samples.append(_Sample(
                faces=found,
                salient=_saliency_point(gray, None if cut else prev_gray, detector),
                cut=cut,
                rows=np.percentile(frame.max(axis=2), 98, axis=1) if n % 2 == 0 else None,
            ))
            prev_gray, prev_hist = gray, hist
    finally:
        if landmarker is not None:
            landmarker.close()
    return samples


def _track(samples: list[_Sample]) -> list[dict[int, tuple]]:
    """Asigna un id estable a cada cara según su posición horizontal (se reinicia en cada plano)."""
    tracks: dict[int, float] = {}
    next_id = 0
    out = []
    for sample in samples:
        if sample.cut:
            tracks = {}
        assigned: dict[int, tuple] = {}
        # Caras grandes primero: son las más fiables.
        for face in sorted(sample.faces, key=lambda f: -f[1]):
            free = [(abs(cx - face[0]), tid) for tid, cx in tracks.items() if tid not in assigned]
            dist, tid = min(free, default=(1.0, -1))
            if dist > max(0.1, face[1]):
                tid, next_id = next_id, next_id + 1
            tracks[tid] = face[0]
            assigned[tid] = face
        out.append(assigned)
    return out


def _activity(tracked: list[dict[int, tuple]], k: int, tid: int, half: int) -> float:
    """Movimiento medio de la mandíbula de una cara alrededor del fotograma k."""
    jaws = [tracked[i][tid][2] for i in range(max(0, k - half), min(len(tracked), k + half + 1)) if tid in tracked[i]]
    if len(jaws) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(jaws)))) + 0.2 * float(np.mean(jaws))


def _targets(samples: list[_Sample], crop_frac: float) -> list[tuple[float, float, float]]:
    """Punto (x, y) 0-1 que hay que encuadrar y ancho de la cara (0 si no hay) en cada fotograma."""
    tracked = _track(samples)
    half = round(0.75 * ANALYSIS_FPS)
    switch_frames = round(SWITCH_SECONDS * ANALYSIS_FPS)
    speaker: int | None = None
    challenger, challenge_count = None, 0
    targets = []
    for k, (sample, faces) in enumerate(zip(samples, tracked)):
        if not faces:
            targets.append((*sample.salient, 0.0))
            continue
        centers = [f[0] for f in faces.values()]
        # Si todas las caras caben en el recorte, se encuadran juntas.
        if len(faces) > 1 and max(centers) - min(centers) + max(f[1] for f in faces.values()) < crop_frac * 0.9:
            targets.append(((max(centers) + min(centers)) / 2, float(np.mean([f[3] for f in faces.values()])), 0.0))
            continue
        scores = {tid: _activity(tracked, k, tid, half) for tid in faces}
        best = max(scores, key=scores.get)
        if speaker not in faces:
            speaker, challenger, challenge_count = best, None, 0
        elif best != speaker and scores[best] > scores[speaker] * 1.4 + 0.005:
            challenge_count = challenge_count + 1 if challenger == best else 1
            challenger = best
            if challenge_count >= switch_frames:
                speaker, challenger, challenge_count = best, None, 0
        else:
            challenger, challenge_count = None, 0
        face = faces[speaker]
        targets.append((face[0], face[3], face[1]))
    return targets


def _shot_zooms(targets: list[tuple[float, float, float]], cuts: list[bool]) -> list[float]:
    """Zoom de cada fotograma según el plano: si la cara de quien habla sale pequeña, se acerca."""
    zooms = [1.0] * len(targets)
    starts = [0] + [k for k, cut in enumerate(cuts) if cut and k] + [len(targets)]
    for a, b in zip(starts, starts[1:]):
        widths = [t[2] for t in targets[a:b] if t[2] > 0]
        if len(widths) >= max(2, (b - a) // 2):
            zoom = min(max(FACE_SHARE / float(np.median(widths)), 1.0), MAX_SMART_ZOOM)
            zooms[a:b] = [zoom] * (b - a)
    return zooms


def _camera(targets: list[tuple[float, float, float]], cuts: list[bool], duration: float) -> list[tuple[float, ...]]:
    """Trayectoria de planos fijos, paneos cortos y cortes (en los cambios de plano y de persona)."""
    step = 1 / ANALYSIS_FPS
    hold = round(0.6 * ANALYSIS_FPS)
    zooms = _shot_zooms(targets, cuts)

    def settle(k: int) -> tuple[float, float]:
        window = targets[k: k + hold] or targets[k: k + 1]
        return float(np.median([t[0] for t in window])), float(np.median([t[1] for t in window]))

    x, y = settle(0)
    z = zooms[0]
    keys = [(0.0, x, y, z)]
    outside = 0
    for k in range(1, len(targets)):
        t = k * step
        if cuts[k]:  # el montaje original cambia de plano: la cámara corta con él
            nx, ny = settle(k)
            keys += [(max(t - 0.001, keys[-1][0]), x, y, z), (t, nx, ny, zooms[k])]
            x, y, z, outside = nx, ny, zooms[k], 0
            continue
        tx, ty = targets[k][:2]
        outside = outside + 1 if abs(tx - x) > DEAD_ZONE or abs(ty - y) > DEAD_ZONE * 2.5 else 0
        if outside >= hold:
            first = k - hold + 1
            nx = float(np.median([p[0] for p in targets[first: k + 1]]))
            ny = float(np.median([p[1] for p in targets[first: k + 1]]))
            t0 = first * step
            if abs(nx - x) >= CUT_DISTANCE:
                keys += [(max(t0 - 0.001, keys[-1][0]), x, y, z), (max(t0, keys[-1][0]), nx, ny, z)]
            else:
                keys += [(max(t0, keys[-1][0]), x, y, z), (min(t0 + PAN_SECONDS, duration), nx, ny, z)]
            x, y, outside = nx, ny, 0
    keys.append((max(duration, keys[-1][0]), x, y, z))
    return keys


def camera(src: Path, start: float, duration: float, src_w: int, src_h: int, ffmpeg: str, model: Path,
           hw: bool = False, layout: str = "auto") -> Camera:
    """Analiza [start, start + duration] del original y devuelve la trayectoria de la cámara."""
    samples = _analyze(src, start, duration, src_w, src_h, ffmpeg, model, hw, faces=layout == "auto")
    box = _letterbox([s.rows for s in samples if s.rows is not None], src_w, src_h)
    width, height = (box[0], box[1]) if box else (src_w, src_h)
    cam = Camera(start=start, width=width, height=height, box=box)
    if layout != "auto" or not samples:
        return cam
    crop_frac = min(1.0, height * 9 / 16 / width)
    targets = _targets(samples, crop_frac)
    if box:  # las caras y la prominencia se midieron en el fotograma entero: se pasan a la zona con imagen
        top, bottom = box[3] / src_h, (box[3] + box[1]) / src_h
        targets = [(x, min(max((y - top) / (bottom - top), 0.0), 1.0), w) for x, y, w in targets]
    cam.keys = _camera(targets, [s.cut for s in samples], duration)
    cam.faces = sum(bool(s.faces) for s in samples) > len(samples) / 2
    return cam
