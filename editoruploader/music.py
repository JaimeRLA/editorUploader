"""Música de fondo por ambiente.

La carpeta music/ tiene una subcarpeta por ambiente (music/energetic/, music/emotional/…).
Cada canal puede tener la suya en channels/<id>/music/ (si tiene temas, se usa en vez de la
global). El LLM asigna un ambiente a cada clip y aquí se elige un tema de esa carpeta, sin
repetir dentro del mismo vídeo. `setup` descarga un pack gratuito de Kevin MacLeod (CC BY 4.0, exige
crédito, que se añade solo a la descripción); puedes añadir tus propios .mp3 en cada carpeta.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

MOODS = {
    "energetic": "enérgico, rápido, con subidón",
    "funny": "humor, situación cómica o absurda",
    "chill": "neutro, conversación tranquila, curiosidad",
    "emotional": "emotivo, triste, íntimo, nostálgico",
    "tense": "tensión, misterio, polémica, conflicto, algo oscuro",
    "inspiring": "inspirador, motivación, superación, épico",
}
# Si los temas de un ambiente ya se usaron en el vídeo, se prueba con estos antes de repetir.
NEIGHBORS = {
    "energetic": ("inspiring", "funny"),
    "funny": ("chill", "energetic"),
    "chill": ("inspiring", "funny"),
    "emotional": ("inspiring", "chill"),
    "tense": ("chill", "energetic"),
    "inspiring": ("emotional", "energetic"),
}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".ogg", ".wav", ".flac", ".opus"}

INCOMPETECH = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"
PACK = {
    "energetic": ["Volatile Reaction", "Movement Proposition", "Cut and Run", "Rising Game", "Funkorama", "Take a Chance"],
    "funny": ["Sneaky Snitch", "Fluffing a Duck", "Monkeys Spinning Monkeys", "Life of Riley", "Wallpaper",
              "Pixelland", "Local Forecast - Elevator", "Lobby Time"],
    "chill": ["Chill Wave", "Easy Lemon", "Bossa Antigua", "Carefree", "Overcast", "Night in Venice",
              "Wholesome", "Thinking Music", "Breaktime"],
    "emotional": ["Long Note Two", "Sad Trio", "Heartwarming", "At Rest", "Dreamer", "Unwritten Return"],
    "tense": ["Investigations", "Hidden Agenda", "Comfortable Mystery", "Hitman", "Deliberate Thought",
              "Danse Macabre", "Aitech", "Lightless Dawn", "Ether Vox"],
    "inspiring": ["Inspired", "Ascending the Vale", "Enchanted Valley", "Impact Moderato"],
}
CREDIT = (
    '"{title}" Kevin MacLeod (incompetech.com)\n'
    "Licensed under Creative Commons: By Attribution 4.0 License\n"
    "http://creativecommons.org/licenses/by/4.0/"
)


def pack_downloads(music_dir: Path) -> list[tuple[str, Path, str]]:
    """(url, destino, crédito) de cada tema del pack que aún no está descargado."""
    missing = []
    for mood, titles in PACK.items():
        for title in titles:
            dest = music_dir / mood / f"{title}.mp3"
            if not dest.exists():
                missing.append((INCOMPETECH + quote(title) + ".mp3", dest, CREDIT.format(title=title)))
    return missing


def save_credits(music_dir: Path, credits: dict[str, str]) -> None:
    """credits.json: «ambiente/fichero» → texto de crédito para la descripción."""
    path = music_dir / "credits.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(credits)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _tracks(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def _has_tracks(music_dir: Path) -> bool:
    return bool(_tracks(music_dir)) or any(_tracks(music_dir / m) for m in MOODS)


def pick(
    music_dirs: list[Path], mood: str, seed: str, avoid: set[str], allowed: list[str] | None = None
) -> tuple[Path, str] | None:
    """Elige un tema del ambiente pedido, sin repetir los de `avoid` mientras queden otros.

    `music_dirs` va de más a menos específica (la del canal y luego la global): se usa la
    primera que tenga música. Si el ambiente se agota, se prueba con ambientes cercanos que
    estén en `allowed`. La elección es determinista con `seed` (relanzar da el mismo tema).
    Devuelve (fichero, crédito) o None si no hay música.
    """
    music_dir = next((d for d in music_dirs if _has_tracks(d)), None)
    if music_dir is None:
        return None
    allowed = allowed or list(MOODS)
    if mood not in allowed:
        mood = allowed[0]
    loose = _tracks(music_dir)  # sueltos en music/: valen para cualquier ambiente
    tracks = _tracks(music_dir / mood) + loose
    fresh = [t for t in tracks if str(t) not in avoid]
    for other in NEIGHBORS.get(mood, ()):
        if fresh:
            break
        if other in allowed:
            fresh = [t for t in _tracks(music_dir / other) if str(t) not in avoid]
    if not fresh:
        tracks = tracks or [t for m in allowed for t in _tracks(music_dir / m)]
        fresh = [t for t in tracks if str(t) not in avoid] or tracks
    if not fresh:
        return None
    index = int(hashlib.sha1(seed.encode()).hexdigest(), 16) % len(fresh)
    track = fresh[index]
    credits_file = music_dir / "credits.json"
    credits = json.loads(credits_file.read_text(encoding="utf-8")) if credits_file.exists() else {}
    return track, credits.get(str(track.relative_to(music_dir)), "")


def start_offset(seed: str, track_seconds: float, clip_seconds: float) -> float:
    """Punto de entrada en el tema: salta la intro y varía entre clips."""
    room = track_seconds - clip_seconds - 5
    if room <= 10:
        return 0.0
    lo, hi = min(8.0, room), min(room, track_seconds * 0.5)
    fraction = int(hashlib.sha1((seed + "offset").encode()).hexdigest(), 16) % 1000 / 1000
    return lo + (hi - lo) * fraction if hi > lo else lo
