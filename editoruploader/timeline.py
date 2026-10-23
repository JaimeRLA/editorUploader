"""Cortes de salto: qué trozos de un tramo del original se quedan al quitar los silencios.

Se usa al elegir momentos (para saber cuánto durará de verdad un Short ya montado) y al
montarlo. Los tiempos de palabra vienen de Whisper.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right

GAP = 0.35  # pausas más largas que esto se quitan
KEEP_BEFORE = 0.08  # margen que se deja antes de la primera palabra de cada trozo
KEEP_AFTER = 0.15  # y después de la última (Whisper suele cortar pronto el final de palabra)
MIN_PIECE = 0.5  # trozos más cortos se unen al anterior (cortes demasiado seguidos marean)


class Words:
    """Palabras de la transcripción con búsqueda rápida por tiempo."""

    def __init__(self, words: list[dict]):
        self.words = words
        self.starts = [w["s"] for w in words]
        self.ends = [w["e"] for w in words]

    def between(self, start: float, end: float) -> list[dict]:
        lo = bisect_right(self.ends, start)
        hi = bisect_left(self.starts, end)
        return self.words[lo:hi]


def speech_pieces(words: Words, start: float, end: float, jump_cuts: bool = True) -> list[tuple[float, float]]:
    """Trozos (inicio, fin) de [start, end] con voz; sin `jump_cuts`, un solo trozo."""
    spoken = words.between(start, end)
    if not spoken:
        return [(start, end)] if end - start > 0.2 else []
    first = max(start - KEEP_BEFORE, spoken[0]["s"] - KEEP_BEFORE)
    last = min(end + KEEP_AFTER, spoken[-1]["e"] + KEEP_AFTER)
    if not jump_cuts:
        return [(first, last)]
    pieces: list[list[float]] = []
    piece_start, prev_end = first, spoken[0]["e"]
    for word in spoken[1:]:
        if word["s"] - prev_end > GAP:
            pieces.append([piece_start, prev_end + KEEP_AFTER])
            piece_start = word["s"] - KEEP_BEFORE
        prev_end = max(prev_end, word["e"])
    pieces.append([piece_start, last])
    merged: list[list[float]] = []
    for piece in pieces:
        if merged and (piece[1] - piece[0] < MIN_PIECE or merged[-1][1] - merged[-1][0] < MIN_PIECE):
            merged[-1][1] = piece[1]
        else:
            merged.append(piece)
    return [(a, b) for a, b in merged]


def duration(pieces: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in pieces)
