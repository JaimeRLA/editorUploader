"""Registro persistente (JSON) de vídeos procesados, clips y subidas.

Permite retomar el trabajo interrumpido y garantiza que ningún clip se sube dos veces.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"sources": {}}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            self.data.setdefault("sources", {})

    @property
    def sources(self) -> dict[str, dict]:
        return self.data["sources"]

    def source(self, url: str) -> dict:
        entry = self.sources.setdefault(url, {})
        entry.setdefault("clips", {})
        return entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
