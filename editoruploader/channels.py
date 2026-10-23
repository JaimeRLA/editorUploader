"""Canales de destino: cada uno con su temática, fuentes, edición y cuenta de YouTube.

channels/<id>/
  channel.toml   configuración del canal (se edita a mano; ver la plantilla de `create`)
  token.json     sesión de YouTube de ese canal (`python -m editoruploader channel auth <id>`)
  youtube.json   id y nombre del canal de YouTube conectado (evita subir al canal equivocado)
  state.json     vídeos procesados, clips y subidas
  music/         opcional: música propia por ambiente (music/<ambiente>/*.mp3); si no, la global
"""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .music import MOODS

PRIVACY_CHOICES = ("private", "unlisted", "public")
LAYOUTS = ("auto", "center")
MODES = ("viral", "latest")
PERIODS = ("week", "month", "year", "any")
LANGUAGES = {"es": "español", "en": "inglés", "pt": "portugués", "fr": "francés", "it": "italiano", "de": "alemán"}
CATEGORIES = {
    "1": "Cine y animación", "2": "Motor", "10": "Música", "15": "Mascotas y animales", "17": "Deportes",
    "19": "Viajes", "20": "Videojuegos", "22": "Personas y blogs", "23": "Comedia", "24": "Entretenimiento",
    "25": "Noticias y política", "26": "Consejos y estilo", "27": "Educación", "28": "Ciencia y tecnología",
}
DEFAULT_DESCRIPTION = """{description}

Fuente: «{source_title}» de {source_channel}
{source_url}

{music_credit}

#Shorts"""
# Plantilla en inglés para los canales que no son en español.
DEFAULT_DESCRIPTION_EN = """{description}

Source: "{source_title}" by {source_channel}
{source_url}

{music_credit}

#Shorts"""


def default_description(language: str) -> str:
    return DEFAULT_DESCRIPTION if language in ("", "es") else DEFAULT_DESCRIPTION_EN
# Resubidas típicas de contenido ajeno mal etiquetado como Creative Commons.
DEFAULT_EXCLUDE = [
    "National Geographic", "BBC", "Discovery Channel", "History Channel", "Netflix", "HBO", "Disney",
    "full movie", "película completa", "pelicula completa", "reupload", "re-upload",
]
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


class ChannelError(RuntimeError):
    pass


@dataclass
class Sources:
    mode: str = "viral"  # viral: lo que más funciona en la temática; latest: lo último de las fuentes
    period: str = "month"  # ventana de tiempo del análisis viral
    urls: list[str] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    videos_per_run: int = 1
    min_minutes: float = 10
    max_minutes: float = 240
    only_creative_commons: bool = True
    languages: list[str] = field(default_factory=list)  # idiomas hablados admitidos; vacío = cualquiera
    exclude: list[str] = field(default_factory=list)  # si el título o el canal contienen esto, se descarta


@dataclass
class ClipRules:
    per_video: int = 0  # 0 = automático: uno cada 8 minutos de vídeo (entre 3 y 20)
    min_seconds: int = 60
    max_seconds: int = 90
    min_score: float = 55
    focus: str = ""
    avoid: str = ""


@dataclass
class EditStyle:
    captions: bool = True
    font: str = "Poppins ExtraBold"
    font_size: int = 100
    text_color: str = "#FFFFFF"
    highlight_color: str = "#FFE600"
    outline_color: str = "#000000"
    uppercase: bool = True
    words_per_line: int = 3
    position: float = 0.68
    hook: bool = True
    hook_seconds: float = 3.0
    music: bool = True
    music_volume: float = -26  # LUFS; la voz va a -14
    music_moods: list[str] = field(default_factory=list)  # vacío = todos (el LLM elige)
    layout: str = "auto"  # auto: sigue caras o lo importante de la imagen; center: recorte central fijo
    # Montaje
    story: bool = True  # gancho inicial (la frase más fuerte) + varios tramos que cuentan una historia
    jump_cuts: bool = True  # quita silencios y pausas
    zoom_effects: bool = True  # zoom de golpe en los cortes y acercamiento lento continuo
    transitions: bool = True  # destello + «whoosh» entre el gancho y la historia y entre tramos
    progress_bar: bool = True
    keyword_color: str = "#00E5FF"  # palabras clave resaltadas en los subtítulos

    def signature(self) -> dict:
        """Lo que, si cambia, obliga a volver a renderizar los clips no subidos."""
        return dict(self.__dict__)


@dataclass
class UploadRules:
    enabled: bool = True
    privacy: str = "private"
    category_id: str = "22"
    tags: list[str] = field(default_factory=lambda: ["shorts"])
    max_per_day: int = 10
    description: str = DEFAULT_DESCRIPTION


@dataclass
class Channel:
    id: str
    dir: Path
    name: str
    niche: str
    language: str
    sources: Sources
    clips: ClipRules
    edit: EditStyle
    upload: UploadRules

    @property
    def config_file(self) -> Path:
        return self.dir / "channel.toml"

    @property
    def token_file(self) -> Path:
        return self.dir / "token.json"

    @property
    def youtube_file(self) -> Path:
        return self.dir / "youtube.json"

    @property
    def state_file(self) -> Path:
        return self.dir / "state.json"

    @property
    def music_dir(self) -> Path:
        return self.dir / "music"

    @property
    def moods(self) -> list[str]:
        return self.edit.music_moods or list(MOODS)

    def youtube(self) -> dict:
        """{"id", "title"} del canal de YouTube conectado, o {} si aún no se ha conectado."""
        if not self.youtube_file.exists():
            return {}
        return json.loads(self.youtube_file.read_text(encoding="utf-8"))

    def save_youtube(self, channel_id: str, title: str) -> None:
        self.youtube_file.write_text(
            json.dumps({"id": channel_id, "title": title}, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _section(data: dict, name: str, cls, path: Path):
    raw = data.get(name, {})
    if not isinstance(raw, dict):
        raise ChannelError(f"{path}: [{name}] debe ser una sección")
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ChannelError(f"{path}: opción desconocida en [{name}]: {', '.join(sorted(unknown))}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ChannelError(f"{path}: [{name}] {exc}") from exc


def load(settings: Settings, channel_id: str) -> Channel:
    folder = settings.channels_dir / channel_id
    path = folder / "channel.toml"
    if not path.exists():
        existing = ", ".join(c.id for c in all_channels(settings)) or "ninguno"
        raise ChannelError(
            f"No existe el canal «{channel_id}» (canales: {existing}). "
            f"Créalo con: python -m editoruploader channel add {channel_id}"
        )
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ChannelError(f"{path} tiene un error de formato: {exc}") from exc
    channel = Channel(
        id=channel_id,
        dir=folder,
        name=str(data.get("name") or channel_id),
        niche=str(data.get("niche") or ""),
        language=str(data.get("language") or ""),
        sources=_section(data, "sources", Sources, path),
        clips=_section(data, "clips", ClipRules, path),
        edit=_section(data, "edit", EditStyle, path),
        upload=_section(data, "upload", UploadRules, path),
    )
    _validate(channel)
    return channel


def _validate(ch: Channel) -> None:
    where = ch.config_file
    problems = []
    if not 1 <= ch.clips.min_seconds < ch.clips.max_seconds <= 180:
        problems.append("[clips] min_seconds debe ser menor que max_seconds, y max_seconds ≤ 180")
    if ch.upload.privacy not in PRIVACY_CHOICES:
        problems.append(f"[upload] privacy debe ser una de: {', '.join(PRIVACY_CHOICES)}")
    if str(ch.upload.category_id) not in CATEGORIES:
        problems.append(f"[upload] category_id no válido; opciones: {', '.join(CATEGORIES)}")
    ch.upload.category_id = str(ch.upload.category_id)
    if ch.edit.layout not in LAYOUTS:
        problems.append(f"[edit] layout debe ser una de: {', '.join(LAYOUTS)} (el vídeo siempre ocupa la pantalla)")
    if ch.sources.mode not in MODES:
        problems.append(f"[sources] mode debe ser una de: {', '.join(MODES)}")
    if ch.sources.period not in PERIODS:
        problems.append(f"[sources] period debe ser una de: {', '.join(PERIODS)}")
    bad_moods = set(ch.edit.music_moods) - set(MOODS)
    if bad_moods:
        problems.append(f"[edit] music_moods desconocidos: {', '.join(bad_moods)}; opciones: {', '.join(MOODS)}")
    for key in ("text_color", "highlight_color", "outline_color", "keyword_color"):
        if not _COLOR.match(getattr(ch.edit, key)):
            problems.append(f"[edit] {key} debe ser un color como \"#FFE600\"")
    if not 0.1 <= ch.edit.position <= 0.95:
        problems.append("[edit] position debe estar entre 0.1 (arriba) y 0.95 (abajo)")
    if ch.sources.videos_per_run < 1:
        problems.append("[sources] videos_per_run debe ser al menos 1")
    if problems:
        raise ChannelError(f"{where}:\n  - " + "\n  - ".join(problems))


def all_channels(settings: Settings) -> list[Channel]:
    if not settings.channels_dir.is_dir():
        return []
    ids = sorted(p.parent.name for p in settings.channels_dir.glob("*/channel.toml"))
    return [load(settings, channel_id) for channel_id in ids]


def valid_id(channel_id: str) -> bool:
    return bool(_ID.match(channel_id))


def _toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)  # una cadena JSON es una cadena TOML válida


def create(
    settings: Settings,
    channel_id: str,
    *,
    name: str,
    niche: str,
    language: str,
    urls: list[str],
    searches: list[str],
    focus: str = "",
    avoid: str = "",
    music_moods: list[str] | None = None,
    tags: list[str] | None = None,
    font: str = "Poppins ExtraBold",
    highlight_color: str = "#FFE600",
    category_id: str = "22",
) -> Path:
    """Escribe channels/<id>/channel.toml con todas las opciones comentadas."""
    folder = settings.channels_dir / channel_id
    path = folder / "channel.toml"
    if path.exists():
        raise ChannelError(f"El canal «{channel_id}» ya existe: {path}")
    from .captions import FONTS  # evita un import circular al cargar el paquete

    categories = ", ".join(f"{k} = {v}" for k, v in CATEGORIES.items())
    text = f'''# Canal «{name}» — configuración de editoruploader.
# Edita lo que quieras; los cambios se aplican en la siguiente ejecución.

name = {_toml(name)}
# Temática del canal: el LLM solo elige momentos que encajen con ella.
niche = {_toml(niche)}
# Idioma de títulos y descripciones ({", ".join(f"{k} = {v}" for k, v in LANGUAGES.items())}).
# Los subtítulos van siempre en el idioma en que se habla en el vídeo.
language = {_toml(language)}

[sources]
# viral: antes de elegir, analiza lo que más está funcionando en la temática (búsquedas ordenadas
# por visitas, rendimiento frente a los suscriptores del canal, participación) y coge de ahí;
# deja el ranking en channels/{channel_id}/viral.txt. latest: lo más reciente de las fuentes.
mode = "viral"
# Ventana del análisis viral: week, month, year o any.
period = "month"
# Canales («https://www.youtube.com/@canal»), playlists o vídeos sueltos de los que coger contenido.
urls = {_toml(urls)}
# Búsquedas en YouTube (con only_creative_commons, solo devuelve vídeos Creative Commons).
searches = {_toml(searches)}
# Vídeos largos nuevos a procesar en cada ejecución (cada uno da varios Shorts).
videos_per_run = 1
# Duración de los vídeos fuente, en minutos.
min_minutes = 10
max_minutes = 240
# Idiomas que se hablan en los vídeos fuente (los subtítulos van en ese idioma).
# Vacío = cualquiera. Se comprueba antes de descargar y otra vez al transcribir.
languages = {_toml([language] if language else [])}
# Solo vídeos con licencia Creative Commons (YouTube permite reutilizarlos citando al autor;
# la cita se añade sola a la descripción). Ponlo a false SOLO si tienes permiso de las fuentes
# (contenido propio, creadores que permiten clips…): resubir vídeos ajenos sin permiso trae
# reclamaciones de Content ID y avisos de copyright que pueden cerrar el canal.
only_creative_commons = true
# Se descartan los vídeos cuyo título o canal contengan alguna de estas palabras. Sirve para
# evitar resubidas de contenido de cadenas o estudios mal etiquetadas como Creative Commons.
exclude = {_toml(DEFAULT_EXCLUDE)}

[clips]
# Shorts por vídeo largo; 0 = automático (uno cada 8 minutos, entre 3 y 20).
per_video = 0
# Duración de cada Short, en segundos (YouTube admite hasta 180).
min_seconds = 60
max_seconds = 90
# Nota mínima (0-100) para aceptar un fragmento.
min_score = 55
# Qué momentos buscar y cuáles evitar en esta temática (se le dice al LLM).
focus = {_toml(focus)}
avoid = {_toml(avoid)}

[edit]
captions = true
# Fuentes incluidas: {", ".join(FONTS)} (o cualquier .ttf que pongas en tools/fonts/).
font = {_toml(font)}
font_size = 100
text_color = "#FFFFFF"
# Color de la palabra que suena en cada momento.
highlight_color = {_toml(highlight_color)}
outline_color = "#000000"
uppercase = true
words_per_line = 3
# Altura de los subtítulos: 0.1 = arriba, 0.95 = abajo.
position = 0.68
# Texto gancho arriba durante los primeros segundos.
hook = true
hook_seconds = 3.0
music = true
# Volumen de la música en LUFS (la voz va a -14): -22 más presente, -30 muy suave.
# Además baja sola mientras alguien habla.
music_volume = -26
# Ambientes de música permitidos ({", ".join(MOODS)}); vacío = el LLM elige entre todos.
music_moods = {_toml(music_moods or [])}
# El vídeo siempre ocupa toda la pantalla vertical. auto: sigue a quien habla o, si no hay caras,
# lo más importante de la imagen (y corta en los cambios de plano); center: recorte central fijo.
layout = "auto"
# Montaje. story: empieza con la frase más fuerte como gancho y monta varios tramos que cuentan
# una historia; jump_cuts: quita silencios; zoom_effects: zoom de golpe en los cortes y acercamiento
# lento; transitions: destello y «whoosh» entre partes; progress_bar: barra de progreso arriba.
story = true
jump_cuts = true
zoom_effects = true
transitions = true
progress_bar = true
# Color de las palabras clave en los subtítulos.
keyword_color = "#00E5FF"

[upload]
# false = solo genera los Shorts en output/{channel_id}/ y no sube nada.
enabled = true
# private | unlisted | public. En privado puedes revisarlos en YouTube Studio antes de publicar.
privacy = "private"
# Categoría de YouTube: {categories}
category_id = {_toml(str(category_id))}
tags = {_toml(tags or ["shorts"])}
# Máximo de Shorts que se suben al día a este canal (el resto espera a la siguiente ejecución).
# Ojo: el límite de la API es de 100 subidas al día para TODOS los canales juntos.
max_per_day = 10
# Marcadores: {{title}} {{description}} {{source_title}} {{source_channel}} {{source_url}}
# {{music_credit}} {{channel_name}}. Si quitas la fuente o el crédito musical, se añaden solos
# al final cuando la licencia (Creative Commons) lo exige.
description = """{default_description(language)}"""
'''
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    load(settings, channel_id)  # comprueba que la plantilla es válida
    return path
