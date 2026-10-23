"""Línea de comandos: python -m editoruploader <comando>."""
from __future__ import annotations

import argparse
import sys

from . import channels
from .channels import Channel, ChannelError
from .config import Settings, load_settings
from .engine import Engine
from .highlights import LLMError
from .sources import SourceError, collect
from .state import State
from .uploads import Uploads, connect, pending, uploaded_today
from .utils import log
from .youtube import NotConnected, YouTubeError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m editoruploader",
        description="Vídeos largos → Shorts con subtítulos y música para varios canales de YouTube. Gratis y en local.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="descarga ffmpeg, modelos, fuentes y música (gratis)")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("channel", help="crea, lista y conecta canales")
    csub = p.add_subparsers(dest="action", required=True)
    c = csub.add_parser("add", help="crea un canal nuevo (pregunta lo que falte)")
    c.add_argument("id", nargs="?", help="identificador corto: minúsculas, números y guiones (p. ej. ciencia-es)")
    c.add_argument("--name", help="nombre del canal")
    c.add_argument("--niche", help="temática, en una frase")
    c.add_argument("--language", help="idioma de títulos y descripciones: es, en, pt…")
    c.add_argument("--url", action="append", default=[], help="fuente: canal, playlist o vídeo (repetible)")
    c.add_argument("--search", action="append", default=[], help="búsqueda de YouTube como fuente (repetible)")
    c.add_argument("--no-ai", action="store_true", help="no pedir al LLM que proponga la configuración")
    c.add_argument("--no-auth", action="store_true", help="no conectar el canal de YouTube al terminar")
    c.set_defaults(func=cmd_channel_add)
    c = csub.add_parser("list", help="lista los canales")
    c.set_defaults(func=cmd_channel_list)
    c = csub.add_parser("auth", help="conecta (o cambia) la cuenta de YouTube de un canal")
    c.add_argument("id")
    c.add_argument("--reset", action="store_true", help="olvida la sesión guardada y vuelve a autorizar")
    c.set_defaults(func=cmd_channel_auth)

    p = sub.add_parser("run", help="busca vídeos nuevos, crea los Shorts y los sube")
    p.add_argument("channel", nargs="?", help="canal a procesar (o --all)")
    p.add_argument("urls", nargs="*", help="vídeos concretos (si no, los nuevos de las fuentes del canal)")
    p.add_argument("--all", action="store_true", help="todos los canales")
    p.add_argument("--videos", type=int, help="vídeos nuevos a procesar (por defecto, videos_per_run del canal)")
    p.add_argument("-n", "--clips", type=int, help="Shorts por vídeo (por defecto, per_video del canal)")
    p.add_argument("--no-upload", action="store_true", help="solo crear los Shorts; subirlos luego con `upload`")
    p.add_argument("--reanalyze", action="store_true", help="vuelve a pedir al LLM los momentos aunque ya estén")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("upload", help="sube los Shorts pendientes (respetando el máximo diario)")
    p.add_argument("channel", nargs="?", help="canal (o --all)")
    p.add_argument("--all", action="store_true", help="todos los canales")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("status", help="resumen de los canales y sus Shorts")
    p.add_argument("channel", nargs="?", help="solo este canal, con el detalle de cada Short")
    p.set_defaults(func=cmd_status)
    return parser


def _targets(args: argparse.Namespace, settings: Settings) -> list[Channel]:
    if args.all:
        if args.channel:
            raise ChannelError("usa un canal o --all, no ambos")
        found = channels.all_channels(settings)
        if not found:
            raise ChannelError("no hay canales; crea uno con: python -m editoruploader channel add")
        return found
    if not args.channel:
        raise ChannelError("indica un canal o --all (canales: python -m editoruploader channel list)")
    if "://" in args.channel:
        raise ChannelError("indica primero el canal: python -m editoruploader run <canal> \"URL\"")
    return [channels.load(settings, args.channel)]


def cmd_setup(args: argparse.Namespace, settings: Settings) -> int:
    from .setup_tools import setup

    setup(settings)
    return 0


def _ask(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer or default


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def cmd_channel_add(args: argparse.Namespace, settings: Settings) -> int:
    interactive = sys.stdin.isatty()
    channel_id = args.id or (interactive and _ask("Identificador corto del canal (p. ej. ciencia-es)")) or ""
    if not channels.valid_id(channel_id):
        raise ChannelError("el identificador debe ser corto, en minúsculas, con números y guiones (p. ej. ciencia-es)")
    name = args.name or (_ask("Nombre del canal", channel_id) if interactive else channel_id)
    niche = args.niche or (_ask("Temática, en una frase (p. ej. «divulgación científica y espacio»)") if interactive else "")
    language = args.language or (_ask("Idioma de títulos y descripciones (es, en, pt…)", "es") if interactive else "es")
    urls, searches = list(args.url), list(args.search)
    if interactive and not urls and not searches:
        urls = _split(_ask("Fuentes: canales o playlists de YouTube, separados por comas (opcional)"))
        searches = _split(_ask("Búsquedas de YouTube, separadas por comas (opcional; si no, se proponen)"))

    extra: dict = {}
    if not args.no_ai and niche:
        from .suggest import suggest

        log(f"Pidiendo a {settings.ollama_model} una configuración para «{niche}»…")
        try:
            extra = suggest(settings, name=name, niche=niche, language=language)
        except LLMError as exc:
            log(f"  aviso: sin propuesta automática ({exc}); se usan los valores por defecto")
    suggested_searches = extra.pop("searches", [])
    if not urls and not searches:
        searches = suggested_searches
    path = channels.create(
        settings, channel_id, name=name, niche=niche, language=language, urls=urls, searches=searches, **extra,
    )
    ch = channels.load(settings, channel_id)
    log(f"✓ Canal «{ch.name}» creado: {path}")
    log(f"  temática: {ch.niche or '(sin definir: edítala en el fichero)'}")
    log(f"  fuentes: {len(ch.sources.urls)} URL/s, búsquedas: {', '.join(ch.sources.searches) or '-'}")
    if ch.clips.focus:
        log(f"  busca: {ch.clips.focus}")
    log(f"  música: {', '.join(ch.moods)} · letra: {ch.edit.font} {ch.edit.highlight_color} · etiquetas: {', '.join(ch.upload.tags)}")
    log("  Revisa y ajusta el fichero cuando quieras; los cambios se aplican en la siguiente ejecución.")

    if not args.no_auth and interactive and _ask("¿Conectar ahora su canal de YouTube? (s/n)", "s").lower().startswith("s"):
        return cmd_channel_auth(argparse.Namespace(id=channel_id, reset=False), settings)
    log(f"Para conectarlo a YouTube: python -m editoruploader channel auth {channel_id}")
    return 0


def cmd_channel_list(args: argparse.Namespace, settings: Settings) -> int:
    found = channels.all_channels(settings)
    if not found:
        print("No hay canales. Crea uno con: python -m editoruploader channel add")
        return 0
    for ch in found:
        yt = ch.youtube().get("title") or "sin conectar"
        print(f"{ch.id:<20} {ch.name}  ·  YouTube: {yt}  ·  {ch.niche}")
    return 0


def cmd_channel_auth(args: argparse.Namespace, settings: Settings) -> int:
    ch = channels.load(settings, args.id)
    if args.reset:
        ch.token_file.unlink(missing_ok=True)
        ch.youtube_file.unlink(missing_ok=True)
    while True:
        if not ch.token_file.exists():
            log(f"En la pantalla de Google, elige el canal de YouTube de «{ch.name}» "
                "(si tu Gmail tiene varios, aparecen en la lista «Elige una cuenta o una cuenta de marca»).")
        connect(settings, ch, interactive=True)
        title = ch.youtube().get("title")
        if not sys.stdin.isatty() or _ask(f"Conectado a «{title}». ¿Es el canal correcto? (s/n)", "s").lower().startswith("s"):
            break
        # Canal equivocado: se olvida la sesión y se vuelve a preguntar.
        ch.token_file.unlink(missing_ok=True)
        ch.youtube_file.unlink(missing_ok=True)
    log(f"✓ «{ch.id}» conectado al canal de YouTube «{ch.youtube().get('title')}»")
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    targets = _targets(args, settings)
    if args.urls and len(targets) > 1:
        raise ChannelError("las URLs concretas solo se pueden usar con un canal, no con --all")
    for ch in targets:
        log(f"━━ Canal «{ch.name}» ({ch.id})")
        uploader = None
        if ch.upload.enabled and not args.no_upload:
            try:
                uploader = connect(settings, ch, interactive=False)
            except NotConnected:
                log(f"  · sin conexión con YouTube: los Shorts quedan pendientes. "
                    f"Conéctalo con: python -m editoruploader channel auth {ch.id}")
            except YouTubeError as exc:
                log(f"  ✗ {exc}")
        videos = collect(args.urls) if args.urls else None
        if args.urls and not videos:
            log("  · ninguna de las URLs es un vídeo válido")
            continue
        engine = Engine(settings, ch, uploader=uploader, reanalyze=args.reanalyze, clips=args.clips)
        engine.run(videos, how_many=args.videos)
    log("Terminado. Resumen: python -m editoruploader status")
    return 0


def cmd_upload(args: argparse.Namespace, settings: Settings) -> int:
    for ch in _targets(args, settings):
        log(f"━━ Canal «{ch.name}» ({ch.id})")
        state = State(ch.state_file)
        if not ch.upload.enabled:
            log(f"  · la subida está desactivada ([upload] enabled = false en {ch.config_file})")
            continue
        if not pending(state):
            log("  · no hay Shorts pendientes de subir")
            continue
        try:
            uploader = connect(settings, ch, interactive=sys.stdin.isatty())
        except NotConnected:
            log(f"  · sin conexión con YouTube. Conéctalo con: python -m editoruploader channel auth {ch.id}")
            continue
        count = Uploads(settings, ch, state, uploader).upload_pending()
        log(f"  {count} Short/s subido/s")
    return 0


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    targets = [channels.load(settings, args.channel)] if args.channel else channels.all_channels(settings)
    if not targets:
        print("No hay canales. Crea uno con: python -m editoruploader channel add")
        return 0
    for ch in targets:
        state = State(ch.state_file)
        counts: dict[str, int] = {}
        for entry in state.sources.values():
            counts[entry.get("status", "?")] = counts.get(entry.get("status", "?"), 0) + 1
        clips = [c for e in state.sources.values() for c in e.get("clips", {}).values()]
        uploaded = sum(1 for c in clips if c.get("youtube_id"))
        waiting = len(pending(state))
        yt = ch.youtube().get("title") or "sin conectar"
        print(f"\n{ch.name} ({ch.id}) → YouTube: {yt}")
        print(f"  temática: {ch.niche or '-'}")
        summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "ninguno"
        print(f"  vídeos fuente: {summary}")
        if ch.upload.enabled:
            print(f"  Shorts: {uploaded} subidos ({uploaded_today(state)} hoy, máx. {ch.upload.max_per_day}/día), "
                  f"{waiting} pendientes de subir")
        else:
            print(f"  Shorts: {waiting} creados en {settings.output_dir / ch.id} (subida desactivada)")
        if not args.channel:
            continue
        for url, entry in state.sources.items():
            print(f"\n  {entry.get('title') or url}\n    {url}  [{entry.get('status', '?')}]")
            if entry.get("error"):
                print(f"    {entry['error']}")
            for clip in sorted(entry.get("clips", {}).values(), key=lambda c: c.get("rank", 0)):
                if clip.get("youtube_id"):
                    where = f"https://youtube.com/shorts/{clip['youtube_id']} ({clip.get('privacy', '?')})"
                elif clip.get("file"):
                    where = f"{clip['file']} (pendiente de subir)"
                else:
                    where = "sin renderizar"
                print(f"    [{clip.get('score', 0):>3}] {clip.get('name')} → {where}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    try:
        return args.func(args, settings)
    except KeyboardInterrupt:
        log("Interrumpido. El progreso está guardado: relanza el mismo comando para continuar.")
        return 130
    except (ChannelError, YouTubeError, SourceError, LLMError) as exc:
        log(f"Error: {exc}")
        return 1
