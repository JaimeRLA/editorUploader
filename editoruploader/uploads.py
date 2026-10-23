"""Subida a YouTube de los Shorts de un canal, con su límite diario y sus créditos obligatorios."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from .channels import Channel
from .config import Settings
from .media import is_creative_commons
from .state import State
from .utils import log
from .youtube import (
    NO_CHANNEL,
    AccountError,
    QuotaExceeded,
    Uploader,
    YouTubeError,
    load_credentials,
    render_description,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(settings: Settings, channel: Channel, interactive: bool = False) -> Uploader:
    """Sesión de YouTube del canal, comprobando que es el canal de YouTube que se conectó.

    Sin `interactive` (ejecuciones automáticas) nunca abre el navegador: si no hay sesión
    válida lanza NotConnected.
    """
    creds = load_credentials(settings.client_secrets, channel.token_file, interactive=interactive)
    uploader = Uploader(creds)
    found = uploader.channel()
    if found is None:
        raise AccountError(NO_CHANNEL.replace("<canal>", channel.id))
    yt_id, yt_title = found
    expected = channel.youtube()
    if expected and expected.get("id") != yt_id:
        raise AccountError(
            f"La sesión de «{channel.id}» es del canal de YouTube «{yt_title}», pero este canal se "
            f"conectó a «{expected.get('title')}». Para cambiarlo: "
            f"python -m editoruploader channel auth {channel.id} --reset"
        )
    if not expected:
        channel.save_youtube(yt_id, yt_title)
    return uploader


def uploaded_today(state: State) -> int:
    today = date.today()
    count = 0
    for entry in state.sources.values():
        for clip in entry.get("clips", {}).values():
            stamp = clip.get("uploaded_at")
            if stamp and datetime.fromisoformat(stamp).astimezone().date() == today:
                count += 1
    return count


def pending(state: State, only: list[str] | None = None) -> list[tuple[str, dict, dict]]:
    """(url, vídeo, clip) renderizados y sin subir, de mejor a peor nota."""
    items = []
    for url, entry in state.sources.items():
        if only is not None and url not in only:
            continue
        for clip in entry.get("clips", {}).values():
            if clip.get("file") and not clip.get("youtube_id"):
                items.append((url, entry, clip))
    return sorted(items, key=lambda item: item[2].get("score", 0), reverse=True)


def description_for(channel: Channel, url: str, entry: dict, clip: dict) -> str:
    spanish = channel.language in ("", "es")
    fields = {
        "title": clip["name"],
        "description": clip.get("description", ""),
        "source_url": url,
        "source_title": entry.get("title", ""),
        "source_channel": entry.get("source_channel", "") or ("su autor" if spanish else "its author"),
        "music_credit": clip.get("music_credit", ""),
        "channel_name": channel.name,
    }
    text = render_description(channel.upload.description, **fields)
    # Las licencias CC BY (vídeo y música) exigen citar al autor aunque la plantilla no lo haga.
    if is_creative_commons(entry.get("license")) and url not in text:
        if spanish:
            text += (f"\n\nFuente: «{fields['source_title']}» de {fields['source_channel']} — {url}\n"
                     f"Licencia: {entry.get('license')}")
        else:
            text += (f"\n\nSource: \"{fields['source_title']}\" by {fields['source_channel']} — {url}\n"
                     f"License: {entry.get('license')}")
    credit = clip.get("music_credit")
    if credit and credit not in text:
        text += f"\n\n{credit}"
    return text


class Uploads:
    def __init__(self, settings: Settings, channel: Channel, state: State, uploader: Uploader | None):
        self.settings = settings
        self.channel = channel
        self.state = state
        self.uploader = uploader
        self.blocked = False  # límite de YouTube alcanzado o cuenta con problemas

    def remaining_today(self) -> int:
        return max(0, self.channel.upload.max_per_day - uploaded_today(self.state))

    def upload_pending(self, only: list[str] | None = None) -> int:
        if self.blocked or not self.uploader:
            return 0
        queue = pending(self.state, only)
        if not queue:
            return 0
        budget = self.remaining_today()
        if budget == 0:
            log(f"  · límite diario de «{self.channel.id}» alcanzado ({self.channel.upload.max_per_day}); "
                f"{len(queue)} Short/s esperan a mañana")
            return 0
        privacy = self.channel.upload.privacy
        uploaded = 0
        for url, entry, clip in queue[:budget]:
            path = Path(clip["file"])
            if not path.exists():
                log(f"  aviso: falta el fichero {path}")
                continue
            log(f"  ⬆ subiendo «{clip['name']}»…")
            try:
                video_id = self.uploader.upload(
                    path,
                    title=clip["name"],
                    description=description_for(self.channel, url, entry, clip),
                    tags=self.channel.upload.tags,
                    category_id=self.channel.upload.category_id,
                    privacy=privacy,
                )
            except QuotaExceeded:
                self.blocked = True
                log("✗ YouTube no admite más subidas hoy (límite de la API o del canal). "
                    "Los Shorts quedan pendientes para la siguiente ejecución.")
                return uploaded
            except AccountError as exc:
                self.blocked = True
                log(f"✗ {exc}")
                return uploaded
            except YouTubeError as exc:
                log(f"  ✗ falló la subida de «{clip['name']}»: {exc}")
                continue
            clip.update(youtube_id=video_id, uploaded_at=_now(), privacy=privacy)
            self.state.save()
            uploaded += 1
            log(f"  ✓ https://youtube.com/shorts/{video_id} ({privacy})")
        left = len(queue) - uploaded
        if left and not self.blocked and len(queue) > budget:
            log(f"  · {left} Short/s pendientes para otro día (máximo {self.channel.upload.max_per_day} al día)")
        return uploaded

