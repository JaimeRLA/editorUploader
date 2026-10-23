"""Autenticación OAuth y subida de vídeos con la YouTube Data API v3."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httplib2
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from .utils import log

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
RETRYABLE_STATUS = {500, 502, 503, 504}
QUOTA_REASONS = {"quotaExceeded", "uploadLimitExceeded", "dailyLimitExceeded"}
# Errores de la cuenta, no del vídeo: no tiene sentido intentar el siguiente clip.
ACCOUNT_REASONS = {"youtubeSignupRequired", "forbidden", "insufficientPermissions", "authError"}
NO_CHANNEL = (
    "La cuenta de Google autorizada no tiene canal de YouTube. Crea el canal en youtube.com con esa "
    "cuenta o, si es una cuenta de marca, vuelve a conectarlo con "
    "`python -m editoruploader channel auth <canal> --reset` y elige el canal (no tu Gmail personal)."
)

MISSING_SECRETS = """No encuentro {path}.
Crea en Google Cloud Console un cliente OAuth de tipo «Aplicación de escritorio»
(con la YouTube Data API v3 activada), descarga su JSON y guárdalo en esa ruta.
Los pasos están en el README."""


class YouTubeError(RuntimeError):
    pass


class AccountError(YouTubeError):
    pass


class QuotaExceeded(YouTubeError):
    pass


class NotConnected(YouTubeError):
    """No hay sesión válida y no se puede pedir (ejecución automática)."""


def load_credentials(client_secrets: Path, token_file: Path, interactive: bool = True) -> Credentials:
    creds = None
    if token_file.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        except ValueError:
            creds = None
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            log("La sesión de YouTube caducó o fue revocada; hay que volver a autorizar.")
        else:
            _save(creds, token_file)
            return creds

    if not interactive:
        raise NotConnected("no hay sesión de YouTube válida (o caducó)")
    if not client_secrets.exists():
        raise YouTubeError(MISSING_SECRETS.format(path=client_secrets))
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
    # open_browser=False: en WSL/servidores no hay navegador que abrir; se imprime la URL.
    creds = flow.run_local_server(
        port=0,
        open_browser=False,
        authorization_prompt_message=(
            "\nAbre esta URL en tu navegador, inicia sesión y elige el canal de YouTube:\n{url}\n"
        ),
        success_message="Autorización completada. Ya puedes cerrar esta pestaña.",
        access_type="offline",
        # select_account: con varios canales, Google debe preguntar cuál conectar cada vez.
        prompt="select_account consent",
    )
    _save(creds, token_file)
    return creds


def _save(creds: Credentials, token_file: Path) -> None:
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(creds.to_json(), encoding="utf-8")
    token_file.chmod(0o600)


def make_title(text: str) -> str:
    title = re.sub(r"[<>]", "", " ".join(text.split()))
    return title[:100].rstrip() or "Short"


class _Blank(dict):
    def __missing__(self, key: str) -> str:
        return ""


def render_description(template: str, **fields: str) -> str:
    text = template.format_map(_Blank(fields))
    text = re.sub(r"[<>]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text.encode("utf-8")[:4900].decode("utf-8", errors="ignore")


class Uploader:
    def __init__(self, creds: Credentials):
        self.service = build("youtube", "v3", credentials=creds, cache_discovery=False)

    def channel(self) -> tuple[str, str] | None:
        """(id, nombre) del canal de YouTube de la cuenta autorizada, o None si no tiene."""
        items = self.service.channels().list(part="snippet", mine=True).execute().get("items") or []
        return (items[0]["id"], items[0]["snippet"]["title"]) if items else None

    def upload(
        self,
        path: Path,
        *,
        title: str,
        description: str,
        tags: list[str],
        category_id: str,
        privacy: str,
    ) -> str:
        body = {
            "snippet": {
                "title": make_title(title),
                "description": description,
                "tags": tags,
                "categoryId": category_id,
            },
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
        }
        media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
        request = self.service.videos().insert(part="snippet,status", body=body, media_body=media)

        response, retries = None, 0
        while response is None:
            try:
                _, response = request.next_chunk()
            except HttpError as exc:
                reason = _reason(exc)
                if reason == "youtubeSignupRequired":
                    raise AccountError(NO_CHANNEL) from exc
                if reason in ACCOUNT_REASONS:
                    raise AccountError(f"YouTube rechazó la cuenta ({reason}): {exc}") from exc
                if reason in QUOTA_REASONS:
                    raise QuotaExceeded(reason) from exc
                if exc.resp.status in RETRYABLE_STATUS and retries < 5:
                    retries += 1
                    time.sleep(2**retries)
                    continue
                raise YouTubeError(f"YouTube rechazó la subida ({exc.resp.status} {reason}): {exc}") from exc
            except (httplib2.HttpLib2Error, OSError) as exc:
                if retries >= 5:
                    raise YouTubeError(f"Error de red subiendo {path.name}: {exc}") from exc
                retries += 1
                time.sleep(2**retries)
        return response["id"]


def _reason(exc: HttpError) -> str:
    try:
        return json.loads(exc.content.decode("utf-8"))["error"]["errors"][0]["reason"]
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return ""
