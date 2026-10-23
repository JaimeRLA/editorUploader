"""Proceso de un canal: lo viral de su temática → Whisper → guion (LLM) → montaje → YouTube.

La descarga y la transcripción de cada vídeo se guardan en work/<vídeo>/ y se comparten entre
canales; la selección y el guion de los Shorts (que dependen de la temática) se guardan por canal.
Relanzar retoma donde se quedó y no repite trabajo.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import channels, editor, media, music, trends
from .channels import Channel
from .config import Settings
from .highlights import Candidate, LLMError, Ollama, Profile, find_highlights, refine_story
from .timeline import Words
from .sources import Skip, SourceError, SourceVideo, WrongLanguage, discover
from .state import State
from .transcribe import transcribe
from .uploads import Uploads, pending
from .utils import log, slugify

# Versión del análisis: si cambia (p. ej. nuevos campos del LLM), se recalcula aunque haya caché.
ANALYSIS_VERSION = 5
RENDER_WORKERS = 2
MAX_ATTEMPTS = 3  # vídeos que fallan: se reintentan en ejecuciones posteriores hasta este límite
BACKLOG_DAYS = 3  # con más de estos días de Shorts pendientes de subir, no se procesan vídeos nuevos


def clips_for(duration: float) -> int:
    """Un Short por cada 8 minutos de vídeo, entre 3 y 20."""
    return max(3, min(20, round(duration / 60 / 8)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Engine:
    def __init__(self, settings: Settings, channel: Channel, *, uploader=None,
                 reanalyze: bool = False, clips: int | None = None):
        self.settings = settings
        self.channel = channel
        self.state = State(channel.state_file)
        self.llm = Ollama(settings.ollama_url, settings.ollama_model)
        self.uploads = Uploads(settings, channel, self.state, uploader)
        self.reanalyze = reanalyze
        self.clips = clips

    def run(self, videos: list[SourceVideo] | None = None, how_many: int | None = None) -> None:
        """Procesa `videos` o, si no se indican, los nuevos de las fuentes del canal."""
        ch = self.channel
        self.uploads.upload_pending()  # primero lo que quedó pendiente de otros días
        if videos is None:
            backlog = len(pending(self.state))
            limit = ch.upload.max_per_day * BACKLOG_DAYS
            if ch.upload.enabled and backlog >= limit:
                log(f"  · {backlog} Shorts esperan a subirse (más de {BACKLOG_DAYS} días al ritmo de "
                    f"{ch.upload.max_per_day}/día): no proceso vídeos nuevos hasta que bajen")
                return
            videos = self._discover(how_many or ch.sources.videos_per_run)
            if not videos:
                log("  · no hay vídeos nuevos en las fuentes del canal")
                return
        self.llm.check()
        for video in videos:
            entry = self.state.source(video.url)
            entry.setdefault("title", video.title)
            log(f"▶ [{ch.id}] «{entry['title']}»")
            try:
                self._process(video, entry)
            except Skip as exc:
                entry.update(status="skipped", error=str(exc))
                self.state.save()
                log(f"  – {exc}")
                continue
            except (SourceError, LLMError, RuntimeError, OSError) as exc:
                entry.update(status="error", error=str(exc), attempts=entry.get("attempts", 0) + 1)
                self.state.save()
                log(f"✗ «{entry['title']}»: {exc}")
                continue
            self.uploads.upload_pending(only=[video.url])

    def _discover(self, want: int) -> list[SourceVideo]:
        src = self.channel.sources
        if not src.urls and not src.searches and not (src.mode == "viral" and self.channel.niche):
            log(f"  · «{self.channel.id}» no tiene fuentes: añade urls o searches en {self.channel.config_file}")
            return []
        # Ya vistos: terminados, descartados o fallidos demasiadas veces. Lo que se quedó a medias
        # (ejecución interrumpida) se retoma.
        known = {
            url for url, entry in self.state.sources.items()
            if entry.get("status") in ("ready", "skipped")
            or (entry.get("status") == "error" and entry.get("attempts", 0) >= MAX_ATTEMPTS)
        }
        # Lo que ya usan tus otros canales tampoco: ni el mismo vídeo ni, a poder ser, el mismo tema.
        recent = [e.get("title", "") for e in list(self.state.sources.values())[-15:] if e.get("status") == "ready"]
        for other in channels.all_channels(self.settings):
            if other.id != self.channel.id:
                used = State(other.state_file).sources
                known |= {url for url, e in used.items() if e.get("status") != "skipped"}
                recent += [e.get("title", "") for e in list(used.values())[-5:] if e.get("status") == "ready"]
        if src.mode == "viral":
            found = trends.find_viral(self.settings, self.channel, want=want, known=known, recent_titles=recent)
            if found:
                return found
            log("  · el análisis viral no encontró vídeos nuevos; pruebo con lo último de las fuentes")
            if not src.urls and not src.searches:
                return []
        checks = src.only_creative_commons or bool(src.languages)
        # Algunos no pasarán la licencia o el idioma: se piden de más y se descartan al comprobarlo.
        found = discover(
            src.urls, src.searches, want=want * (4 if checks else 1), known=known,
            min_seconds=src.min_minutes * 60, max_seconds=src.max_minutes * 60,
            creative_commons=src.only_creative_commons,
        )
        banned = [word.lower() for word in src.exclude]

        def excluded(*texts: str) -> str | None:
            text = " ".join(texts).lower()
            return next((word for word in banned if word in text), None)

        valid = []
        for video in found:
            word = excluded(video.title)
            if word:
                self.state.source(video.url).update(title=video.title, status="skipped", error=f"excluido («{word}»)")
                log(f"  – «{video.title}»: contiene «{word}» (exclude); se descarta")
                continue
            if not checks:
                valid.append(video)
                if len(valid) >= want:
                    break
                continue
            if len(valid) >= want:
                break
            try:
                meta = media.identity(video.url, self.settings)
            except SourceError as exc:
                log(f"  – {exc}")
                continue
            reason = None
            spoken = meta.get("spoken") or []
            word = excluded(meta.get("title") or "", meta.get("channel") or "")
            if word:
                reason = f"el título o el canal contienen «{word}» (exclude)"
            elif src.only_creative_commons and not media.is_creative_commons(meta.get("license")):
                reason = "sin licencia Creative Commons"
            elif src.languages and spoken and not set(spoken) & set(src.languages):
                reason = f"se habla {', '.join(spoken)} y el canal admite {', '.join(src.languages)}"
            if reason:
                entry = self.state.source(video.url)
                entry.update(title=video.title, status="skipped", error=reason)
                log(f"  – «{video.title}»: {reason}; se descarta")
            else:
                valid.append(video)
        self.state.save()
        return valid

    def _profile(self) -> Profile:
        ch = self.channel
        return Profile(
            channel=ch.name, niche=ch.niche, focus=ch.clips.focus, avoid=ch.clips.avoid,
            language=ch.language, moods=ch.moods,
            min_duration=ch.clips.min_seconds, max_duration=ch.clips.max_seconds,
            story=ch.edit.story, jump_cuts=ch.edit.jump_cuts,
        )

    def _process(self, video: SourceVideo, entry: dict) -> None:
        ch, settings = self.channel, self.settings
        folder = settings.work_dir / f"{slugify(video.title, 40)}-{hashlib.sha1(video.url.encode()).hexdigest()[:8]}"
        wanted_language = ch.sources.languages[0] if ch.sources.languages else None
        src, meta = media.download(video.url, folder, settings, require_cc=ch.sources.only_creative_commons,
                                   language=wanted_language)
        entry.update(
            status="downloaded", error=None, title=meta.get("title") or entry["title"],
            source_channel=meta.get("channel", ""), license=meta.get("license", ""),
        )
        info = media.probe(src, settings)

        # Si YouTube dice qué se habla y coincide con el canal, Whisper no tiene que adivinarlo.
        spoken = meta.get("spoken") or []
        matches = [lang for lang in ch.sources.languages if lang in spoken]
        language = matches[0] if matches else (spoken[0] if len(spoken) == 1 else None)
        transcript = transcribe(
            src, folder / "transcript.json", settings.whisper_model, settings.whisper_device,
            language, settings.ffmpeg,
        )
        words = transcript["words"]
        spoken = transcript.get("language") or ""
        if ch.sources.languages and spoken and spoken not in ch.sources.languages:
            raise WrongLanguage(
                f"en el vídeo se habla «{spoken}» y el canal admite {', '.join(ch.sources.languages)}"
            )
        if len(words) < 30:
            raise SourceError("el vídeo apenas tiene voz; no hay nada que subtitular ni recortar")
        entry["status"] = "transcribed"
        self.state.save()

        candidates = self._highlights(folder, words, meta)
        wanted = self.clips or ch.clips.per_video or clips_for(info.duration)
        chosen = [c for c in candidates if c.score >= ch.clips.min_score][:wanted]
        self._refine(folder, words, candidates, chosen)
        out_dir = settings.output_dir / ch.id / slugify(entry["title"])
        _write_report(out_dir / "candidatos.txt", candidates, entry["title"], ch)
        log(
            f"  {len(candidates)} candidato/s; me quedo con {len(chosen)} de {wanted} buscados "
            f"(nota ≥ {ch.clips.min_score:g}; informe: {out_dir / 'candidatos.txt'})"
        )

        # Los clips de ejecuciones anteriores que ya no están entre los elegidos (y no se han
        # subido) se descartan; si no, se subirían junto con los nuevos.
        keep = {f"{c.start:.2f}-{c.end:.2f}" for c in chosen}
        for clip_id, old in list(entry["clips"].items()):
            if clip_id not in keep and not old.get("youtube_id"):
                if old.get("file"):
                    Path(old["file"]).unlink(missing_ok=True)
                del entry["clips"][clip_id]
        self.state.save()

        edit = ch.edit.signature()
        jobs = []
        for rank, cand in enumerate(chosen, 1):
            clip_id = f"{cand.start:.2f}-{cand.end:.2f}"
            clip = entry["clips"].setdefault(clip_id, {})
            clip.update(
                name=cand.title or entry["title"], score=round(cand.score), rank=rank,
                start=cand.start, end=cand.end, description=cand.description, hook=cand.hook, mood=cand.mood,
            )
            if clip.get("youtube_id"):
                continue
            # Se vuelve a renderizar si cambió la edición del canal.
            if clip.get("file") and Path(clip["file"]).exists() and clip.get("edit") == edit:
                continue
            track = None
            if ch.edit.music:
                used = {c.get("music_file") for c in entry["clips"].values() if c is not clip}
                track = music.pick([ch.music_dir, settings.music_dir], cand.mood, video.url + clip_id, used, ch.moods)
                if track is None and rank == 1:
                    log("  aviso: no hay música; ejecuta `python -m editoruploader setup` o añade tus .mp3")
            dest = out_dir / f"{rank:02d}-{slugify(clip['name'], 50)}.mp4"
            if clip.get("file") and Path(clip["file"]) != dest:
                Path(clip["file"]).unlink(missing_ok=True)
            # Se anota ya para que el siguiente clip no elija el mismo tema.
            clip.update(file=None, music_file=str(track[0]) if track else None,
                        music_credit=track[1] if track else "")
            jobs.append((clip, cand, dest, track, video.url + clip_id))

        # Varios clips a la vez: el análisis de caras usa la CPU y el render NVDEC/NVENC,
        # así que solaparlos aprovecha mejor el equipo.
        hw = media.nvdec_ok(src, settings)
        with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as pool:
            futures = {}
            for clip, cand, dest, track, seed in jobs:
                song = f" ♪ {track[0].stem} ({cand.mood})" if track else ""
                shape = f"{len(cand.segments)} tramo/s" + (" + gancho" if cand.cold_open and ch.edit.story else "")
                log(f"  ✂ [{clip['score']}] {clip['name']} ({cand.duration:.0f} s, {shape}){song}")
                future = pool.submit(self._render, src, info, words, cand, dest, folder, track, seed, hw)
                futures[future] = (clip, dest)
            for future in as_completed(futures):
                clip, dest = futures[future]
                try:
                    future.result()
                except (RuntimeError, OSError) as exc:
                    log(f"  ✗ no pude renderizar «{clip['name']}»: {exc}")
                    continue
                clip.update(file=str(dest), rendered_at=_now(), edit=edit)
                self.state.save()
                log(f"    → {dest}")
        entry["status"] = "ready"
        self.state.save()

    def _refine(self, folder: Path, words: list[dict], candidates: list[Candidate], chosen: list[Candidate]) -> None:
        """Reedición de guion (gancho + cortes de relleno) solo para los Shorts elegidos."""
        todo = [c for c in chosen if not c.refined]
        if not self.channel.edit.story or not todo:
            return
        log(f"  escribiendo el guion de {len(todo)} Short/s…")
        profile = self._profile()
        for cand in todo:
            refine_story(cand, words, self.llm, profile)
            cand.refined = True
        self.llm.unload()
        cache = folder / f"highlights-{self.channel.id}.json"
        data = json.loads(cache.read_text(encoding="utf-8"))
        data["candidates"] = [c.to_dict() for c in candidates]
        cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _highlights(self, folder: Path, words: list[dict], meta: dict) -> list[Candidate]:
        profile = self._profile()
        cache = folder / f"highlights-{self.channel.id}.json"
        params = {"profile": profile.key(), "model": self.llm.model, "v": ANALYSIS_VERSION}
        if cache.exists() and not self.reanalyze:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("params") == params:
                return [Candidate(**c) for c in data["candidates"]]
        candidates = find_highlights(
            words, self.llm, video_title=meta.get("title") or "", heatmap=meta.get("heatmap"), profile=profile,
        )
        self.llm.unload()
        if not candidates:
            raise LLMError("el LLM no encontró ningún fragmento que encaje con la temática del canal")
        cache.write_text(
            json.dumps({"params": params, "candidates": [c.to_dict() for c in candidates]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return candidates

    def _render(self, src: Path, info: media.VideoInfo, words: list[dict], cand: Candidate,
                dest: Path, folder: Path, track: tuple[Path, str] | None, seed: str, hw: bool = False) -> None:
        editor.make_short(
            src=src, info=info, cand=cand, words=Words(words), edit=self.channel.edit, settings=self.settings,
            dest=dest, work=folder / f"edit-{self.channel.id}", hw=hw, track=track, seed=seed,
        )


def _mmss(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _write_report(path: Path, candidates: list[Candidate], title: str, channel: Channel) -> None:
    lines = [f"Candidatos a Short de «{title}» para el canal «{channel.name}» (de mejor a peor)", ""]
    for n, c in enumerate(candidates, 1):
        heat = "-" if c.heat_score is None else f"{c.heat_score:.0f}"
        rank = "-" if c.rank_score is None else f"{c.rank_score:.0f}"
        parts = " + ".join(f"{_mmss(x)}-{_mmss(y)}" for x, y in c.segments) or f"{_mmss(c.start)}-{_mmss(c.end)}"
        cold = f"gancho {_mmss(c.cold_open[0])} → " if c.cold_open else ""
        lines += [
            f"{n:>2}. [{c.score:.0f}] {c.title}  ({c.duration:.0f} s montado)",
            f"    {cold}{parts}",
            f"    rúbrica {c.window_score:.0f} · comparación {rank} · mapa de calor {heat} · claves: {', '.join(c.keywords)}",
            f"    gancho: {c.hook}  ·  música: {c.mood}",
            f"    por qué: {c.reason}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
