import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time as _t
from pathlib import Path

import cv2
import numpy as np
import torch

import comfy.utils
import folder_paths

from ..utils.ltxv_train_env import (
    free_vram,
    get_script_path,
    get_trainer_env,
    run_training_subprocess,
    validate_submodule,
    validate_text_encoder_path,
)

logger = logging.getLogger(__name__)

# PromptServer is used to push live status updates to the node's on-canvas
# widget while the dataset builds. Guard the import so the node still loads
# if the API moves around.
try:
    from server import PromptServer  # type: ignore
except Exception:
    PromptServer = None  # type: ignore


def _emit_prepper_status(
    node_id,
    char_counts: dict[str, int],
    total_clips: int,
    max_samples: int,
    pool_remaining: int = 0,
    pool_total: int = 0,
) -> None:
    """Send the current character counts + running total to the frontend so
    the node UI can update live. No-op if PromptServer isn't available or
    node_id wasn't provided.

    `total` on the panel tracks APPEARANCES (sum of per-character counts),
    not clip count — that's what max_samples caps. `clips` and
    `chunk pool` show separately so the operator can see capacity usage,
    dataset size, and how much of the source pool remains to search."""
    if PromptServer is None or node_id is None:
        return
    lines = []
    for name in sorted(char_counts):
        lines.append(f"{name}: {char_counts[name]}")
    total_appearances = sum(char_counts.values())
    if max_samples > 0:
        lines.append(f"total: {total_appearances}/{max_samples}")
    else:
        lines.append(f"total: {total_appearances}")
    # Blank line visually separates the per-char + total block from the
    # dataset/pool counters below.
    lines.append("")
    lines.append(f"clips: {total_clips}")
    if pool_total > 0:
        lines.append(f"chunk pool: {pool_remaining}/{pool_total}")
    try:
        PromptServer.instance.send_sync("rs.prepper.status", {
            "node_id": str(node_id),
            "text": "\n".join(lines),
        })
    except Exception:
        pass


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}

# Markers that indicate a valid HF text encoder directory
_HF_DIR_MARKERS = {"tokenizer.model", "config.json", "tokenizer_config.json"}


AUTO_DOWNLOAD_SENTINEL = "auto (download gemma-3-12b-it)"


def _list_text_encoder_dirs() -> list[str]:
    """Scan ComfyUI model folders for HF text encoder directories
    (subdirectories containing tokenizer.model or config.json).
    Always includes an auto-download option as fallback."""
    dirs = []
    search_paths = set()

    # Gather all clip/text_encoder base directories
    for folder_name in ("clip", "text_encoders"):
        try:
            for p in folder_paths.get_folder_paths(folder_name):
                search_paths.add(Path(p))
        except Exception:
            pass

    for base in search_paths:
        if not base.exists():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            # Check if it looks like an HF model directory
            child_files = {f.name for f in child.iterdir() if f.is_file()}
            if child_files & _HF_DIR_MARKERS:
                dirs.append(str(child))

    # Always offer auto-download as an option
    dirs.append(AUTO_DOWNLOAD_SENTINEL)

    return dirs


# Whisper speech transcription (lazy-loaded)
_whisper_model = None
_whisper_model_size_loaded: str | None = None
# Default to large-v3 (~3GB) for best transcription quality.  Smaller sizes
# (base, small, medium) hallucinate badly on music/SFX/chants which is exactly
# the failure mode that pollutes character voice-attribution downstream.
# The user can override via the prep node's `whisper_model` input.
_WHISPER_MODEL_SIZE = "large-v3"
_WORD_CUT_THRESHOLD = 0.3  # seconds — if last word ends within this of clip end, it's likely cut

# Demucs vocal isolation (lazy-loaded)
_demucs_model = None
_demucs_device = None

# Speechbrain ECAPA-TDNN speaker embedding (lazy-loaded).
# Used for voice-attribution: embed each Whisper segment and match to enrolled
# voice references via cosine distance.  No HF token required, no torchcodec.
_speechbrain_embedder = None
_VOICE_MATCH_THRESHOLD = 0.70  # cosine distance — lower = stricter
_VOICE_MATCH_MARGIN = 0.05     # require winner to beat runner-up by this much
_FACE_HINT_BONUS = 0.15        # subtract from distance when character is on-screen
                                # (per face detection) — biases ambiguous matches
                                # toward the visible character without overriding
                                # strong off-screen voice evidence.


def _get_whisper_model(size: str | None = None):
    """Load Whisper model on first use. Cached after first call.

    If size differs from the cached model's size, the old model is dropped
    and the requested size is loaded fresh.
    """
    global _whisper_model, _whisper_model_size_loaded
    target = size or _WHISPER_MODEL_SIZE
    if _whisper_model is not None and _whisper_model_size_loaded == target:
        return _whisper_model
    if _whisper_model is not None and _whisper_model_size_loaded != target:
        # Different size requested — drop the old model and reload.
        logger.info(f"Whisper model size changed: {_whisper_model_size_loaded} -> {target}, reloading")
        del _whisper_model
        _whisper_model = None
    try:
        import whisper
        logger.info(f"Loading Whisper model ({target})...")
        _whisper_model = whisper.load_model(target)
        _whisper_model_size_loaded = target
        logger.info(f"Whisper model loaded ({target})")
        return _whisper_model
    except ImportError:
        logger.error("openai-whisper not installed. Run: pip install openai-whisper")
        return None


def _isolate_vocals(clip_path: Path) -> Path | None:
    """Use Demucs to isolate vocals from a clip's audio.
    Returns path to a temporary WAV file with vocals only,
    or None if demucs isn't available or isolation fails."""
    global _demucs_model, _demucs_device
    try:
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError:
        return None

    try:
        # Load demucs model (cached after first call)
        if _demucs_model is None:
            logger.info("Loading Demucs model for vocal isolation...")
            _demucs_model = get_model("htdemucs")
            _demucs_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Demucs model loaded")

        # Extract audio from clip to WAV
        audio_tmp = clip_path.with_suffix(".tmp_audio.wav")
        ffmpeg_result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(clip_path),
             "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
             str(audio_tmp)],
            capture_output=True, text=True,
        )
        if ffmpeg_result.returncode != 0 or not audio_tmp.exists():
            return None

        # Load and separate
        waveform, sr = torchaudio.load(str(audio_tmp))
        # Demucs expects [batch, channels, samples]
        waveform = waveform.unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(_demucs_model, waveform, device=_demucs_device)

        # sources shape: [batch, n_sources, channels, samples]
        # htdemucs sources: drums, bass, other, vocals (index 3)
        vocals = sources[0, 3]  # [channels, samples]

        # Save vocals to temp file
        vocals_path = clip_path.with_suffix(".tmp_vocals.wav")
        torchaudio.save(str(vocals_path), vocals.cpu(), sr)

        # Clean up intermediate audio
        try:
            audio_tmp.unlink()
        except OSError:
            pass

        return vocals_path

    except Exception as e:
        logger.warning(f"Demucs vocal isolation failed for {clip_path.name}: {e}")
        # Clean up on failure
        for tmp in [clip_path.with_suffix(".tmp_audio.wav"),
                    clip_path.with_suffix(".tmp_vocals.wav")]:
            try:
                tmp.unlink()
            except OSError:
                pass
        return None


_ECAPA_TARGET_SR = 16000  # speechbrain ECAPA-TDNN expects 16 kHz mono


def _get_speechbrain_embedder():
    """Lazy-load speechbrain ECAPA-TDNN speaker-embedding model.
    No HF token required; weights are downloaded and cached on first use."""
    global _speechbrain_embedder
    if _speechbrain_embedder is not None:
        return _speechbrain_embedder
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        logger.error(
            "speechbrain not installed. Run 'pip install speechbrain' in "
            "ComfyUI's venv (or re-run install.bat). Voice attribution disabled."
        )
        return None
    try:
        logger.info("Loading speechbrain ECAPA-TDNN speaker embedder...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Cache weights under the custom-node folder so they're co-located with
        # the rest of the project (and easy to wipe on reinstall).
        savedir = Path(__file__).parent.parent / "models" / "spkrec-ecapa-voxceleb"
        savedir.mkdir(parents=True, exist_ok=True)
        # LocalStrategy.COPY avoids the default symlink behaviour, which fails
        # on Windows without Developer Mode (WinError 1314: A required
        # privilege is not held by the client).  Costs ~80MB extra disk for
        # the duplicate vs the HF cache; portable across all platforms.
        kwargs = dict(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts={"device": device},
        )
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kwargs["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass  # older speechbrain — fall through to default symlink
        embedder = EncoderClassifier.from_hparams(**kwargs)
        _speechbrain_embedder = embedder
        logger.info("Speechbrain ECAPA-TDNN loaded")
        return embedder
    except Exception as e:
        logger.error(f"Failed to load speechbrain embedder: {e}")
        return None


def _load_audio_for_embedding(audio_path: Path):
    """Load audio as a 1-channel 16 kHz tensor suitable for ECAPA-TDNN.
    Returns (waveform[1, samples], sample_rate=16000) or (None, 0) on failure."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception as e:
        logger.warning(f"Could not load audio {audio_path.name}: {e}")
        return None, 0
    if waveform.numel() == 0:
        return None, 0
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != _ECAPA_TARGET_SR:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, _ECAPA_TARGET_SR)
    return waveform, _ECAPA_TARGET_SR


def _embed_audio_segment(audio_path: Path, start: float, end: float) -> np.ndarray | None:
    """Embed audio [start, end] seconds with ECAPA-TDNN.
    Returns 192-d numpy array (L2-normalized) or None on failure."""
    embedder = _get_speechbrain_embedder()
    if embedder is None:
        return None
    waveform, sr = _load_audio_for_embedding(audio_path)
    if waveform is None:
        return None
    s = max(0, int(start * sr))
    e = min(waveform.shape[1], int(end * sr))
    if e - s < int(0.3 * sr):  # need at least 0.3s for a stable embedding
        return None
    slice_wave = waveform[:, s:e]
    try:
        emb = embedder.encode_batch(slice_wave).squeeze().detach().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as e:
        logger.warning(f"Embedding failed for {audio_path.name} [{start:.2f}-{end:.2f}]: {e}")
        return None


def _embed_audio_full(audio_path: Path) -> np.ndarray | None:
    """Embed an entire audio file (used for voice reference enrollment)."""
    embedder = _get_speechbrain_embedder()
    if embedder is None:
        return None
    waveform, _ = _load_audio_for_embedding(audio_path)
    if waveform is None:
        return None
    try:
        emb = embedder.encode_batch(waveform).squeeze().detach().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception as e:
        logger.warning(f"Voice reference embedding failed for {audio_path.name}: {e}")
        return None


def _format_speaker_display(trigger: str) -> str:
    """pee-wee → Pee-wee, miss-yvonne → Miss Yvonne, cowboy-curtis → Cowboy Curtis."""
    if not trigger or trigger == "unknown":
        return "Someone"
    parts = trigger.replace("_", "-").split("-")
    return " ".join(p[:1].upper() + p[1:] for p in parts if p)


def _transcribe_clip(clip_path: Path, voice_refs: dict | None = None,
                     face_chars: set | None = None) -> dict | None:
    """Run Demucs vocal isolation then Whisper on a clip.
    Returns dict with 'text', 'words' list of {word, start, end}, 'duration'.
    Returns None if transcription fails or no speech detected.

    When voice_refs is provided, also runs pyannote speaker diarization +
    embedding to attribute each segment to a known speaker.  In that case
    the returned dict additionally contains 'segments': list of {speaker,
    text, start, end} and 'text' is rewritten with speaker labels of the
    form `<Speaker Display Name>: "<line>"`, joined sequentially.  Speakers
    that don't match any enrolled voice above threshold are tagged as
    'unknown' (rendered as "Someone").
    """
    model = _get_whisper_model()
    if model is None:
        return None

    import whisper

    # Isolate vocals to remove music/sfx before transcribing.  We keep the
    # vocals file alive across both Whisper and pyannote so diarization and
    # speaker embedding see the same isolated audio Whisper transcribed.
    vocals_path = _isolate_vocals(clip_path)
    audio_source = str(vocals_path) if vocals_path else str(clip_path)

    whisper_segs: list[dict] = []
    text = ""
    try:
        try:
            result = whisper.transcribe(model, audio_source, word_timestamps=True)
        except Exception as e:
            logger.warning(f"Whisper transcription failed for {clip_path.name}: {e}")
            return None

        text = result.get("text", "").strip()
        if not text:
            return None

        # Detect Whisper hallucinations — non-Latin gibberish when there's no speech
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0 and latin_chars / total_alpha < 0.5:
            logger.info(f"  Whisper hallucination detected, no usable speech: {text[:60]}")
            return {"text": "", "words": [], "duration": 0.0, "hallucination": True}

        # Extract word-level timestamps
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"],
                })

        # Get clip duration via ffprobe
        duration = 0.0
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(clip_path)],
                capture_output=True, text=True,
            )
            duration = float(probe.stdout.strip())
        except (ValueError, OSError):
            pass

        whisper_segs = [
            {"start": float(s["start"]), "end": float(s["end"]),
             "text": s.get("text", "").strip()}
            for s in result.get("segments", [])
            if s.get("text", "").strip()
        ]

        # No voice references → return flat-text format (legacy callers).
        if not voice_refs:
            return {"text": text, "words": words, "duration": duration}

        # Voice attribution: embed each Whisper segment with ECAPA-TDNN and
        # match against enrolled voice prints.  When face_chars is provided
        # (the visible characters from face detection), it biases ambiguous
        # matches toward on-screen speakers.
        embedded_audio = Path(audio_source)
        segments = _attribute_whisper_segments(
            embedded_audio, whisper_segs, voice_refs, face_chars=face_chars,
        )
        if segments is None:
            # Diarization unavailable — fall back to flat text but flag it.
            logger.info(
                f"  Voice attribution unavailable for {clip_path.name} — "
                f"emitting flat transcript without speaker labels"
            )
            return {"text": text, "words": words, "duration": duration}

        # Rebuild flat text with `<Speaker>: "<line>"` labels so any consumer
        # reading the legacy 'text' field still gets the speaker context.
        flat_parts = []
        for s in segments:
            sp_disp = _format_speaker_display(s["speaker"])
            line = s["text"].strip().rstrip(".!?")
            flat_parts.append(f'{sp_disp}: "{line}".')
        flat_text = " ".join(flat_parts) if flat_parts else text

        return {
            "text": flat_text,
            "words": words,
            "duration": duration,
            "segments": segments,
        }
    finally:
        # Clean up vocals temp file (after both Whisper AND any diarization)
        if vocals_path:
            try:
                vocals_path.unlink()
            except OSError:
                pass


def _attribute_whisper_segments(
    audio_path: Path, whisper_segs: list[dict], voice_refs: dict,
    face_chars: set | None = None,
) -> list[dict] | None:
    """For each Whisper segment, embed its audio slice and match against
    enrolled voice_refs via cosine distance. Returns one entry per Whisper
    segment tagged with the matched speaker name (or 'unknown').

    Whisper's segments are already speech-bounded, so we don't need a
    separate diarization pass — each segment gets independently attributed.
    Trade-off: segments that contain TWO speakers (overlap) get a single
    label.  In practice, Whisper's segmenter cuts on speaker change boundaries
    fairly often, and our use case has mostly single-speaker clips anyway.

    Returns None if the embedder isn't available — caller falls back to
    flat-text transcripts.
    """
    embedder = _get_speechbrain_embedder()
    if embedder is None:
        return None

    waveform, sr = _load_audio_for_embedding(audio_path)
    if waveform is None:
        return None

    segments_out: list[dict] = []
    for ws in whisper_segs:
        s_frame = max(0, int(ws["start"] * sr))
        e_frame = min(waveform.shape[1], int(ws["end"] * sr))
        slice_dur = (e_frame - s_frame) / sr if sr else 0.0
        speaker = "unknown"
        match_diag = ""
        if slice_dur >= 0.3:  # need ~0.3s for a stable embedding
            try:
                slice_wave = waveform[:, s_frame:e_frame]
                emb = embedder.encode_batch(slice_wave).squeeze().detach().cpu().numpy()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                    # Compute raw + face-hint-adjusted distance per candidate.
                    # Face hint subtracts a small bonus when the character is
                    # on-screen for this clip, biasing ambiguous matches toward
                    # visible speakers without overriding strong off-screen
                    # voice evidence.
                    scored = []
                    for name, ref_emb in voice_refs.items():
                        raw = 1.0 - float(np.dot(emb, ref_emb))
                        adj = raw - _FACE_HINT_BONUS if (face_chars and name in face_chars) else raw
                        scored.append((name, raw, adj))
                    # Rank by adjusted distance (lower = better match)
                    scored.sort(key=lambda p: p[2])
                    best_name, best_raw, best_adj = scored[0]
                    runner_adj = scored[1][2] if len(scored) > 1 else float("inf")
                    margin = runner_adj - best_adj
                    if best_adj <= _VOICE_MATCH_THRESHOLD and margin >= _VOICE_MATCH_MARGIN:
                        speaker = best_name
                    # Log raw and adjusted (face-hinted) distances so threshold
                    # tuning is informed by real data.
                    summary = ", ".join(
                        f"{n}={r:.2f}" + (f"→{a:.2f}*" if a != r else "")
                        for n, r, a in scored
                    )
                    match_diag = (
                        f" [match: best={best_name}@{best_adj:.2f} "
                        f"margin={margin:.2f} → {speaker}; all: {summary}]"
                    )
            except Exception as e:
                logger.warning(f"Speaker match failed for segment "
                               f"[{ws['start']:.2f}-{ws['end']:.2f}]: {e}")
        if match_diag:
            logger.info(f"  seg [{ws['start']:.1f}-{ws['end']:.1f}]{match_diag}")
        segments_out.append({
            "speaker": speaker,
            "text": ws["text"],
            "start": ws["start"],
            "end": ws["end"],
        })
    return segments_out


# InsightFace (SCRFD detection + ArcFace recognition, antelopev2 model pack).
# One FaceAnalysis call returns bboxes + 512-d L2-normalized embeddings together;
# detection helpers cache the last analyzed frame so the embedding lookup is free.
_face_app = None
_face_app_checked = False
_FACE_PADDING = 0.6  # 60% padding around detected face for head+shoulders context
_FACE_MATCH_THRESHOLD = 0.40  # Cosine similarity threshold for face matching (lower = stricter)
# Two-level confidence thresholds:
#   * _FACE_DET_MIN_CONFIDENCE: minimum score for a detection to be kept
#     at all. We match these against character refs — a known character
#     face might score as low as 0.5 at an odd angle, so this has to be
#     permissive.
#   * _FACE_UNKNOWN_MIN_CONFIDENCE: minimum score for an UNMATCHED face
#     to count as an unknown human extra. Tiki masks, cactus faces,
#     posters, painted props typically score in the 0.5-0.8 range. Real
#     human faces almost always score 0.85+. Being strict here avoids
#     false-positive "unknown faces" from a prop-heavy set like Pee-wee's
#     Playhouse where real extras are rare.
_FACE_DET_MIN_CONFIDENCE = 0.5
_FACE_UNKNOWN_MIN_CONFIDENCE = 0.85

_last_analysis_frame_id: int | None = None
_last_analysis_faces: list = []


def _get_face_app():
    """Load InsightFace FaceAnalysis (antelopev2). Lazy, cached, one-shot."""
    global _face_app, _face_app_checked
    if _face_app is not None:
        return _face_app
    if _face_app_checked:
        return None
    _face_app_checked = True

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        logger.warning(
            "insightface not installed — face detection disabled. "
            "Run: pip install insightface onnxruntime-gpu"
        )
        return None

    try:
        app = FaceAnalysis(
            name="antelopev2",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
        logger.info("Loaded InsightFace (antelopev2) — SCRFD detect + ArcFace embed")
        return app
    except Exception as e:
        logger.warning(f"Failed to load InsightFace: {e}")
        return None


def _analyze_frame(frame: np.ndarray) -> list:
    """Run InsightFace on frame; memoize by id(frame) so repeat calls are
    free. Drops low-confidence detections (cactus/tiki/poster/cartoon
    patterns) so they don't pollute hit counts or unknown-face counts."""
    global _last_analysis_frame_id, _last_analysis_faces
    app = _get_face_app()
    if app is None:
        _last_analysis_frame_id = id(frame)
        _last_analysis_faces = []
        return []
    if id(frame) == _last_analysis_frame_id:
        return _last_analysis_faces
    raw = app.get(frame)
    # Filter by detector confidence — non-human face-like patterns
    # typically score below _FACE_DET_MIN_CONFIDENCE.
    faces = [
        f for f in raw
        if float(getattr(f, "det_score", 1.0)) >= _FACE_DET_MIN_CONFIDENCE
    ]
    _last_analysis_frame_id = id(frame)
    _last_analysis_faces = faces
    return faces


def _detect_face_dnn(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest face bbox (x, y, w, h) or None."""
    faces = _detect_all_faces_dnn(frame)
    if not faces:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def _detect_all_faces_dnn(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """All face bboxes as (x, y, w, h)."""
    out: list[tuple[int, int, int, int]] = []
    for f in _analyze_frame(frame):
        x1, y1, x2, y2 = f.bbox.astype(int).tolist()
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        out.append((x1, y1, w, h))
    return out


def _get_face_embedding(frame: np.ndarray, face_rect: tuple[int, int, int, int]) -> np.ndarray | None:
    """Return the 512-d L2-normalized ArcFace embedding for the face at face_rect.
    Matches face_rect against the cached analysis by IoU."""
    faces = _analyze_frame(frame)
    if not faces:
        return None
    fx, fy, fw, fh = face_rect
    fa = fw * fh
    best = None
    best_iou = 0.0
    for f in faces:
        x1, y1, x2, y2 = f.bbox.tolist()
        ix1 = max(fx, x1)
        iy1 = max(fy, y1)
        ix2 = min(fx + fw, x2)
        iy2 = min(fy + fh, y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = fa + (x2 - x1) * (y2 - y1) - inter
        if union <= 0:
            continue
        iou = inter / union
        if iou > best_iou:
            best_iou = iou
            best = f
    if best is None or best_iou < 0.3:
        return None
    return np.asarray(best.normed_embedding, dtype=np.float32)


def _match_face(embedding: np.ndarray, target_embedding: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings. Range [-1, 1]; ~0.5+ = same person for ArcFace."""
    return float(np.dot(embedding, target_embedding))


def _has_unknown_face(
    frame: np.ndarray,
    character_refs: dict[str, dict],
    threshold: float,
) -> bool:
    """True if the frame contains at least one face that (a) scores at or
    above _FACE_UNKNOWN_MIN_CONFIDENCE — meaning it's very likely a real
    human face, not a decorative face-like prop — AND (b) does not match
    any known face-ref at or above `threshold`. Returns False when there
    are no face-refs to adjudicate against or when no qualifying faces
    are detected."""
    face_refs = [r["embedding"] for r in character_refs.values() if r["type"] == "face"]
    if not face_refs:
        return False
    for f in _analyze_frame(frame):
        score = float(getattr(f, "det_score", 1.0))
        if score < _FACE_UNKNOWN_MIN_CONFIDENCE:
            # Low-confidence face-like pattern (prop/mask/poster/cartoon).
            # Don't flag as an unknown human extra.
            continue
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            continue
        emb = np.asarray(emb, dtype=np.float32)
        best_sim = max(_match_face(emb, ref) for ref in face_refs)
        if best_sim < threshold:
            return True
    return False


def _compute_face_crop(
    face_x: int, face_y: int, face_w: int, face_h: int,
    frame_w: int, frame_h: int,
    target_w: int, target_h: int,
) -> tuple[int, int, int, int]:
    """Compute a crop region centered on the face with the target aspect ratio.
    Adds padding around the face for head+shoulders context.
    Returns (crop_x, crop_y, crop_w, crop_h).
    """
    target_aspect = target_w / target_h

    # Face center
    cx = face_x + face_w // 2
    cy = face_y + face_h // 2

    # Start with padded face region
    padded_size = max(face_w, face_h) * (1 + _FACE_PADDING)

    # Compute crop dimensions matching target aspect ratio
    if target_aspect >= 1.0:
        crop_w = padded_size * target_aspect
        crop_h = padded_size
    else:
        crop_w = padded_size
        crop_h = padded_size / target_aspect

    # Ensure crop doesn't exceed frame
    crop_w = min(crop_w, frame_w)
    crop_h = min(crop_h, frame_h)

    # Re-enforce aspect ratio after clamping
    if crop_w / crop_h > target_aspect:
        crop_w = crop_h * target_aspect
    else:
        crop_h = crop_w / target_aspect

    crop_w = int(crop_w)
    crop_h = int(crop_h)

    # Center crop on face, clamped to frame bounds
    crop_x = max(0, min(cx - crop_w // 2, frame_w - crop_w))
    crop_y = max(0, min(cy - crop_h // 2, frame_h - crop_h))

    return crop_x, crop_y, crop_w, crop_h


def _compute_pan_and_scan(
    face_x: int, face_y: int, face_w: int, face_h: int,
    frame_w: int, frame_h: int,
    target_w: int, target_h: int,
) -> tuple[int, int, int, int]:
    """Compute the largest crop at target aspect ratio that includes the face.
    The face stays at its natural position — the crop just slides to contain it.
    Returns (crop_x, crop_y, crop_w, crop_h).
    """
    target_aspect = target_w / target_h

    # Max crop size at target aspect ratio that fits in the frame
    if frame_w / frame_h > target_aspect:
        crop_h = frame_h
        crop_w = int(frame_h * target_aspect)
    else:
        crop_w = frame_w
        crop_h = int(frame_w / target_aspect)

    # Face center
    cx = face_x + face_w // 2
    cy = face_y + face_h // 2

    # Default: center the crop in the frame
    crop_x = (frame_w - crop_w) // 2
    crop_y = (frame_h - crop_h) // 2

    # Slide horizontally if face is outside crop
    if cx < crop_x + face_w // 2:
        crop_x = max(0, cx - face_w // 2)
    elif cx > crop_x + crop_w - face_w // 2:
        crop_x = min(frame_w - crop_w, cx + face_w // 2 - crop_w)

    # Slide vertically if face is outside crop
    if cy < crop_y + face_h // 2:
        crop_y = max(0, cy - face_h // 2)
    elif cy > crop_y + crop_h - face_h // 2:
        crop_y = min(frame_h - crop_h, cy + face_h // 2 - crop_h)

    # Clamp
    crop_x = max(0, min(crop_x, frame_w - crop_w))
    crop_y = max(0, min(crop_y, frame_h - crop_h))

    return crop_x, crop_y, crop_w, crop_h


class RSLTXVPrepareDataset:
    """Scan a folder of videos/images, detect faces, crop around them,
    generate dataset JSON, and run process_dataset.py to precompute latents."""

    # A character only counts as present in a clip if they appear in at
    # least this fraction of the MOST-SEEN character's positions. Prevents
    # brief cameos (1-2 frames at the edge of another character's shot)
    # from inflating the clip's `characters` list and confusing quota
    # accounting. Tuned relative to the dominant character so a short
    # clip of pee-wee alone (e.g. 2/8 positions with nothing else
    # detected) still labels him correctly.
    _CHAR_DOMINANCE_RATIO = 0.5

    @classmethod
    def _filter_dominant_chars(cls, char_position_counts: dict) -> set:
        """Keep only characters whose position count is at least
        _CHAR_DOMINANCE_RATIO * max(counts). When only one character is
        detected, that character always passes. Empty input -> empty set."""
        if not char_position_counts:
            return set()
        max_count = max(char_position_counts.values())
        threshold = max_count * cls._CHAR_DOMINANCE_RATIO
        return {c for c, n in char_position_counts.items() if n >= threshold}

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        te_dirs = _list_text_encoder_dirs()
        return {
            "required": {
                "media_folder": ("STRING", {"default": "", "tooltip": "Path to folder containing videos and/or images"}),
                "model_path": (checkpoints, {"tooltip": "LTX-2 model checkpoint"}),
                "text_encoder_path": (te_dirs, {"tooltip": "Full Gemma-3 HF directory. Select 'auto' to download on first run."}),
                "output_name": ("STRING", {"default": "my_dataset", "tooltip": "Name for the preprocessed output folder"}),
            },
            "optional": {
                "resolution_buckets": ("STRING", {"default": "576x576x49", "tooltip": "WxHxF resolution buckets, semicolon-separated (e.g. '576x576x49;768x512x25')"}),
                "lora_trigger": ("STRING", {"default": "", "tooltip": "Trigger word prepended to all captions"}),
                "caption_mode": (["ollama", "skip", "auto_filename"], {"default": "ollama", "tooltip": "ollama=vision model captions, skip=use filenames, auto_filename=clean filenames"}),
                "caption_style": (["subject", "subject + style", "style", "motion", "general", "multi_character"], {"default": "subject", "tooltip": "Captioning focus: subject=person identity, subject+style=person+cinematography, style=visual aesthetics only, motion=movement/action, general=balanced, multi_character=name every recognized character equally (pairs with character_refs_folder)"}),
                "ollama_url": ("STRING", {"default": "http://localhost:11434", "tooltip": "Ollama server URL (only used when caption_mode=ollama)"}),
                "ollama_model": ("STRING", {"default": "gemma4:26b", "tooltip": "Ollama vision model for captioning (only used when caption_mode=ollama)"}),
                "skip_id_pass": ("BOOLEAN", {"default": False, "tooltip": "Skip cast/location ID passes — send ALL reference images directly to the captioner and let it identify characters and locations itself in one shot."}),
                "with_audio": ("BOOLEAN", {"default": False, "tooltip": "Extract and encode audio latents"}),
                "transcribe_speech": ("BOOLEAN", {"default": False, "tooltip": "Transcribe speech in clips using Whisper and append to captions. Also adjusts clip boundaries to avoid cutting words."}),
                "whisper_model": (["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"], {"default": "large-v3", "tooltip": "Whisper model size for transcription. Larger = far better accuracy, especially on music/SFX/unusual vocabulary, at the cost of more VRAM (large-v3 ≈ 3GB) and slower per-clip processing. Drop to 'base' or 'small' only for quick tests."}),
                "load_text_encoder_in_8bit": ("BOOLEAN", {"default": True, "tooltip": "Load text encoder in 8-bit to save VRAM"}),
                "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.001, "tooltip": "Target framerate for output clips (0=keep source fps). Drops frames to match target without affecting audio speed."}),
                "max_samples": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1, "tooltip": "Maximum total character appearances across the dataset (0=unlimited). With N characters defined, each character gets a quota of max_samples/N appearances. Total clip count can be less than max_samples when characters share clips — the goal is even per-character distribution, not a specific clip count."}),
                "crop_mode": (["face_crop", "pan_and_scan", "full_frame"], {"default": "face_crop", "tooltip": "face_crop=tight crop around face, pan_and_scan=max crop at target aspect ratio slid to include face (preserves natural framing), full_frame=scale entire frame. All modes use face detection for clip selection when enabled."}),
                "face_detection": ("BOOLEAN", {"default": True, "tooltip": "Enable face detection: crop around faces, discard clips with no faces"}),
                "target_face": ("IMAGE", {"tooltip": "Reference face image. When connected, only keeps clips matching this specific face."}),
                "character_refs_folder": ("STRING", {"default": "", "tooltip": "Multi-character mode: path to a folder of reference face images. Filename (minus extension) is used as that character's trigger word. Clips are kept if ANY reference matches. All matched characters are named in captions."}),
                "voice_refs_folder": ("STRING", {"default": "", "tooltip": "Optional: path to a folder of voice reference clips (one per character). Filename stem must match the corresponding character_refs_folder entry. When provided, each line of dialogue is attributed to a known speaker via speechbrain ECAPA-TDNN voice matching; the captioner then weaves the dialogue into the caption with speaker attribution. Recommended: 30+ seconds of clean isolated speech per character."}),
                "location_refs_folder": ("STRING", {"default": "", "tooltip": "Optional: path to a folder of reference images for distinct locations/sets (e.g. 'Main Room', 'Kitchen'). Filename (minus extension) is the location name. Gemma picks the best match per clip and uses it in the caption. Soft — clips with no location match are still captioned normally."}),
                "skip_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 1.0, "tooltip": "Skip the first N seconds of every video (useful for cutting out repetitive intros)."}),
                "skip_end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 1.0, "tooltip": "Skip the last N seconds of every video (useful for cutting out end credits)."}),
                "face_similarity": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Face match threshold (0-1). Higher = stricter matching. 0.40 is a good default."}),
                "face_padding": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Padding around detected face (0.6 = 60% extra for head+shoulders)"}),
                "sample_count": ("INT", {"default": 4, "min": 2, "max": 16, "step": 1, "tooltip": "Number of evenly-spaced sample positions per chunk for character detection. Higher = less likely to miss a character who appears briefly between samples (e.g. at 10% and 40%), but slower. Positions are laid out as 0%, 1/N, 2/N, ..., (N-1)/N."}),
                "char_positions_required": (["10%", "25%", "50%", "75%", "100%"], {"default": "75%", "tooltip": "Fraction of sample positions that must contain ANY named character. Scales with sample_count (e.g. 75% of 4 = 3, 75% of 8 = 6). Any mix of named characters across positions counts — they don't need to be the same character."}),
                "allow_unknown_faces_in": (["0%", "10%", "25%", "50%", "75%", "100%"], {"default": "25%", "tooltip": "Fraction of sample positions allowed to contain unknown (non-reference) faces. Scales with sample_count (e.g. 25% of 4 = 1 position, 25% of 8 = 2). 0% = reject any extras at all, 100% = allow extras anywhere. 25% (default) tolerates one detection blip in a 4-sample chunk."}),
                "clip_check": ("BOOLEAN", {"default": False, "tooltip": "When ON, re-scan every existing clip in the dataset and rewrite its `characters` field based on current character refs. Useful when you've added a new reference character after the dataset was already partly built — existing clips that contain the new character get their characters list updated. Runs once at the start of the run; leave OFF for subsequent runs."}),
                "conditioning_folder": ("STRING", {"default": "", "tooltip": "IC-LoRA: folder of conditioning input videos (depth maps, edge maps, poses). Matched to media_folder by filename. Media folder is the ground truth output."}),
                "clip": ("CLIP", {"tooltip": "Text encoder (from CheckpointLoaderSimple). When connected, encodes captions in-process instead of slow subprocess."}),
                "vae": ("VAE", {"tooltip": "VAE (from CheckpointLoaderSimple). When connected, encodes latents in-process instead of slow subprocess."}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "Optional CLIP Vision model (from CLIPVisionLoader). Enables matching of non-human characters (puppets, props, objects) in character_refs_folder. Human characters use face matching regardless."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("preprocessed_path", "dataset_json_path")
    FUNCTION = "prepare"
    CATEGORY = "RS Nodes"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, text_encoder_path):
        # The dropdown only enumerates HF model dirs currently on disk, but
        # when `clip` is plugged in the widget value is unused (CLIP supplies
        # the encoder in-process). Accept any string so workflows that saved
        # an old path -- or were built when CLIP is connected and the widget
        # is irrelevant -- still validate. When CLIP is NOT connected, the
        # runtime resolves the path via validate_text_encoder_path which has
        # its own sanity checks.
        return True

    @staticmethod
    def _normalize_loaded_entries(entries, output_dir):
        """Resolve every entry's media_path to an absolute path in-memory so
        the rest of the prepare flow operates on absolute paths as before.

        Three cases:
        - Relative path (post-fix layout): resolve against output_dir
        - Absolute and exists: use as-is
        - Absolute and missing: try basename under output_dir/clips/
          (legacy migration -- handles moved-folder scenario where the saved
          absolute path points at the OLD location but the clip is in the
          new clips/ next to dataset.json)

        Mutates entries in-place. Idempotent.
        """
        clips_dir = output_dir / "clips"
        for e in entries:
            mp = e.get("media_path", "")
            if not mp:
                continue
            p = Path(mp)
            if p.is_absolute():
                if p.exists():
                    continue
                candidate = clips_dir / p.name
                if candidate.exists():
                    e["media_path"] = str(candidate)
            else:
                e["media_path"] = str((output_dir / p).resolve())

    @staticmethod
    def _entries_for_write(entries, output_dir):
        """Return a copy of entries with each media_path stored relative to
        output_dir when the clip lives under it (app-generated artifact).
        Paths outside output_dir stay absolute (user-supplied source data).

        Use forward slashes in the relative form so paths are portable
        between Windows and POSIX. Does NOT mutate the input list -- in
        memory, callers continue to see absolute paths.
        """
        output_dir_resolved = output_dir.resolve()
        out = []
        for e in entries:
            e2 = dict(e)
            mp = e2.get("media_path", "")
            if mp:
                p = Path(mp)
                if p.is_absolute():
                    try:
                        rel = p.resolve().relative_to(output_dir_resolved)
                        e2["media_path"] = rel.as_posix()
                    except ValueError:
                        pass  # outside output_dir -- keep absolute
            out.append(e2)
        return out

    def prepare(
        self,
        media_folder: str,
        model_path: str,
        text_encoder_path: str,
        output_name: str,
        resolution_buckets: str = "576x576x49",
        lora_trigger: str = "",
        caption_mode: str = "ollama",
        caption_style: str = "subject",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "gemma4:26b",
        skip_id_pass: bool = False,
        with_audio: bool = False,
        load_text_encoder_in_8bit: bool = True,
        crop_mode: str = "face_crop",
        face_detection: bool = True,
        transcribe_speech: bool = False,
        whisper_model: str = "large-v3",
        target_face=None,
        character_refs_folder: str = "",
        voice_refs_folder: str = "",
        location_refs_folder: str = "",
        skip_start_seconds: float = 0.0,
        skip_end_seconds: float = 0.0,
        face_similarity: float = 0.40,
        face_padding: float = 0.6,
        sample_count: int = 4,
        char_positions_required: str = "75%",
        allow_unknown_faces_in: str = "25%",
        clip_check: bool = False,
        conditioning_folder: str = "",
        clip=None,
        vae=None,
        clip_vision=None,
        target_fps: float = 0.0,
        max_samples: int = 0,
        unique_id=None,
    ):
        validate_submodule()
        # Set the global whisper model size from the input — _get_whisper_model
        # uses this when loading.  Changing this between runs forces a reload.
        global _WHISPER_MODEL_SIZE
        _WHISPER_MODEL_SIZE = whisper_model
        # Only validate/download text encoder if we'll need it for the subprocess
        # (when CLIP is provided, text encoding is done in-process and doesn't need Gemma on disk)
        if clip is None:
            text_encoder_path = validate_text_encoder_path(text_encoder_path)

        media_folder = Path(media_folder)
        if not media_folder.exists():
            raise ValueError(f"Media folder does not exist: {media_folder}")

        model_full_path = folder_paths.get_full_path("checkpoints", model_path)

        # Parse target resolution from first bucket
        bucket = resolution_buckets.split(";")[0].strip()
        bucket_parts = bucket.split("x")
        target_w, target_h = int(bucket_parts[0]), int(bucket_parts[1])
        target_frames = int(bucket_parts[2]) if len(bucket_parts) > 2 else 49

        # Multi-character mode: load a folder of reference images.  Each file's
        # stem becomes that character's trigger word.  Face-containing refs are
        # matched via face embeddings; non-face refs (puppets, props, objects)
        # fall back to CLIP vision embedding if clip_vision is connected.
        # When populated, clip selection accepts chunks where any reference
        # matches in at least one sample frame.
        character_refs: dict[str, dict] = {}
        if character_refs_folder and face_detection:
            character_refs = self._load_character_refs(character_refs_folder, clip_vision=clip_vision)
            if character_refs:
                n_face = sum(1 for r in character_refs.values() if r["type"] == "face")
                n_clip = sum(1 for r in character_refs.values() if r["type"] == "clip")
                logger.info(
                    f"Multi-character mode: loaded {len(character_refs)} references "
                    f"({n_face} face, {n_clip} clip-vision) "
                    f"— {', '.join(sorted(character_refs.keys()))}"
                )

        # Voice references: optional folder of audio clips per character (one
        # clip per character, filename stem = trigger). When populated, the
        # transcriber runs pyannote diarization + speaker embedding to attribute
        # each line of dialogue to a known speaker.
        voice_refs: dict[str, np.ndarray] = {}
        if voice_refs_folder and transcribe_speech:
            voice_refs = self._load_voice_refs(voice_refs_folder)
            if voice_refs:
                logger.info(
                    f"Voice attribution: loaded {len(voice_refs)} voice reference(s) "
                    f"— {', '.join(sorted(voice_refs.keys()))}"
                )

        # target_face pin: wired through the same character_refs pipeline as the
        # folder path, using lora_trigger as the trigger name (equivalent to
        # dropping one image in a folder named <lora_trigger>.jpg).
        target_embedding = None
        if target_face is not None and face_detection:
            target_embedding = self._compute_target_embedding(target_face)
            if target_embedding is None:
                logger.warning("Could not extract face from target_face image, face matching disabled")
            else:
                trigger = (lora_trigger or "").strip().lower().replace("_", "-")
                if not trigger:
                    trigger = "subject"
                    logger.warning("target_face connected without lora_trigger — defaulting trigger to 'subject'")
                # Encode target_face tensor as base64 JPEG so Gemma can see the
                # reference at caption time (matches _load_character_refs format).
                import base64 as _b64
                frame = target_face[0].cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok:
                    image_b64 = _b64.b64encode(buf.tobytes()).decode("utf-8")
                    character_refs[trigger] = {
                        "type": "face",
                        "embedding": target_embedding,
                        "image_b64": image_b64,
                    }
                    logger.info(f"target_face pin registered as character reference: {trigger} (similarity threshold: {face_similarity})")
                    # Route exclusively through character_refs — avoid double-filtering
                    # via the legacy target_embedding codepath.
                    target_embedding = None
                else:
                    logger.warning("Could not JPEG-encode target_face — falling back to legacy target_embedding filter (no Gemma cast entry)")

        # Location references: labeled images of distinct sets/locations.
        # Gemma is shown these alongside the clip frames during captioning
        # and picks the best match (if any).  Much simpler than character
        # refs — just name + image, no embeddings.
        location_refs: dict[str, str] = {}
        if location_refs_folder:
            location_refs = self._load_location_refs(location_refs_folder)
            if location_refs:
                logger.info(
                    f"Location mode: loaded {len(location_refs)} location refs "
                    f"— {', '.join(sorted(location_refs.keys()))}"
                )

        output_dir = Path(folder_paths.get_output_directory()) / "ltxv_training" / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_json_path = output_dir / "dataset.json"
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        latents_dir = output_dir / "latents"
        conditions_dir = output_dir / "conditions"
        audio_latents_dir = output_dir / "audio_latents"

        global _FACE_PADDING
        _FACE_PADDING = face_padding

        # ----------------------------------------------------------------
        # Early-exit: if all preprocessing artifacts are already on disk
        # AND media_folder has no NEW source videos to ingest, treat as
        # complete and return immediately. dataset.json is only the
        # manifest used to BUILD these artifacts; once they exist, it's
        # not consulted again. This skips the cleanup / captioning / write
        # paths entirely so a wiped or stale dataset.json can never cause
        # damage on resume.
        # ----------------------------------------------------------------
        if clips_dir.exists() and latents_dir.exists():
            clip_files = [p for p in clips_dir.iterdir() if p.is_file()]
            latent_files = [p for p in latents_dir.iterdir() if p.is_file()]
            artifacts_ok = bool(clip_files) and bool(latent_files)
            if artifacts_ok and with_audio:
                artifacts_ok = (
                    audio_latents_dir.exists()
                    and any(p.is_file() for p in audio_latents_dir.iterdir())
                )
            if artifacts_ok and conditioning_folder:
                artifacts_ok = (
                    conditions_dir.exists()
                    and any(p.is_file() for p in conditions_dir.iterdir())
                )
            if artifacts_ok:
                # Derive processed source stems from clip filenames
                # (clips look like "video_chunk0001.mp4" etc.)
                processed_stems = set()
                for cp in clip_files:
                    stem = cp.stem
                    if "_chunk" in stem:
                        processed_stems.add(stem.rsplit("_chunk", 1)[0])
                    else:
                        processed_stems.add(stem)
                # Are all source videos in media_folder represented?
                no_new_media = True
                for item in self._scan_media(media_folder):
                    if Path(item["path"]).stem not in processed_stems:
                        no_new_media = False
                        break
                if no_new_media:
                    logger.info(
                        f"Preprocessing already complete ({len(clip_files)} clips, "
                        f"{len(latent_files)} latents) -- skipping prepare, "
                        f"dataset.json not touched"
                    )
                    return (str(output_dir), str(dataset_json_path))

        # Scan current media folder
        media_files = self._scan_media(media_folder)
        if not media_files:
            raise ValueError(f"No video or image files found in: {media_folder}")

        current_sources = {str(item["path"]) for item in media_files}

        # --- PHASE 1: Generate clips ---
        # Load existing JSON to see what's already processed
        existing_entries = []
        if dataset_json_path.exists():
            try:
                with open(dataset_json_path) as f:
                    existing_entries = json.load(f)
            except (json.JSONDecodeError, KeyError):
                existing_entries = []

        # Resolve any relative media_path to absolute (and migrate legacy
        # absolute paths whose target moved -- try basename in clips/) so
        # the rest of this method operates on absolute paths exactly as
        # before. Writes go through _entries_for_write which converts back
        # to relative on the way out.
        self._normalize_loaded_entries(existing_entries, output_dir)

        # Track which source files are already in the JSON
        processed_sources = {e.get("source_file", "") for e in existing_entries}

        # Mutable container so the closure and the pool-building code can
        # share state without nonlocal boilerplate.
        pool_state = {"remaining": 0, "total": 0}

        def _emit_status() -> None:
            """Compute per-character counts from existing_entries and push to
            the node UI so the user sees progress live."""
            counts: dict[str, int] = {}
            for e in existing_entries:
                for c in e.get("characters", []):
                    counts[c] = counts.get(c, 0) + 1
            _emit_prepper_status(
                unique_id, counts, len(existing_entries), max_samples,
                pool_remaining=pool_state["remaining"],
                pool_total=pool_state["total"],
            )

        # Show the starting state immediately so the widget isn't blank while
        # the long setup runs.
        _emit_status()

        # Load rejected clips so we don't re-process them. Two kinds of
        # rejection coexist in rejected.json:
        #   * Chunk-level: the entry has media_path AND source_file. We
        #     want to skip THIS chunk but still allow the rest of that
        #     video to be drawn from — so only the clip path / filename
        #     goes into rejection sets, NEVER the source_file.
        #   * Video-level (classic mode, "all chunks rejected"): entry has
        #     only source_file. The whole video is skipped.
        rejected_path = output_dir / "rejected.json"
        rejected_clips = set()       # paths (videos or clips) to blacklist whole
        rejected_chunk_files = set()  # specific clip filenames to exclude from chunk pool
        if rejected_path.exists():
            try:
                with open(rejected_path) as rf:
                    for r in json.load(rf):
                        mp = r.get("media_path", "")
                        sf = r.get("source_file", "")
                        if mp:
                            # Chunk-level rejection — don't blacklist the
                            # parent video, just the chunk.
                            rejected_clips.add(mp)
                            rejected_chunk_files.add(Path(mp).name)
                        elif sf:
                            # Video-level rejection — blacklist the video.
                            rejected_clips.add(sf)
            except (json.JSONDecodeError, KeyError):
                pass

        # Find new media files not yet processed (and not rejected)
        new_media = [
            item for item in media_files
            if str(item["path"]) not in processed_sources
            and str(item["path"]) not in rejected_clips
        ]

        # Verify existing clips still exist on disk, remove entries for missing clips.
        # For rolling dataset mode, record deleted clips in rejected.json with
        # chunk-level info so we can audit what was purged.
        valid_entries = []
        removed_missing = 0
        deleted_rejected: list[dict] = []
        for entry in existing_entries:
            clip_path = Path(entry["media_path"])
            if clip_path.exists():
                valid_entries.append(entry)
            else:
                logger.info(f"Clip missing from disk, removing: {clip_path.name}")
                removed_missing += 1
                # Build a chunk-level rejection record so the raw-pool loop
                # knows this slot freed up.  We derive frame range from the
                # clip filename when possible; fall back to target_frames math
                # if the filename doesn't carry chunk info.
                rej_entry: dict = {
                    "source_file": entry.get("source_file", ""),
                    "media_path": str(clip_path),
                    "reason": "deleted",
                }
                # Try to extract chunk index from stem (e.g. "video_chunk0042")
                stem = clip_path.stem
                if "_chunk" in stem:
                    try:
                        chunk_num = int(stem.rsplit("_chunk", 1)[1])
                        rej_entry["start_frame"] = chunk_num * target_frames
                        rej_entry["end_frame"] = rej_entry["start_frame"] + target_frames
                    except (ValueError, IndexError):
                        pass
                deleted_rejected.append(rej_entry)

        existing_entries = valid_entries
        if removed_missing > 0:
            with open(dataset_json_path, "w") as f:
                json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)
            logger.info(f"Removed {removed_missing} missing-clip entries from dataset.json")
            # Append chunk-level deletion records to rejected.json
            if deleted_rejected:
                existing_rejected: list[dict] = []
                if rejected_path.exists():
                    try:
                        with open(rejected_path) as rf:
                            existing_rejected = json.load(rf)
                    except (json.JSONDecodeError, KeyError):
                        pass
                existing_rejected.extend(deleted_rejected)
                with open(rejected_path, "w") as rf:
                    json.dump(existing_rejected, rf, indent=2)

        # Reconcile: add dataset entries for any clips on disk missing from JSON.
        # Clips on disk are the source of truth.
        known_media = {e["media_path"] for e in existing_entries}
        orphans_added = 0
        orphan_clips = []
        for clip_file in sorted(clips_dir.iterdir()):
            if clip_file.suffix != ".mp4" or str(clip_file) in known_media:
                continue
            # Derive source file from clip name (strip _chunkNNNN suffix)
            stem = clip_file.stem
            source_file = ""
            if "_chunk" in stem:
                base = stem.rsplit("_chunk", 1)[0]
                for item in media_files:
                    if item["path"].stem == base:
                        source_file = str(item["path"])
                        break
            entry = {"caption": "", "media_path": str(clip_file), "source_file": source_file}
            existing_entries.append(entry)
            orphan_clips.append((entry, clip_file))
            orphans_added += 1
        if orphans_added:
            logger.info(f"Reconciling {orphans_added} clips on disk missing from dataset.json")
            for entry, clip_file in orphan_clips:
                # Detect characters from clip's first frame
                if character_refs:
                    frame = self._read_frame(clip_file, 0)
                    if frame is not None:
                        chars = self._match_characters_in_frame(
                            frame, character_refs, face_similarity,
                            clip_vision=clip_vision,
                            first_match_only=False,
                        )
                        if chars:
                            entry["characters"] = sorted(chars)
                            logger.info(f"  Detected {', '.join(sorted(chars))} in {clip_file.name}")
                # Transcribe if speech transcription is enabled.
                # Use key existence so silent clips (transcript=="") don't
                # trigger another transcription pass on subsequent runs.
                if transcribe_speech and "transcript" not in entry:
                    _face_chars = set(entry.get("characters") or [])
                    tr = _transcribe_clip(clip_file, voice_refs=voice_refs, face_chars=_face_chars)
                    if tr and tr.get("hallucination"):
                        logger.info(f"  No usable speech, quarantining orphan: {clip_file.name}")
                        q_path = self._quarantine_clip(clip_file, output_dir, "speech_hallucination")
                        self._record_clip_rejection(rejected_path, entry, "speech_hallucination", quarantined_path=q_path)
                        existing_entries.remove(entry)
                        orphans_added -= 1
                        continue
                    if tr and tr["text"]:
                        entry["transcript"] = tr["text"]
                        if tr.get("segments"):
                            entry["transcript_segments"] = tr["segments"]
                        logger.info(f"  Transcribed {clip_file.name}: {tr['text'][:80]}{'...' if len(tr['text']) > 80 else ''}")
                    else:
                        # Silent clip — record the attempt so we don't re-load whisper.
                        entry["transcript"] = ""
            with open(dataset_json_path, "w") as f:
                json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)
            if orphans_added > 0:
                logger.info(f"Reconciled {orphans_added} clips into dataset.json")

        # Backfill characters and transcripts for any existing entries missing them
        if character_refs:
            chars_backfilled = 0
            for entry in existing_entries:
                if entry.get("characters"):
                    continue
                clip_file = Path(entry["media_path"])
                if not clip_file.exists():
                    continue
                frame = self._read_frame(clip_file, 0)
                if frame is not None:
                    chars = self._match_characters_in_frame(
                        frame, character_refs, face_similarity,
                        clip_vision=clip_vision,
                        first_match_only=False,
                    )
                    if chars:
                        entry["characters"] = sorted(chars)
                        chars_backfilled += 1
            if chars_backfilled:
                with open(dataset_json_path, "w") as f:
                    json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)
                logger.info(f"Backfilled characters for {chars_backfilled} entries")

        # clip_check: re-scan EVERY existing clip and rewrite its
        # characters field. Uses identical settings to the main
        # scanner — same character_refs, same face_similarity
        # threshold, same clip_vision, same sample_count, same
        # first_match_only=False enumeration. The SCRFD detection
        # confidence floor (_FACE_DET_MIN_CONFIDENCE) applies
        # automatically through _analyze_frame. Identification
        # results here will exactly match what extraction would
        # produce for the same clip.
        if character_refs and clip_check:
            n_check = max(2, sample_count)
            total_clips = len(existing_entries)
            logger.info(
                f"clip_check: re-scanning {total_clips} clips at {n_check} sample positions, "
                f"face_similarity={face_similarity}, "
                f"refs={sorted(character_refs.keys())}"
            )
            changed = 0
            added_by_char: dict[str, int] = {}
            removed_by_char: dict[str, int] = {}
            for idx, entry in enumerate(existing_entries, start=1):
                clip_file = Path(entry["media_path"])
                if not clip_file.exists():
                    logger.info(f"  [{idx}/{total_clips}] {clip_file.name}: missing file, skipping")
                    continue
                cap = cv2.VideoCapture(str(clip_file))
                if not cap.isOpened():
                    logger.info(f"  [{idx}/{total_clips}] {clip_file.name}: couldn't open, skipping")
                    continue
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if total <= 0:
                    logger.info(f"  [{idx}/{total_clips}] {clip_file.name}: 0-frame clip, skipping")
                    continue
                # Evenly-spaced sample positions: 0, 1/N, 2/N, ..., (N-1)/N.
                # Track per-character position counts so we can apply the
                # same dominance filter the scanner uses when recording
                # characters — a character seen in only 1-2 positions of
                # an 8-position scan won't be labeled as "in" the clip.
                char_position_counts: dict[str, int] = {}
                for i in range(n_check):
                    frame = self._read_frame(clip_file, (total * i) // n_check)
                    if frame is None:
                        continue
                    found = self._match_characters_in_frame(
                        frame, character_refs, face_similarity,
                        clip_vision=clip_vision,
                        first_match_only=False,
                    )
                    for c in found:
                        char_position_counts[c] = char_position_counts.get(c, 0) + 1
                found_chars = self._filter_dominant_chars(char_position_counts)
                prev_chars = set(entry.get("characters") or [])
                if found_chars != prev_chars:
                    added = found_chars - prev_chars
                    removed = prev_chars - found_chars
                    for c in added:
                        added_by_char[c] = added_by_char.get(c, 0) + 1
                    for c in removed:
                        removed_by_char[c] = removed_by_char.get(c, 0) + 1
                    if found_chars:
                        entry["characters"] = sorted(found_chars)
                    elif "characters" in entry:
                        del entry["characters"]
                    changed += 1
                    change_parts = []
                    if added:
                        change_parts.append(f"+{sorted(added)}")
                    if removed:
                        change_parts.append(f"-{sorted(removed)}")
                    logger.info(
                        f"  [{idx}/{total_clips}] {clip_file.name}: "
                        f"{sorted(prev_chars)} -> {sorted(found_chars)} "
                        f"({' '.join(change_parts)})"
                    )
                    # Persist after every change so a crash mid-scan
                    # doesn't lose the work already done.
                    with open(dataset_json_path, "w") as f:
                        json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)
                    _emit_status()
                else:
                    logger.info(
                        f"  [{idx}/{total_clips}] {clip_file.name}: "
                        f"{sorted(prev_chars) if prev_chars else 'no chars'} (unchanged)"
                    )
            if changed:
                logger.info(
                    f"clip_check: updated {changed}/{total_clips} entries. "
                    f"Added: {dict(sorted(added_by_char.items()))}. "
                    f"Removed: {dict(sorted(removed_by_char.items()))}."
                )
            else:
                logger.info(f"clip_check: no entries needed updating ({total_clips} scanned)")

        # Pre-check: only enter the transcription path if there's actually work
        # to do.  The lazy whisper/demucs load fires on the first _transcribe_clip
        # call — if every existing entry has been transcribed (or has no clip
        # on disk), we want zero model loads, not one model load + 168 no-op
        # iterations.  Use key existence (not truthiness) so silent clips with
        # an empty transcript don't get re-attempted on every run.
        backfill_targets = (
            [e for e in existing_entries
             if "transcript" not in e and Path(e["media_path"]).exists()]
            if transcribe_speech else []
        )
        if transcribe_speech and not backfill_targets:
            logger.info("Transcript backfill: nothing to do (skipping whisper/demucs load)")

        if backfill_targets:
            logger.info(f"Transcript backfill: {len(backfill_targets)} clip(s) need transcripts")
            backfilled = 0
            silent_marked = 0
            hallucination_purge = []
            for entry in backfill_targets:
                clip_file = Path(entry["media_path"])
                _face_chars = set(entry.get("characters") or [])
                tr = _transcribe_clip(clip_file, voice_refs=voice_refs, face_chars=_face_chars)
                if tr and tr.get("hallucination"):
                    logger.info(f"  No usable speech, quarantining: {clip_file.name}")
                    q_path = self._quarantine_clip(clip_file, output_dir, "speech_hallucination")
                    self._record_clip_rejection(rejected_path, entry, "speech_hallucination", quarantined_path=q_path)
                    hallucination_purge.append(entry)
                    # Don't mutate existing_entries mid-iteration; the purge
                    # list is processed after the loop.  Skip the per-clip
                    # save here — the entry will be removed at end of loop.
                    continue
                if tr and tr["text"]:
                    entry["transcript"] = tr["text"]
                    if tr.get("segments"):
                        entry["transcript_segments"] = tr["segments"]
                    backfilled += 1
                    logger.info(f"  Backfill transcript {clip_file.name}: {tr['text'][:80]}{'...' if len(tr['text']) > 80 else ''}")
                else:
                    # Silent clip — mark as attempted with empty string so we
                    # don't re-load whisper for it on every subsequent run.
                    entry["transcript"] = ""
                    silent_marked += 1
                # Persist after every successful clip so an interrupted run
                # doesn't waste the work done so far.  Atomic write via temp +
                # rename so the file is never half-written.
                try:
                    tmp_path = dataset_json_path.with_suffix(dataset_json_path.suffix + ".tmp")
                    with open(tmp_path, "w") as f:
                        json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)
                    os.replace(tmp_path, dataset_json_path)
                except OSError as e:
                    logger.warning(f"Could not write dataset.json mid-run: {e}")
            for entry in hallucination_purge:
                existing_entries.remove(entry)
            if backfilled or hallucination_purge or silent_marked:
                # Final save after hallucination purge
                try:
                    tmp_path = dataset_json_path.with_suffix(dataset_json_path.suffix + ".tmp")
                    with open(tmp_path, "w") as f:
                        json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)
                    os.replace(tmp_path, dataset_json_path)
                except OSError as e:
                    logger.warning(f"Could not write dataset.json: {e}")
                if backfilled:
                    logger.info(f"Backfilled {backfilled} missing transcripts")
                if silent_marked:
                    logger.info(f"Marked {silent_marked} silent clip(s) as no-dialogue")
                if hallucination_purge:
                    logger.info(f"Purged {len(hallucination_purge)} clips with no usable speech")

        # --- Rolling dataset budget check ---
        # When max_samples is set, skip clip extraction entirely if we're full.
        if max_samples > 0 and len(existing_entries) >= max_samples:
            logger.info(
                f"Dataset at capacity ({len(existing_entries)}/{max_samples}), "
                "skipping clip extraction"
            )
            # Jump straight to captioning / encoding below.
        else:
            # Determine whether we're in rolling-dataset mode (max_samples set)
            # or classic sequential mode (process all new media from media_folder).
            use_rolling = max_samples > 0
            ref_folder = Path(conditioning_folder) if conditioning_folder else None

            if use_rolling:
                # ---- Rolling dataset mode ----
                # Balanced per-character budgets when character_refs_folder is set.
                char_names = sorted(character_refs.keys()) if character_refs else []
                per_char_quota = {}

                # Use media_folder as the source pool — video-only for rolling
                # mode because images have nothing to resume from.
                pool_media = [
                    item for item in media_files
                    if item["type"] == "video"
                ]
                if not pool_media:
                    logger.warning(f"No videos found in media_folder: {media_folder}")

                # Outer loop: sweep deletions → rebalance → extract → check equilibrium → repeat
                balance_pass = 0
                total_clips_produced = 0
                while pool_media:
                    balance_pass += 1

                    # Sweep for clips deleted from disk since last pass
                    sweep_valid = []
                    sweep_removed = 0
                    sweep_rejected = []
                    for entry in existing_entries:
                        cp = Path(entry["media_path"])
                        if cp.exists():
                            sweep_valid.append(entry)
                        else:
                            sweep_removed += 1
                            logger.info(f"Clip deleted, rejecting: {cp.name}")
                            rejected_chunk_files.add(cp.name)
                            rej = {"source_file": entry.get("source_file", ""),
                                   "media_path": str(cp), "reason": "user_deleted"}
                            stem = cp.stem
                            if "_chunk" in stem:
                                try:
                                    rej["chunk_idx"] = int(stem.rsplit("_chunk", 1)[1])
                                except (ValueError, IndexError):
                                    pass
                            sweep_rejected.append(rej)
                    if sweep_removed:
                        existing_entries = sweep_valid
                        with open(dataset_json_path, "w") as f:
                            json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)
                        rej_list: list[dict] = []
                        if rejected_path.exists():
                            try:
                                with open(rejected_path) as rf:
                                    rej_list = json.load(rf)
                            except (json.JSONDecodeError, KeyError):
                                pass
                        rej_list.extend(sweep_rejected)
                        with open(rejected_path, "w") as rf:
                            json.dump(rej_list, rf, indent=2)
                        logger.info(f"Swept {sweep_removed} deleted clips into rejected.json")

                    clips_budget = max_samples - len(existing_entries)

                    # Compute per-character quotas and counts
                    char_counts = {}
                    if char_names and max_samples > 0:
                        quota_each = max_samples // len(char_names)
                        remainder = max_samples % len(char_names)
                        per_char_quota = {}
                        for ci, name in enumerate(char_names):
                            per_char_quota[name] = quota_each + (1 if ci < remainder else 0)
                            char_counts[name] = 0
                        for entry in existing_entries:
                            for name in entry.get("characters", []):
                                if name in char_counts:
                                    char_counts[name] += 1
                        logger.info(
                            f"Pass {balance_pass} — {', '.join(f'{n}={char_counts[n]}/{per_char_quota[n]}' for n in char_names)}"
                        )

                        # Rebalance: remove excess clips from over-quota
                        # characters. Runs whenever ANY character is over
                        # quota (even if no character is under quota) so that
                        # startup state with extra appearances of a character
                        # gets trimmed to the quota automatically.
                        over_quota = {n for n in char_names if char_counts[n] > per_char_quota[n]}
                        under_quota = {n for n in char_names if char_counts[n] < per_char_quota[n]}
                        if over_quota:
                            removed_for_balance = 0
                            removable: list[dict] = []
                            for entry in existing_entries:
                                entry_chars = set(entry.get("characters", []))
                                if not entry_chars:
                                    continue
                                if entry_chars.issubset(over_quota):
                                    excess = max(char_counts[c] - per_char_quota[c] for c in entry_chars)
                                    removable.append((excess, entry))
                            removable.sort(key=lambda x: -x[0])

                            for _, entry in removable:
                                entry_chars = set(entry.get("characters", []))
                                if not all(char_counts.get(c, 0) > per_char_quota.get(c, 0) for c in entry_chars):
                                    continue
                                clip_path = Path(entry["media_path"])
                                if clip_path.exists():
                                    clip_path.unlink()
                                    logger.info(f"  Rebalance: removed {clip_path.name} ({', '.join(sorted(entry_chars))})")
                                rejected_chunk_files.add(clip_path.name)
                                for c in entry_chars:
                                    if c in char_counts:
                                        char_counts[c] -= 1
                                existing_entries.remove(entry)
                                removed_for_balance += 1

                            if removed_for_balance:
                                clips_budget = max_samples - len(existing_entries)
                                with open(dataset_json_path, "w") as f:
                                    json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)
                                _emit_status()
                                logger.info(
                                    f"Rebalanced: removed {removed_for_balance} clips, "
                                    f"{clips_budget} slots now available"
                                )
                                logger.info(
                                    f"After rebalance: {', '.join(f'{n}={char_counts[n]}/{per_char_quota[n]}' for n in char_names)}"
                                )

                    # Budget is measured in character APPEARANCES, not clips.
                    # Total appearances = sum of char_counts; max appearances
                    # is max_samples. A clip contributes one appearance per
                    # named character, so total clip count can be less than
                    # max_samples when characters share clips.
                    appearances = sum(char_counts.values()) if per_char_quota else len(existing_entries)
                    clips_budget = max(0, max_samples - appearances) if max_samples > 0 else 0
                    if per_char_quota:
                        if all(char_counts.get(n, 0) >= per_char_quota[n] for n in char_names):
                            logger.info(
                                f"All characters at or above quota "
                                f"({', '.join(f'{n}={char_counts[n]}/{per_char_quota[n]}' for n in char_names)})"
                            )
                            break
                    elif max_samples > 0 and clips_budget <= 0:
                        logger.info(f"Dataset at capacity ({len(existing_entries)}/{max_samples})")
                        break

                    logger.info(
                        f"Rolling dataset: {appearances}/{max_samples} appearances, "
                        f"{len(existing_entries)} clips"
                    )

                    # Load video_progress.json
                    vp_path = output_dir / "video_progress.json"
                    video_progress: dict[str, dict] = {}
                    if vp_path.exists():
                        try:
                            with open(vp_path) as vpf:
                                video_progress = json.load(vpf)
                        except (json.JSONDecodeError, KeyError):
                            video_progress = {}

                    incomplete = [
                        item for item in pool_media
                        if video_progress.get(item["path"].name, {}).get("status") != "complete"
                        and item["path"].exists()
                        and str(item["path"]) not in rejected_clips
                    ]

                    if not incomplete:
                        logger.info("All source videos marked complete — nothing left to draw from")
                        break

                    pbar = comfy.utils.ProgressBar(clips_budget)
                    clips_produced_this_pass = 0
                    known_paths = {e["media_path"] for e in existing_entries}

                    # Build (or resume) chunk pool. Persisted to chunk_pool.json
                    # so a crashed or interrupted run can pick up where it left
                    # off without reshuffling and re-trying the same chunks.
                    pool_path = output_dir / "chunk_pool.json"
                    pool_items_by_path = {str(item["path"]): item for item in incomplete}
                    chunk_pool: list[tuple[dict, int]] = []

                    if pool_path.exists():
                        try:
                            with open(pool_path) as pf:
                                saved = json.load(pf)
                        except (json.JSONDecodeError, OSError):
                            saved = []
                        for entry in saved:
                            vp = entry.get("video")
                            ci = entry.get("chunk")
                            if vp is None or ci is None:
                                continue
                            item = pool_items_by_path.get(vp)
                            if item is None:
                                continue  # video no longer in incomplete set
                            clip_file = clips_dir / f"{item['path'].stem}_chunk{ci:04d}.mp4"
                            # Skip if already extracted (on disk OR recorded in
                            # dataset.json), rejected in a prior run, or
                            # absorbed by a shifted extraction.
                            if clip_file.exists():
                                continue
                            if str(clip_file) in known_paths:
                                continue
                            if clip_file.name in rejected_chunk_files:
                                continue
                            chunk_pool.append((item, ci))
                        if chunk_pool:
                            logger.info(
                                f"Resumed chunk pool from disk: "
                                f"{len(chunk_pool)} chunks remaining"
                            )

                    if not chunk_pool:
                        _skipped_known = 0
                        for item in incomplete:
                            cap = cv2.VideoCapture(str(item["path"]))
                            if not cap.isOpened():
                                continue
                            vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.release()
                            if vid_total <= 0:
                                continue

                            skip_s = int(round(skip_start_seconds * vid_fps))
                            skip_e = int(round(skip_end_seconds * vid_fps))
                            ff = max(0, skip_s)
                            lf = max(ff, vid_total - skip_e)
                            if target_fps > 0 and abs(vid_fps - target_fps) > 0.5:
                                # Always overshoot — fps filter rounds DOWN, undershoot kills clips
                                # at the bucket frame minimum.  Buffer of +2 source frames ensures
                                # we never come up short after fps conversion.
                                vid_chunk_frames = math.ceil(target_frames * (vid_fps / target_fps)) + 2
                            else:
                                vid_chunk_frames = target_frames
                            n_chunks = (lf - ff) // vid_chunk_frames

                            for ci in range(n_chunks):
                                clip_file = clips_dir / f"{item['path'].stem}_chunk{ci:04d}.mp4"
                                # Existing dataset entries point at files like
                                # `/full/path/.../clips/name_chunkNNNN.mp4`.
                                # Filter those out so a run with an existing
                                # dataset.json but no chunk_pool.json doesn't
                                # re-process every already-extracted chunk.
                                if clip_file.exists():
                                    continue
                                if str(clip_file) in known_paths:
                                    _skipped_known += 1
                                    continue
                                if clip_file.name in rejected_chunk_files:
                                    continue
                                chunk_pool.append((item, ci))

                        random.shuffle(chunk_pool)
                        _extra = f", {_skipped_known} already in dataset.json" if _skipped_known else ""
                        logger.info(
                            f"Chunk pool: {len(chunk_pool)} available chunks across "
                            f"{len(incomplete)} videos{_extra}"
                        )

                    # Publish the pool's initial size so the live panel can
                    # show progress through the pool.
                    pool_state["total"] = len(chunk_pool)
                    pool_state["remaining"] = len(chunk_pool)
                    _emit_status()

                    def _save_pool() -> None:
                        try:
                            with open(pool_path, "w") as pf:
                                json.dump(
                                    [{"video": str(it["path"]), "chunk": c} for it, c in chunk_pool],
                                    pf,
                                )
                        except OSError as e:
                            logger.warning(f"Could not write chunk_pool.json: {e}")

                    _save_pool()

                    def _any_under_quota() -> bool:
                        if per_char_quota:
                            return any(
                                char_counts.get(n, 0) < per_char_quota[n]
                                for n in char_names
                            )
                        if max_samples > 0:
                            return len(existing_entries) < max_samples
                        return True

                    while chunk_pool and _any_under_quota():
                        item, ci = chunk_pool.pop()
                        _save_pool()
                        pool_state["remaining"] = len(chunk_pool)
                        # Push an update on every pop so the panel's chunk-pool
                        # counter ticks down even when chunks get rejected and
                        # never reach the add-entry path.
                        _emit_status()
                        # Skip chunks that a previous shifted extraction in this
                        # pass has already absorbed — their frames are largely
                        # covered by an earlier clip so re-extracting would
                        # produce mostly-duplicate content.
                        _popped_file = f"{item['path'].stem}_chunk{ci:04d}.mp4"
                        if _popped_file in rejected_chunk_files:
                            logger.info(
                                f"Skipping pool entry {_popped_file} — absorbed "
                                f"by a previous shifted extraction"
                            )
                            continue
                        source_path = str(item["path"])

                        # Target characters are those still below their quota.
                        # Characters at quota are identified (for captioning /
                        # accounting) but don't drive the gate or shift logic,
                        # so we stop wasting time rescuing clips for them.
                        _target_chars = (
                            {n for n in char_names if char_counts.get(n, 0) < per_char_quota.get(n, 0)}
                            if per_char_quota else None
                        )

                        results = self._process_video(
                            item["path"], clips_dir, target_w, target_h, target_frames,
                            face_detection,
                            target_embedding=target_embedding,
                            face_similarity=face_similarity,
                            crop_mode=crop_mode, with_audio=with_audio,
                            character_refs=character_refs,
                            clip_vision=clip_vision,
                            skip_start_seconds=skip_start_seconds,
                            skip_end_seconds=skip_end_seconds,
                            target_fps=target_fps,
                            transcribe_speech=transcribe_speech,
                            voice_refs=voice_refs,
                            target_chunk_idx=ci,
                            sample_count=sample_count,
                            char_positions_required=char_positions_required,
                            allow_unknown_faces_in=allow_unknown_faces_in,
                            target_chars=_target_chars,
                        )
                        # Persist any content-level rejections recorded by
                        # _process_video, and fold any absorbed-neighbor chunks
                        # into rejected_chunk_files for the rest of this pass.
                        self._flush_rejected_chunks(rejected_path, rejected_chunk_files)
                        self._flush_consumed_chunks(rejected_chunk_files)

                        for result_path, result_transcript in results:
                            if str(result_path) in known_paths:
                                continue
                            clip_chars = getattr(self, "_clip_characters", {}).get(str(result_path), [])
                            clip_set = set(clip_chars)

                            # Quota-aware intake: the new clip must (a)
                            # contain at least one under-quota character
                            # and (b) if it contains any at-quota
                            # characters, we must be able to swap out a
                            # SUBSET clip whose char set is a proper subset
                            # of the new clip AND contains every at-quota
                            # character. That way the at-quota chars stay
                            # exactly at quota (remove once, add once)
                            # while under-quota chars gain (+1).
                            swap_target = None
                            if per_char_quota and clip_set:
                                under_in_clip = [
                                    c for c in clip_set
                                    if char_counts.get(c, 0) < per_char_quota.get(c, 0)
                                ]
                                if not under_in_clip:
                                    # No under-quota char in this clip — it
                                    # contributes nothing useful. Reject.
                                    try:
                                        result_path.unlink()
                                    except OSError:
                                        pass
                                    rejected_chunk_files.add(result_path.name)
                                    continue
                                at_quota_in_clip = {
                                    c for c in clip_set
                                    if char_counts.get(c, 0) >= per_char_quota.get(c, 0)
                                }
                                if at_quota_in_clip:
                                    # Need to swap. Look for an existing clip
                                    # whose char set contains every at-quota
                                    # char in the new clip AND is a proper
                                    # subset of the new clip's char set.
                                    # Prefer the smallest such subset (solo
                                    # first) so we keep maximal clip variety.
                                    candidates = []
                                    for old in existing_entries:
                                        old_chars = set(old.get("characters", []))
                                        if not old_chars:
                                            continue
                                        if old_chars == clip_set:
                                            continue
                                        if not old_chars.issubset(clip_set):
                                            continue
                                        if not at_quota_in_clip.issubset(old_chars):
                                            continue
                                        candidates.append((len(old_chars), old))
                                    if not candidates:
                                        try:
                                            result_path.unlink()
                                        except OSError:
                                            pass
                                        rejected_chunk_files.add(result_path.name)
                                        logger.info(
                                            f"Rejected {result_path.name}: "
                                            f"{sorted(at_quota_in_clip)} at quota and "
                                            f"no subset clip available to swap"
                                        )
                                        continue
                                    candidates.sort(key=lambda t: t[0])
                                    swap_target = candidates[0][1]

                            # Perform the swap, if any, BEFORE appending so
                            # counts stay consistent throughout.
                            if swap_target is not None:
                                swap_chars = set(swap_target.get("characters", []))
                                swap_path = Path(swap_target["media_path"])
                                if swap_path.exists():
                                    try:
                                        swap_path.unlink()
                                    except OSError:
                                        pass
                                rejected_chunk_files.add(swap_path.name)
                                known_paths.discard(swap_target["media_path"])
                                existing_entries.remove(swap_target)
                                for c in swap_chars:
                                    if c in char_counts:
                                        char_counts[c] -= 1
                                logger.info(
                                    f"Auto-balance: swapped {sorted(swap_chars)} clip "
                                    f"({swap_path.name}) for new {sorted(clip_set)} clip"
                                )

                            entry = {
                                "caption": "",
                                "media_path": str(result_path),
                                "source_file": source_path,
                            }
                            if transcribe_speech:
                                # Always record the transcript field (even "")
                                # so silent clips don't trigger re-transcription
                                # on subsequent runs.
                                entry["transcript"] = result_transcript or ""
                                clip_segs = getattr(self, "_clip_segments", {}).get(str(result_path))
                                if clip_segs:
                                    entry["transcript_segments"] = clip_segs
                            if clip_chars:
                                entry["characters"] = clip_chars
                            if ref_folder and ref_folder.exists():
                                for ext in VIDEO_EXTENSIONS:
                                    ref_file = ref_folder / (result_path.stem + ext)
                                    if ref_file.exists():
                                        entry["reference_path"] = str(ref_file)
                                        break
                            existing_entries.append(entry)
                            known_paths.add(str(result_path))
                            for c in clip_chars:
                                if c in char_counts:
                                    char_counts[c] += 1
                            clips_produced_this_pass += 1
                            total_clips_produced += 1

                            pbar.update_absolute(clips_produced_this_pass, max_samples - len(existing_entries) + clips_produced_this_pass)
                            _emit_status()

                        # Save dataset.json after each chunk
                        with open(dataset_json_path, "w") as f:
                            json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)

                    # Pass concluded normally — remove chunk_pool.json so the
                    # next pass rebuilds a fresh shuffle. (If the run crashed
                    # inside the loop, the file is left in place and a restart
                    # will resume from it.)
                    if pool_path.exists():
                        try:
                            pool_path.unlink()
                        except OSError:
                            pass

                    logger.info(
                        f"Pass {balance_pass}: added {clips_produced_this_pass} clips "
                        f"({len(existing_entries)}/{max_samples} total)"
                    )
                    if per_char_quota:
                        logger.info(
                            f"Counts: {', '.join(f'{n}={char_counts[n]}/{per_char_quota[n]}' for n in char_names)}"
                        )

                    # Check if we need another rebalance pass
                    if clips_produced_this_pass == 0:
                        logger.info("No new clips produced this pass, stopping")
                        break
                    if not per_char_quota:
                        break  # No character balancing needed
                    # Goal: every character at or above quota. Stop once that
                    # holds (target distribution reached).
                    all_balanced = all(
                        char_counts.get(n, 0) >= per_char_quota.get(n, 0)
                        for n in char_names
                    )
                    if all_balanced:
                        break
                    # Still under-quota somewhere. If nothing is over-quota to
                    # trim AND nothing is under, we're stuck; otherwise loop
                    # back through the rebalance + extract steps.
                    over_quota = {n for n in char_names if char_counts[n] > per_char_quota[n]}
                    under_quota = {n for n in char_names if char_counts[n] < per_char_quota[n]}
                    if not under_quota:
                        break
                    logger.info("Characters still imbalanced, starting rebalance pass...")

                if per_char_quota:
                    logger.info(
                        f"Final counts: {', '.join(f'{n}={char_counts[n]}/{per_char_quota[n]}' for n in char_names)}"
                    )

            else:
                # ---- Classic sequential mode ----
                if new_media:
                    logger.info(f"Processing {len(new_media)} new media files ({len(existing_entries)} already done)")

                    pbar = comfy.utils.ProgressBar(len(new_media))

                    for i, item in enumerate(new_media):
                        # Budget check: honour max_samples even in classic mode
                        if max_samples > 0 and len(existing_entries) >= max_samples:
                            logger.info(
                                f"Reached max_samples={max_samples}, stopping early"
                            )
                            break

                        source_path = str(item["path"])
                        produced_clips = False
                        if item["type"] == "image":
                            result = self._process_image(
                                item["path"], clips_dir, target_w, target_h, face_detection,
                                target_embedding=target_embedding, face_similarity=face_similarity,
                                crop_mode=crop_mode,
                                character_refs=character_refs,
                                clip_vision=clip_vision,
                            )
                            if result:
                                produced_clips = True
                                entry = {"caption": "", "media_path": str(result), "source_file": source_path}
                                if ref_folder and ref_folder.exists():
                                    for ext in VIDEO_EXTENSIONS:
                                        ref_file = ref_folder / (result.stem + ext)
                                        if ref_file.exists():
                                            entry["reference_path"] = str(ref_file)
                                            break
                                existing_entries.append(entry)
                                _emit_status()
                        else:
                            results = self._process_video(
                                item["path"], clips_dir, target_w, target_h, target_frames,
                                face_detection,
                                target_embedding=target_embedding, face_similarity=face_similarity,
                                crop_mode=crop_mode, with_audio=with_audio,
                                character_refs=character_refs,
                                clip_vision=clip_vision,
                                skip_start_seconds=skip_start_seconds,
                                skip_end_seconds=skip_end_seconds,
                                target_fps=target_fps,
                                transcribe_speech=transcribe_speech,
                                voice_refs=voice_refs,
                                sample_count=sample_count,
                                char_positions_required=char_positions_required,
                                allow_unknown_faces_in=allow_unknown_faces_in,
                            )
                            self._flush_rejected_chunks(rejected_path, rejected_chunk_files)
                            if results:
                                produced_clips = True
                            for result_path, result_transcript in results:
                                if max_samples > 0 and len(existing_entries) >= max_samples:
                                    break
                                entry = {"caption": "", "media_path": str(result_path), "source_file": source_path}
                                if transcribe_speech:
                                    # Always record the transcript field (even "")
                                    # so silent clips don't trigger re-transcription.
                                    entry["transcript"] = result_transcript or ""
                                    clip_segs = getattr(self, "_clip_segments", {}).get(str(result_path))
                                    if clip_segs:
                                        entry["transcript_segments"] = clip_segs
                                clip_chars = getattr(self, "_clip_characters", {}).get(str(result_path), [])
                                if clip_chars:
                                    entry["characters"] = clip_chars
                                if ref_folder and ref_folder.exists():
                                    for ext in VIDEO_EXTENSIONS:
                                        ref_file = ref_folder / (result_path.stem + ext)
                                        if ref_file.exists():
                                            entry["reference_path"] = str(ref_file)
                                            break
                                existing_entries.append(entry)
                                _emit_status()

                        # Record fully-rejected source files so they aren't reprocessed
                        if not produced_clips:
                            existing_rejected = []
                            if rejected_path.exists():
                                try:
                                    with open(rejected_path) as rf:
                                        existing_rejected = json.load(rf)
                                except (json.JSONDecodeError, KeyError):
                                    pass
                            existing_rejected.append({
                                "source_file": source_path,
                                "reason": "no clips produced (all chunks rejected)",
                            })
                            with open(rejected_path, "w") as rf:
                                json.dump(existing_rejected, rf, indent=2)
                            logger.info(f"No clips from {item['path'].name} — added to rejected.json")

                        # Save JSON after each source file so progress isn't lost
                        with open(dataset_json_path, "w") as f:
                            json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)

                        pbar.update_absolute(i + 1, len(new_media))

                    # New clips will get their own conditions/latents encoded —
                    # the encode functions skip files that already exist on disk.
                else:
                    logger.info(f"All {len(existing_entries)} clips up to date, no new media to process")
                    # Make sure JSON is written even if no new media
                    if not dataset_json_path.exists():
                        with open(dataset_json_path, "w") as f:
                            json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)

        # Post-extraction cleanup: remove entries for clips deleted during the run
        # and add them to rejected.json so they won't be re-extracted
        post_valid = []
        post_removed = 0
        post_rejected = []
        for entry in existing_entries:
            clip_path = Path(entry["media_path"])
            if clip_path.exists():
                post_valid.append(entry)
            else:
                post_removed += 1
                logger.info(f"Clip deleted during run, rejecting: {clip_path.name}")
                rej_entry: dict = {
                    "source_file": entry.get("source_file", ""),
                    "media_path": str(clip_path),
                    "reason": "user_deleted",
                }
                stem = clip_path.stem
                if "_chunk" in stem:
                    try:
                        chunk_num = int(stem.rsplit("_chunk", 1)[1])
                        rej_entry["chunk_idx"] = chunk_num
                    except (ValueError, IndexError):
                        pass
                post_rejected.append(rej_entry)
        if post_removed:
            existing_entries = post_valid
            with open(dataset_json_path, "w") as f:
                json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)
            # Append to rejected.json
            existing_rejected_list: list[dict] = []
            if rejected_path.exists():
                try:
                    with open(rejected_path) as rf:
                        existing_rejected_list = json.load(rf)
                except (json.JSONDecodeError, KeyError):
                    pass
            existing_rejected_list.extend(post_rejected)
            with open(rejected_path, "w") as rf:
                json.dump(existing_rejected_list, rf, indent=2)
            logger.info(f"Removed {post_removed} deleted clips, added to rejected.json ({len(existing_entries)} remain)")

        if not existing_entries:
            raise RuntimeError(
                "No usable clips produced. All media was discarded "
                "(no faces detected or processing failed)."
            )

        clip_paths = [Path(e["media_path"]) for e in existing_entries]

        # Transcripts are now stored directly in dataset entries during extraction

        # Unload whisper + demucs + speechbrain before captioning to free VRAM
        global _whisper_model, _demucs_model, _demucs_device
        global _speechbrain_embedder
        freed = []
        if _whisper_model is not None:
            del _whisper_model
            _whisper_model = None
            freed.append("Whisper")
        if _demucs_model is not None:
            del _demucs_model
            _demucs_model = None
            _demucs_device = None
            freed.append("Demucs")
        if _speechbrain_embedder is not None:
            try:
                del _speechbrain_embedder
            except Exception:
                pass
            _speechbrain_embedder = None
            freed.append("speechbrain-ecapa")
        if freed:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded {' + '.join(freed)} to free VRAM for captioning")

        # --- PHASE 2: Caption uncaptioned clips (with LLM-based QC if target face provided) ---
        # Encode target face as base64 for Ollama verification
        target_face_b64 = None
        if target_face is not None and caption_mode == "ollama":
            import base64
            frame = target_face[0].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".png", frame)
            target_face_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        # Build cast list for Gemma identification + caption: when
        # character_refs is populated, Gemma is shown each reference image
        # labeled with its name, then the clip frames, and asked to identify
        # which referenced characters appear.  The confirmed subset is then
        # used to drive caption naming.
        cast_names: list[str] = sorted(character_refs.keys()) if character_refs else []
        cast_refs: list[tuple[str, str]] = (
            [(n, character_refs[n]["image_b64"]) for n in cast_names]
            if character_refs else []
        )
        location_refs_list: list[tuple[str, str]] = (
            [(n, location_refs[n]) for n in sorted(location_refs.keys())]
            if location_refs else []
        )

        # Drop references to ComfyUI models BEFORE captioning so Ollama
        # has the full GPU.  We've already extracted everything we need
        # (image bytes for refs) into plain Python data above.  Without
        # this, ComfyUI's CLIP Vision and the embedded face models stay
        # resident in VRAM and ComfyUI's dynamic loader keeps shuffling
        # them in and out, fighting Ollama for memory and slowing every
        # vision call to a crawl.
        character_refs = None  # noqa: F841
        location_refs = None  # noqa: F841
        clip_vision = None  # noqa: F841
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            if hasattr(mm, "cleanup_models"):
                try:
                    mm.cleanup_models(keep_clone_weights_loaded=False)
                except TypeError:
                    mm.cleanup_models()
            if hasattr(mm, "current_loaded_models"):
                try:
                    mm.current_loaded_models.clear()
                except Exception:
                    pass
            mm.soft_empty_cache()
            import gc
            gc.collect()
            mm.free_memory(1e18, mm.get_torch_device())
            logger.info("Cleared ComfyUI VRAM before Ollama captioning phase")
        except Exception as e:
            logger.warning(f"Could not fully free VRAM before captioning: {e}")

        self._caption_dataset_json(
            dataset_json_path, clip_paths, caption_mode, lora_trigger,
            ollama_url=ollama_url, ollama_model=ollama_model,
            target_face_b64=target_face_b64, caption_style=caption_style,
            cast_names=cast_names, cast_refs=cast_refs,
            location_refs=location_refs_list,
            skip_id_pass=skip_id_pass,
        )

        # --- PHASE 3: Encode conditions and latents ---
        # Reload entries from JSON — captioning may have rejected some clips,
        # and existing_entries in memory still has the pre-rejection list.
        with open(dataset_json_path) as f:
            existing_entries = json.load(f)

        # Merge transcripts into captions for text encoding.
        # The combined text goes into a separate "caption_with_transcript" field
        # so the original caption stays clean.  For subprocess encoding we
        # temporarily write the combined text into "caption" then restore it.
        # Voice-attributed clips skip this step — their captioner already wove
        # the dialogue inline (see _prepare_clip_for_caption / DIALOGUE CONTEXT).
        if transcribe_speech:
            for entry in existing_entries:
                if entry.get("transcript_segments"):
                    continue
                transcript = entry.get("transcript", "")
                if transcript:
                    cap = entry.get("caption", "")
                    entry["_caption_original"] = cap
                    entry["caption"] = f'{cap} Dialogue: "{transcript}"' if cap else f'Dialogue: "{transcript}"'
            with open(dataset_json_path, "w") as f:
                json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)

        conditions_dir = output_dir / "conditions"
        latents_dir = output_dir / "latents"
        need_subprocess = False

        # Phase 3a: Text encoding
        if clip is not None:
            self._encode_conditions_inprocess(
                clip, dataset_json_path, conditions_dir,
                character_names=list(character_refs.keys()) if character_refs else None,
            )
        else:
            need_subprocess = True

        # Phase 3b: VAE encoding
        # When with_audio is set, the subprocess needs to write combined video+audio
        # latent files.  It auto-skips files that already exist on disk, so we must
        # NOT pre-encode video-only latents in-process — doing so would cause audio
        # encoding to be skipped entirely.
        if vae is not None and not with_audio:
            self._encode_latents_inprocess(vae, dataset_json_path, latents_dir, existing_entries,
                                             target_w=target_w, target_h=target_h)
        else:
            need_subprocess = True

        # Reference latents (IC-LoRA) and audio still require subprocess
        if conditioning_folder or with_audio:
            need_subprocess = True

        if need_subprocess:
            # process_dataset.py auto-skips files that already exist on disk,
            # so this is safe to run even when in-process encoding already ran.
            # Ensure text encoder path is validated (may have been skipped when clip was provided)
            text_encoder_path = validate_text_encoder_path(text_encoder_path)
            script = get_script_path("process_dataset.py")
            cmd = [
                sys.executable, script,
                str(dataset_json_path),
                "--resolution-buckets", self._resolve_resolution_buckets(resolution_buckets, existing_entries),
                "--model-path", model_full_path,
                "--text-encoder-path", text_encoder_path,
                "--output-dir", str(output_dir),
            ]

            if with_audio:
                cmd.append("--with-audio")
            if load_text_encoder_in_8bit:
                cmd.append("--load-text-encoder-in-8bit")
            if conditioning_folder:
                cmd.extend(["--reference-column", "reference_path"])

            free_vram()
            env = get_trainer_env()

            pbar2 = comfy.utils.ProgressBar(100)
            returncode = run_training_subprocess(cmd, env, progress_bar=pbar2, total_steps=100)

            if returncode != 0:
                raise RuntimeError(f"Dataset preprocessing failed with return code {returncode}")

        # Restore original captions after encoding (remove merged transcript)
        if transcribe_speech:
            for entry in existing_entries:
                if "_caption_original" in entry:
                    entry["caption"] = entry.pop("_caption_original")
            with open(dataset_json_path, "w") as f:
                json.dump(self._entries_for_write(existing_entries, output_dir), f, indent=2)

        # Final unload — drop everything prep loaded so unattended training that
        # follows in the same workflow starts with a clean VRAM slate.
        self._unload_all_prepper_models()

        return (str(output_dir), str(dataset_json_path))

    @staticmethod
    def _unload_all_prepper_models():
        """Free everything the prepper holds: InsightFace, Whisper, Demucs,
        speechbrain ECAPA-TDNN, memoization caches, and ComfyUI's currently-
        loaded models. Safe to call even if nothing was loaded — each unload
        is guarded.
        """
        global _whisper_model, _demucs_model, _demucs_device
        global _face_app, _face_app_checked
        global _last_analysis_frame_id, _last_analysis_faces
        global _speechbrain_embedder

        freed = []
        if _whisper_model is not None:
            del _whisper_model
            _whisper_model = None
            freed.append("Whisper")
        if _demucs_model is not None:
            del _demucs_model
            _demucs_model = None
            _demucs_device = None
            freed.append("Demucs")
        if _face_app is not None:
            try:
                del _face_app
            except Exception:
                pass
            _face_app = None
            _face_app_checked = False
            freed.append("InsightFace")
        if _speechbrain_embedder is not None:
            try:
                del _speechbrain_embedder
            except Exception:
                pass
            _speechbrain_embedder = None
            freed.append("speechbrain-ecapa")

        # Drop frame memoization (holds a reference to the most recent frame)
        _last_analysis_frame_id = None
        _last_analysis_faces = []

        import gc
        gc.collect()
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as e:
            logger.warning(f"Final unload: ComfyUI model_management cleanup failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if freed:
            logger.info(f"Final unload: dropped {' + '.join(freed)} + ComfyUI loaded models")
        else:
            logger.info("Final unload: dropped ComfyUI loaded models (no prepper models were resident)")

    # Strip ' / ' / ' / ` that quote words but keep apostrophes inside words.
    # 'pee-wee' → pee-wee, `pee-wee` → pee-wee, said 'no'. → said no.
    # it's / Pee-wee's / don't all stay intact (apostrophe between letters).
    _QUOTE_STRIP_RE = re.compile(r"(?<![A-Za-z])['‘’`]|['‘’`](?![A-Za-z])")

    @staticmethod
    def _titlecase_name(name: str) -> str:
        """Capitalize the first letter of each space-separated token; leave
        hyphenated forms with single capitalization (pee-wee → Pee-wee, not
        Pee-Wee).  miss yvonne → Miss Yvonne; cowboy curtis → Cowboy Curtis."""
        return " ".join(w[:1].upper() + w[1:] for w in name.split(" ") if w)

    @classmethod
    def _normalize_caption_for_encode(cls, caption: str, character_names=None) -> str:
        """Caption preprocessing applied at text-encode time only — leaves
        dataset.json untouched.

        - Strips word-wrapping quotes so trigger words don't bind to a quoted
          form (e.g. 'pee-wee' would train differently from pee-wee).
        - Capitalizes known character names (case-insensitive match) so the
          training tokens for character triggers are stable proper nouns
          regardless of how Gemma rendered them in any individual caption.
        """
        if not caption:
            return caption
        caption = cls._QUOTE_STRIP_RE.sub("", caption)
        if character_names:
            # Longest-first so multi-word names (miss yvonne) win against
            # single-word substrings (yvonne) inside the same scan.
            for name in sorted(character_names, key=len, reverse=True):
                if not name:
                    continue
                pattern = r"\b" + re.escape(name) + r"\b"
                caption = re.sub(
                    pattern,
                    cls._titlecase_name(name),
                    caption,
                    flags=re.IGNORECASE,
                )
        return caption

    def _encode_conditions_inprocess(self, clip, dataset_json_path: Path, conditions_dir: Path,
                                       character_names=None):
        """Encode captions in-process using ComfyUI's already-loaded CLIP text encoder.

        ComfyUI's LTX CLIP runs Gemma + feature extractor (blocks 1+2 of the text
        encoder pipeline). This produces the same output as process_captions.py's
        text_encoder.encode() + embeddings_processor.feature_extractor() but without
        loading Gemma from disk.

        LTX-2   (single_linear): cond is [B, seq, 3840] — video and audio share features.
        LTX-2.3 (dual_linear):   cond is [B, seq, 6144] — split at 4096 for video/audio.
        """
        conditions_dir = Path(conditions_dir)
        conditions_dir.mkdir(parents=True, exist_ok=True)

        with open(dataset_json_path) as f:
            entries = json.load(f)

        data_root = dataset_json_path.parent
        to_encode = []
        for entry in entries:
            media_path = Path(entry["media_path"])
            # Preserve relative directory structure (e.g. clips/filename.pt)
            # to match how latents are stored — PrecomputedDataset requires matching paths
            try:
                rel_path = media_path.relative_to(data_root).with_suffix(".pt")
            except ValueError:
                rel_path = Path(media_path.stem + ".pt")
            output_file = conditions_dir / rel_path
            if not output_file.exists():
                to_encode.append((entry, output_file))

        if not to_encode:
            logger.info("All conditions already exist, skipping text encoding")
            return

        logger.info(f"Encoding {len(to_encode)} captions in-process with ComfyUI CLIP...")
        pbar = comfy.utils.ProgressBar(len(to_encode))

        for i, (entry, output_file) in enumerate(to_encode):
            caption = entry.get("caption", "")
            caption = self._normalize_caption_for_encode(caption, character_names)

            tokens = clip.tokenize(caption)
            result = clip.encode_from_tokens(tokens, return_dict=True)
            cond = result["cond"]  # [1, seq_len, proj_dim]

            proj_dim = cond.shape[2]
            if proj_dim == 3840:
                # LTX-2: same features for both modalities
                video_features = cond
                audio_features = cond
            elif proj_dim >= 6144:
                # LTX-2.3: split at 4096 for separate video/audio connectors
                video_features = cond[:, :, :4096]
                audio_features = cond[:, :, 4096:]
            else:
                video_features = cond
                audio_features = cond

            orig_seq_len = cond.shape[1]

            # EmbeddingsProcessor.create_embeddings (block 3, applied during training)
            # requires seq_len divisible by num_learnable_registers (128).
            # ComfyUI's CLIP doesn't pad like Gemma's tokenizer, so we left-pad here.
            pad_to = 128
            pad_len = (pad_to - orig_seq_len % pad_to) % pad_to
            if pad_len > 0:
                # Left-pad features with zeros (matching Gemma's left-padding convention)
                video_features = torch.nn.functional.pad(video_features, (0, 0, pad_len, 0))
                if audio_features is not video_features:
                    audio_features = torch.nn.functional.pad(audio_features, (0, 0, pad_len, 0))

            padded_len = orig_seq_len + pad_len
            # Attention mask: False for padding (left), True for real tokens (right)
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.bool),
                torch.ones(orig_seq_len, dtype=torch.bool),
            ]) if pad_len > 0 else torch.ones(orig_seq_len, dtype=torch.bool)

            embedding_data = {
                "video_prompt_embeds": video_features[0].cpu().contiguous(),
                "prompt_attention_mask": attention_mask.cpu().contiguous(),
            }
            if proj_dim >= 6144:
                embedding_data["audio_prompt_embeds"] = audio_features[0].cpu().contiguous()

            output_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embedding_data, output_file)
            pbar.update_absolute(i + 1, len(to_encode))

        logger.info(f"Encoded {len(to_encode)} captions in-process")

    def _encode_latents_inprocess(
        self, vae, dataset_json_path: Path, latents_dir: Path, existing_entries: list[dict],
        target_w: int = 0, target_h: int = 0,
    ):
        """Encode video/image clips in-process using ComfyUI's already-loaded VAE.

        ComfyUI's VAE.encode() handles device management, batching, and normalisation.
        Input format expected: [F, H, W, C] float [0,1] for video, [1, H, W, C] for images.
        Output: 5-D latent [1, C, F', H', W'] from 3D video VAE.
        """
        latents_dir = Path(latents_dir)
        latents_dir.mkdir(parents=True, exist_ok=True)

        data_root = dataset_json_path.parent
        to_encode = []
        for entry in existing_entries:
            media_path = Path(entry["media_path"])
            try:
                rel_path = media_path.relative_to(data_root).with_suffix(".pt")
            except ValueError:
                rel_path = Path(media_path.stem + ".pt")
            output_file = latents_dir / rel_path
            if not output_file.exists():
                to_encode.append((entry, media_path, output_file))

        if not to_encode:
            logger.info("All latents already exist, skipping VAE encoding")
            return

        logger.info(f"Encoding {len(to_encode)} clips in-process with ComfyUI VAE...")
        pbar = comfy.utils.ProgressBar(len(to_encode))

        for i, (entry, media_path, output_file) in enumerate(to_encode):
            print(f"[VAE encode {i+1}/{len(to_encode)}] {media_path.name}")
            ext = media_path.suffix.lower()

            if ext in IMAGE_EXTENSIONS:
                # Read image as [1, H, W, 3] float [0,1]
                img = cv2.imread(str(media_path))
                if img is None:
                    logger.warning(f"Could not read image: {media_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pixels = torch.from_numpy(img).float() / 255.0
                pixels = pixels.unsqueeze(0)  # [1, H, W, 3]
                fps = 1.0
            else:
                pixels, fps = self._load_video_frames(media_path)
                if pixels is None:
                    logger.warning(f"Could not read video: {media_path}")
                    continue

            # Resize to target resolution if needed (allows re-encoding at different res
            # without re-extracting clips). Images keep native resolution (tiny tensors).
            if target_w > 0 and target_h > 0 and pixels.shape[0] > 1:
                _, cur_h, cur_w, _ = pixels.shape
                if cur_w != target_w or cur_h != target_h:
                    # Aspect-preserving scale + letterbox (handles full_frame clips
                    # that may have different aspect ratios than target)
                    scale = min(target_w / cur_w, target_h / cur_h)
                    new_w = int(cur_w * scale)
                    new_h = int(cur_h * scale)
                    scaled = torch.nn.functional.interpolate(
                        pixels.permute(0, 3, 1, 2),  # [F, C, H, W]
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if new_w == target_w and new_h == target_h:
                        pixels = scaled.permute(0, 2, 3, 1)
                    else:
                        # Pad to exact target size with black
                        padded = torch.zeros(pixels.shape[0], 3, target_h, target_w,
                                             dtype=scaled.dtype, device=scaled.device)
                        pad_y = (target_h - new_h) // 2
                        pad_x = (target_w - new_w) // 2
                        padded[:, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled
                        pixels = padded.permute(0, 2, 3, 1)  # [F, H, W, C]

            # VAE.encode() expects [F, H, W, C] and returns [1, C, F', H', W']
            latents = vae.encode(pixels)

            if latents.ndim == 5:
                latents_save = latents[0]  # [C, F', H', W']
            else:
                latents_save = latents

            _, num_frames_lat, h_lat, w_lat = latents_save.shape

            latent_data = {
                "latents": latents_save.cpu().contiguous(),
                "num_frames": num_frames_lat,
                "height": h_lat,
                "width": w_lat,
                "fps": fps,
            }
            output_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(latent_data, output_file)
            pbar.update_absolute(i + 1, len(to_encode))

        logger.info(f"Encoded {len(to_encode)} clips in-process")

    def _load_video_frames(self, video_path: Path):
        """Load all frames from a video as [F, H, W, 3] float [0,1] tensor.
        Returns (tensor, fps) on success, or (None, 0) on failure.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            return None, 0

        video = np.stack(frames)  # [F, H, W, 3] uint8
        video = torch.from_numpy(video).float() / 255.0
        return video, fps

    @staticmethod
    def _resolve_resolution_buckets(resolution_buckets: str, entries: list[dict]) -> str:
        """Add 1-frame image buckets derived from existing video buckets.

        Uses the spatial dimensions of each video bucket with frames=1,
        so images are resized/cropped to match the video resolution.
        This prevents the bucketing algorithm from creating a native-
        resolution image bucket that hijacks video clips due to a closer
        aspect ratio match.
        """
        has_images = any(
            entry.get("media_path", "").lower().endswith((".png", ".jpg", ".jpeg"))
            for entry in entries
        )
        if not has_images:
            return resolution_buckets

        existing_buckets = set(resolution_buckets.split(";"))
        for bucket_str in list(existing_buckets):
            parts = bucket_str.strip().split("x")
            if len(parts) >= 3 and int(parts[2]) > 1:
                image_bucket = f"{parts[0]}x{parts[1]}x1"
                if image_bucket not in existing_buckets:
                    resolution_buckets = f"{resolution_buckets};{image_bucket}"
                    existing_buckets.add(image_bucket)
                    logger.info(f"Added image resolution bucket: {image_bucket}")
        return resolution_buckets

    def _scan_media(self, folder: Path) -> list[dict]:
        """Scan folder for video and image files."""
        files = []
        for f in sorted(folder.iterdir()):
            if f.is_file() and not f.name.startswith("_"):
                ext = f.suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    files.append({"path": f, "type": "video"})
                elif ext in IMAGE_EXTENSIONS:
                    files.append({"path": f, "type": "image"})
        logger.info(
            f"Found {len(files)} media files "
            f"({sum(1 for f in files if f['type'] == 'video')} videos, "
            f"{sum(1 for f in files if f['type'] == 'image')} images)"
        )
        return files

    def _compute_target_embedding(self, target_face_tensor) -> np.ndarray | None:
        """Compute face embedding from a ComfyUI IMAGE tensor."""
        # Convert IMAGE tensor [B, H, W, C] (0-1 float) to BGR uint8
        frame = target_face_tensor[0].cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face = _detect_face_dnn(frame)
        if face is None:
            return None

        embedding = _get_face_embedding(frame, face)
        return embedding

    def _load_character_refs(
        self, refs_folder: str, clip_vision=None,
    ) -> dict[str, dict]:
        """Load reference images from a folder.
        Each file's stem becomes the character's trigger word.  For references
        where a human face is detected, a face embedding is stored.  Otherwise,
        if a CLIP vision model is provided, a CLIP image embedding is stored
        for whole-frame visual matching (puppets, props, objects).  We also
        store a base64-encoded copy of the reference image so Gemma can see
        the actual character when doing identification at caption time.

        Returns {trigger: {"type": "face"|"clip", "embedding": np.ndarray,
                           "image_b64": str}}.
        """
        import base64 as _b64
        refs: dict[str, dict] = {}
        folder = Path(refs_folder)
        if not folder.exists() or not folder.is_dir():
            logger.warning(f"Character refs folder not found: {refs_folder}")
            return refs

        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            frame = cv2.imread(str(f))
            if frame is None:
                logger.warning(f"Could not read character reference: {f.name}")
                continue
            trigger = f.stem.lower().replace("_", "-")

            # Encode the reference image at full resolution so Gemma can
            # see fine details for identification.
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                logger.warning(f"Could not encode reference image for {f.name}")
                continue
            image_b64 = _b64.b64encode(buf.tobytes()).decode("utf-8")

            # Try face detection first — most reliable for humans
            face = _detect_face_dnn(frame)
            if face is not None:
                embedding = _get_face_embedding(frame, face)
                if embedding is not None:
                    refs[trigger] = {
                        "type": "face",
                        "embedding": embedding,
                        "image_b64": image_b64,
                    }
                    logger.info(f"Loaded character reference (face): {trigger}")
                    continue

            # Fallback: CLIP vision embedding for non-human characters/objects
            if clip_vision is None:
                logger.warning(
                    f"No face in '{f.name}' and no clip_vision provided — skipping. "
                    f"Connect a CLIPVisionLoader to match non-human characters."
                )
                continue
            emb = self._clip_vision_encode(clip_vision, frame)
            if emb is None:
                logger.warning(f"Could not compute CLIP embedding for: {f.name}")
                continue
            refs[trigger] = {
                "type": "clip",
                "embedding": emb,
                "image_b64": image_b64,
            }
            logger.info(f"Loaded character reference (clip-vision): {trigger}")
        return refs

    # Audio + video extensions accepted for voice references.  Video files
    # are handled the same way as audio files — _isolate_vocals already runs
    # ffmpeg to pull audio out of any container, then Demucs isolates vocals.
    _VOICE_REF_EXTENSIONS = {
        ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus",
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
    }

    def _load_voice_refs(self, refs_folder: str) -> dict[str, np.ndarray]:
        """Load voice reference clips. Each file's stem becomes the speaker's
        trigger (matching character_refs keys). Vocals are isolated via Demucs
        before embedding so reference embeddings are computed from clean
        speech, matching how clip-time embeddings are computed.

        Returns {trigger: l2_normalized_embedding (np.ndarray)}.
        """
        refs: dict[str, np.ndarray] = {}
        folder = Path(refs_folder)
        if not folder.exists() or not folder.is_dir():
            logger.warning(f"Voice refs folder not found: {refs_folder}")
            return refs
        if _get_speechbrain_embedder() is None:
            logger.warning(
                "Voice attribution requested but speechbrain embedder unavailable — "
                "skipping voice reference enrollment."
            )
            return refs

        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in self._VOICE_REF_EXTENSIONS:
                continue
            trigger = f.stem.lower().replace("_", "-")
            # Demucs isolation matches what _transcribe_clip does so the
            # enrollment domain matches the inference domain.
            vocals_path = _isolate_vocals(f)
            embed_source = vocals_path if vocals_path else f
            try:
                emb = _embed_audio_full(embed_source)
            finally:
                if vocals_path:
                    try:
                        vocals_path.unlink()
                    except OSError:
                        pass
            if emb is None:
                logger.warning(f"Could not embed voice reference: {f.name}")
                continue
            refs[trigger] = emb
            logger.info(f"Loaded voice reference: {trigger}")
        return refs

    def _load_location_refs(self, refs_folder: str) -> dict[str, str]:
        """Load reference images for distinct locations/sets.  Filename
        stem (underscores -> spaces) becomes the location label.  Returns
        {label: base64_jpeg}."""
        import base64 as _b64
        refs: dict[str, str] = {}
        folder = Path(refs_folder)
        if not folder.exists() or not folder.is_dir():
            logger.warning(f"Location refs folder not found: {refs_folder}")
            return refs

        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            frame = cv2.imread(str(f))
            if frame is None:
                logger.warning(f"Could not read location reference: {f.name}")
                continue
            label = f.stem.replace("_", " ").strip()
            # Keep full resolution for location refs — small details like
            # doors, furniture, and wall decorations matter for identification.
            # Only JPEG-compress to reduce payload size.
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                logger.warning(f"Could not encode location reference: {f.name}")
                continue
            refs[label] = _b64.b64encode(buf.tobytes()).decode("utf-8")
            logger.info(f"Loaded location reference: {label}")
        return refs

    def _clip_vision_encode(self, clip_vision, frame: np.ndarray) -> np.ndarray | None:
        """Encode a BGR uint8 frame with a ComfyUI CLIPVision model.
        Returns a unit-normalised 1-D numpy vector, or None on failure."""
        try:
            import torch
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().unsqueeze(0) / 255.0  # [1,H,W,3]
            output = clip_vision.encode_image(tensor)
            # ComfyUI's CLIPVision returns an object with image_embeds / last_hidden_state
            if hasattr(output, "image_embeds"):
                vec = output.image_embeds
            elif isinstance(output, dict) and "image_embeds" in output:
                vec = output["image_embeds"]
            elif hasattr(output, "last_hidden_state"):
                # Mean-pool the token sequence as a fallback
                vec = output.last_hidden_state.mean(dim=1)
            else:
                return None
            vec = vec[0].detach().cpu().float().numpy()
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
            return vec
        except Exception as e:
            logger.warning(f"CLIP vision encode failed: {e}")
            return None

    def _match_characters_in_frame(
        self, frame: np.ndarray,
        character_refs: dict[str, dict],
        threshold: float,
        clip_vision=None,
        clip_threshold: float = 0.75,
        first_match_only: bool = False,
    ) -> set[str]:
        """Return the set of character trigger words whose reference matches
        something in the given frame.  Face refs are matched against detected
        faces in the frame; clip-vision refs are matched against the
        whole-frame CLIP embedding.

        When first_match_only=True, returns as soon as ANY match is found —
        skips CLIP vision if a face already matched.  This is the fast path
        for clip-selection gating."""
        matches: set[str] = set()
        if not character_refs:
            return matches

        face_refs = {n: r["embedding"] for n, r in character_refs.items() if r["type"] == "face"}
        clip_refs = {n: r["embedding"] for n, r in character_refs.items() if r["type"] == "clip"}

        # --- Face matching ---
        if face_refs:
            faces = _detect_all_faces_dnn(frame)
            for rect in faces:
                emb = _get_face_embedding(frame, rect)
                if emb is None:
                    continue
                best_name = None
                best_sim = threshold
                for name, ref in face_refs.items():
                    sim = _match_face(emb, ref)
                    if sim >= best_sim:
                        best_sim = sim
                        best_name = name
                if best_name is not None:
                    matches.add(best_name)
                    if first_match_only:
                        return matches

        # --- CLIP vision matching (non-human refs) ---
        if clip_refs and clip_vision is not None:
            frame_emb = self._clip_vision_encode(clip_vision, frame)
            if frame_emb is not None:
                for name, ref in clip_refs.items():
                    sim = float(np.dot(frame_emb, ref))
                    if sim >= clip_threshold:
                        matches.add(name)
                        if first_match_only:
                            return matches

        return matches

    def _check_face_match(
        self, frame: np.ndarray, face_rect: tuple, target_embedding: np.ndarray, threshold: float,
    ) -> bool:
        """Check if a detected face matches the target face."""
        embedding = _get_face_embedding(frame, face_rect)
        if embedding is None:
            return False
        similarity = _match_face(embedding, target_embedding)
        return similarity >= threshold

    def _process_image(
        self, img_path: Path, clips_dir: Path,
        target_w: int, target_h: int, face_detection: bool,
        target_embedding: np.ndarray | None = None, face_similarity: float = 0.40,
        crop_mode: str = "face_crop",
        character_refs: dict[str, dict] | None = None,
        clip_vision=None,
    ) -> Path | None:
        """Process a single image: detect face, crop or scale, save as PNG.
        Returns output path or None if no face found (when face_detection is on).
        """
        out_path = clips_dir / (img_path.stem + "_img.png")
        if out_path.exists():
            return out_path

        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Could not read image: {img_path}")
            return None

        # Face detection only for face_crop mode — stills in full_frame mode
        # are assumed to be the subject (user curated the folder)
        if crop_mode != "full_frame" and face_detection:
            if character_refs:
                # Multi-character mode: accept if ANY known character is
                # present (face or clip-vision match), no bare face required.
                matches = self._match_characters_in_frame(
                    frame, character_refs, face_similarity,
                    clip_vision=clip_vision,
                )
                if not matches:
                    logger.info(f"No known characters in {img_path.name}, skipping")
                    return None
            else:
                face = _detect_face_dnn(frame)
                if face is None:
                    logger.info(f"No face detected, skipping: {img_path.name}")
                    return None
                if target_embedding is not None:
                    if not self._check_face_match(frame, face, target_embedding, face_similarity):
                        logger.info(f"Face doesn't match target, skipping: {img_path.name}")
                        return None

        if crop_mode == "full_frame":
            # Keep native resolution — VAE encode step handles resize
            output = frame
        elif crop_mode == "pan_and_scan":
            face = _detect_face_dnn(frame) if face_detection else None
            h, w = frame.shape[:2]
            if face is not None:
                crop = _compute_pan_and_scan(*face, w, h, target_w, target_h)
            else:
                crop = self._center_crop(w, h, target_w, target_h)
            cx, cy, cw, ch = crop
            output = frame[cy:cy+ch, cx:cx+cw]
        else:
            # face_crop: tight crop around face
            crop = self._get_face_crop(frame, target_w, target_h, face_detection)
            if crop is None:
                logger.info(f"No face detected, skipping: {img_path.name}")
                return None
            cx, cy, cw, ch = crop
            output = frame[cy:cy+ch, cx:cx+cw]

        cv2.imwrite(str(out_path), output)
        logger.info(f"Processed image: {img_path.name} -> {out_path.name}")
        return out_path

    def _flush_consumed_chunks(self, rejected_chunk_files: set) -> None:
        """Drain self._consumed_chunks (chunks absorbed by a shifted
        extraction) into the in-memory rejected_chunk_files set so they're
        skipped at pop time. Not persisted to rejected.json — these aren't
        rejected content, just already-covered-by-another-extraction."""
        pending = getattr(self, "_consumed_chunks", [])
        for cf in pending:
            rejected_chunk_files.add(cf)
        self._consumed_chunks = []

    @staticmethod
    def _quarantine_clip(clip_path: Path, output_dir: Path, reason: str) -> Path | None:
        """Move a clip to <output_dir>/rejected_clips/<reason>/ instead of
        deleting it.  Files stay recoverable for inspection — the user can
        review what got auto-rejected and restore false positives.  Returns
        the new path (or None if the move failed).
        """
        if not clip_path.exists():
            return None
        # Sanitize reason into a folder name (e.g. "speech_hallucination")
        safe_reason = "".join(c if c.isalnum() or c in "_-" else "_" for c in reason)
        dest_dir = output_dir / "rejected_clips" / safe_reason
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create quarantine dir {dest_dir}: {e}")
            return None
        dest = dest_dir / clip_path.name
        # If a file already exists at the destination, append a counter
        if dest.exists():
            i = 1
            while True:
                alt = dest_dir / f"{clip_path.stem}_{i}{clip_path.suffix}"
                if not alt.exists():
                    dest = alt
                    break
                i += 1
        try:
            shutil.move(str(clip_path), str(dest))
            logger.info(f"  Quarantined to {dest_dir.name}/: {clip_path.name}")
            return dest
        except OSError as e:
            logger.warning(f"Could not quarantine {clip_path.name}: {e}")
            return None

    @staticmethod
    def _record_clip_rejection(rejected_path: Path, entry: dict, reason: str,
                                quarantined_path: Path | None = None) -> None:
        """Append a rejection record for a single clip to rejected.json with
        chunk-level info. Used when a clip is removed mid-run (hallucination,
        manual delete, etc.) so we have a paper trail and the chunk pool
        knows the slot is freed."""
        clip_path = Path(entry.get("media_path", ""))
        rej = {
            "source_file": entry.get("source_file", ""),
            "media_path": str(clip_path),
            "chunk_file": clip_path.name,
            "reason": reason,
        }
        if quarantined_path is not None:
            rej["quarantined_path"] = str(quarantined_path)
        stem = clip_path.stem
        if "_chunk" in stem:
            try:
                rej["chunk_idx"] = int(stem.rsplit("_chunk", 1)[1])
            except (ValueError, IndexError):
                pass
        existing: list[dict] = []
        if rejected_path.exists():
            try:
                with open(rejected_path) as rf:
                    existing = json.load(rf)
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(rej)
        try:
            with open(rejected_path, "w") as rf:
                json.dump(existing, rf, indent=2)
        except OSError as e:
            logger.warning(f"Could not write rejected.json: {e}")

    def _flush_rejected_chunks(
        self,
        rejected_path: Path,
        rejected_chunk_files: set,
    ) -> None:
        """Drain self._rejected_chunks (populated by _process_video on content
        failures) into rejected.json and the in-memory rejected_chunk_files set,
        then clear the list. Safe no-op when the list is empty."""
        pending = getattr(self, "_rejected_chunks", [])
        if not pending:
            return
        for rej in pending:
            cf = rej.get("chunk_file")
            if cf:
                rejected_chunk_files.add(cf)
        existing: list[dict] = []
        if rejected_path.exists():
            try:
                with open(rejected_path) as rf:
                    existing = json.load(rf)
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.extend(pending)
        try:
            with open(rejected_path, "w") as rf:
                json.dump(existing, rf, indent=2)
        except OSError as e:
            logger.warning(f"Could not write rejected.json: {e}")
        self._rejected_chunks = []

    def _process_video(
        self, video_path: Path, clips_dir: Path,
        target_w: int, target_h: int, target_frames: int,
        face_detection: bool,
        target_embedding: np.ndarray | None = None, face_similarity: float = 0.40,
        crop_mode: str = "face_crop", with_audio: bool = False,
        character_refs: dict[str, dict] | None = None,
        clip_vision=None,
        skip_start_seconds: float = 0.0,
        skip_end_seconds: float = 0.0,
        target_fps: float = 0.0,
        transcribe_speech: bool = False,
        voice_refs: dict | None = None,
        max_new_clips: int = 0,
        target_chunk_idx: int = -1,
        sample_count: int = 4,
        char_positions_required: str = "75%",
        allow_unknown_faces_in: str = "25%",
        target_chars: set[str] | None = None,
    ) -> list[Path]:
        """Split a video into chunks, detect faces, crop or scale.
        Returns list of output clip paths (skips chunks with no face when face_detection is on).
        max_new_clips: stop after producing this many NEW clips (0=unlimited).
        target_chunk_idx: if >= 0, jump straight to this chunk and process only it.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if total_frames <= 0:
            return []

        # Per-clip character tracking for balanced sampling.
        # Populated when character_refs is set; keyed by clip path string.
        if not hasattr(self, "_clip_characters"):
            self._clip_characters = {}
        # Content-based chunk rejections collected during this call so the
        # caller can flush them into rejected.json. Each entry is a dict with
        # media_path, source_file, reason, chunk_file.
        if not hasattr(self, "_rejected_chunks"):
            self._rejected_chunks = []
        # Chunk filenames absorbed into a shifted extraction (the shifted
        # chunk's frames overlap with these chunks' territory, so re-extracting
        # them would produce mostly-duplicate content). Drained by the caller
        # into rejected_chunk_files after each _process_video call.
        if not hasattr(self, "_consumed_chunks"):
            self._consumed_chunks = []

        # Apply intro / outro skip ranges (useful for cutting show intros and
        # end credits that would otherwise be duplicated across every episode).
        skip_start_frames = int(round(skip_start_seconds * fps))
        skip_end_frames = int(round(skip_end_seconds * fps))
        first_frame = max(0, skip_start_frames)
        last_frame = max(first_frame, total_frames - skip_end_frames)
        if first_frame > 0 or last_frame < total_frames:
            logger.info(
                f"{video_path.name}: scanning frames "
                f"{first_frame}-{last_frame} of {total_frames} "
                f"(skip_start={skip_start_seconds}s, skip_end={skip_end_seconds}s)"
            )

        # When target_fps is set and differs from source, each chunk must span
        # more source frames to produce target_frames output frames at the lower
        # rate.  E.g. 49 output frames at 24fps = 2.04s → need 61 source frames
        # at 30fps to cover the same duration.
        use_fps_conversion = target_fps > 0 and abs(fps - target_fps) > 0.5
        if use_fps_conversion:
            # Always overshoot: fps filter rounds DOWN, undershoot drops clips at
            # the bucket frame minimum.  Buffer of +2 source frames ensures we
            # never come up short after fps conversion; the output is then
            # capped to exactly target_frames via -frames:v.
            source_chunk_frames = math.ceil(target_frames * (fps / target_fps)) + 2
            logger.info(
                f"{video_path.name}: fps conversion {fps:.3f} -> {target_fps:.3f} "
                f"({source_chunk_frames} source frames per {target_frames}-frame output chunk)"
            )
        else:
            source_chunk_frames = target_frames

        # Split into chunks of source_chunk_frames (yields target_frames output frames)
        clips = []
        new_clips_produced = 0

        # Resume support: we keep a per-video `.progress` file in clips_dir
        # that records the NEXT chunk index to process.  Unlike a glob of
        # extracted clips, this tracks progress through skipped chunks too, so
        # on resume we jump straight past all previously-scanned chunks even
        # if most of them were rejected by the character filter.
        progress_file = clips_dir / f"{video_path.stem}.progress"
        # NOTE: do not use Path.glob here — video filenames frequently contain
        # characters like `[iEgv1K7nPCM]` which glob interprets as character
        # classes, causing zero matches.  Scan the directory and match literal
        # prefixes instead.
        chunk_prefix = f"{video_path.stem}_chunk"
        existing_chunks = sorted(
            (
                p for p in clips_dir.iterdir()
                if p.is_file()
                and p.suffix == ".mp4"
                and p.name.startswith(chunk_prefix)
            ),
            key=lambda p: p.name,
        )
        # In targeted mode, don't return existing chunks — only the targeted one.
        # Otherwise all on-disk clips get added as new entries, blowing past max_samples.
        if target_chunk_idx < 0:
            clips.extend((p, None) for p in existing_chunks)

        # When target_chunk_idx is set, jump straight to that chunk.
        # Otherwise use sequential resume logic.
        if target_chunk_idx >= 0:
            chunk_idx = target_chunk_idx
            start_frame = first_frame + chunk_idx * source_chunk_frames
            # Force max_new_clips=1 for targeted extraction
            max_new_clips = 1
        else:
            resume_chunk_idx = 0
            if progress_file.exists():
                try:
                    resume_chunk_idx = int(progress_file.read_text().strip())
                except (ValueError, OSError):
                    resume_chunk_idx = 0
            # Fallback: if no progress file but extracted clips exist, derive a
            # lower bound from the highest extracted chunk index so we don't
            # reprocess clearly-complete chunks on old runs.
            if resume_chunk_idx == 0 and existing_chunks:
                for p in existing_chunks:
                    try:
                        idx = int(p.stem.rsplit("_chunk", 1)[1])
                        if idx + 1 > resume_chunk_idx:
                            resume_chunk_idx = idx + 1
                    except (ValueError, IndexError):
                        continue

            chunk_idx = resume_chunk_idx
            start_frame = first_frame + chunk_idx * source_chunk_frames

            if chunk_idx > 0:
                logger.info(
                    f"{video_path.name}: resuming from chunk {chunk_idx} "
                    f"(frame {start_frame}) — {len(existing_chunks)} existing clips kept"
                )

        while start_frame < last_frame:
            end_frame = min(start_frame + source_chunk_frames, last_frame)
            # Skip chunks that are too short (less than half the target)
            if end_frame - start_frame < source_chunk_frames // 2:
                break
            clip_transcript = None

            # Persist resume progress for sequential mode only.
            # Rolling mode tracks by clip file existence instead.
            if target_chunk_idx < 0:
                try:
                    progress_file.write_text(str(chunk_idx))
                except OSError:
                    pass

            crop = None
            matched_sample = None
            matched_names = set()

            # Clip-selection gating.  Two modes:
            #   - Multi-character: chunk must contain at least one known
            #     character from character_refs (face OR clip-vision match).
            #     Face crop uses the first detected face if available,
            #     otherwise falls back to center crop.
            #   - Single-character / default: requires a detected face; if a
            #     target_embedding is set, that face must match the target.
            # Sample positions are evenly spaced from start_frame over the
            # chunk — 0%, 1/N, 2/N, ..., (N-1)/N. At N=4 this reproduces the
            # old 0%, 25%, 50%, 75% layout.
            chunk_len = end_frame - start_frame
            n_samples = max(2, sample_count)
            sample_positions = [start_frame + i * chunk_len // n_samples for i in range(n_samples)]
            sample_positions = [p for p in sample_positions if start_frame <= p < end_frame]

            if face_detection and character_refs:
                # Parse the percentage settings once per chunk.
                try:
                    _pct = int(str(char_positions_required).rstrip("%")) / 100.0
                except ValueError:
                    _pct = 0.75
                try:
                    _unknown_pct = int(str(allow_unknown_faces_in).rstrip("%")) / 100.0
                except ValueError:
                    _unknown_pct = 0.25

                def _validate_at(_start: int, _end: int):
                    """Run full character + unknown-face validation at a
                    given [start, end) range.

                    Target-aware counting: `hits`/`pos_has_hit` only count
                    positions where a TARGET character was detected (one
                    we still need more of). Over-quota characters still
                    appear in `matched_names` for the final caption and
                    still count as KNOWN (so they don't inflate the
                    unknown-face count) — they're just not what the seek
                    is chasing. Which character eventually rolls in along
                    with the target is handled later by the subset-swap
                    intake filter.

                    When target_chars is None (classic / single-target
                    mode), every reference character counts as a hit."""
                    n = max(2, sample_count)
                    clen = _end - _start
                    positions = [_start + i * clen // n for i in range(n)]
                    positions = [p for p in positions if _start <= p < _end]
                    char_position_counts: dict[str, int] = {}
                    hits = 0
                    has_hit = [False] * len(positions)
                    anchor = None
                    unknown_at: list[int] = []
                    for idx, sp in enumerate(positions):
                        sample = self._read_frame(video_path, sp)
                        if sample is None:
                            continue
                        found = self._match_characters_in_frame(
                            sample, character_refs, face_similarity,
                            clip_vision=clip_vision, first_match_only=False,
                        )
                        if found:
                            for c in found:
                                char_position_counts[c] = char_position_counts.get(c, 0) + 1
                            is_target_hit = (
                                bool(found & target_chars)
                                if target_chars is not None
                                else True
                            )
                            if is_target_hit:
                                hits += 1
                                has_hit[idx] = True
                                if anchor is None:
                                    face = _detect_face_dnn(sample)
                                    if face is not None:
                                        anchor = (sample, face)
                        # Full character_refs used for unknown detection —
                        # any KNOWN character (target or over-quota) is
                        # still "known" and doesn't count as unknown.
                        if _has_unknown_face(sample, character_refs, face_similarity):
                            unknown_at.append(idx)
                    # Dominance filter: a character is only recorded as
                    # present if they were seen in at least
                    # _CHAR_DOMINANCE_RATIO * max_count positions. Cameos
                    # at the fringe of another character's shot are
                    # filtered out of the recorded label.
                    matched = self._filter_dominant_chars(char_position_counts)
                    return {
                        "sample_positions": positions,
                        "matched_names": matched,
                        "hits_per_pos": hits,
                        "pos_has_hit": has_hit,
                        "face_anchor": anchor,
                        "unknown_face_positions": len(unknown_at),
                        "unknown_at": unknown_at,
                        "char_position_counts": char_position_counts,
                    }

                info = _validate_at(start_frame, end_frame)
                sample_positions = info["sample_positions"]
                matched_names = info["matched_names"]
                hits_per_pos = info["hits_per_pos"]
                face_anchor = info["face_anchor"]
                unknown_face_positions = info["unknown_face_positions"]
                pos_has_hit = info["pos_has_hit"]
                unknown_at = info["unknown_at"]

                min_hits = max(1, int(round(_pct * len(sample_positions))))
                # Unknown-face tolerance converts from a percentage of sample
                # positions. 0% = no extras tolerated; 100% = extras anywhere.
                unknown_tol = int(round(_unknown_pct * len(sample_positions)))
                chunk_file = f"{video_path.stem}_chunk{chunk_idx:04d}.mp4"

                # Pre-compute shift helpers so both the main search and the
                # extension block can use them, even when the main search
                # is skipped (e.g. original already at 100% coverage).
                _chunk_len = end_frame - start_frame
                _snap8 = lambda x: (x // 8) * 8

                # Early reject only if we found NO target character at all.
                # If matched_names has entries they're all over-quota — we're
                # not going to chase them. No direction clue for rescue.
                if hits_per_pos == 0:
                    if matched_names:
                        # Only over-quota chars here. Skip in-memory only so
                        # quota changes between runs can revisit the chunk.
                        logger.info(
                            f"Chunk {chunk_idx} of {video_path.name}: "
                            f"only over-quota chars ({sorted(matched_names)}), skipping"
                        )
                        self._consumed_chunks.append(chunk_file)
                    else:
                        # Truly no known character — persist so future runs
                        # don't retry.
                        logger.info(f"No known characters in chunk {chunk_idx} of {video_path.name}, skipping")
                        self._rejected_chunks.append({
                            "source_file": str(video_path),
                            "media_path": str(clips_dir / chunk_file),
                            "chunk_file": chunk_file,
                            "reason": "no_known_character",
                        })
                    if target_chunk_idx >= 0:
                        break
                    start_frame = end_frame
                    chunk_idx += 1
                    continue

                # Unified shift logic — covers two cases in rolling/targeted
                # mode (sequential mode can't mutate start/end without
                # cascading):
                #   1. Rescue: original gate fails (hits below min or too many
                #      unknowns) but at least one character was detected.
                #      Search before and after for a position where the
                #      character is better captured.
                #   2. Balance: original gate passes but hits are concentrated
                #      in one half. Shift in that direction to center the
                #      character's on-screen window.
                # First shift (across all tried fractions and directions) that
                # lands in a passing + balanced position wins. Progressive
                # fractions (25/50/75/100%) handle short character windows
                # that need only a small shift as well as long ones that need
                # a full chunk move.
                # Seek for the best position: evaluate the original plus
                # every candidate shift (both directions, all fractions),
                # then pick the candidate with the highest hit count, using
                # lowest unknown-face count as tiebreaker. The floor
                # (min_hits) and unknown_tol act as pass/fail gates — they
                # only mark candidates ineligible, not chosen. So 50% is the
                # minimum; a position with 100% coverage will always beat
                # a position with 50% coverage, regardless of which one we
                # "found first".
                hit_indices = [i for i, h in enumerate(pos_has_hit) if h]
                n_pos = len(sample_positions)
                original_passes = (
                    hits_per_pos >= min_hits
                    and unknown_face_positions <= unknown_tol
                )

                candidates: list[dict] = []
                if original_passes:
                    candidates.append({
                        "shift": 0,
                        "start": start_frame,
                        "end": end_frame,
                        "sample_positions": sample_positions,
                        "matched_names": matched_names,
                        "hits_per_pos": hits_per_pos,
                        "pos_has_hit": pos_has_hit,
                        "face_anchor": face_anchor,
                        "unknown_face_positions": unknown_face_positions,
                        "unknown_at": unknown_at,
                    })

                # Shifts only meaningful when we already found at least one
                # character hit at the original position (direction cue) and
                # we're in rolling/targeted mode (sequential mode can't
                # mutate start/end without cascading into the next chunk).
                # Skip the search entirely if the original is already 100%
                # covered — no shift can do better.
                _original_perfect = (
                    original_passes and hits_per_pos == n_pos
                )
                if _original_perfect:
                    logger.info(
                        f"Chunk {chunk_idx}: original at 100% coverage — "
                        f"no shift search needed"
                    )
                if (
                    target_chunk_idx >= 0
                    and hit_indices
                    and n_pos >= 2
                    and not _original_perfect
                ):
                    logger.info(
                        f"Chunk {chunk_idx}: seeking better coverage — "
                        f"original hits {hits_per_pos}/{n_pos} at positions "
                        f"{hit_indices}, unknowns at {unknown_at}/{n_pos}"
                    )
                    # _chunk_len and _snap8 already defined above.
                    # Seek up to 2 full chunk lengths in either direction —
                    # enough to catch a character whose on-screen window
                    # spans multiple chunk slots, but not so wide that we're
                    # effectively scanning the whole video. A real editor
                    # would scrub this far to find a clean in/out point.
                    # If any shift hits 100% coverage we early-exit both
                    # loops — no better result possible.
                    _perfect_found = False
                    for _sign in (-1, 1):
                        if _perfect_found:
                            break
                        for _frac in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
                            _shift = _sign * _snap8(int(_chunk_len * _frac))
                            if _shift == 0:
                                continue
                            _ns, _ne = start_frame + _shift, end_frame + _shift
                            if _ns < first_frame or _ne > last_frame:
                                logger.info(
                                    f"Chunk {chunk_idx}: shift {_shift:+d} "
                                    f"({int(_frac * 100)}%) aborted — new range "
                                    f"[{_ns}, {_ne}) crosses video bounds "
                                    f"[{first_frame}, {last_frame})"
                                )
                                continue
                            alt = _validate_at(_ns, _ne)
                            _alt_hits = [i for i, h in enumerate(alt["pos_has_hit"]) if h]
                            _alt_passes = (
                                alt["hits_per_pos"] >= min_hits
                                and alt["unknown_face_positions"] <= unknown_tol
                            )
                            if _alt_passes:
                                logger.info(
                                    f"Chunk {chunk_idx}: shift {_shift:+d} "
                                    f"({int(_frac * 100)}%) candidate — hits "
                                    f"{alt['hits_per_pos']}/{n_pos} at {_alt_hits}, "
                                    f"unknowns {alt['unknown_face_positions']}/{n_pos}, "
                                    f"chars {sorted(alt['matched_names'])}"
                                )
                                candidates.append({
                                    "shift": _shift,
                                    "start": _ns,
                                    "end": _ne,
                                    **alt,
                                })
                                if alt["hits_per_pos"] == n_pos:
                                    logger.info(
                                        f"Chunk {chunk_idx}: shift {_shift:+d} "
                                        f"hit 100% coverage — stopping search"
                                    )
                                    _perfect_found = True
                                    break
                            else:
                                _reasons = []
                                if alt["hits_per_pos"] < min_hits:
                                    _reasons.append(f"hits {alt['hits_per_pos']}<{min_hits}")
                                if alt["unknown_face_positions"] > unknown_tol:
                                    _reasons.append(
                                        f"unknown {alt['unknown_face_positions']}>{unknown_tol}"
                                    )
                                logger.info(
                                    f"Chunk {chunk_idx}: shift {_shift:+d} "
                                    f"({int(_frac * 100)}%) ineligible — {', '.join(_reasons)}"
                                )

                # If the best candidate so far is pinned at the outer
                # boundary (±200%) AND still below 100% hits, the
                # character's on-screen window probably extends further.
                # Keep pushing in that direction in 25% steps until we
                # hit 100%, hits drop (past the peak), bounds cross, or
                # the extended cap (400% = 4 chunks) is reached.
                if candidates and target_chunk_idx >= 0:
                    candidates.sort(
                        key=lambda c: (
                            -c["hits_per_pos"],
                            c["unknown_face_positions"],
                            abs(c["shift"]),
                        )
                    )
                    _best_now = candidates[0]
                    _max_initial_shift = _snap8(int(_chunk_len * 2.0))
                    if (
                        _best_now["shift"] != 0
                        and abs(_best_now["shift"]) >= _max_initial_shift
                        and _best_now["hits_per_pos"] < n_pos
                    ):
                        _ext_sign = 1 if _best_now["shift"] > 0 else -1
                        _ext_prev_hits = _best_now["hits_per_pos"]
                        logger.info(
                            f"Chunk {chunk_idx}: best so far pinned at "
                            f"{_best_now['shift']:+d} ({_best_now['hits_per_pos']}/{n_pos} hits) — "
                            f"extending search in that direction"
                        )
                        for _frac in (2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0):
                            _shift = _ext_sign * _snap8(int(_chunk_len * _frac))
                            _ns, _ne = start_frame + _shift, end_frame + _shift
                            if _ns < first_frame or _ne > last_frame:
                                logger.info(
                                    f"Chunk {chunk_idx}: extended shift {_shift:+d} "
                                    f"({int(_frac * 100)}%) stopped — range "
                                    f"[{_ns}, {_ne}) crosses video bounds"
                                )
                                break
                            alt = _validate_at(_ns, _ne)
                            _alt_hits = [i for i, h in enumerate(alt["pos_has_hit"]) if h]
                            _alt_passes = (
                                alt["hits_per_pos"] >= min_hits
                                and alt["unknown_face_positions"] <= unknown_tol
                            )
                            if not _alt_passes:
                                logger.info(
                                    f"Chunk {chunk_idx}: extended shift {_shift:+d} "
                                    f"({int(_frac * 100)}%) ineligible — stopping extension"
                                )
                                break
                            if alt["hits_per_pos"] < _ext_prev_hits:
                                # Strict decline = past the peak. The
                                # character's window is receding; no gain
                                # from pushing further.
                                logger.info(
                                    f"Chunk {chunk_idx}: extended shift {_shift:+d} "
                                    f"({int(_frac * 100)}%) past the peak "
                                    f"({alt['hits_per_pos']}<{_ext_prev_hits}) — stopping extension"
                                )
                                break
                            # Plateau (same hit count as previous step) is
                            # NOT a stop — the character window may have a
                            # brief dip and then improve again further out.
                            # Register it as a candidate and keep pushing
                            # until we find a strict improvement, a strict
                            # decline, or 100% coverage.
                            logger.info(
                                f"Chunk {chunk_idx}: extended shift {_shift:+d} "
                                f"({int(_frac * 100)}%) candidate — hits "
                                f"{alt['hits_per_pos']}/{n_pos} at {_alt_hits}, "
                                f"unknowns {alt['unknown_face_positions']}/{n_pos}"
                            )
                            candidates.append({
                                "shift": _shift,
                                "start": _ns,
                                "end": _ne,
                                **alt,
                            })
                            _ext_prev_hits = alt["hits_per_pos"]
                            if alt["hits_per_pos"] == n_pos:
                                logger.info(
                                    f"Chunk {chunk_idx}: extended shift {_shift:+d} "
                                    f"reached 100% coverage — stopping extension"
                                )
                                break

                # Hit-centered probe: if nothing in the initial search hit
                # 100%, we already know the exact video frames where the
                # character was detected. Align the chunk on the center of
                # the best candidate's hit frames — no guessing at
                # half-step fractions, just center the window on where the
                # character actually is. One additional validation per
                # chunk. Skipped entirely when 100% was already found.
                if (
                    candidates
                    and target_chunk_idx >= 0
                    and not any(c["hits_per_pos"] == n_pos for c in candidates)
                ):
                    _best_sorted = sorted(
                        candidates,
                        key=lambda c: (-c["hits_per_pos"], c["unknown_face_positions"]),
                    )
                    _best_seed = _best_sorted[0]
                    # Pool hit frames from every candidate whose shift is
                    # within one chunk_len of the best shift — same local
                    # on-screen window. Candidates far from the best
                    # (e.g. a +2 chunk shift that found a DIFFERENT
                    # window) are excluded so their hits don't drag the
                    # center off the current cluster.
                    _seed_hit_frames: list[int] = []
                    for c in candidates:
                        if abs(c["shift"] - _best_seed["shift"]) > _chunk_len:
                            continue
                        for _i, _h in enumerate(c["pos_has_hit"]):
                            if _h:
                                _seed_hit_frames.append(c["sample_positions"][_i])
                    if _seed_hit_frames:
                        # Center of the hit frame range — min + max halved
                        # is more robust than the mean when hits cluster at
                        # the edges of the window.
                        _hit_center = (min(_seed_hit_frames) + max(_seed_hit_frames)) // 2
                        _original_center = start_frame + _chunk_len // 2
                        _raw_shift = _hit_center - _original_center
                        # Snap to nearest multiple of 8 for VAE alignment.
                        _centered_shift = int(round(_raw_shift / 8.0)) * 8
                        _existing_shifts = {c["shift"] for c in candidates}
                        if (
                            _centered_shift != 0
                            and _centered_shift not in _existing_shifts
                        ):
                            _ns = start_frame + _centered_shift
                            _ne = end_frame + _centered_shift
                            if first_frame <= _ns and _ne <= last_frame:
                                alt = _validate_at(_ns, _ne)
                                _alt_hits = [j for j, h in enumerate(alt["pos_has_hit"]) if h]
                                _alt_passes = (
                                    alt["hits_per_pos"] >= min_hits
                                    and alt["unknown_face_positions"] <= unknown_tol
                                )
                                _seed_tag = (
                                    f"centered on {len(_seed_hit_frames)} "
                                    f"pooled hit frames near shift "
                                    f"{_best_seed['shift']:+d} (center {_hit_center})"
                                )
                                if _alt_passes:
                                    logger.info(
                                        f"Chunk {chunk_idx}: {_centered_shift:+d} "
                                        f"{_seed_tag} candidate — hits "
                                        f"{alt['hits_per_pos']}/{n_pos} at {_alt_hits}, "
                                        f"unknowns {alt['unknown_face_positions']}/{n_pos}"
                                    )
                                    candidates.append({
                                        "shift": _centered_shift,
                                        "start": _ns,
                                        "end": _ne,
                                        **alt,
                                    })
                                else:
                                    _reasons = []
                                    if alt["hits_per_pos"] < min_hits:
                                        _reasons.append(
                                            f"hits {alt['hits_per_pos']}<{min_hits}"
                                        )
                                    if alt["unknown_face_positions"] > unknown_tol:
                                        _reasons.append(
                                            f"unknown {alt['unknown_face_positions']}>{unknown_tol}"
                                        )
                                    logger.info(
                                        f"Chunk {chunk_idx}: {_centered_shift:+d} "
                                        f"{_seed_tag} ineligible — {', '.join(_reasons)}"
                                    )

                if candidates:
                    # Best = max hits, tiebreak min unknowns, tiebreak shift
                    # magnitude ascending (stay closer to the original pool
                    # position when truly tied).
                    candidates.sort(
                        key=lambda c: (
                            -c["hits_per_pos"],
                            c["unknown_face_positions"],
                            abs(c["shift"]),
                        )
                    )
                    best = candidates[0]
                    if best["shift"] != 0:
                        logger.info(
                            f"Chunk {chunk_idx}: chose shift {best['shift']:+d} — "
                            f"hits {best['hits_per_pos']}/{n_pos}, "
                            f"unknowns {best['unknown_face_positions']}/{n_pos}"
                        )
                        start_frame = best["start"]
                        end_frame = best["end"]
                        sample_positions = best["sample_positions"]
                        matched_names = best["matched_names"]
                        hits_per_pos = best["hits_per_pos"]
                        face_anchor = best["face_anchor"]
                        unknown_face_positions = best["unknown_face_positions"]
                        pos_has_hit = best["pos_has_hit"]
                        # The shifted range crosses into neighbor pool
                        # entries' territory. Mark every chunk that the
                        # new range overlaps as consumed so we don't
                        # re-extract near-duplicate content. A shift of
                        # N chunk lengths touches up to ceil(N) neighbors
                        # in the shift direction.
                        _shift = best["shift"]
                        _direction = 1 if _shift > 0 else -1
                        # ceil(|shift| / chunk_len) = number of neighbors touched
                        _num_adj = (abs(_shift) + _chunk_len - 1) // _chunk_len
                        for _offset in range(1, _num_adj + 1):
                            _adj_ci = chunk_idx + _direction * _offset
                            if _adj_ci >= 0:
                                self._consumed_chunks.append(
                                    f"{video_path.stem}_chunk{_adj_ci:04d}.mp4"
                                )
                    else:
                        logger.info(
                            f"Chunk {chunk_idx}: original is best — "
                            f"hits {hits_per_pos}/{n_pos}, unknowns {unknown_face_positions}/{n_pos}"
                        )
                elif target_chunk_idx >= 0 and hit_indices:
                    logger.info(
                        f"Chunk {chunk_idx}: no eligible position found "
                        f"(floor {min_hits}/{n_pos} hits, tolerance {unknown_tol}/{n_pos} unknowns)"
                    )

                # Final gate checks (may have been satisfied by a shift).
                if hits_per_pos < min_hits:
                    logger.info(
                        f"Chunk {chunk_idx}: rejected — {hits_per_pos}/{len(sample_positions)} "
                        f"hits below required {min_hits}"
                    )
                    self._rejected_chunks.append({
                        "source_file": str(video_path),
                        "media_path": str(clips_dir / chunk_file),
                        "chunk_file": chunk_file,
                        "reason": "insufficient_character_presence",
                    })
                    if target_chunk_idx >= 0:
                        break
                    start_frame = end_frame
                    chunk_idx += 1
                    continue
                if unknown_face_positions > unknown_tol:
                    logger.info(
                        f"Chunk {chunk_idx}: rejected — unknown faces in "
                        f"{unknown_face_positions}/{len(sample_positions)} sample positions "
                        f"(tolerance: {unknown_tol} / {allow_unknown_faces_in})"
                    )
                    self._rejected_chunks.append({
                        "source_file": str(video_path),
                        "media_path": str(clips_dir / chunk_file),
                        "chunk_file": chunk_file,
                        "reason": "unknown_face",
                    })
                    if target_chunk_idx >= 0:
                        break
                    start_frame = end_frame
                    chunk_idx += 1
                    continue
                logger.info(f"Chunk {chunk_idx}: matched {', '.join(sorted(matched_names))}")
                if crop_mode in ("face_crop", "pan_and_scan"):
                    if face_anchor is not None:
                        _, face = face_anchor
                        if crop_mode == "pan_and_scan":
                            crop = _compute_pan_and_scan(*face, frame_w, frame_h, target_w, target_h)
                        else:
                            crop = _compute_face_crop(*face, frame_w, frame_h, target_w, target_h)
                    else:
                        # Non-human-only match (e.g. Chairry solo shot) — no
                        # face to anchor on, fall back to center crop.
                        crop = self._center_crop(frame_w, frame_h, target_w, target_h)
            elif face_detection:
                face_found = False
                for try_idx in sample_positions:
                    sample = self._read_frame(video_path, try_idx)
                    if sample is None:
                        continue
                    face = _detect_face_dnn(sample)
                    if face is not None:
                        matched_sample = sample
                        if crop_mode == "face_crop":
                            crop = _compute_face_crop(*face, frame_w, frame_h, target_w, target_h)
                        elif crop_mode == "pan_and_scan":
                            crop = _compute_pan_and_scan(*face, frame_w, frame_h, target_w, target_h)
                        face_found = True
                        break
                if not face_found:
                    logger.info(f"No face in chunk {chunk_idx} of {video_path.name}, skipping")
                    cf = f"{video_path.stem}_chunk{chunk_idx:04d}.mp4"
                    self._rejected_chunks.append({
                        "source_file": str(video_path),
                        "media_path": str(clips_dir / cf),
                        "chunk_file": cf,
                        "reason": "no_face",
                    })
                    if target_chunk_idx >= 0:
                        break
                    start_frame = end_frame
                    chunk_idx += 1
                    continue

                if target_embedding is not None and matched_sample is not None:
                    face = _detect_face_dnn(matched_sample)
                    if face is None or not self._check_face_match(matched_sample, face, target_embedding, face_similarity):
                        logger.info(f"Face doesn't match target in chunk {chunk_idx} of {video_path.name}, skipping")
                        cf = f"{video_path.stem}_chunk{chunk_idx:04d}.mp4"
                        self._rejected_chunks.append({
                            "source_file": str(video_path),
                            "media_path": str(clips_dir / cf),
                            "chunk_file": cf,
                            "reason": "face_mismatch",
                        })
                        if target_chunk_idx >= 0:
                            break
                        start_frame = end_frame
                        chunk_idx += 1
                        continue
            elif crop_mode in ("face_crop", "pan_and_scan"):
                # No face detection + face_crop/pan_and_scan: center crop
                crop = self._center_crop(frame_w, frame_h, target_w, target_h)

            out_path = clips_dir / f"{video_path.stem}_chunk{chunk_idx:04d}.mp4"
            start_time = start_frame / fps

            if not out_path.exists():
                num_source_frames = end_frame - start_frame
                clip_duration = num_source_frames / fps

                # Build video filter chain
                vf_parts = []
                if crop_mode != "full_frame":
                    cx, cy, cw, ch = crop
                    vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
                if use_fps_conversion:
                    vf_parts.append(f"fps={target_fps}")

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start_time:.4f}",
                    "-i", str(video_path),
                ]
                # When decimating fps, use -t (duration) instead of -frames:v
                # so ffmpeg reads all source frames before the fps filter drops them.
                if use_fps_conversion:
                    cmd += ["-t", f"{clip_duration:.4f}"]
                else:
                    cmd += ["-frames:v", str(num_source_frames)]
                if vf_parts:
                    cmd += ["-vf", ",".join(vf_parts)]
                cmd += [
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                ]
                # Force exactly target_frames in output to defeat fps-filter
                # rounding drift — source duration is padded above so this
                # always trims rather than coming up short.
                if use_fps_conversion:
                    cmd += ["-frames:v", str(target_frames)]
                if with_audio:
                    # Audio duration matches the exact video output length.
                    audio_duration = (
                        target_frames / target_fps if use_fps_conversion
                        else num_source_frames / fps
                    )
                    cmd += [
                        "-t", f"{audio_duration:.4f}",
                        "-c:a", "aac",
                        "-b:a", "128k",
                    ]
                else:
                    cmd += ["-an"]
                cmd += [str(out_path)]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"ffmpeg failed for chunk {chunk_idx} of {video_path.name}: {result.stderr}")
                    start_frame = end_frame
                    chunk_idx += 1
                    continue

            # --- Transcribe speech and fix word boundaries ---
            if transcribe_speech and out_path.exists():
                # Use the face-detected characters as a hint for speaker matching.
                _face_chars = set(matched_names) if matched_names else None
                tr = _transcribe_clip(out_path, voice_refs=voice_refs, face_chars=_face_chars)
                if tr and tr.get("hallucination"):
                    # No usable speech — quarantine clip (don't delete) so it
                    # can be inspected later for false positives.
                    logger.info(f"  No usable speech, quarantining clip: {out_path.name}")
                    q_path = self._quarantine_clip(out_path, clips_dir.parent, "speech_hallucination")
                    rej_record = {
                        "source_file": str(video_path),
                        "media_path": str(out_path),
                        "chunk_file": out_path.name,
                        "reason": "speech_hallucination",
                    }
                    if q_path is not None:
                        rej_record["quarantined_path"] = str(q_path)
                    self._rejected_chunks.append(rej_record)
                    start_frame = end_frame
                    chunk_idx += 1
                    continue
                if tr and tr["text"]:
                    # Check if the last word is cut off — if so, shift the
                    # clip start backward so the end lands in the gap before
                    # the cut word.  Frame count stays exactly the same.
                    if tr["words"]:
                        last_word = tr["words"][-1]
                        clip_dur = tr["duration"]
                        if clip_dur > 0 and (clip_dur - last_word["end"]) < _WORD_CUT_THRESHOLD:
                            # Find the gap before the cut word to land in
                            if len(tr["words"]) >= 2:
                                prev_word_end = tr["words"][-2]["end"]
                            else:
                                prev_word_end = 0.0
                            # Shift so clip ends at the midpoint of the gap
                            gap_mid = (prev_word_end + last_word["start"]) / 2.0
                            shift_seconds = clip_dur - gap_mid
                            shift_frames = int(round(shift_seconds * fps))
                            new_start = max(first_frame, start_frame - shift_frames)

                            if new_start < start_frame:
                                new_start_time = new_start / fps
                                logger.info(
                                    f"  Word '{last_word['word']}' cut at {last_word['end']:.2f}s "
                                    f"— shifting start back by {shift_frames} frames"
                                )
                                # Re-extract from shifted position, same frame count
                                vf_parts_re = []
                                if crop_mode != "full_frame" and crop is not None:
                                    cx, cy, cw, ch = crop
                                    vf_parts_re.append(f"crop={cw}:{ch}:{cx}:{cy}")
                                if use_fps_conversion:
                                    vf_parts_re.append(f"fps={target_fps}")

                                re_cmd = [
                                    "ffmpeg", "-y",
                                    "-ss", f"{new_start_time:.4f}",
                                    "-i", str(video_path),
                                ]
                                if use_fps_conversion:
                                    re_cmd += ["-t", f"{source_chunk_frames / fps:.4f}"]
                                else:
                                    re_cmd += ["-frames:v", str(source_chunk_frames)]
                                if vf_parts_re:
                                    re_cmd += ["-vf", ",".join(vf_parts_re)]
                                re_cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
                                # Same overshoot-then-trim safeguard as the main extraction.
                                if use_fps_conversion:
                                    re_cmd += ["-frames:v", str(target_frames)]
                                if with_audio:
                                    re_audio_duration = (
                                        target_frames / target_fps if use_fps_conversion
                                        else source_chunk_frames / fps
                                    )
                                    re_cmd += ["-t", f"{re_audio_duration:.4f}", "-c:a", "aac", "-b:a", "128k"]
                                else:
                                    re_cmd += ["-an"]
                                re_cmd += [str(out_path)]

                                re_result = subprocess.run(re_cmd, capture_output=True, text=True)
                                if re_result.returncode == 0:
                                    # Re-transcribe the shifted clip
                                    _face_chars = set(matched_names) if matched_names else None
                                    tr = _transcribe_clip(out_path, voice_refs=voice_refs, face_chars=_face_chars)

                    # Store transcript text to return to caller
                    if tr and tr["text"]:
                        clip_transcript = tr["text"]
                        logger.info(f"  Transcript: {clip_transcript[:80]}{'...' if len(clip_transcript) > 80 else ''}")
                    # Stash speaker-attributed segments on self so the caller
                    # can read them by clip path (parallels _clip_characters).
                    if tr and tr.get("segments"):
                        if not hasattr(self, "_clip_segments"):
                            self._clip_segments = {}
                        self._clip_segments[str(out_path)] = tr["segments"]

            clips.append((out_path, clip_transcript if transcribe_speech else None))
            # Store matched characters for balanced sampling
            if face_detection and character_refs and matched_names:
                self._clip_characters[str(out_path)] = sorted(matched_names)
            if out_path not in existing_chunks:
                new_clips_produced += 1
            logger.info(f"Extracted clip: {video_path.name} chunk {chunk_idx} -> {out_path.name}")

            start_frame = end_frame
            chunk_idx += 1

            # Stop early if we've produced enough new clips for this video
            if max_new_clips > 0 and new_clips_produced >= max_new_clips:
                break

        return clips

    def _read_frame(self, video_path: Path, frame_idx: int) -> np.ndarray | None:
        """Read a single frame from a video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _get_face_crop(
        self, frame: np.ndarray, target_w: int, target_h: int, face_detection: bool,
    ) -> tuple[int, int, int, int] | None:
        """Detect face and compute crop region. Returns (x, y, w, h) or None."""
        if not face_detection:
            h, w = frame.shape[:2]
            return self._center_crop(w, h, target_w, target_h)

        face = _detect_face_dnn(frame)
        if face is None:
            return None

        fx, fy, fw, fh = face
        frame_h, frame_w = frame.shape[:2]
        return _compute_face_crop(fx, fy, fw, fh, frame_w, frame_h, target_w, target_h)

    def _center_crop(
        self, frame_w: int, frame_h: int, target_w: int, target_h: int,
    ) -> tuple[int, int, int, int]:
        """Compute center crop matching target aspect ratio."""
        target_aspect = target_w / target_h
        frame_aspect = frame_w / frame_h

        if frame_aspect > target_aspect:
            # Frame is wider: crop width
            crop_h = frame_h
            crop_w = int(frame_h * target_aspect)
        else:
            # Frame is taller: crop height
            crop_w = frame_w
            crop_h = int(frame_w / target_aspect)

        crop_x = (frame_w - crop_w) // 2
        crop_y = (frame_h - crop_h) // 2
        return crop_x, crop_y, crop_w, crop_h

    def _caption_dataset_json(
        self,
        dataset_json_path: Path,
        clip_paths: list[Path],
        caption_mode: str,
        lora_trigger: str,
        ollama_url: str = "",
        ollama_model: str = "",
        target_face_b64: str | None = None,
        caption_style: str = "subject",
        cast_names: list[str] | None = None,
        cast_refs: list[tuple[str, str]] | None = None,
        location_refs: list[tuple[str, str]] | None = None,
        skip_id_pass: bool = False,
    ):
        """Caption entries in the dataset JSON. Reads existing JSON, skips
        already-captioned entries, saves incrementally after each new caption.
        If target_face_b64 is provided, LLM verifies the person matches."""
        import re

        with open(dataset_json_path) as f:
            entries = json.load(f)

        # Check if all entries already have captions (and none are marked mismatch)
        uncaptioned = [i for i, e in enumerate(entries) if not e.get("caption")]
        if not uncaptioned:
            logger.info("All clips already captioned, skipping")
            return

        logger.info(f"{len(uncaptioned)} clips need captioning, {len(entries) - len(uncaptioned)} already done")

        # Make sure the requested Ollama model is installed before we
        # start the loop.  If it's not, pull it (streaming progress to
        # the log).  Without this, every clip would 404 and get rejected.
        if caption_mode == "ollama":
            self._ensure_ollama_model(ollama_url, ollama_model)

        removed_count = 0
        # Persistent multi-turn conversation state for ollama captions.
        # The model's thinking from earlier clips carries forward.
        ollama_messages: list[dict] | None = None
        caption_first_time: float = 0.0  # gen time of first caption in session
        caption_gen_times: list[float] = []  # all gen times in session for averaging

        i = 0
        while i < len(entries):
            entry = entries[i]
            if entry.get("caption"):
                i += 1
                continue

            vf = Path(entry["media_path"])

            if caption_mode == "ollama":
                logger.info(f"[{i+1}/{len(entries)}] Captioning: {vf.name}")
                prep = self._prepare_clip_for_caption(
                    vf, ollama_url, ollama_model, lora_trigger,
                    target_face_b64=target_face_b64,
                    caption_style=caption_style,
                    cast_names=cast_names,
                    cast_refs=cast_refs,
                    location_refs=location_refs,
                    skip_id_pass=skip_id_pass,
                    expected_cast=entry.get("characters") or None,
                    transcript_segments=entry.get("transcript_segments"),
                )

                if prep is None:
                    # MISMATCH — reject this entry
                    logger.info(f"  LLM QC: MISMATCH — removing {vf.name}")
                    rejected_entry = entries.pop(i)
                    removed_count += 1
                    rejected_path = dataset_json_path.parent / "rejected.json"
                    rejected = []
                    if rejected_path.exists():
                        try:
                            with open(rejected_path) as rf:
                                rejected = json.load(rf)
                        except (json.JSONDecodeError, KeyError):
                            pass
                    rejected.append({
                        "media_path": str(vf),
                        "source_file": rejected_entry.get("source_file", ""),
                        "reason": "llm_mismatch",
                    })
                    with open(rejected_path, "w") as rf:
                        json.dump(rejected, rf, indent=2)
                    with open(dataset_json_path, "w") as f:
                        json.dump(entries, f, indent=2)
                    continue

                # Handle fallback (frame extraction failed)
                if "fallback_caption" in prep:
                    fc = prep["fallback_caption"]
                    if lora_trigger:
                        fc = f"{lora_trigger} {fc}"
                    entry["caption"] = fc
                    with open(dataset_json_path, "w") as f:
                        json.dump(entries, f, indent=2)
                    i += 1
                    continue

                # Caption via persistent multi-turn conversation
                caption, ollama_messages, cap_gen_time, token_count = self._caption_single_ollama(
                    prep, ollama_url, ollama_model, ollama_messages,
                )

                # Detect slowdown or context limit — reset session if either:
                # 1. Average gen time exceeds 40% of first caption's gen time
                # 2. Token count approaches 75% of the 128K context window
                CTX_THRESHOLD = int(131072 * 0.75)  # ~98K tokens
                reset_reason = None

                if caption_first_time == 0.0:
                    caption_first_time = cap_gen_time
                caption_gen_times.append(cap_gen_time)

                if len(caption_gen_times) >= 2 and caption_first_time > 0:
                    avg_gen = sum(caption_gen_times) / len(caption_gen_times)
                    if avg_gen > caption_first_time * 1.25:
                        reset_reason = f"slowdown: avg {avg_gen:.1f}s gen vs first {caption_first_time:.1f}s"

                if token_count > CTX_THRESHOLD:
                    reset_reason = f"context: {token_count}/{131072} tokens"

                if reset_reason:
                    logger.info(f"  Session reset ({reset_reason}) — starting new session")
                    ollama_messages = None
                    caption_first_time = 0.0
                    caption_gen_times.clear()

                entry["caption"] = caption
                with open(dataset_json_path, "w") as f:
                    json.dump(entries, f, indent=2)
                i += 1

            elif caption_mode == "auto_filename":
                name = vf.stem
                for suffix in ["_img"]:
                    name = name.removesuffix(suffix)
                name = re.sub(r'_chunk\d+$', '', name)
                caption = name.replace("_", " ").replace("-", " ")
                caption = " ".join(caption.split())
                if lora_trigger:
                    caption = f"{lora_trigger} {caption}"
                entry["caption"] = caption
                with open(dataset_json_path, "w") as f:
                    json.dump(entries, f, indent=2)
                i += 1
            else:
                caption = vf.stem
                if lora_trigger:
                    caption = f"{lora_trigger} {caption}"
                entry["caption"] = caption
                with open(dataset_json_path, "w") as f:
                    json.dump(entries, f, indent=2)
                i += 1

        if removed_count:
            logger.info(f"LLM QC removed {removed_count} mismatched clips")
        logger.info(f"Captioning complete for {len(entries)} entries")

    def _ensure_ollama_model(self, base_url: str, model: str) -> None:
        """Check whether the requested Ollama model is installed; if not,
        pull it.  Streams pull progress to the logger so the user can
        watch the download.  Raises RuntimeError on failure so the node
        aborts cleanly instead of marking every clip as MISMATCH."""
        import urllib.request
        import urllib.error

        base = base_url.rstrip("/")

        # Check what's installed via /api/tags
        try:
            req = urllib.request.Request(
                f"{base}/api/tags",
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
            installed = {m.get("name", "") for m in tags.get("models", [])}
            # Match either the bare name or with the :latest suffix.
            if model in installed or f"{model}:latest" in installed or any(
                n == model or n.startswith(f"{model}:") for n in installed
            ):
                logger.info(f"Ollama model '{model}' is installed")
                return
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Could not query Ollama at {base}: {e}. "
                "Make sure the Ollama server is running."
            ) from e

        logger.info(f"Ollama model '{model}' not found, pulling now...")
        try:
            pull_payload = json.dumps({"model": model, "stream": True}).encode("utf-8")
            pull_req = urllib.request.Request(
                f"{base}/api/pull",
                data=pull_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            last_status = ""
            with urllib.request.urlopen(pull_req, timeout=3600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    status = evt.get("status", "")
                    total = evt.get("total")
                    completed = evt.get("completed")
                    if total and completed is not None:
                        pct = (completed / total) * 100
                        msg = f"  pulling {status}: {pct:.1f}% ({completed/1e9:.2f}/{total/1e9:.2f} GB)"
                    else:
                        msg = f"  {status}" if status else ""
                    if msg and msg != last_status:
                        logger.info(msg)
                        last_status = msg
                    if evt.get("error"):
                        raise RuntimeError(f"Ollama pull error: {evt['error']}")
            logger.info(f"Ollama model '{model}' pulled successfully")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            raise RuntimeError(
                f"Failed to pull Ollama model '{model}': {e}. "
                f"Check that the tag exists at https://ollama.com/library"
            ) from e

    # ---- Caption prompt definitions (shared by prep + batch) ----
    _CAPTION_PROMPTS = {
        "subject": (
            "You are a video training caption generator for character/subject learning."
            " Describe what you see in a single detailed paragraph."
            " Describe in detail: lighting, color grading, background, environment, "
            "camera angle, pose, expression, and clothing."
            " Do NOT describe the subject's defining physical features (face shape, hair color/style, "
            "skin tone, eye color, body type) — the trigger word handles identity."
            " Be factual and specific. "
            "Do not use poetic language or speculation. Do not start with 'The image shows' or "
            "'In this frame'."
        ),
        "subject + style": (
            "You are a video training caption generator for character + style learning."
            " Describe what you see in a single detailed paragraph."
            " Describe: background, environment, camera angle, pose, expression, and clothing."
            " Do NOT describe the subject's defining physical features (face shape, hair color/style, "
            "skin tone, eye color, body type) — the trigger word handles identity."
            " Do NOT describe lighting, color grading, contrast, shadows, film grain, "
            "depth of field, or visual mood — the LoRA should learn the visual style."
            " If you recognize any other named characters in the scene besides the main subject, "
            "refer to them by name (e.g. 'standing next to <name>'). Only name characters you are "
            "confident about; do not guess."
            " Be factual and specific. "
            "Do not use poetic language or speculation. Do not start with 'The image shows' or "
            "'In this frame'."
        ),
        "style": (
            "You are a video training caption generator for visual style learning."
            " Describe what you see in a single detailed paragraph."
            " Describe the scene content: subjects, people, objects, actions, setting, "
            "and composition."
            " Do NOT describe the visual treatment — no lighting style, color grading, contrast, "
            "shadows, highlights, film grain, texture, depth of field, or visual mood."
            " The LoRA should learn all visual style characteristics."
            " Be factual and specific. "
            "Do not use poetic language or speculation. Do not start with 'The image shows' or "
            "'In this frame'."
        ),
        "motion": (
            "You are a video training caption generator for motion learning."
            " Describe what you see in a single detailed paragraph."
            " Describe: the subject's appearance, setting, lighting, and background."
            " Do NOT describe camera movement (pan, tilt, dolly, zoom, tracking), "
            "subject motion, speed, or direction of movement — the LoRA should learn motion."
            " Be factual and specific. "
            "Do not use poetic language or speculation. Do not start with 'The image shows' or "
            "'In this frame'."
        ),
        "general": (
            "You are a video training caption generator."
            " Describe what you see in a single detailed paragraph."
            " Give a balanced description covering: subjects, actions, setting, "
            "lighting, composition, and camera angle. Be factual and specific. "
            "Do not use poetic language or speculation. Do not start with 'The image shows' or "
            "'In this frame'."
        ),
        "multi_character": (
            "You are writing captions for LoRA training. The purpose of "
            "these captions is to describe ONLY things the base video model "
            "already understands — the LoRA will learn everything else from "
            "the visual data directly. The base model already knows what "
            "colors are, what objects are, what actions look like, what "
            "environments look like. It does NOT know these specific "
            "characters, this specific visual style, or this specific "
            "lighting/cinematography.\n\n"
            "IMPORTANT: Images labeled 'REF:' are reference photos for "
            "character recognition ONLY. NEVER describe anything from a "
            "REF image. NEVER include a character in your caption unless "
            "they actually appear in the numbered clip frames. If a REF "
            "character is not visible in the clip frames, ignore them "
            "completely.\n\n"
            "The same applies to locations: the identified location is a "
            "suggestion based on reference matching, not a guarantee. If "
            "the clip frames clearly show a different environment than the "
            "suggested location, describe what you actually see instead of "
            "forcing the suggested location name. If the clip contains "
            "cuts between a known location and an unknown one, use the "
            "known location name where it applies and describe the unknown "
            "location generically.\n\n"
            "Therefore:\n"
            "- ONLY describe what you see in the numbered clip frames.\n"
            "- DO describe: background, environment, camera angle, poses, "
            "interactions between characters, expressions, actions, "
            "clothing (the model knows these concepts).\n"
            "- DO refer to named characters strictly by their given name "
            "— never by physical description. Never put character names in "
            "quotes. Character reference names may include descriptive hints "
            "after the name; use ONLY the name portion when writing captions.\n"
            "- Do NOT describe any character's defining physical features "
            "(face shape, hair, skin tone, eye color, body type) — the "
            "character names handle identity and the LoRA learns appearance.\n"
            "- Do NOT describe lighting, color grading, contrast, shadows, "
            "film grain, depth of field, or visual mood — the LoRA should "
            "learn the visual style.\n\n"
            "Write a single detailed paragraph. Be factual and specific. "
            "Do not use poetic language or speculation. Do not start with "
            "'The image shows' or 'In this frame'.\n\n"
            "You will be given clips one at a time. Apply the same rules "
            "consistently to each clip."
        ),
    }

    def _prepare_clip_for_caption(
        self, clip_path: Path, ollama_url: str, ollama_model: str, lora_trigger: str,
        target_face_b64: str | None = None, caption_style: str = "subject",
        cast_names: list[str] | None = None,
        cast_refs: list[tuple[str, str]] | None = None,
        location_refs: list[tuple[str, str]] | None = None,
        skip_id_pass: bool = False,
        expected_cast: list[str] | None = None,
        transcript_segments: list[dict] | None = None,
    ) -> dict | None:
        """Prepare a clip for captioning: extract frames, run cast ID and
        location ID.  Returns a dict with all info needed for the caption
        call, or None if the clip should be rejected (MISMATCH).
        If skip_id_pass=True, skips cast/location ID and sends ALL refs
        directly to the captioner."""
        import base64

        frames = self._extract_caption_frames(clip_path, num_frames=5)
        if not frames:
            single = self._extract_caption_frame(clip_path)
            if single is None:
                logger.warning(f"Could not extract frame from {clip_path.name}, using filename")
                return {"fallback_caption": clip_path.stem}
            frames = [single]

        b64_images: list[str] = []
        for frame_idx, fr in enumerate(frames, start=1):
            label = f"Frame {frame_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(fr, (0, 0), (tw + 8, th + 10), (0, 0, 0), -1)
            cv2.putText(fr, label, (4, th + 6), font, font_scale, (255, 255, 255), thickness)
            ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                b64_images.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        if not b64_images:
            return {"fallback_caption": clip_path.stem}

        b64_image = b64_images[len(b64_images) // 2]
        base = ollama_url.rstrip("/")

        if skip_id_pass:
            # --- Skip ID: send ALL refs + frames, let captioner do everything ---
            trigger_instruction = ""
            if lora_trigger:
                trigger_instruction = (
                    f" Always refer to the main subject as '{lora_trigger}'."
                    f" Start the caption with '{lora_trigger}'."
                )

            system_prompt = self._CAPTION_PROMPTS["multi_character"] + trigger_instruction

            # Label and collect all cast refs
            all_cast_refs = []
            if cast_refs:
                all_cast_refs = [
                    (name, self._label_ref_image(b64, name))
                    for name, b64 in cast_refs
                ]
            # Label and collect all location refs
            all_loc_refs = []
            if location_refs:
                all_loc_refs = [
                    (name, self._label_ref_image(b64, name))
                    for name, b64 in location_refs
                ]

            num_cast = len(all_cast_refs)
            num_loc = len(all_loc_refs)
            num_frames = len(b64_images)

            # Build ref intro
            ref_lines = []
            img_idx = 1
            for name, _ in all_cast_refs:
                ref_lines.append(f"Reference image {img_idx}: character '{name}'.")
                img_idx += 1
            for name, _ in all_loc_refs:
                ref_lines.append(f"Reference image {img_idx}: location '{name}'.")
                img_idx += 1
            ref_block = "\n".join(ref_lines)
            total_refs = num_cast + num_loc

            frame_listing = ", ".join(f"Frame {i + 1}" for i in range(num_frames))
            skip_hint_block = ""
            if expected_cast:
                skip_hint_block = (
                    f"HINT: a face-detection pre-pass flagged these characters "
                    f"as likely present: {', '.join(expected_cast)}. Treat this "
                    f"as a prior — confirm each one visually, and still check "
                    f"every other reference in case the pre-pass missed someone.\n\n"
                )
            user_content = (
                f"You will first see {total_refs} labeled reference images "
                f"for character and location identification:\n\n"
                f"{ref_block}\n\n"
                f"After the references, you will see {num_frames} frames "
                f"from a video clip in chronological order: {frame_listing}. "
                f"Frame 1 is the opening shot and the last frame is the end.\n\n"
                f"{skip_hint_block}"
                f"YOUR TASK:\n"
                f"1. Identify which characters from the reference images "
                f"actually appear in the clip frames. Only name characters "
                f"you can confidently match — if a reference doesn't match "
                f"anyone in the clip, ignore it completely.\n"
                f"2. Identify which location from the reference images best "
                f"matches the clip's setting. If none match, describe the "
                f"location generically.\n"
                f"3. Write ONE detailed caption paragraph describing the clip.\n\n"
                f"IMPORTANT: The reference images are ONLY for identification. "
                f"NEVER describe anything from a reference image. Only describe "
                f"what you see in the numbered clip frames.\n\n"
                f"Write ONE caption that describes the whole clip as a single "
                f"scene. If the clip cuts between shots or characters, include "
                f"what appears across the different frames. "
                f"Do NOT mention frame numbers in the caption.\n\n"
                "NAMING RULES:\n"
                "- Name only characters you have confidently matched to a "
                "reference image. Use their given name without quotes.\n"
                "- Describe any unmatched characters generically "
                "(e.g. 'a child', 'a woman in a dress').\n"
                "- Wrong names are worse than no names. When in doubt, "
                "describe without naming.\n"
                "- Character reference names may include descriptive hints "
                "after the name; use ONLY the name portion in captions."
            )

            images_payload = (
                [b for _, b in all_cast_refs]
                + [b for _, b in all_loc_refs]
                + list(b64_images)
            )
        else:
            # --- Standard path: separate ID passes then caption ---

            # --- Face verification ---
            if target_face_b64:
                if self._ollama_verify_face(base, ollama_model, target_face_b64, b64_image) == "MISMATCH":
                    return None

            # --- Cast identification ---
            confirmed_cast: list[str] | None = None
            if cast_names and cast_refs:
                confirmed_cast = self._ollama_identify_cast(
                    base, ollama_model, cast_refs, b64_images,
                    expected_cast=expected_cast,
                )
                if not confirmed_cast:
                    confirmed_cast = None

            # --- Location identification ---
            identified_location: str | None = None
            if location_refs:
                _loc_t0 = _t.time()
                logger.info(f"  Identifying location...")
                identified_location = self._ollama_identify_location(base, ollama_model, location_refs, b64_images)
                logger.info(f"  Location: {identified_location or 'NONE'} ({_t.time() - _loc_t0:.1f}s)")

            # --- Build user message + images for caption ---
            trigger_instruction = ""
            if lora_trigger:
                trigger_instruction = (
                    f" Always refer to the main subject as '{lora_trigger}'."
                    f" Start the caption with '{lora_trigger}'."
                )

            effective_style = "multi_character" if confirmed_cast else caption_style
            system_prompt = self._CAPTION_PROMPTS.get(effective_style, self._CAPTION_PROMPTS["subject"]) + trigger_instruction

            num_clip_frames = len(b64_images)
            frame_listing = ", ".join(f"Frame {i + 1}" for i in range(num_clip_frames))
            user_content = (
                f"You will see {num_clip_frames} frames sampled from a single "
                f"short video clip, in chronological order: {frame_listing}. "
                "Frame 1 is the opening shot and the last frame is the end. "
                "Write ONE caption that describes the whole clip as a single "
                "scene. If the clip cuts between shots or characters, include "
                "what appears across the different frames. "
                "Do NOT mention frame numbers in the caption — the labels are "
                "only for your reference to understand temporal order."
            )
            images_payload: list[str] = list(b64_images)

            if confirmed_cast:
                ref_by_name = {n.lower(): b for n, b in (cast_refs or [])}
                confirmed_refs: list[tuple[str, str]] = []
                for name in confirmed_cast:
                    img = ref_by_name.get(name.lower())
                    if img is not None:
                        confirmed_refs.append((name, img))

                names_csv = ", ".join(confirmed_cast)
                confirmed_refs = [
                    (name, self._label_ref_image(b64, name))
                    for name, b64 in confirmed_refs
                ]
                ref_intro_lines = []
                for i, (name, _) in enumerate(confirmed_refs, start=1):
                    ref_intro_lines.append(f"Reference image {i}: this is {name}.")
                ref_block = "\n".join(ref_intro_lines)
                num_refs = len(confirmed_refs)
                num_frames = len(b64_images)

                user_content = (
                    f"Confirmed characters visible in this clip: {names_csv}.\n\n"
                    + (
                        (
                            f"You will first see {num_refs} labeled reference "
                            f"images, one per confirmed character:\n\n"
                            f"{ref_block}\n\n"
                            f"After the reference images, you will see "
                            f"{num_frames} frames from the clip in chronological "
                            f"order (Frame 1 through Frame {num_frames}). "
                            f"Frame 1 is the opening shot.\n\n"
                            f"IMPORTANT: Only the numbered frames (Frame 1 "
                            f"through Frame {num_frames}) are the actual clip "
                            f"to caption. The reference images are ONLY for "
                            f"identifying who is who — do NOT describe or "
                            f"include the reference images in your caption.\n\n"
                        ) if confirmed_refs else ""
                    )
                    + user_content
                    + "\n\nNAMING RULES — READ CAREFULLY:\n"
                    "0. BEFORE writing your caption, double-check the confirmed "
                    "character list against what you actually see in the clip "
                    "frames. Compare each confirmed name's reference image to "
                    "the figures in the clip. If a confirmed name does NOT "
                    "actually match any living, active character in the clip "
                    "(e.g. it was matched to a statue, toy, decoration, or "
                    "the wrong person), DROP that name and describe the figure "
                    "generically instead. The confirmed list is a suggestion, "
                    "not an obligation — accuracy matters more.\n"
                    "1. For characters you have verified are truly present, "
                    "refer to them by their given names without quotes — never 'the subject', "
                    "'the man', 'the person', or any physical description. "
                    "Use the labeled reference images to correctly match each "
                    "name to its character.\n"
                    "2. If you recognize a character from previous clips in "
                    "this conversation who is NOT on the confirmed list, you "
                    "MAY name them — continuity is valuable.\n"
                    "3. Any OTHER character you do NOT recognize MUST be "
                    "described generically (e.g. 'a child', 'a musician', "
                    "'a group of kids', 'a person in a costume', 'a woman "
                    "at the door'). NEVER invent or guess a name.\n"
                    "4. Do NOT fabricate names. Only use names from the "
                    "confirmed list or from characters you have already "
                    "identified in previous clips.\n"
                    "4. Wrong names are worse than no names. When in doubt, "
                    "describe without naming."
                )
                images_payload = [b for _, b in confirmed_refs] + list(b64_images)

            if identified_location:
                user_content = (
                    f"The setting of this clip has been identified as "
                    f"'{identified_location}'. When describing the environment "
                    f"or location, refer to it by that exact name.\n\n"
                    + user_content
                )

        # Voice-attributed dialogue: when speaker diarization produced per-line
        # speaker tags, give them to the captioner as DIALOGUE CONTEXT and
        # instruct it to weave the lines naturally into the prose.
        if transcript_segments:
            dialogue_lines = []
            for s in transcript_segments:
                sp = _format_speaker_display(s.get("speaker", "unknown"))
                line = (s.get("text") or "").strip().rstrip(".!?")
                if line:
                    dialogue_lines.append(f'- {sp}: "{line}"')
            if dialogue_lines:
                dialogue_block = (
                    "DIALOGUE CONTEXT (detected speech in this clip, with speaker "
                    "attribution from voice analysis — treat as ground truth):\n"
                    + "\n".join(dialogue_lines)
                    + "\n\nWeave these lines naturally into the caption prose. "
                    "Attribute each line to the correct speaker by name and "
                    "render the dialogue as part of the action — e.g. "
                    "\"<speaker> stands by the window and says, '<line>'\". "
                    "Do NOT list the lines separately at the end. If a line is "
                    "tagged as 'Someone' (unmatched speaker), describe it "
                    "generically (e.g. \"a voice off-screen says, '<line>'\")."
                )
                user_content = user_content + "\n\n" + dialogue_block
                system_prompt = system_prompt + (
                    " When DIALOGUE CONTEXT is provided in the user message, "
                    "incorporate every line into the caption inline, attributed "
                    "to the named speaker. The dialogue is ground-truth audio "
                    "from the clip — never invent or omit lines."
                )

        return {
            "system_prompt": system_prompt,
            "user_content": user_content,
            "images_payload": images_payload,
            "clip_name": clip_path.name,
            "lora_trigger": lora_trigger,
        }

    def _caption_single_ollama(
        self, prep: dict, ollama_url: str, ollama_model: str,
        conversation_messages: list[dict] | None,
    ) -> tuple[str, list[dict], float, int]:
        """Caption a single clip, continuing a persistent multi-turn
        conversation.  Previous turns keep their text (captions +
        thinking context) but images are stripped so only the current
        clip's frames are sent/processed.
        Returns (caption, updated_messages, caption_only_time, token_count).
        caption_only_time excludes thinking — only measures generation."""
        import urllib.request
        import urllib.error
        import sys as _sys
        import time as _t
        import re

        base = ollama_url.rstrip("/")
        _cap_t0 = _t.time()
        logger.info(f"  Generating caption for {prep['clip_name']}...")

        # Start or continue conversation
        if conversation_messages is None:
            messages = [
                {"role": "system", "content": prep["system_prompt"]},
            ]
            is_continuation = False
        else:
            messages = conversation_messages
            is_continuation = True

        content = prep["user_content"]
        if is_continuation:
            content = (
                "--- NEXT CLIP ---\n"
                "Apply the same captioning rules as before.\n\n"
                + content
            )

        messages.append({
            "role": "user",
            "content": content,
            "images": prep["images_payload"],
        })

        # Strip images from all previous turns — only the current
        # (last) user message keeps its images. Text context
        # (captions, thinking) carries forward for free.
        send_messages = []
        for msg in messages:
            if "images" in msg and msg is not messages[-1]:
                send_messages.append({k: v for k, v in msg.items() if k != "images"})
            else:
                send_messages.append(msg)

        payload = json.dumps({
            "model": ollama_model,
            "messages": send_messages,
            "stream": True,
            "think": True,
            "keep_alive": "10m",
            "options": {"num_ctx": 131072},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            thinking_started = False
            content_started = False
            caption_parts: list[str] = []
            token_count = 0
            _think_t0 = 0.0  # track thinking duration to subtract
            _think_total = 0.0
            with urllib.request.urlopen(req, timeout=1800) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = evt.get("message", {})
                    think_chunk = msg.get("thinking", "")
                    content_chunk = msg.get("content", "")
                    if think_chunk:
                        if not thinking_started:
                            _sys.stderr.write("  [thinking] ")
                            thinking_started = True
                            _think_t0 = _t.time()
                        _sys.stderr.write(think_chunk)
                        _sys.stderr.flush()
                    if content_chunk:
                        if thinking_started and not content_started:
                            # Thinking just ended — record duration
                            _think_total = _t.time() - _think_t0
                            _sys.stderr.write("\n  [caption] ")
                        elif not content_started:
                            _sys.stderr.write("  [caption] ")
                        content_started = True
                        _sys.stderr.write(content_chunk)
                        _sys.stderr.flush()
                        caption_parts.append(content_chunk)
                    if evt.get("done"):
                        _sys.stderr.write("\n")
                        _sys.stderr.flush()
                        token_count = evt.get("prompt_eval_count", 0) + evt.get("eval_count", 0)
                        # If it was still thinking when done fired
                        if thinking_started and not content_started:
                            _think_total = _t.time() - _think_t0
                        break

            caption = "".join(caption_parts).strip()
            caption = re.sub(r"<think>.*?</think>", "", caption, flags=re.DOTALL)
            caption = re.sub(r"<think>.*", "", caption, flags=re.DOTALL)
            caption = caption.strip()

            if not caption:
                caption = f"{prep.get('lora_trigger', '')} {prep['clip_name']}".strip()

            total_time = _t.time() - _cap_t0
            work_time = total_time - _think_total  # total minus thinking
            logger.info(f"  Caption done ({total_time:.1f}s total, {_think_total:.1f}s think, {work_time:.1f}s work, {token_count} tokens)")
            messages.append({"role": "assistant", "content": caption})
            return caption, messages, work_time, token_count

        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.error(f"Ollama captioning failed for {prep['clip_name']}: {e}")
            fallback = f"{prep.get('lora_trigger', '')} {prep['clip_name']}".strip()
            messages.append({"role": "assistant", "content": fallback})
            return fallback, messages, 0.0, 0

    @staticmethod
    def _label_ref_image(b64: str, label: str) -> str:
        """Burn a 'REF: label' tag onto the top-left of a base64 JPEG so
        the vision model can visually distinguish reference images from
        numbered clip frames."""
        import base64 as _b64
        raw = _b64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return b64
        tag = f"REF: {label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(tag, font, font_scale, thickness)
        cv2.rectangle(img, (0, 0), (tw + 8, th + 10), (0, 0, 0), -1)
        cv2.putText(img, tag, (4, th + 6), font, font_scale, (255, 255, 255), thickness)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return _b64.b64encode(buf.tobytes()).decode("utf-8") if ok else b64

    def _ollama_identify_cast(
        self, base_url: str, model: str,
        cast_refs: list[tuple[str, str]],
        clip_image_b64s: list[str],
        expected_cast: list[str] | None = None,
    ) -> list[str]:
        """Ask Gemma to identify which referenced characters are visible in
        the clip.  We give Gemma the actual reference images labeled by name
        first, then the clip frames, so it has visual ground truth for each
        character instead of guessing from text names alone.

        cast_refs: list of (name, base64_image) pairs — one per character.
        clip_image_b64s: frames sampled from the clip being captioned.

        Returns a subset of cast names (possibly empty).  An empty list
        means the clip should be rejected as a false positive."""
        import urllib.request
        import urllib.error

        cast_names = [n for n, _ in cast_refs]
        num_refs = len(cast_refs)
        num_frames = len(clip_image_b64s)

        # Label reference images with "REF: name" and clip frames already
        # have "Frame N" burned on, so the model can visually distinguish them.
        labeled_cast_refs = [
            (name, self._label_ref_image(b64, name))
            for name, b64 in cast_refs
        ]
        ref_lines = []
        for i, (name, _) in enumerate(labeled_cast_refs, start=1):
            ref_lines.append(f"Reference image {i}: this is {name}.")
        ref_block = "\n".join(ref_lines)

        hint_block = ""
        if expected_cast:
            hint_names = ", ".join(expected_cast)
            hint_block = (
                f"HINT: a face-detection pre-pass flagged these characters as "
                f"likely present in this clip: {hint_names}. Treat this as a "
                f"prior, not a guarantee — confirm each one visually against "
                f"the reference images before naming them, and still check "
                f"EVERY other reference character too in case the pre-pass "
                f"missed someone.\n\n"
            )

        user_text = (
            f"First, {num_refs} reference images of known characters "
            f"(labeled 'REF:'):\n\n"
            f"{ref_block}\n\n"
            f"Then, {num_frames} frames from a video clip (labeled "
            f"'Frame 1' through 'Frame {num_frames}').\n\n"
            f"{hint_block}"
            "TASK: Go through EVERY reference character one by one and "
            "decide if they appear in the clip. List EVERY character you "
            "are confident appears — not just the first or most prominent "
            "one. A clip can contain many characters at once.\n\n"
            "Rules:\n"
            "- ONLY count a character if they are a real, active participant "
            "in the scene — moving, speaking, or interacting. Statues, "
            "figurines, toys, dolls, posters, photos, costumes on racks, "
            "decorations, or any other inanimate representation do NOT "
            "count, even if they resemble a known character.\n"
            "- Match primarily by face. If you cannot confidently match a "
            "face but the outfit is identical to a reference character, "
            "assume it is that character.\n"
            "- Each name can only appear ONCE. Do not assign the same name "
            "to two different figures in the clip.\n"
            "- Ignore characters not in the reference list.\n"
            "- When unsure about a character, leave them out. A missing "
            "name is far better than a wrong name.\n\n"
            "Respond with a comma-separated list of ALL matching names, "
            "spelled exactly as given. If none match, respond NONE. "
            "No other text."
        )

        # Image order: labeled references first, then clip frames.
        all_images = [b64 for _, b64 in labeled_cast_refs] + list(clip_image_b64s)

        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict visual cast identification system. "
                        "You will be shown labeled reference images of "
                        "characters, followed by frames from a video clip. "
                        "You must identify which of those specific "
                        "reference characters appear in the clip by visual "
                        "comparison. Do not guess based on names alone — "
                        "always compare to the reference images. If a "
                        "character in the clip does not visually match any "
                        "reference, do not assign them a name."
                    ),
                },
                {
                    "role": "user",
                    "content": user_text,
                    "images": all_images,
                },
            ],
            "stream": False,
            "keep_alive": "10m",
            "think": False,
            "options": {"num_ctx": 131072, "num_predict": 256},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_answer = result.get("message", {}).get("content", "")
                # If response is empty, log diagnostic fields from Ollama
                # (done_reason, prompt_eval_count, etc.) so we can see why.
                if not raw_answer.strip():
                    diag = {
                        k: result.get(k) for k in (
                            "done", "done_reason", "total_duration",
                            "prompt_eval_count", "eval_count", "error",
                        ) if k in result
                    }
                    logger.warning(f"  Cast ID EMPTY response. Diag: {diag}")
                answer = raw_answer.strip()
                import re
                # Strip common wrappers any model might add.
                answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
                answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)
                # Strip markdown formatting (**bold**, *italic*, `code`,
                # _underscore_) so the token regex can match cleanly.
                answer = re.sub(r"[*_`]+", "", answer)
                # Strip JSON/list brackets so names inside arrays match.
                answer = answer.replace("[", " ").replace("]", " ")
                answer = answer.replace("{", " ").replace("}", " ")
                answer = answer.replace('"', " ").replace("'", " ")
                answer = answer.strip()
                logger.info(f"  Cast ID raw response: {answer!r}")

                # Universal "no match" detection: if the model explicitly
                # says NONE or declares no match, return empty.
                ans_upper = answer.upper()
                if not answer or ans_upper == "NONE":
                    return []
                none_phrases = (
                    "NONE OF THE", "NO CHARACTERS", "NOT PRESENT",
                    "NOT VISIBLE", "CANNOT IDENTIFY", "UNABLE TO",
                    "NO MATCH", "I DO NOT SEE", "I DON'T SEE",
                )
                if any(p in ans_upper for p in none_phrases):
                    return []

                # Parse the response: whole-word match each known cast
                # name against the full response.  Works regardless of
                # format — CSV, prose, bullet list, numbered list, JSON,
                # yaml, markdown, or any mix.  Much more forgiving than
                # splitting on commas.
                answer_lower = answer.lower()
                cast_lower = {n.lower(): n for n in cast_names}
                confirmed: list[str] = []
                for key, canonical in cast_lower.items():
                    pattern = r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])"
                    if re.search(pattern, answer_lower) and canonical not in confirmed:
                        confirmed.append(canonical)
                if not confirmed:
                    logger.info(f"  No known cast names found in response")
                return confirmed
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.warning(f"Cast identification failed: {e}, rejecting clip")
            return []

    def _ollama_identify_location(
        self, base_url: str, model: str,
        location_refs: list[tuple[str, str]],
        clip_image_b64s: list[str],
    ) -> str | None:
        """Ask Gemma which labeled location the clip takes place in, by
        visually comparing clip frames to the reference images.  Returns
        the matched location name, or None if no confident match.  Soft
        — never rejects a clip."""
        import urllib.request
        import urllib.error

        loc_names = [n for n, _ in location_refs]
        num_refs = len(location_refs)
        num_frames = len(clip_image_b64s)

        labeled_loc_refs = [
            (name, self._label_ref_image(b64, name))
            for name, b64 in location_refs
        ]
        ref_lines = []
        for i, (name, _) in enumerate(labeled_loc_refs, start=1):
            ref_lines.append(f"Reference image {i}: this is the {name}.")
        ref_block = "\n".join(ref_lines)

        user_text = (
            f"First, {num_refs} reference images of known locations "
            f"(labeled 'REF:'):\n\n"
            f"{ref_block}\n\n"
            f"Then, {num_frames} frames from a video clip (labeled "
            f"'Frame 1' through 'Frame {num_frames}').\n\n"
            "Which ONE location does the clip take place in? Pay close "
            "attention to distinguishing details: wall colors and patterns, "
            "doors, windows, furniture, props, floor patterns, shelving, "
            "and overall room layout. Compare these specific details against "
            "the reference images. If the clip clearly matches one of the "
            "reference locations, respond with that name. Otherwise respond "
            "NONE.\n\n"
            "Respond with the single name or NONE. No other text."
        )

        all_images = [b64 for _, b64 in labeled_loc_refs] + list(clip_image_b64s)

        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict visual location identification "
                        "system. You will be shown labeled reference "
                        "images of sets, followed by frames from a video "
                        "clip. You must identify which of those specific "
                        "reference locations the clip takes place in by "
                        "visual comparison. Do not guess based on names "
                        "alone — always compare to the reference images."
                    ),
                },
                {
                    "role": "user",
                    "content": user_text,
                    "images": all_images,
                },
            ],
            "stream": False,
            "keep_alive": "10m",
            "think": False,
            "options": {"num_ctx": 131072, "num_predict": 64},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                answer = result.get("message", {}).get("content", "").strip()
                import re
                answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
                answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)
                answer = re.sub(r"[*_`]+", "", answer)
                answer = answer.replace("[", " ").replace("]", " ")
                answer = answer.replace("{", " ").replace("}", " ")
                answer = answer.replace('"', " ").replace("'", " ")
                answer = answer.strip().strip(".").strip()
                logger.info(f"  Location ID raw response: {answer!r}")

                ans_upper = answer.upper()
                if not answer or ans_upper == "NONE":
                    return None
                none_phrases = (
                    "NONE OF THE", "NO MATCH", "NOT A MATCH",
                    "CANNOT IDENTIFY", "UNABLE TO", "I DO NOT",
                    "I DON'T",
                )
                if any(p in ans_upper for p in none_phrases):
                    return None

                # Whole-word match each known location name against the
                # response.  Format-agnostic: handles prose, bullets,
                # JSON, or a single name.
                answer_lower = answer.lower()
                for name in loc_names:
                    pattern = r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?![a-z0-9])"
                    if re.search(pattern, answer_lower):
                        return name
                return None
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.warning(f"Location identification failed: {e}, skipping")
            return None

    def _ollama_verify_face(
        self, base_url: str, model: str, ref_b64: str, clip_b64: str,
    ) -> str:
        """Ask Ollama if two images show the same person. Returns 'MATCH' or 'MISMATCH'."""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a face verification system. You will receive two images. "
                        "Determine if the primary person in both images is the same individual. "
                        "Respond with ONLY the word MATCH or MISMATCH. No other text."
                    ),
                },
                {
                    "role": "user",
                    "content": "Is the person in the second image the same person as in the first image?",
                    "images": [ref_b64, clip_b64],
                },
            ],
            "stream": False,
            "keep_alive": "10m",
            "think": False,
            "options": {"num_ctx": 131072, "num_predict": 64},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                answer = result.get("message", {}).get("content", "").strip().upper()
                import re
                answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
                answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)
                answer = answer.strip().upper()
                if "MISMATCH" in answer:
                    return "MISMATCH"
                return "MATCH"
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.warning(f"Face verification failed: {e}, assuming match")
            return "MATCH"

    # Captioner frame extraction pulls in from each edge by this many
    # frames before sampling. The seek-and-shift extractor produces
    # clean takes but can leave a frame or two of neighbouring-scene
    # content at the extreme head / tail. The LLM will happily describe
    # whatever it sees even if it only exists for one frame; skipping
    # the edges keeps the caption tied to the main on-screen content.
    # Clamped at 25% of the clip length so short clips still have a
    # usable sample range.
    _CAPTION_EDGE_MARGIN = 10

    @classmethod
    def _caption_frame_range(cls, total: int) -> tuple[int, int]:
        """Return (lo, hi) inclusive frame indices to sample for captions."""
        safe = min(cls._CAPTION_EDGE_MARGIN, max(0, (total - 1) // 4))
        lo = safe
        hi = max(lo, total - 1 - safe)
        return lo, hi

    def _extract_caption_frame(self, clip_path: Path) -> np.ndarray | None:
        """Extract the best frame from a clip for captioning.
        Tries middle frame first, then samples others looking for a face.
        Skips the first/last _CAPTION_EDGE_MARGIN frames so transition
        artifacts at the clip edges don't end up in the caption."""
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        lo, hi = self._caption_frame_range(total)
        mid = (lo + hi) // 2
        q1 = (lo + mid) // 2
        q3 = (mid + hi) // 2
        # Sample positions: middle, 1/4 in, 3/4 in, safe start, safe end
        candidates = [mid, q1, q3, lo, hi]

        for idx in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            # If face detection is available, prefer frames with a visible face
            face = _detect_face_dnn(frame)
            if face is not None:
                cap.release()
                return frame

        # Fallback: return middle frame even without a face
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _extract_caption_frames(self, clip_path: Path, num_frames: int = 5) -> list[np.ndarray]:
        """Extract N evenly-spaced frames from a clip for multi-frame captioning.
        This lets the LLM see shot changes and secondary characters that may only
        appear in part of the clip. Sample range is inset from each end by
        _CAPTION_EDGE_MARGIN frames so neighbouring-scene transitions at the
        clip boundaries don't leak into the caption. Returns an empty list on
        failure."""
        # Handle image files directly
        if clip_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"):
            frame = cv2.imread(str(clip_path))
            return [frame] if frame is not None else []
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        num_frames = max(1, num_frames)
        lo, hi = self._caption_frame_range(total)
        span = hi - lo + 1
        if span <= num_frames:
            indices = list(range(lo, hi + 1))
        elif num_frames == 1:
            indices = [(lo + hi) // 2]
        else:
            # Evenly spaced across the SAFE range, inclusive of lo and hi.
            indices = [lo + int(i * (hi - lo) / (num_frames - 1)) for i in range(num_frames)]

        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames
