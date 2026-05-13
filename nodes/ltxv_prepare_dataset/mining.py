"""Clip extraction / mining: turns source videos into the dataset's
ingredients (clips, latents-eligible files, transcripts, character-tagged
entries).

Calls into face, audio, and status modules. Stateful helpers
(`_clip_characters`, `_clip_segments`) live as module globals here
because they bridge data between phases of clip processing.
"""
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .dataset_io import normalize_loaded_entries
from .face import (
    analyze_frame,
    compute_face_crop,
    compute_pan_and_scan,
    detect_all_faces_dnn,
    detect_face_dnn,
    get_face_app,
    get_face_embedding,
    has_unknown_face,
    match_face,
)
from .audio import transcribe_clip
from .status import emit_prepper_status

logger = logging.getLogger(__name__)

# File extensions accepted by the media scan. Mirrors the SOURCE constants
# (mining is the only place that reads these from this module).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}

# Audio + video extensions accepted for voice references.  Video files
# are handled the same way as audio files — _isolate_vocals already runs
# ffmpeg to pull audio out of any container, then Demucs isolates vocals.
_VOICE_REF_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
}

# Dominance filter: a character is recorded as present in a chunk only if
# they were seen in at least _CHAR_DOMINANCE_RATIO * max_count positions.
# Cameos at the fringe of another character's shot are filtered out of
# the recorded label.
_CHAR_DOMINANCE_RATIO = 0.5

# Whisper word-boundary heuristic: if the last word ends within this many
# seconds of the clip's end, it's likely cut and we should shift the clip.
_WORD_CUT_THRESHOLD = 0.3

# Cross-phase state populated during process_video / process_image,
# read by the orchestrator after mining completes.
#
# `_clip_characters`: {clip_path_str: sorted list of matched character triggers}
# `_clip_segments`:  {clip_path_str: list of speaker-attributed transcript segments}
#
# Module-level (rather than per-call return values) because they bridge
# data across many process_video calls and into the captioning phase.
_clip_characters: dict[str, list[str]] = {}
_clip_segments: dict[str, list] = {}

# Process-wide buffers populated by process_video and drained by the
# orchestrator after each call. They mirror the per-instance attributes
# the original RSLTXVPrepareDataset class held:
#   * _rejected_chunks: chunks rejected for content reasons (no face,
#     unknown face, hallucinated speech, etc.) — flushed into rejected.json.
#   * _consumed_chunks: chunks absorbed into a shifted extraction's
#     territory so they're skipped in-memory but NOT persisted as rejected.
_rejected_chunks: list[dict] = []
_consumed_chunks: list[str] = []


def filter_dominant_chars(char_position_counts: dict) -> set:
    """Keep only characters whose position count is at least
    _CHAR_DOMINANCE_RATIO * max(counts). When only one character is
    detected, that character always passes. Empty input -> empty set."""
    if not char_position_counts:
        return set()
    max_count = max(char_position_counts.values())
    threshold = max_count * _CHAR_DOMINANCE_RATIO
    return {c for c, n in char_position_counts.items() if n >= threshold}

def scan_media(folder: Path) -> list[dict]:
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


def compute_target_embedding(target_face_tensor) -> np.ndarray | None:
    """Compute face embedding from a ComfyUI IMAGE tensor."""
    # Convert IMAGE tensor [B, H, W, C] (0-1 float) to BGR uint8
    frame = target_face_tensor[0].cpu().numpy()
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    face = detect_face_dnn(frame)
    if face is None:
        return None

    embedding = get_face_embedding(frame, face)
    return embedding


def load_character_refs(
    refs_folder: str, clip_vision=None,
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
        face = detect_face_dnn(frame)
        if face is not None:
            embedding = get_face_embedding(frame, face)
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
        emb = clip_vision_encode(clip_vision, frame)
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


def load_voice_refs(refs_folder: str) -> dict[str, np.ndarray]:
    """Load voice reference clips. Each file's stem becomes the speaker's
    trigger (matching character_refs keys). Vocals are isolated via Demucs
    before embedding so reference embeddings are computed from clean
    speech, matching how clip-time embeddings are computed.

    Returns {trigger: l2_normalized_embedding (np.ndarray)}.
    """
    # Lazy import — voice attribution machinery lives in the audio module
    # and pulls in heavy deps (speechbrain, demucs). Keep it out of mining's
    # import path so a face-only run doesn't pay that cost.
    from .audio import (
        embed_audio_full as _embed_audio_full,
        get_speechbrain_embedder as _get_speechbrain_embedder,
        isolate_vocals as _isolate_vocals,
    )

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
        if f.suffix.lower() not in _VOICE_REF_EXTENSIONS:
            continue
        trigger = f.stem.lower().replace("_", "-")
        # Demucs isolation matches what transcribe_clip does so the
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


def load_location_refs(refs_folder: str) -> dict[str, str]:
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


def clip_vision_encode(clip_vision, frame: np.ndarray) -> np.ndarray | None:
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


def match_characters_in_frame(
    frame: np.ndarray,
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
        faces = detect_all_faces_dnn(frame)
        for rect in faces:
            emb = get_face_embedding(frame, rect)
            if emb is None:
                continue
            best_name = None
            best_sim = threshold
            for name, ref in face_refs.items():
                sim = match_face(emb, ref)
                if sim >= best_sim:
                    best_sim = sim
                    best_name = name
            if best_name is not None:
                matches.add(best_name)
                if first_match_only:
                    return matches

    # --- CLIP vision matching (non-human refs) ---
    if clip_refs and clip_vision is not None:
        frame_emb = clip_vision_encode(clip_vision, frame)
        if frame_emb is not None:
            for name, ref in clip_refs.items():
                sim = float(np.dot(frame_emb, ref))
                if sim >= clip_threshold:
                    matches.add(name)
                    if first_match_only:
                        return matches

    return matches


def check_face_match(
    frame: np.ndarray, face_rect: tuple, target_embedding: np.ndarray, threshold: float,
) -> bool:
    """Check if a detected face matches the target face."""
    embedding = get_face_embedding(frame, face_rect)
    if embedding is None:
        return False
    similarity = match_face(embedding, target_embedding)
    return similarity >= threshold


def process_image(
    img_path: Path, clips_dir: Path,
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
            matches = match_characters_in_frame(
                frame, character_refs, face_similarity,
                clip_vision=clip_vision,
            )
            if not matches:
                logger.info(f"No known characters in {img_path.name}, skipping")
                return None
        else:
            face = detect_face_dnn(frame)
            if face is None:
                logger.info(f"No face detected, skipping: {img_path.name}")
                return None
            if target_embedding is not None:
                if not check_face_match(frame, face, target_embedding, face_similarity):
                    logger.info(f"Face doesn't match target, skipping: {img_path.name}")
                    return None

    if crop_mode == "full_frame":
        # Keep native resolution — VAE encode step handles resize
        output = frame
    elif crop_mode == "pan_and_scan":
        face = detect_face_dnn(frame) if face_detection else None
        h, w = frame.shape[:2]
        if face is not None:
            crop = compute_pan_and_scan(*face, w, h, target_w, target_h)
        else:
            crop = center_crop(w, h, target_w, target_h)
        cx, cy, cw, ch = crop
        output = frame[cy:cy+ch, cx:cx+cw]
    else:
        # face_crop: tight crop around face
        crop = get_face_crop(frame, target_w, target_h, face_detection)
        if crop is None:
            logger.info(f"No face detected, skipping: {img_path.name}")
            return None
        cx, cy, cw, ch = crop
        output = frame[cy:cy+ch, cx:cx+cw]

    cv2.imwrite(str(out_path), output)
    logger.info(f"Processed image: {img_path.name} -> {out_path.name}")
    return out_path


def flush_consumed_chunks(rejected_chunk_files: set) -> None:
    """Drain the module-level _consumed_chunks list (chunks absorbed by a
    shifted extraction) into the in-memory rejected_chunk_files set so
    they're skipped at pop time. Not persisted to rejected.json — these
    aren't rejected content, just already-covered-by-another-extraction."""
    global _consumed_chunks
    for cf in _consumed_chunks:
        rejected_chunk_files.add(cf)
    _consumed_chunks = []


def quarantine_clip(clip_path: Path, output_dir: Path, reason: str) -> Path | None:
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


def record_clip_rejection(rejected_path: Path, entry: dict, reason: str,
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


def flush_rejected_chunks(
    rejected_path: Path,
    rejected_chunk_files: set,
) -> None:
    """Drain the module-level _rejected_chunks list (populated by
    process_video on content failures) into rejected.json and the
    in-memory rejected_chunk_files set, then clear the list. Safe no-op
    when the list is empty."""
    global _rejected_chunks
    if not _rejected_chunks:
        return
    for rej in _rejected_chunks:
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
    existing.extend(_rejected_chunks)
    try:
        with open(rejected_path, "w") as rf:
            json.dump(existing, rf, indent=2)
    except OSError as e:
        logger.warning(f"Could not write rejected.json: {e}")
    _rejected_chunks = []


def process_video(
    video_path: Path, clips_dir: Path,
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
) -> list:
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

    # Per-clip character tracking for balanced sampling lives on the
    # module globals `_clip_characters` and `_clip_segments`. Content-
    # based chunk rejections accumulate in `_rejected_chunks` and are
    # drained by the orchestrator after each call. Chunks absorbed into
    # a shifted extraction are queued in `_consumed_chunks` (skipped
    # in-memory but not persisted as rejected).

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
                    sample = read_frame(video_path, sp)
                    if sample is None:
                        continue
                    found = match_characters_in_frame(
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
                                face = detect_face_dnn(sample)
                                if face is not None:
                                    anchor = (sample, face)
                    # Full character_refs used for unknown detection —
                    # any KNOWN character (target or over-quota) is
                    # still "known" and doesn't count as unknown.
                    if has_unknown_face(sample, character_refs, face_similarity):
                        unknown_at.append(idx)
                # Dominance filter: a character is only recorded as
                # present if they were seen in at least
                # _CHAR_DOMINANCE_RATIO * max_count positions. Cameos
                # at the fringe of another character's shot are
                # filtered out of the recorded label.
                matched = filter_dominant_chars(char_position_counts)
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
                    _consumed_chunks.append(chunk_file)
                else:
                    # Truly no known character — persist so future runs
                    # don't retry.
                    logger.info(f"No known characters in chunk {chunk_idx} of {video_path.name}, skipping")
                    _rejected_chunks.append({
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
                            _consumed_chunks.append(
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
                _rejected_chunks.append({
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
                _rejected_chunks.append({
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
                        crop = compute_pan_and_scan(*face, frame_w, frame_h, target_w, target_h)
                    else:
                        crop = compute_face_crop(*face, frame_w, frame_h, target_w, target_h)
                else:
                    # Non-human-only match (e.g. Chairry solo shot) — no
                    # face to anchor on, fall back to center crop.
                    crop = center_crop(frame_w, frame_h, target_w, target_h)
        elif face_detection:
            face_found = False
            for try_idx in sample_positions:
                sample = read_frame(video_path, try_idx)
                if sample is None:
                    continue
                face = detect_face_dnn(sample)
                if face is not None:
                    matched_sample = sample
                    if crop_mode == "face_crop":
                        crop = compute_face_crop(*face, frame_w, frame_h, target_w, target_h)
                    elif crop_mode == "pan_and_scan":
                        crop = compute_pan_and_scan(*face, frame_w, frame_h, target_w, target_h)
                    face_found = True
                    break
            if not face_found:
                logger.info(f"No face in chunk {chunk_idx} of {video_path.name}, skipping")
                cf = f"{video_path.stem}_chunk{chunk_idx:04d}.mp4"
                _rejected_chunks.append({
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
                face = detect_face_dnn(matched_sample)
                if face is None or not check_face_match(matched_sample, face, target_embedding, face_similarity):
                    logger.info(f"Face doesn't match target in chunk {chunk_idx} of {video_path.name}, skipping")
                    cf = f"{video_path.stem}_chunk{chunk_idx:04d}.mp4"
                    _rejected_chunks.append({
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
            crop = center_crop(frame_w, frame_h, target_w, target_h)

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
            tr = transcribe_clip(out_path, voice_refs=voice_refs, face_chars=_face_chars)
            if tr and tr.get("hallucination"):
                # No usable speech — quarantine clip (don't delete) so it
                # can be inspected later for false positives.
                logger.info(f"  No usable speech, quarantining clip: {out_path.name}")
                q_path = quarantine_clip(out_path, clips_dir.parent, "speech_hallucination")
                rej_record = {
                    "source_file": str(video_path),
                    "media_path": str(out_path),
                    "chunk_file": out_path.name,
                    "reason": "speech_hallucination",
                }
                if q_path is not None:
                    rej_record["quarantined_path"] = str(q_path)
                _rejected_chunks.append(rej_record)
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
                                tr = transcribe_clip(out_path, voice_refs=voice_refs, face_chars=_face_chars)

                # Store transcript text to return to caller
                if tr and tr["text"]:
                    clip_transcript = tr["text"]
                    logger.info(f"  Transcript: {clip_transcript[:80]}{'...' if len(clip_transcript) > 80 else ''}")
                # Stash speaker-attributed segments on the module global
                # _clip_segments so the caller can read them by clip path
                # (parallels _clip_characters).
                if tr and tr.get("segments"):
                    _clip_segments[str(out_path)] = tr["segments"]

        clips.append((out_path, clip_transcript if transcribe_speech else None))
        # Store matched characters for balanced sampling
        if face_detection and character_refs and matched_names:
            _clip_characters[str(out_path)] = sorted(matched_names)
        if out_path not in existing_chunks:
            new_clips_produced += 1
        logger.info(f"Extracted clip: {video_path.name} chunk {chunk_idx} -> {out_path.name}")

        start_frame = end_frame
        chunk_idx += 1

        # Stop early if we've produced enough new clips for this video
        if max_new_clips > 0 and new_clips_produced >= max_new_clips:
            break

    return clips


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    """Read a single frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_face_crop(
    frame: np.ndarray, target_w: int, target_h: int, face_detection: bool,
) -> tuple[int, int, int, int] | None:
    """Detect face and compute crop region. Returns (x, y, w, h) or None."""
    if not face_detection:
        h, w = frame.shape[:2]
        return center_crop(w, h, target_w, target_h)

    face = detect_face_dnn(frame)
    if face is None:
        return None

    fx, fy, fw, fh = face
    frame_h, frame_w = frame.shape[:2]
    return compute_face_crop(fx, fy, fw, fh, frame_w, frame_h, target_w, target_h)


def center_crop(
    frame_w: int, frame_h: int, target_w: int, target_h: int,
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
