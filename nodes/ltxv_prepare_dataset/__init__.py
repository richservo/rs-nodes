"""RSLTXVPrepareDataset — orchestrator for the prepare-dataset node.

Coordinates the audio, face, captioning, encoding, mining, and dataset_io
modules to turn raw source videos + character refs into the precomputed
artifacts (latents, conditions, audio_latents) that LTX-2 LoRA training
consumes. dataset.json is the source of truth ONLY for encoding work
that hasn't happened yet — once artifacts exist on disk, they win.
"""
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
import time as _t
from pathlib import Path

import cv2
import numpy as np
import torch

import comfy.utils
import folder_paths

from ...utils.ltxv_train_env import (
    free_vram,
    get_script_path,
    get_trainer_env,
    run_training_subprocess,
    validate_submodule,
    validate_text_encoder_path,
)

from . import audio, captioning, dataset_io, encoding, face, mining, status

logger = logging.getLogger(__name__)

# Self-heal / quota-recovery tuning knobs (see Reference/dataset-self-heal-spec.md).
#
# COND_TOKEN_LIMIT — token-length ceiling for a sample's text-conditioning.
# The encoder pads to multiples of this; anything that comes out larger means
# the captioner ran away. A single sample with 16x the tokens of the rest
# blows up attention memory ~256x at training time and triggers C-level
# aborts when shuffled in. Match this to the encoder's pad multiple.
#
# MAX_CAPTION_CHARS — fast pre-filter before the encoder sees a caption.
# 600 chars ≈ 128 tokens after BPE; tuned to comfortably fit COND_TOKEN_LIMIT.
#
# MAX_CAPTION_RETRIES — re-rolls per clip before giving up and rejecting it.
# Gemma-class models are probabilistic, so wiping context and re-rolling
# usually produces a completely different caption.
#
# MAX_OUTER_PASSES — outer mine→caption cycles before giving up on quota.
# Each pass beyond the first pays the VRAM-cycle cost of swapping Ollama
# in/out, so we keep this small.
COND_TOKEN_LIMIT = 128
MAX_CAPTION_CHARS = 600
MAX_CAPTION_RETRIES = 3
MAX_OUTER_PASSES = 3

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


def _unload_all_prepper_models():
    """Free everything the prepper holds across all submodules.

    Each domain module (audio, face) owns its own state and exposes an
    unload function. We call each, then do the cross-cutting cleanup
    (gc, ComfyUI's mm.unload_all_models, cuda empty_cache).
    """
    freed = []
    try:
        freed.extend(audio.unload_audio_models())
    except Exception as e:
        logger.warning(f"audio.unload_audio_models failed: {e}")
    try:
        freed.extend(face.unload_face_models())
    except Exception as e:
        logger.warning(f"face.unload_face_models failed: {e}")

    # Cross-cutting: ComfyUI's model management + GC
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as e:
        logger.warning(f"comfy mm cleanup failed: {e}")
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if freed:
        logger.info(f"Unloaded prepper models: {', '.join(freed)}")


class RSLTXVPrepareDataset:
    """Scan a folder of videos/images, detect faces, crop around them,
    generate dataset JSON, and run process_dataset.py to precompute latents."""

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
                "vae": ("VAE", {"tooltip": "Video VAE (from CheckpointLoaderSimple). When connected, encodes video latents in-process instead of slow subprocess."}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE (from LTXV-AV checkpoint loader). When connected, audio latents are encoded in-process; without it, with_audio=True falls back to the LTX-2 subprocess + downloaded text encoder."}),
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
        audio_vae=None,
        clip_vision=None,
        target_fps: float = 0.0,
        max_samples: int = 0,
        unique_id=None,
    ):
        validate_submodule()
        # Set the audio module's whisper model size from the input — its
        # get_whisper_model uses this when loading. Changing this between
        # runs forces a reload.
        audio._WHISPER_MODEL_SIZE = whisper_model
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

        # NOTE: heavy reference-asset loads (character_refs / voice_refs /
        # target_embedding / location_refs) used to live here, but they
        # spin up InsightFace + speechbrain + clip-vision and waste GPU
        # memory + several seconds of startup time on runs that turn out
        # to need no work (the established-dataset / fully-consistent
        # case). Those loads now happen AFTER the audit + reconciliation
        # below decides we actually have something to do — see the block
        # right after the early-exit return.
        character_refs: dict[str, dict] = {}
        voice_refs: dict[str, np.ndarray] = {}
        target_embedding = None
        location_refs: dict[str, str] = {}

        output_dir = Path(folder_paths.get_output_directory()) / "ltxv_training" / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_json_path = output_dir / "dataset.json"
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        latents_dir = output_dir / "latents"
        conditions_dir = output_dir / "conditions"
        audio_latents_dir = output_dir / "audio_latents"

        face.set_face_padding(face_padding)

        # ===== PHASE 0: AUDIT + RECONCILIATION =====
        # Audit pass: reconcile dataset.json against on-disk state. Drops
        # orphan entries, stubs missing entries, blanks captions whose
        # condition file is over the token limit (those are the runaway
        # captions that crash training when shuffled in). Runs BEFORE the
        # early-exit check so any repairs surfaced here force the rest of
        # the pipeline to do the corresponding refill work.
        audit_dropped, audit_stubbed, audit_blanked = dataset_io.audit_and_repair_dataset(
            output_dir, dataset_json_path,
        )
        # Only stubs and blanks require pipeline follow-up (re-caption,
        # re-encode). Orphan drops are pure metadata cleanup — the missing
        # files are already gone, so there's nothing left to refill, and
        # forcing the full pipeline to run would needlessly re-trigger
        # entry-driven phases (transcript backfill etc.) that aren't
        # idempotent against fields-vs-artifacts mismatches.
        audit_needs_followup = (audit_stubbed + audit_blanked) > 0

        # ----------------------------------------------------------------
        # Artifact-stem reconciliation. The encoded files (latents,
        # conditions, audio_latents) are the source of truth for "is the
        # dataset complete?" — NOT dataset.json. dataset.json can be
        # corrupted, partially restored from backup, or missing fields,
        # and none of that affects training as long as the encoded files
        # exist for every clip on disk.
        #
        # Build the stem set from each kind by looking at <kind>/clips/,
        # which is the actual layout the encoders write to. The previous
        # check used <kind>.iterdir() which only sees the clips/ subdir
        # itself (a directory) and filtered it out — meaning early-exit
        # NEVER fired in the standard layout. Bug, fixed.
        #
        # Decision tree:
        #   clip_stems == all encoded stems AND no new source media
        #     -> early exit, do nothing
        #   clip_stems != encoded stems
        #     -> targeted repair: skip mining, skip the captioning loop
        #        for stems that already have a condition (artifact-trumps-
        #        JSON guards), encoders fill in only what's missing
        #   new source media exists
        #     -> mining runs to ingest the new sources
        # ----------------------------------------------------------------
        def _stems_with_pt(kind_dir: Path) -> set[str]:
            """Stems of .pt files under kind_dir/clips/ (the encoder layout)."""
            sub = kind_dir / "clips"
            if not sub.exists():
                return set()
            return {p.stem for p in sub.iterdir() if p.is_file() and p.suffix == ".pt"}

        clip_video_suffixes = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
        clip_stems: set[str] = set()
        if clips_dir.exists():
            clip_stems = {
                p.stem for p in clips_dir.iterdir()
                if p.is_file() and p.suffix.lower() in clip_video_suffixes
            }
        latent_stems = _stems_with_pt(latents_dir)
        condition_stems = _stems_with_pt(conditions_dir)
        audio_latent_stems = _stems_with_pt(audio_latents_dir) if with_audio else clip_stems

        missing_latents = clip_stems - latent_stems
        missing_conditions = clip_stems - condition_stems
        missing_audio_latents = (clip_stems - audio_latent_stems) if with_audio else set()

        # Source media check — for deciding whether to skip mining.
        processed_sources = {
            (s.rsplit("_chunk", 1)[0] if "_chunk" in s else s)
            for s in clip_stems
        }
        new_source_media = []
        if media_folder.exists():
            for item in mining.scan_media(media_folder):
                if Path(item["path"]).stem not in processed_sources:
                    new_source_media.append(item)

        # State summary for the log so the user sees exactly what we decided.
        artifacts_complete = (
            bool(clip_stems)
            and not missing_latents
            and not missing_conditions
            and not missing_audio_latents
        )

        # "Established dataset" — clips exist on disk AND at least some
        # encoded files exist. This is the recovery / fix-only state. In
        # this state dataset.json is NOT a source of truth (it can be
        # wiped, partial, restored from backup, missing fields). Only the
        # encoded artifacts are. The only legitimate work in this state is
        # filling in missing encodes for clips that are already on disk.
        # New source media in media_folder is explicitly ignored — the
        # user must start a fresh dataset to ingest those, not mix mining
        # into a recovery run.
        dataset_is_established = bool(clip_stems) and bool(
            latent_stems or condition_stems or audio_latent_stems
        )

        if (
            not audit_needs_followup
            and artifacts_complete
            and dataset_is_established
        ):
            logger.info(
                f"prepare: dataset complete — {len(clip_stems)} clips, all "
                f"latents/conditions{'/audio_latents' if with_audio else ''} "
                f"present. Skipping all phases. "
                f"(dataset.json not consulted; encoded artifacts are the "
                f"source of truth.)"
            )
            return (str(output_dir), str(dataset_json_path))

        # ===== HEAVY ASSET LOADS (deferred until we know there's work) =====
        # These were originally at the top of prepare(), but the audit +
        # reconciliation above can decide there's nothing to do, and in
        # that case loading face / speechbrain / clip-vision models was
        # pure waste — both startup latency and GPU VRAM that the trainer
        # then has to re-claim. So they happen here, only on the path
        # that's actually going to mine / caption / encode.

        # Multi-character mode: load a folder of reference images.  Each file's
        # stem becomes that character's trigger word.  Face-containing refs are
        # matched via face embeddings; non-face refs (puppets, props, objects)
        # fall back to CLIP vision embedding if clip_vision is connected.
        # When populated, clip selection accepts chunks where any reference
        # matches in at least one sample frame.
        if character_refs_folder and face_detection:
            character_refs = mining.load_character_refs(character_refs_folder, clip_vision=clip_vision)
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
        if voice_refs_folder and transcribe_speech:
            voice_refs = mining.load_voice_refs(voice_refs_folder)
            if voice_refs:
                logger.info(
                    f"Voice attribution: loaded {len(voice_refs)} voice reference(s) "
                    f"— {', '.join(sorted(voice_refs.keys()))}"
                )

        # target_face pin: wired through the same character_refs pipeline as the
        # folder path, using lora_trigger as the trigger name (equivalent to
        # dropping one image in a folder named <lora_trigger>.jpg).
        if target_face is not None and face_detection:
            target_embedding = mining.compute_target_embedding(target_face)
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
        if location_refs_folder:
            location_refs = mining.load_location_refs(location_refs_folder)
            if location_refs:
                logger.info(
                    f"Location mode: loaded {len(location_refs)} location refs "
                    f"— {', '.join(sorted(location_refs.keys()))}"
                )

        # If we get here, there is real work to do. Surface what's actually
        # missing so the rest of the run is interpretable, and so the user
        # sees that we're NOT redoing the entire dataset.
        if clip_stems:
            n_miss = len(missing_latents | missing_conditions | missing_audio_latents)
            mode_str = (
                "ESTABLISHED DATASET (recovery mode — JSON ignored, fixing missing encodes only)"
                if dataset_is_established
                else f"FRESH DATASET (will mine {len(new_source_media)} new source video(s))"
            )
            logger.info(
                f"Reconciliation: {len(clip_stems)} clips on disk, "
                f"{len(missing_latents)} missing latents, "
                f"{len(missing_conditions)} missing conditions, "
                f"{len(missing_audio_latents)} missing audio_latents. "
                f"Mode: {mode_str}. "
                f"Will encode {n_miss} clip(s)."
            )
        elif new_source_media:
            logger.info(
                f"Reconciliation: no clips yet, {len(new_source_media)} "
                f"new source video(s) to mine."
            )

        # Skip mining entirely when the dataset is established. Mining
        # exists to add NEW clips from NEW source videos; in recovery mode
        # we only fix existing clips' missing encodes. If the user wants
        # to ingest new source media on top of an established dataset,
        # they need to start a fresh dataset (different output_name) or
        # explicitly wipe encoded artifacts first.
        skip_mining = dataset_is_established

        # Scan current media folder
        media_files = mining.scan_media(media_folder)
        if not media_files:
            raise ValueError(f"No video or image files found in: {media_folder}")

        current_sources = {str(item["path"]) for item in media_files}

        # ===== PHASE 1: GENERATE CLIPS =====
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
        # before. Writes go through entries_for_write which converts back
        # to relative on the way out.
        dataset_io.normalize_loaded_entries(existing_entries, output_dir)

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
            status.emit_prepper_status(
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
                json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)
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
                    frame = mining.read_frame(clip_file, 0)
                    if frame is not None:
                        chars = mining.match_characters_in_frame(
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
                    tr = audio.transcribe_clip(clip_file, voice_refs=voice_refs, face_chars=_face_chars)
                    if tr and tr.get("hallucination"):
                        logger.info(f"  No usable speech, quarantining orphan: {clip_file.name}")
                        q_path = mining.quarantine_clip(clip_file, output_dir, "speech_hallucination")
                        mining.record_clip_rejection(rejected_path, entry, "speech_hallucination", quarantined_path=q_path)
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
                json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)
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
                frame = mining.read_frame(clip_file, 0)
                if frame is not None:
                    chars = mining.match_characters_in_frame(
                        frame, character_refs, face_similarity,
                        clip_vision=clip_vision,
                        first_match_only=False,
                    )
                    if chars:
                        entry["characters"] = sorted(chars)
                        chars_backfilled += 1
            if chars_backfilled:
                with open(dataset_json_path, "w") as f:
                    json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)
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
                    frame = mining.read_frame(clip_file, (total * i) // n_check)
                    if frame is None:
                        continue
                    found = mining.match_characters_in_frame(
                        frame, character_refs, face_similarity,
                        clip_vision=clip_vision,
                        first_match_only=False,
                    )
                    for c in found:
                        char_position_counts[c] = char_position_counts.get(c, 0) + 1
                found_chars = mining.filter_dominant_chars(char_position_counts)
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
                        json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)
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
        # to do.  The lazy whisper/demucs load fires on the first transcribe_clip
        # call — if every existing entry has been transcribed (or has no clip
        # on disk), we want zero model loads, not one model load + 168 no-op
        # iterations.  Use key existence (not truthiness) so silent clips with
        # an empty transcript don't get re-attempted on every run.
        #
        # Artifact-trumps-JSON guard: if the condition file already exists
        # for this clip, the transcript was already merged into the caption
        # at the time the condition was encoded — re-transcribing would be
        # cosmetic only (the transcript gets written back into dataset.json
        # but does not change training data). Skip those entries so a wiped
        # JSON doesn't trigger re-transcription of an already-complete dataset.
        backfill_targets = (
            [e for e in existing_entries
             if "transcript" not in e
             and Path(e["media_path"]).exists()
             and not dataset_io.condition_path_for_clip(
                 output_dir, Path(e["media_path"])
             ).exists()]
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
                tr = audio.transcribe_clip(clip_file, voice_refs=voice_refs, face_chars=_face_chars)
                if tr and tr.get("hallucination"):
                    logger.info(f"  No usable speech, quarantining: {clip_file.name}")
                    q_path = mining.quarantine_clip(clip_file, output_dir, "speech_hallucination")
                    mining.record_clip_rejection(rejected_path, entry, "speech_hallucination", quarantined_path=q_path)
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
                        json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)
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
                        json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)
                    os.replace(tmp_path, dataset_json_path)
                except OSError as e:
                    logger.warning(f"Could not write dataset.json: {e}")
                if backfilled:
                    logger.info(f"Backfilled {backfilled} missing transcripts")
                if silent_marked:
                    logger.info(f"Marked {silent_marked} silent clip(s) as no-dialogue")
                if hallucination_purge:
                    logger.info(f"Purged {len(hallucination_purge)} clips with no usable speech")

        # ===== MINING GATE =====
        # Skip the entire mining phase when:
        #   - the rolling dataset is already at capacity, OR
        #   - reconciliation showed there's no new source media to ingest
        #     (skip_mining is set above based on artifact-stem comparison)
        # The second case is critical when dataset.json is partially
        # corrupted: its per-character quota counts may be wrong (chars
        # field missing on entries) and would falsely re-trigger mining
        # for a dataset whose ENCODED artifacts are already complete or
        # nearly complete. Encoded artifacts are the source of truth;
        # don't mine to satisfy a JSON-derived quota when the artifacts
        # already exist.
        if (max_samples > 0 and len(existing_entries) >= max_samples) or skip_mining:
            if skip_mining:
                logger.info(
                    "Skipping mining: no new source media. Will only fix "
                    "missing encodes on existing clips (if any). "
                    f"({len(missing_latents | missing_conditions | missing_audio_latents)} "
                    f"clip(s) need encoding work.)"
                )
            else:
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
                            json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)
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
                                    json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)
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

                        results = mining.process_video(
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
                        # process_video, and fold any absorbed-neighbor chunks
                        # into rejected_chunk_files for the rest of this pass.
                        mining.flush_rejected_chunks(rejected_path, rejected_chunk_files)
                        mining.flush_consumed_chunks(rejected_chunk_files)

                        for result_path, result_transcript in results:
                            if str(result_path) in known_paths:
                                continue
                            clip_chars = mining._clip_characters.get(str(result_path), [])
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
                                clip_segs = mining._clip_segments.get(str(result_path))
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
                            json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)

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
                            result = mining.process_image(
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
                            results = mining.process_video(
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
                            mining.flush_rejected_chunks(rejected_path, rejected_chunk_files)
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
                                    clip_segs = mining._clip_segments.get(str(result_path))
                                    if clip_segs:
                                        entry["transcript_segments"] = clip_segs
                                clip_chars = mining._clip_characters.get(str(result_path), [])
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
                            json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)

                        pbar.update_absolute(i + 1, len(new_media))

                    # New clips will get their own conditions/latents encoded —
                    # the encode functions skip files that already exist on disk.
                else:
                    logger.info(f"All {len(existing_entries)} clips up to date, no new media to process")
                    # Make sure JSON is written even if no new media
                    if not dataset_json_path.exists():
                        with open(dataset_json_path, "w") as f:
                            json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)

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
                json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)
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
        freed = audio.unload_audio_models()
        if freed:
            logger.info(f"Unloaded {' + '.join(freed)} to free VRAM for captioning")

        # ===== PHASE 2: CAPTION UNCAPTIONED CLIPS =====
        # (with LLM-based QC if target face provided)
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
            gc.collect()
            mm.free_memory(1e18, mm.get_torch_device())
            logger.info("Cleared ComfyUI VRAM before Ollama captioning phase")
        except Exception as e:
            logger.warning(f"Could not fully free VRAM before captioning: {e}")

        captioning.caption_dataset_json(
            dataset_json_path, clip_paths, caption_mode, lora_trigger,
            ollama_url=ollama_url, ollama_model=ollama_model,
            target_face_b64=target_face_b64, caption_style=caption_style,
            cast_names=cast_names, cast_refs=cast_refs,
            location_refs=location_refs_list,
            skip_id_pass=skip_id_pass,
        )

        # Post-captioning quota check. If clips got rejected (MISMATCH face
        # QC, or length-overshoot caption rejection) and the chunk pool
        # still has unused candidates, the dataset is now under quota.
        # Automatic re-mining is non-trivial because mining state is
        # interleaved with character-mode setup throughout prepare(); for
        # now we surface a clear warning so the user knows to re-run
        # prepare_dataset, which will resume from the persisted chunk_pool
        # and fill the gap. See Reference/dataset-self-heal-spec.md for
        # the deferred auto-remine plan.
        try:
            with open(dataset_json_path) as _qf:
                _post_cap_entries = json.load(_qf)
        except (OSError, json.JSONDecodeError):
            _post_cap_entries = []
        _pool_remaining = pool_state.get("remaining", 0)
        if max_samples > 0 and len(_post_cap_entries) < max_samples:
            shortfall = max_samples - len(_post_cap_entries)
            if _pool_remaining > 0:
                logger.warning(
                    f"Post-caption quota check: {len(_post_cap_entries)}/{max_samples} "
                    f"samples (short by {shortfall}). chunk_pool has "
                    f"{_pool_remaining} clips remaining — re-run prepare_dataset "
                    f"to mine replacements for the rejected clips."
                )
            else:
                logger.warning(
                    f"Post-caption quota check: {len(_post_cap_entries)}/{max_samples} "
                    f"samples (short by {shortfall}). chunk_pool is empty — "
                    f"add more source media to fill the remaining slots."
                )

        # ===== PHASE 3: ENCODE CONDITIONS AND LATENTS =====
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
                json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)

        conditions_dir = output_dir / "conditions"
        latents_dir = output_dir / "latents"
        audio_latents_dir = output_dir / "audio_latents"
        need_subprocess = False

        # Phase 3a: Text encoding
        if clip is not None:
            encoding.encode_conditions_inprocess(
                clip, dataset_json_path, conditions_dir,
                character_names=list(character_refs.keys()) if character_refs else None,
            )
        else:
            need_subprocess = True

        # Phase 3b: VAE encoding (video latents).
        # The previous design forced subprocess whenever with_audio was set
        # because we had no in-process audio encoder, and the subprocess
        # writes combined video+audio passes. Now that audio_vae is wired
        # (Phase 3c below), the video VAE can run in-process even with
        # audio enabled — they're independent files.
        if vae is not None:
            encoding.encode_latents_inprocess(
                vae, dataset_json_path, latents_dir, existing_entries,
                target_w=target_w, target_h=target_h,
            )
        else:
            need_subprocess = True

        # Phase 3c: Audio latent encoding.
        # When audio_vae is connected, encode in-process — same on-disk
        # format as the LTX-2 subprocess. This eliminates the subprocess
        # invocation (and its ~25 GB Gemma3 download) when all four
        # ComfyUI inputs (clip, vae, audio_vae, optional clip_vision)
        # are wired.
        if with_audio:
            if audio_vae is not None:
                encoding.encode_audio_latents_inprocess(
                    audio_vae, dataset_json_path, audio_latents_dir,
                )
            else:
                # Recompute missing audio_latents NOW (post-mining /
                # post-captioning state) by stem comparison. Only invoke
                # the subprocess if there's actual audio work to do —
                # the subprocess loads its own Gemma + VAEs upfront
                # regardless of remaining work, so dodging it when no
                # audio_latent files are missing avoids a pointless
                # multi-GB model load.
                _audio_stems = {
                    p.stem for p in (audio_latents_dir / "clips").glob("*.pt")
                } if (audio_latents_dir / "clips").exists() else set()
                _clip_stems_now = {
                    p.stem for p in clips_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {
                        ".mp4", ".mov", ".webm", ".avi", ".mkv",
                    }
                } if clips_dir.exists() else set()
                _missing_audio = _clip_stems_now - _audio_stems
                if _missing_audio:
                    logger.info(
                        f"with_audio=True, audio_vae not connected, "
                        f"{len(_missing_audio)} audio_latent(s) missing — "
                        f"falling back to LTX-2 subprocess. Connect audio_vae "
                        f"to keep audio encoding in-process and skip this."
                    )
                    need_subprocess = True
                else:
                    logger.info(
                        "Audio latents already complete — skipping subprocess "
                        "even though audio_vae isn't connected."
                    )

        # Reference latents (IC-LoRA conditioning_folder) still require the
        # subprocess; there's no in-process IC-LoRA encoder yet.
        if conditioning_folder:
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
                "--resolution-buckets", encoding.resolve_resolution_buckets(resolution_buckets, existing_entries),
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
                json.dump(dataset_io.entries_for_write(existing_entries, output_dir), f, indent=2)

        # Final unload — drop everything prep loaded so unattended training that
        # follows in the same workflow starts with a clean VRAM slate.
        _unload_all_prepper_models()

        return (str(output_dir), str(dataset_json_path))
