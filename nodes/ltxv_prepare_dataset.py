import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
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


# OpenCV DNN face detector
_face_net = None
_face_detector_checked = False
_FACE_CONFIDENCE_THRESHOLD = 0.5
_FACE_PADDING = 0.6  # 60% padding around detected face for head+shoulders context
_FACE_MATCH_THRESHOLD = 0.40  # Cosine similarity threshold for face matching (lower = stricter)

# Face recognition DNN (OpenFace embedding model)
_face_recognizer = None
_face_recognizer_checked = False
_RECOGNIZER_MODEL_URL = "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7"

_DNN_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
_DNN_CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"


def _get_face_detector():
    """Load OpenCV DNN face detector (res10_300x300_ssd). Downloads model if needed. Cached after first call."""
    global _face_net, _face_detector_checked
    if _face_net is not None:
        return _face_net
    if _face_detector_checked:
        return None  # Already tried and failed, don't retry

    _face_detector_checked = True

    # Store models alongside this file
    model_dir = Path(__file__).parent.parent / "models" / "face_detector"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_file = model_dir / "deploy.prototxt"

    # Download if missing
    if not model_file.exists():
        logger.info("Downloading OpenCV DNN face detector model...")
        try:
            import urllib.request
            urllib.request.urlretrieve(_DNN_MODEL_URL, str(model_file))
            logger.info(f"Downloaded face detector model to {model_file}")
        except Exception as e:
            logger.warning(f"Failed to download DNN face model: {e}. Using Haar cascade.")
            return None

    if not config_file.exists():
        logger.info("Downloading OpenCV DNN face detector config...")
        try:
            import urllib.request
            urllib.request.urlretrieve(_DNN_CONFIG_URL, str(config_file))
            logger.info(f"Downloaded face detector config to {config_file}")
        except Exception as e:
            logger.warning(f"Failed to download DNN face config: {e}. Using Haar cascade.")
            return None

    try:
        _face_net = cv2.dnn.readNetFromCaffe(str(config_file), str(model_file))
        logger.info("Loaded OpenCV DNN face detector")
        return _face_net
    except Exception as e:
        logger.warning(f"Failed to load DNN face detector: {e}. Using Haar cascade.")
        return None


def _detect_face_dnn(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect the largest face using OpenCV DNN. Returns (x, y, w, h) or None."""
    faces = _detect_all_faces_dnn(frame)
    if not faces:
        return None
    # Return largest
    return max(faces, key=lambda r: r[2] * r[3])


def _detect_all_faces_dnn(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect all faces using OpenCV DNN. Returns list of (x, y, w, h)."""
    net = _get_face_detector()
    if net is None:
        haar = _detect_face_haar(frame)
        return [haar] if haar is not None else []

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces: list[tuple[int, int, int, int]] = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < _FACE_CONFIDENCE_THRESHOLD:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        fw, fh = x2 - x1, y2 - y1
        if fw <= 0 or fh <= 0:
            continue
        faces.append((x1, y1, fw, fh))
    return faces


def _detect_face_haar(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Fallback: detect largest face using Haar cascade."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Return largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    return tuple(faces[idx])


def _get_face_recognizer():
    """Load OpenFace DNN embedding model for face recognition. Downloads if needed."""
    global _face_recognizer, _face_recognizer_checked
    if _face_recognizer is not None:
        return _face_recognizer
    if _face_recognizer_checked:
        return None

    _face_recognizer_checked = True

    model_dir = Path(__file__).parent.parent / "models" / "face_detector"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "openface.nn4.small2.v1.t7"

    if not model_file.exists():
        logger.info("Downloading OpenFace face recognition model...")
        try:
            import urllib.request
            urllib.request.urlretrieve(_RECOGNIZER_MODEL_URL, str(model_file))
            logger.info(f"Downloaded face recognition model to {model_file}")
        except Exception as e:
            logger.warning(f"Failed to download face recognition model: {e}")
            return None

    try:
        _face_recognizer = cv2.dnn.readNetFromTorch(str(model_file))
        logger.info("Loaded OpenFace face recognition model")
        return _face_recognizer
    except Exception as e:
        logger.warning(f"Failed to load face recognition model: {e}")
        return None


def _get_face_embedding(frame: np.ndarray, face_rect: tuple[int, int, int, int]) -> np.ndarray | None:
    """Extract a 128-d face embedding from a detected face region."""
    net = _get_face_recognizer()
    if net is None:
        return None

    x, y, w, h = face_rect
    # Clamp to frame bounds
    fh, fw = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    w = min(w, fw - x)
    h = min(h, fh - y)
    if w <= 0 or h <= 0:
        return None

    face_crop = frame[y:y+h, x:x+w]
    blob = cv2.dnn.blobFromImage(face_crop, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    embedding = net.forward().flatten()
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def _match_face(embedding: np.ndarray, target_embedding: np.ndarray) -> float:
    """Compute cosine similarity between two face embeddings. Returns 0-1 (1 = identical)."""
    return float(np.dot(embedding, target_embedding))


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
                "ollama_model": ("STRING", {"default": "gemma3:27b", "tooltip": "Ollama vision model for captioning (only used when caption_mode=ollama)"}),
                "with_audio": ("BOOLEAN", {"default": False, "tooltip": "Extract and encode audio latents"}),
                "load_text_encoder_in_8bit": ("BOOLEAN", {"default": True, "tooltip": "Load text encoder in 8-bit to save VRAM"}),
                "crop_mode": (["face_crop", "full_frame"], {"default": "face_crop", "tooltip": "face_crop=crop around detected face, full_frame=scale entire frame to target resolution (aspect-preserving, letterboxed). Both modes use face detection for clip selection when enabled."}),
                "face_detection": ("BOOLEAN", {"default": True, "tooltip": "Enable face detection: crop around faces, discard clips with no faces"}),
                "target_face": ("IMAGE", {"tooltip": "Reference face image. When connected, only keeps clips matching this specific face."}),
                "character_refs_folder": ("STRING", {"default": "", "tooltip": "Multi-character mode: path to a folder of reference face images. Filename (minus extension) is used as that character's trigger word. Clips are kept if ANY reference matches. All matched characters are named in captions."}),
                "skip_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 1.0, "tooltip": "Skip the first N seconds of every video (useful for cutting out repetitive intros)."}),
                "skip_end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 1.0, "tooltip": "Skip the last N seconds of every video (useful for cutting out end credits)."}),
                "face_similarity": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Face match threshold (0-1). Higher = stricter matching. 0.40 is a good default."}),
                "face_padding": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Padding around detected face (0.6 = 60% extra for head+shoulders)"}),
                "conditioning_folder": ("STRING", {"default": "", "tooltip": "IC-LoRA: folder of conditioning input videos (depth maps, edge maps, poses). Matched to media_folder by filename. Media folder is the ground truth output."}),
                "clip": ("CLIP", {"tooltip": "Text encoder (from CheckpointLoaderSimple). When connected, encodes captions in-process instead of slow subprocess."}),
                "vae": ("VAE", {"tooltip": "VAE (from CheckpointLoaderSimple). When connected, encodes latents in-process instead of slow subprocess."}),
                "clip_vision": ("CLIP_VISION", {"tooltip": "Optional CLIP Vision model (from CLIPVisionLoader). Enables matching of non-human characters (puppets, props, objects) in character_refs_folder. Human characters use face matching regardless."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("preprocessed_path", "dataset_json_path")
    FUNCTION = "prepare"
    CATEGORY = "RS Nodes"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

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
        ollama_model: str = "gemma3:27b",
        with_audio: bool = False,
        load_text_encoder_in_8bit: bool = True,
        crop_mode: str = "face_crop",
        face_detection: bool = True,
        target_face=None,
        character_refs_folder: str = "",
        skip_start_seconds: float = 0.0,
        skip_end_seconds: float = 0.0,
        face_similarity: float = 0.40,
        face_padding: float = 0.6,
        conditioning_folder: str = "",
        clip=None,
        vae=None,
        clip_vision=None,
    ):
        validate_submodule()
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

        # Compute target face embedding if provided
        target_embedding = None
        if target_face is not None and face_detection:
            target_embedding = self._compute_target_embedding(target_face)
            if target_embedding is not None:
                logger.info(f"Target face embedding computed, similarity threshold: {face_similarity}")
            else:
                logger.warning("Could not extract face from target_face image, face matching disabled")

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

        output_dir = Path(folder_paths.get_output_directory()) / "ltxv_training" / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_json_path = output_dir / "dataset.json"
        clips_dir = output_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        latents_dir = output_dir / "latents"
        conditions_dir = output_dir / "conditions"

        global _FACE_PADDING
        _FACE_PADDING = face_padding

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

        # Track which source files are already in the JSON
        processed_sources = {e.get("source_file", "") for e in existing_entries}

        # Load rejected clips so we don't re-process them
        rejected_path = output_dir / "rejected.json"
        rejected_clips = set()
        if rejected_path.exists():
            try:
                with open(rejected_path) as rf:
                    for r in json.load(rf):
                        rejected_clips.add(r.get("media_path", ""))
                        rejected_clips.add(r.get("source_file", ""))
            except (json.JSONDecodeError, KeyError):
                pass

        # Find new media files not yet processed (and not rejected)
        new_media = [
            item for item in media_files
            if str(item["path"]) not in processed_sources
            and str(item["path"]) not in rejected_clips
        ]

        # Verify existing clips still exist on disk, remove entries for missing clips
        valid_entries = []
        for entry in existing_entries:
            clip_path = Path(entry["media_path"])
            if clip_path.exists():
                valid_entries.append(entry)
            else:
                logger.info(f"Clip missing from disk, removing: {clip_path.name}")
        existing_entries = valid_entries

        if new_media:
            logger.info(f"Processing {len(new_media)} new media files ({len(existing_entries)} already done)")
            entries_before = len(existing_entries)

            pbar = comfy.utils.ProgressBar(len(new_media))
            ref_folder = Path(conditioning_folder) if conditioning_folder else None

            for i, item in enumerate(new_media):
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
                    )
                    if results:
                        produced_clips = True
                    for result in results:
                        entry = {"caption": "", "media_path": str(result), "source_file": source_path}
                        if ref_folder and ref_folder.exists():
                            for ext in VIDEO_EXTENSIONS:
                                ref_file = ref_folder / (result.stem + ext)
                                if ref_file.exists():
                                    entry["reference_path"] = str(ref_file)
                                    break
                        existing_entries.append(entry)

                # Record fully-rejected source files so they aren't reprocessed
                if not produced_clips:
                    rejected_path = output_dir / "rejected.json"
                    rejected = []
                    if rejected_path.exists():
                        try:
                            with open(rejected_path) as rf:
                                rejected = json.load(rf)
                        except (json.JSONDecodeError, KeyError):
                            pass
                    rejected.append({
                        "source_file": source_path,
                        "reason": "no clips produced (all chunks rejected)",
                    })
                    with open(rejected_path, "w") as rf:
                        json.dump(rejected, rf, indent=2)
                    logger.info(f"No clips from {item['path'].name} — added to rejected.json")

                # Save JSON after each source file so progress isn't lost
                with open(dataset_json_path, "w") as f:
                    json.dump(existing_entries, f, indent=2)

                pbar.update_absolute(i + 1, len(new_media))

            # Invalidate preprocessing only if new clips were actually added
            new_clips_added = len(existing_entries) > entries_before
            if new_clips_added:
                if latents_dir.exists():
                    import shutil
                    shutil.rmtree(latents_dir)
                    logger.info("Cleared latents — new clips added, preprocessing needed")
                if conditions_dir.exists():
                    import shutil
                    shutil.rmtree(conditions_dir)
                    logger.info("Cleared conditions — new clips added, preprocessing needed")
        else:
            logger.info(f"All {len(existing_entries)} clips up to date, no new media to process")
            # Make sure JSON is written even if no new media
            if not dataset_json_path.exists():
                with open(dataset_json_path, "w") as f:
                    json.dump(existing_entries, f, indent=2)

        if not existing_entries:
            raise RuntimeError(
                "No usable clips produced. All media was discarded "
                "(no faces detected or processing failed)."
            )

        clip_paths = [Path(e["media_path"]) for e in existing_entries]

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

        # Build cast list for Gemma QC: when character_refs is populated, the
        # caption step asks Gemma whether any known character is visible and
        # drops the clip if not.  Names are derived from ref filenames.
        cast_names: list[str] = sorted(character_refs.keys()) if character_refs else []

        self._caption_dataset_json(
            dataset_json_path, clip_paths, caption_mode, lora_trigger,
            ollama_url=ollama_url, ollama_model=ollama_model,
            target_face_b64=target_face_b64, caption_style=caption_style,
            cast_names=cast_names,
        )

        # --- PHASE 3: Encode conditions and latents ---
        # Reload entries from JSON — captioning may have rejected some clips,
        # and existing_entries in memory still has the pre-rejection list.
        with open(dataset_json_path) as f:
            existing_entries = json.load(f)

        conditions_dir = output_dir / "conditions"
        latents_dir = output_dir / "latents"
        need_subprocess = False

        # Phase 3a: Text encoding
        if clip is not None:
            self._encode_conditions_inprocess(clip, dataset_json_path, conditions_dir)
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

        return (str(output_dir), str(dataset_json_path))

    def _encode_conditions_inprocess(self, clip, dataset_json_path: Path, conditions_dir: Path):
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
        """Add 1-frame image buckets for each unique image resolution in the dataset."""
        existing_buckets = set(resolution_buckets.split(";"))
        for entry in entries:
            media_path = entry.get("media_path", "")
            if not media_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            # Read actual image dimensions
            img = cv2.imread(media_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            image_bucket = f"{w}x{h}x1"
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
        for whole-frame visual matching (puppets, props, objects).

        Returns {trigger: {"type": "face"|"clip", "embedding": np.ndarray}}.
        """
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

            # Try face detection first — most reliable for humans
            face = _detect_face_dnn(frame)
            if face is not None:
                embedding = _get_face_embedding(frame, face)
                if embedding is not None:
                    refs[trigger] = {"type": "face", "embedding": embedding}
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
            refs[trigger] = {"type": "clip", "embedding": emb}
            logger.info(f"Loaded character reference (clip-vision): {trigger}")
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
        else:
            # Crop to target aspect ratio but keep source resolution — no downscale
            crop = self._get_face_crop(frame, target_w, target_h, face_detection)
            if crop is None:
                logger.info(f"No face detected, skipping: {img_path.name}")
                return None
            cx, cy, cw, ch = crop
            output = frame[cy:cy+ch, cx:cx+cw]

        cv2.imwrite(str(out_path), output)
        logger.info(f"Processed image: {img_path.name} -> {out_path.name}")
        return out_path

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
    ) -> list[Path]:
        """Split a video into chunks, detect faces, crop or scale.
        Returns list of output clip paths (skips chunks with no face when face_detection is on).
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

        # Split into chunks of target_frames
        clips = []

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
        clips.extend(existing_chunks)

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
        start_frame = first_frame + chunk_idx * target_frames

        if chunk_idx > 0:
            logger.info(
                f"{video_path.name}: resuming from chunk {chunk_idx} "
                f"(frame {start_frame}) — {len(existing_chunks)} existing clips kept"
            )

        while start_frame < last_frame:
            end_frame = min(start_frame + target_frames, last_frame)
            # Skip chunks that are too short (less than half the target)
            if end_frame - start_frame < target_frames // 2:
                break

            # Persist resume progress before doing any work on this chunk.
            # On a crash / interrupt, the next run resumes at exactly this
            # chunk index and fast-forwards everything before it.
            try:
                progress_file.write_text(str(chunk_idx))
            except OSError:
                pass

            # Sample a frame from the middle of the chunk for face detection
            sample_frame_idx = start_frame + (end_frame - start_frame) // 2
            crop = None
            matched_sample = None

            # Clip-selection gating.  Two modes:
            #   - Multi-character: chunk must contain at least one known
            #     character from character_refs (face OR clip-vision match).
            #     Face crop uses the first detected face if available,
            #     otherwise falls back to center crop.
            #   - Single-character / default: requires a detected face; if a
            #     target_embedding is set, that face must match the target.
            sample_positions = [
                sample_frame_idx,
                start_frame,
                start_frame + (end_frame - start_frame) // 4,
                start_frame + 3 * (end_frame - start_frame) // 4,
            ]
            sample_positions = [p for p in sample_positions if start_frame <= p < end_frame]

            if face_detection and character_refs:
                matched_names: set[str] = set()
                face_anchor = None  # (frame, rect) for crop if we find one
                for try_idx in sample_positions:
                    sample = self._read_frame(video_path, try_idx)
                    if sample is None:
                        continue
                    hits = self._match_characters_in_frame(
                        sample, character_refs, face_similarity,
                        clip_vision=clip_vision,
                        first_match_only=True,
                    )
                    if hits:
                        matched_names |= hits
                        # Remember the first face we see in any matched sample
                        # so face_crop mode still gets a proper anchor.
                        if face_anchor is None:
                            face = _detect_face_dnn(sample)
                            if face is not None:
                                face_anchor = (sample, face)
                        break
                if not matched_names:
                    logger.info(f"No known characters in chunk {chunk_idx} of {video_path.name}, skipping")
                    start_frame = end_frame
                    chunk_idx += 1
                    continue
                logger.info(f"Chunk {chunk_idx}: matched {', '.join(sorted(matched_names))}")
                if crop_mode == "face_crop":
                    if face_anchor is not None:
                        _, face = face_anchor
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
                        face_found = True
                        break
                if not face_found:
                    logger.info(f"No face in chunk {chunk_idx} of {video_path.name}, skipping")
                    start_frame = end_frame
                    chunk_idx += 1
                    continue

                if target_embedding is not None and matched_sample is not None:
                    face = _detect_face_dnn(matched_sample)
                    if face is None or not self._check_face_match(matched_sample, face, target_embedding, face_similarity):
                        logger.info(f"Face doesn't match target in chunk {chunk_idx} of {video_path.name}, skipping")
                        start_frame = end_frame
                        chunk_idx += 1
                        continue
            elif crop_mode == "face_crop":
                # No face detection + face_crop: center crop
                crop = self._center_crop(frame_w, frame_h, target_w, target_h)

            out_path = clips_dir / f"{video_path.stem}_chunk{chunk_idx:04d}.mp4"

            if not out_path.exists():
                start_time = start_frame / fps
                num_frames = end_frame - start_frame

                if crop_mode == "full_frame":
                    # No resize — keep source resolution, VAE encode step handles resize
                    vf = None
                else:
                    # Crop to target aspect ratio at source resolution (no downscale)
                    cx, cy, cw, ch = crop
                    vf = f"crop={cw}:{ch}:{cx}:{cy}"

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start_time:.4f}",
                    "-i", str(video_path),
                    "-frames:v", str(num_frames),
                ]
                if vf:
                    cmd += ["-vf", vf]
                cmd += [
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                ]
                if with_audio:
                    # Use -frames:v above to bound video; cap audio to same duration
                    # so the clip ends cleanly when the video does.
                    clip_duration = num_frames / fps
                    cmd += [
                        "-t", f"{clip_duration:.4f}",
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

            clips.append(out_path)
            logger.info(f"Extracted clip: {video_path.name} chunk {chunk_idx} -> {out_path.name}")

            start_frame = end_frame
            chunk_idx += 1

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

        removed_count = 0
        i = 0
        while i < len(entries):
            entry = entries[i]
            if entry.get("caption"):
                i += 1
                continue

            vf = Path(entry["media_path"])

            if caption_mode == "ollama":
                logger.info(f"[{i+1}/{len(entries)}] Captioning: {vf.name}")
                caption = self._caption_with_ollama(
                    vf, ollama_url, ollama_model, lora_trigger,
                    target_face_b64=target_face_b64,
                    caption_style=caption_style,
                    cast_names=cast_names,
                )

                # LLM QC: if it returned MISMATCH, remove this entry and track it
                if caption.strip().upper().startswith("MISMATCH"):
                    logger.info(f"  LLM QC: MISMATCH — removing {vf.name}")
                    rejected_entry = entries.pop(i)
                    removed_count += 1
                    # Save rejection to rejected.json so it doesn't get re-processed
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
            elif caption_mode == "auto_filename":
                name = vf.stem
                for suffix in ["_img"]:
                    name = name.removesuffix(suffix)
                name = re.sub(r'_chunk\d+$', '', name)
                caption = name.replace("_", " ").replace("-", " ")
                caption = " ".join(caption.split())
                if lora_trigger:
                    caption = f"{lora_trigger} {caption}"
            else:
                caption = vf.stem
                if lora_trigger:
                    caption = f"{lora_trigger} {caption}"

            entry["caption"] = caption

            # Save JSON incrementally after each new caption
            with open(dataset_json_path, "w") as f:
                json.dump(entries, f, indent=2)

            i += 1

        if removed_count:
            logger.info(f"LLM QC removed {removed_count} mismatched clips")
        logger.info(f"Captioning complete for {len(entries)} entries")

    def _caption_with_ollama(
        self, clip_path: Path, ollama_url: str, ollama_model: str, lora_trigger: str,
        target_face_b64: str | None = None, caption_style: str = "subject",
        cast_names: list[str] | None = None,
    ) -> str:
        """Caption a clip frame via Ollama. If target_face_b64 is provided,
        first verifies the person matches (separate call), then captions clean.
        Returns 'MISMATCH' if the person doesn't match."""
        import base64
        import urllib.request
        import urllib.error

        # Extract multiple evenly-spaced frames so the LLM can observe shot
        # changes and secondary characters that may not be present in any single
        # frame.  Falls back to the single-frame picker on failure.
        frames = self._extract_caption_frames(clip_path, num_frames=5)
        if not frames:
            single = self._extract_caption_frame(clip_path)
            if single is None:
                logger.warning(f"Could not extract frame from {clip_path.name}, using filename")
                return clip_path.stem
            frames = [single]

        b64_images: list[str] = []
        for fr in frames:
            ok, buf = cv2.imencode(".png", fr)
            if ok:
                b64_images.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        if not b64_images:
            return clip_path.stem
        # Single-frame handle for face verification (use the middle-ish frame)
        b64_image = b64_images[len(b64_images) // 2]

        base = ollama_url.rstrip("/")

        # --- Step 1: Verify face match (if target provided) ---
        if target_face_b64:
            verify_result = self._ollama_verify_face(
                base, ollama_model, target_face_b64, b64_image,
            )
            if verify_result == "MISMATCH":
                return "MISMATCH"

        # --- Step 1b: Cast QC ---
        # When a character refs folder was used, ask Gemma (over all extracted
        # frames) whether any of the known cast members are visible.  Reject
        # the clip if it says no.  This catches false positives from the
        # face/clip-vision filter at extraction time.
        if cast_names:
            qc_result = self._ollama_verify_cast(
                base, ollama_model, cast_names, b64_images,
            )
            if qc_result == "MISMATCH":
                return "MISMATCH"

        # --- Step 2: Caption (clean, no reference image) ---
        trigger_instruction = ""
        if lora_trigger:
            trigger_instruction = (
                f" Always refer to the main subject as '{lora_trigger}'."
                f" Start the caption with '{lora_trigger}'."
            )

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
                "You are a video training caption generator for a multi-character scene."
                " Describe what you see in a single detailed paragraph."
                " Every named character should be referred to strictly by their given name — "
                "never by physical description."
                " Describe: background, environment, camera angle, poses, interactions between "
                "characters, expressions, actions, and clothing context."
                " Do NOT describe any character's defining physical features (face shape, hair, "
                "skin tone, eye color, body type, costume colors) — the character names handle "
                "identity."
                " Do NOT describe lighting, color grading, contrast, shadows, film grain, "
                "depth of field, or visual mood — the LoRA should learn the visual style."
                " If you recognize additional well-known named characters beyond the ones already "
                "confirmed, name them too. Only name characters you are confident about; do not "
                "guess."
                " Be factual and specific. "
                "Do not use poetic language or speculation. Do not start with 'The image shows' or "
                "'In this frame'."
            ),
        }

        system_prompt = _CAPTION_PROMPTS.get(caption_style, _CAPTION_PROMPTS["subject"]) + trigger_instruction

        payload = json.dumps({
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "These images are evenly-spaced frames sampled from a single "
                        "short video clip, shown in chronological order. Write ONE "
                        "caption that describes the whole clip as a single scene. "
                        "If the clip cuts between shots or characters, include what "
                        "appears across the different frames."
                    ),
                    "images": b64_images,
                },
            ],
            "stream": False,
            "keep_alive": 0,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                caption = result.get("message", {}).get("content", "").strip()
                import re
                caption = re.sub(r"<think>.*?</think>", "", caption, flags=re.DOTALL)
                caption = re.sub(r"<think>.*", "", caption, flags=re.DOTALL)
                caption = caption.strip()
                if caption:
                    logger.info(f"Caption: {caption[:100]}...")
                    return caption
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.error(f"Ollama captioning failed for {clip_path.name}: {e}")

        return f"{lora_trigger} {clip_path.stem}" if lora_trigger else clip_path.stem

    def _ollama_verify_cast(
        self, base_url: str, model: str, cast_names: list[str], image_b64s: list[str],
    ) -> str:
        """Ask Gemma whether any character from the given cast list appears in
        the provided frames.  Returns 'MATCH' or 'MISMATCH'.  On any failure,
        returns 'MATCH' so we don't accidentally drop good clips."""
        import urllib.request
        import urllib.error

        names_csv = ", ".join(cast_names)
        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cast verification system for a TV show dataset. "
                        "You will receive a list of known character names and several "
                        "frames sampled from a short video clip. Decide whether ANY "
                        "of the named characters visibly appears in the clip. "
                        "Respond with ONLY the word MATCH or MISMATCH. No other text. "
                        "If you are unsure, say MATCH."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Known characters: {names_csv}.\n\n"
                        "Does at least one of these characters appear in these frames? "
                        "Answer MATCH or MISMATCH."
                    ),
                    "images": image_b64s,
                },
            ],
            "stream": False,
            "keep_alive": 0,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
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
            logger.warning(f"Cast verification failed: {e}, assuming match")
            return "MATCH"

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
            "keep_alive": 0,
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

    def _extract_caption_frame(self, clip_path: Path) -> np.ndarray | None:
        """Extract the best frame from a clip for captioning.
        Tries middle frame first, then samples others looking for a face."""
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        # Sample positions: middle, 1/4, 3/4, start, end
        candidates = [total // 2, total // 4, (total * 3) // 4, 0, max(0, total - 1)]

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
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _extract_caption_frames(self, clip_path: Path, num_frames: int = 5) -> list[np.ndarray]:
        """Extract N evenly-spaced frames from a clip for multi-frame captioning.
        This lets the LLM see shot changes and secondary characters that may only
        appear in part of the clip.  Returns an empty list on failure."""
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        num_frames = max(1, num_frames)
        if total <= num_frames:
            indices = list(range(total))
        else:
            # Evenly spaced across the clip, inclusive of ends trimmed slightly
            step = total / (num_frames + 1)
            indices = [int(step * (i + 1)) for i in range(num_frames)]

        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames
