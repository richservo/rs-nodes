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
    net = _get_face_detector()
    if net is None:
        return _detect_face_haar(frame)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    best = None
    best_area = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < _FACE_CONFIDENCE_THRESHOLD:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2 - x1, y2 - y1)

    return best


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
                "caption_style": (["subject", "subject + style", "style", "motion", "general"], {"default": "subject", "tooltip": "Captioning focus: subject=person identity, subject+style=person+cinematography, style=visual aesthetics only, motion=movement/action, general=balanced"}),
                "ollama_url": ("STRING", {"default": "http://localhost:11434", "tooltip": "Ollama server URL (only used when caption_mode=ollama)"}),
                "ollama_model": ("STRING", {"default": "gemma3:27b", "tooltip": "Ollama vision model for captioning (only used when caption_mode=ollama)"}),
                "with_audio": ("BOOLEAN", {"default": False, "tooltip": "Extract and encode audio latents"}),
                "load_text_encoder_in_8bit": ("BOOLEAN", {"default": True, "tooltip": "Load text encoder in 8-bit to save VRAM"}),
                "crop_mode": (["face_crop", "full_frame"], {"default": "face_crop", "tooltip": "face_crop=crop around detected face, full_frame=scale entire frame to target resolution (aspect-preserving, letterboxed). Both modes use face detection for clip selection when enabled."}),
                "face_detection": ("BOOLEAN", {"default": True, "tooltip": "Enable face detection: crop around faces, discard clips with no faces"}),
                "target_face": ("IMAGE", {"tooltip": "Reference face image. When connected, only keeps clips matching this specific face."}),
                "face_similarity": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Face match threshold (0-1). Higher = stricter matching. 0.40 is a good default."}),
                "face_padding": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Padding around detected face (0.6 = 60% extra for head+shoulders)"}),
                "conditioning_folder": ("STRING", {"default": "", "tooltip": "IC-LoRA: folder of conditioning input videos (depth maps, edge maps, poses). Matched to media_folder by filename. Media folder is the ground truth output."}),
                "clip": ("CLIP", {"tooltip": "Text encoder (from CheckpointLoaderSimple). When connected, encodes captions in-process instead of slow subprocess."}),
                "vae": ("VAE", {"tooltip": "VAE (from CheckpointLoaderSimple). When connected, encodes latents in-process instead of slow subprocess."}),
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
        face_similarity: float = 0.40,
        face_padding: float = 0.6,
        conditioning_folder: str = "",
        clip=None,
        vae=None,
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
                        crop_mode=crop_mode,
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

        self._caption_dataset_json(
            dataset_json_path, clip_paths, caption_mode, lora_trigger,
            ollama_url=ollama_url, ollama_model=ollama_model,
            target_face_b64=target_face_b64, caption_style=caption_style,
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
        if vae is not None:
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
        crop_mode: str = "face_crop",
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

        # Split into chunks of target_frames
        clips = []
        chunk_idx = 0
        start_frame = 0

        while start_frame < total_frames:
            end_frame = min(start_frame + target_frames, total_frames)
            # Skip chunks that are too short (less than half the target)
            if end_frame - start_frame < target_frames // 2:
                break

            # Sample a frame from the middle of the chunk for face detection
            sample_frame_idx = start_frame + (end_frame - start_frame) // 2
            crop = None
            matched_sample = None

            # Face detection for clip selection (both modes use this)
            if face_detection:
                face_found = False
                for try_idx in [sample_frame_idx,
                                start_frame,
                                start_frame + (end_frame - start_frame) // 4,
                                start_frame + 3 * (end_frame - start_frame) // 4]:
                    if not (start_frame <= try_idx < end_frame):
                        continue
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

                # Check face identity match if target provided
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
                    "-an",
                    str(out_path),
                ]

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
    ) -> str:
        """Caption a clip frame via Ollama. If target_face_b64 is provided,
        first verifies the person matches (separate call), then captions clean.
        Returns 'MISMATCH' if the person doesn't match."""
        import base64
        import urllib.request
        import urllib.error

        # Extract a frame from the clip
        frame = self._extract_caption_frame(clip_path)
        if frame is None:
            logger.warning(f"Could not extract frame from {clip_path.name}, using filename")
            return clip_path.stem

        # Encode frame as base64 PNG
        _, buf = cv2.imencode(".png", frame)
        b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")

        base = ollama_url.rstrip("/")

        # --- Step 1: Verify face match (if target provided) ---
        if target_face_b64:
            verify_result = self._ollama_verify_face(
                base, ollama_model, target_face_b64, b64_image,
            )
            if verify_result == "MISMATCH":
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
        }

        system_prompt = _CAPTION_PROMPTS.get(caption_style, _CAPTION_PROMPTS["subject"]) + trigger_instruction

        payload = json.dumps({
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Describe this frame for video model training.",
                    "images": [b64_image],
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
