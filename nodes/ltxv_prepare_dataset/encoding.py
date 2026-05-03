"""Text + latent + audio-latent encoding for the prepare-dataset node.

In-process encoders that use ComfyUI's already-loaded CLIP / VAE /
audio_vae nodes. Skips files already on disk (idempotent). When
audio_vae is connected, audio latents are encoded in-process here in
the trainer's expected on-disk format, and the LTX-2 subprocess is
not needed for audio.
"""
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

import comfy.utils

from .dataset_io import (
    audio_latent_path_for_clip,
    condition_path_for_clip,
    latent_path_for_clip,
    normalize_loaded_entries,
    COND_TOKEN_LIMIT,
)

# captioning is imported lazily inside encode_conditions_inprocess to avoid
# circular import at module-load time
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def encode_conditions_inprocess(clip, dataset_json_path: Path, conditions_dir: Path,
                                 character_names=None):
    """Encode captions in-process using ComfyUI's already-loaded CLIP text encoder.

    ComfyUI's LTX CLIP runs Gemma + feature extractor (blocks 1+2 of the text
    encoder pipeline). This produces the same output as process_captions.py's
    text_encoder.encode() + embeddings_processor.feature_extractor() but without
    loading Gemma from disk.

    LTX-2   (single_linear): cond is [B, seq, 3840] — video and audio share features.
    LTX-2.3 (dual_linear):   cond is [B, seq, 6144] — split at 4096 for video/audio.
    """
    from .captioning import normalize_caption_for_encode  # avoid circular import

    conditions_dir = Path(conditions_dir)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_json_path) as f:
        entries = json.load(f)

    data_root = dataset_json_path.parent
    # Normalize relative media_paths to absolute so downstream path
    # logic is consistent — without this, _condition_path_for_clip
    # used to fall back to a flat <stem>.pt root path and the encoder
    # would write to conditions/foo.pt instead of conditions/clips/foo.pt,
    # corrupting the dataset layout. _condition_path_for_clip is now
    # robust to relative paths too, but normalizing here means the
    # encoder works correctly even if that helper changes.
    normalize_loaded_entries(entries, data_root)
    to_encode = []
    for entry in entries:
        media_path = Path(entry["media_path"])
        # Reuse the shared path helper so audit / encoder always look
        # at exactly the same condition file for a given clip.
        output_file = condition_path_for_clip(data_root, media_path)
        if not output_file.exists():
            to_encode.append((entry, output_file))

    if not to_encode:
        logger.info("All conditions already exist, skipping text encoding")
        return

    logger.info(f"Encoding {len(to_encode)} captions in-process with ComfyUI CLIP...")
    pbar = comfy.utils.ProgressBar(len(to_encode))

    for i, (entry, output_file) in enumerate(to_encode):
        caption = entry.get("caption", "")
        caption = normalize_caption_for_encode(caption, character_names)

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


def encode_latents_inprocess(
    vae, dataset_json_path: Path, latents_dir: Path, existing_entries: list[dict],
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
    # Normalize relative media_paths to absolute so cv2.VideoCapture can
    # read them without depending on the process's current working
    # directory. Without this a relative "clips\foo.mp4" silently fails
    # to open and the encoder writes a corrupt latent.
    normalize_loaded_entries(existing_entries, data_root)
    to_encode = []
    for entry in existing_entries:
        media_path = Path(entry["media_path"])
        output_file = latent_path_for_clip(data_root, media_path)
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
            pixels, fps = load_video_frames(media_path)
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

        # Match the LTX-2 subprocess's on-disk dtype (bfloat16). ComfyUI's
        # VAE returns float32 by default; if we save float32 alongside
        # bfloat16 files written by previous subprocess runs, the dataset
        # has a single anomalous-dtype file that may cause dataloader
        # collate to upcast everything (memory hit) or to fail strict
        # dtype checks. Cast here so every latent file in the dataset is
        # the same dtype regardless of which encoder produced it.
        latents_save = latents_save.to(torch.bfloat16)

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


def load_video_frames(video_path: Path):
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


def resolve_resolution_buckets(resolution_buckets: str, entries: list[dict]) -> str:
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


# ---------------------------------------------------------------------------
# Audio latent encoding
# ---------------------------------------------------------------------------

def _extract_audio_waveform(
    video_path: Path, target_duration: float
) -> dict | None:
    """Extract audio from a video file using ffmpeg, returned as ComfyUI's
    AUDIO type: ``{"waveform": Tensor[1, C, samples], "sample_rate": int}``.

    Mirrors the LTX-2 subprocess's ``_extract_audio`` behavior so the
    waveform that hits audio_vae.encode is byte-equivalent: stereo,
    44.1 kHz, pcm_s16le → float32, padded to ``target_duration`` if
    shorter. Returns None if ffmpeg is missing or extraction fails (the
    caller falls back to a zero-tensor placeholder, matching the
    subprocess's silent-clip path).
    """
    import soundfile as sf

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        logger.warning("ffmpeg not found on PATH — cannot extract audio")
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            ffmpeg_bin, "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            "-t", str(target_duration),
            "-y", tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return None

        data, sample_rate = sf.read(tmp_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            # mono: [samples] -> [1, samples]
            waveform = waveform.unsqueeze(0)
        else:
            # stereo: [samples, 2] -> [2, samples]
            waveform = waveform.T

        target_samples = int(target_duration * sample_rate)
        current_samples = waveform.shape[-1]
        if current_samples < target_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_samples - current_samples)
            )

        # ComfyUI's AUDIO type expects a batch dim: [1, channels, samples]
        return {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": int(sample_rate),
        }
    except Exception as e:
        logger.debug(f"Could not extract audio from {video_path}: {e}")
        return None
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _video_duration_seconds(video_path: Path) -> float | None:
    """Best-effort video duration in seconds. Returns None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_count <= 0 or fps <= 0:
            return None
        return float(frame_count / fps)
    finally:
        cap.release()


def encode_audio_latents_inprocess(
    audio_vae,
    dataset_json_path: Path,
    audio_latents_dir: Path,
) -> None:
    """Encode audio waveforms in-process using ComfyUI's audio_vae node.

    Mirrors the on-disk format the LTX-2 subprocess writes (see
    process_videos.py::encode_audio) so the trainer's dataloader reads
    these files identically:

    .. code:: python

        {
            "latents": Tensor[C, T, F],   # batch dim squeezed
            "num_time_steps": int,
            "frequency_bins": int,
            "duration": float,
        }

    Idempotent: skips clips whose audio_latent file already exists.
    For clips with no extractable audio (silent / image / ffmpeg
    failure) writes a zeros placeholder, matching the subprocess's
    silent-clip path so the dataloader doesn't see missing files.

    Args:
        audio_vae: ComfyUI VAE-typed node (the audio_vae input from
            an LTXV-AV checkpoint loader). Must implement ``.encode(audio)``
            taking the AUDIO type and returning latents shaped
            [B, C, T, F].
        dataset_json_path: Path to dataset.json. Read fresh; entries are
            normalized to absolute paths internally.
        audio_latents_dir: Output directory; clip-relative subdirs are
            preserved (e.g. ``audio_latents/clips/foo.pt``).
    """
    audio_latents_dir = Path(audio_latents_dir)
    audio_latents_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_json_path) as f:
        entries = json.load(f)
    data_root = dataset_json_path.parent
    normalize_loaded_entries(entries, data_root)

    to_encode = []
    for entry in entries:
        media_path = Path(entry["media_path"])
        output_file = audio_latent_path_for_clip(data_root, media_path)
        if not output_file.exists():
            to_encode.append((entry, media_path, output_file))

    if not to_encode:
        logger.info("All audio latents already exist, skipping audio encoding")
        return

    logger.info(
        f"Encoding {len(to_encode)} audio clips in-process with ComfyUI audio_vae..."
    )
    pbar = comfy.utils.ProgressBar(len(to_encode))

    encoded = 0
    placeholders = 0
    for i, (entry, media_path, output_file) in enumerate(to_encode):
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Image clips never have audio; write a placeholder so the
        # dataloader's collate doesn't see a missing file.
        if media_path.suffix.lower() in IMAGE_EXTENSIONS:
            torch.save(_audio_placeholder(), output_file)
            placeholders += 1
            pbar.update_absolute(i + 1, len(to_encode))
            continue

        duration = _video_duration_seconds(media_path)
        if duration is None or duration <= 0.0:
            logger.warning(
                f"Could not determine duration of {media_path.name}; "
                f"writing audio placeholder"
            )
            torch.save(_audio_placeholder(), output_file)
            placeholders += 1
            pbar.update_absolute(i + 1, len(to_encode))
            continue

        audio = _extract_audio_waveform(media_path, duration)
        if audio is None:
            torch.save(_audio_placeholder(duration=duration), output_file)
            placeholders += 1
            pbar.update_absolute(i + 1, len(to_encode))
            continue

        # Match ComfyUI's canonical audio encode path
        # (comfy_extras/nodes_audio.py::VAEEncodeAudio.execute):
        # resample to the VAE's expected rate if needed, then pass a
        # channels-LAST waveform tensor (not the AUDIO dict). The VAE
        # wrapper handles channel-axis flipping + dispatch to the
        # underlying autoencoder internally.
        sample_rate = int(audio["sample_rate"])
        vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 44100))
        if vae_sample_rate != sample_rate:
            import torchaudio
            waveform = torchaudio.functional.resample(
                audio["waveform"], sample_rate, vae_sample_rate,
            )
        else:
            waveform = audio["waveform"]

        with torch.inference_mode():
            latents = audio_vae.encode(waveform.movedim(1, -1))

        # Tolerate either a tensor or a {"samples": Tensor} dict — different
        # VAE wrapper versions return different things.
        if isinstance(latents, dict) and "samples" in latents:
            latents = latents["samples"]
        if not torch.is_tensor(latents):
            raise TypeError(
                f"audio_vae.encode returned unexpected type {type(latents)}; "
                f"expected Tensor or dict-with-samples"
            )

        # Squeeze batch dim to match the trainer's on-disk shape [C, T, F].
        if latents.dim() == 4 and latents.shape[0] == 1:
            latents = latents.squeeze(0)

        if latents.dim() != 3:
            raise ValueError(
                f"audio latent shape {tuple(latents.shape)} is not [C,T,F] "
                f"after squeezing batch — file format would be wrong"
            )

        c, t, f_bins = latents.shape
        torch.save(
            {
                "latents": latents.cpu().contiguous(),
                "num_time_steps": int(t),
                "frequency_bins": int(f_bins),
                "duration": float(duration),
            },
            output_file,
        )
        encoded += 1
        pbar.update_absolute(i + 1, len(to_encode))

    logger.info(
        f"Audio encoding complete: {encoded} encoded, {placeholders} placeholders"
    )


def _audio_placeholder(duration: float = 0.0) -> dict:
    """Zeros-tensor stand-in for clips with no audio. Matches the shape
    the LTX-2 subprocess uses for silent clips so the dataloader's
    collate never sees a missing file."""
    return {
        "latents": torch.zeros(8, 1, 16, dtype=torch.float32),
        "num_time_steps": 1,
        "frequency_bins": 16,
        "duration": float(duration),
    }
