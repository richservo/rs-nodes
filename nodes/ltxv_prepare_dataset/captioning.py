"""Captioning / vision-language layer for the prepare-dataset node.

Wraps Ollama / Gemma (vision LLM) for clip captioning, identity
verification, location matching, and a length-check + re-roll loop
that prevents runaway captions from corrupting the dataset.

dataset.json is consulted only when an entry needs work — entries
whose condition file already exists on disk are skipped (artifact
trumps JSON).
"""
import json
import logging
import re
import sys as _sys
import time as _t
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .audio import format_speaker_display
from .dataset_io import (
    condition_path_for_clip,
    normalize_loaded_entries,
    reject_entry,
)
from .face import detect_face_dnn

logger = logging.getLogger(__name__)

# Self-heal / quota-recovery tuning knobs (see Reference/dataset-self-heal-spec.md).
#
# MAX_CAPTION_CHARS — fast pre-filter before the encoder sees a caption.
# 600 chars ≈ 128 tokens after BPE; tuned to comfortably fit COND_TOKEN_LIMIT.
#
# MAX_CAPTION_RETRIES — re-rolls per clip before giving up and rejecting it.
# Gemma-class models are probabilistic, so wiping context and re-rolling
# usually produces a completely different caption.
MAX_CAPTION_CHARS = 600
MAX_CAPTION_RETRIES = 3

# Skip the first/last _CAPTION_EDGE_MARGIN frames when sampling for captions
# so transition artifacts at the clip edges (e.g. tail-end of a neighbouring
# scene leaking in via cut detection) don't end up dominating the caption
# content at the extreme head / tail. The LLM will happily describe
# whatever it sees even if it only exists for one frame; skipping
# the edges keeps the caption tied to the main on-screen content.
# Clamped at 25% of the clip length so short clips still have a
# usable sample range.
_CAPTION_EDGE_MARGIN = 10

# Strip ' / ' / ' / ` that quote words but keep apostrophes inside words.
# 'pee-wee' → pee-wee, `pee-wee` → pee-wee, said 'no'. → said no.
# it's / Pee-wee's / don't all stay intact (apostrophe between letters).
_QUOTE_STRIP_RE = re.compile(r"(?<![A-Za-z])['‘’`]|['‘’`](?![A-Za-z])")


# ---- Caption prompt definitions (shared by prep + batch) ----
CAPTION_PROMPTS = {
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


def titlecase_name(name: str) -> str:
    """Capitalize the first letter of each space-separated token; leave
    hyphenated forms with single capitalization (pee-wee → Pee-wee, not
    Pee-Wee).  miss yvonne → Miss Yvonne; cowboy curtis → Cowboy Curtis."""
    return " ".join(w[:1].upper() + w[1:] for w in name.split(" ") if w)


def normalize_caption_for_encode(caption: str, character_names=None) -> str:
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
    caption = _QUOTE_STRIP_RE.sub("", caption)
    if character_names:
        # Longest-first so multi-word names (miss yvonne) win against
        # single-word substrings (yvonne) inside the same scan.
        for name in sorted(character_names, key=len, reverse=True):
            if not name:
                continue
            pattern = r"\b" + re.escape(name) + r"\b"
            caption = re.sub(
                pattern,
                titlecase_name(name),
                caption,
                flags=re.IGNORECASE,
            )
    return caption


def caption_frame_range(total: int) -> tuple[int, int]:
    """Return (lo, hi) inclusive frame indices to sample for captions."""
    safe = min(_CAPTION_EDGE_MARGIN, max(0, (total - 1) // 4))
    lo = safe
    hi = max(lo, total - 1 - safe)
    return lo, hi


def extract_caption_frame(clip_path: Path) -> np.ndarray | None:
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

    lo, hi = caption_frame_range(total)
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
        face = detect_face_dnn(frame)
        if face is not None:
            cap.release()
            return frame

    # Fallback: return middle frame even without a face
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_caption_frames(clip_path: Path, num_frames: int = 5) -> list[np.ndarray]:
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
    lo, hi = caption_frame_range(total)
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


def ensure_ollama_model(base_url: str, model: str) -> None:
    """Check whether the requested Ollama model is installed; if not,
    pull it.  Streams pull progress to the logger so the user can
    watch the download.  Raises RuntimeError on failure so the node
    aborts cleanly instead of marking every clip as MISMATCH."""
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


def label_ref_image(b64: str, label: str) -> str:
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


def ollama_verify_face(
    base_url: str, model: str, ref_b64: str, clip_b64: str,
) -> str:
    """Ask Ollama if two images show the same person. Returns 'MATCH' or 'MISMATCH'."""
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
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
            answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)
            answer = answer.strip().upper()
            if "MISMATCH" in answer:
                return "MISMATCH"
            return "MATCH"
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        logger.warning(f"Face verification failed: {e}, assuming match")
        return "MATCH"


def ollama_identify_cast(
    base_url: str, model: str,
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
    cast_names = [n for n, _ in cast_refs]
    num_refs = len(cast_refs)
    num_frames = len(clip_image_b64s)

    # Label reference images with "REF: name" and clip frames already
    # have "Frame N" burned on, so the model can visually distinguish them.
    labeled_cast_refs = [
        (name, label_ref_image(b64, name))
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


def ollama_identify_location(
    base_url: str, model: str,
    location_refs: list[tuple[str, str]],
    clip_image_b64s: list[str],
) -> str | None:
    """Ask Gemma which labeled location the clip takes place in, by
    visually comparing clip frames to the reference images.  Returns
    the matched location name, or None if no confident match.  Soft
    — never rejects a clip."""
    loc_names = [n for n, _ in location_refs]
    num_refs = len(location_refs)
    num_frames = len(clip_image_b64s)

    labeled_loc_refs = [
        (name, label_ref_image(b64, name))
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


def prepare_clip_for_caption(
    clip_path: Path, ollama_url: str, ollama_model: str, lora_trigger: str,
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

    frames = extract_caption_frames(clip_path, num_frames=5)
    if not frames:
        single = extract_caption_frame(clip_path)
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

        system_prompt = CAPTION_PROMPTS["multi_character"] + trigger_instruction

        # Label and collect all cast refs
        all_cast_refs = []
        if cast_refs:
            all_cast_refs = [
                (name, label_ref_image(b64, name))
                for name, b64 in cast_refs
            ]
        # Label and collect all location refs
        all_loc_refs = []
        if location_refs:
            all_loc_refs = [
                (name, label_ref_image(b64, name))
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
            if ollama_verify_face(base, ollama_model, target_face_b64, b64_image) == "MISMATCH":
                return None

        # --- Cast identification ---
        confirmed_cast: list[str] | None = None
        if cast_names and cast_refs:
            confirmed_cast = ollama_identify_cast(
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
            identified_location = ollama_identify_location(base, ollama_model, location_refs, b64_images)
            logger.info(f"  Location: {identified_location or 'NONE'} ({_t.time() - _loc_t0:.1f}s)")

        # --- Build user message + images for caption ---
        trigger_instruction = ""
        if lora_trigger:
            trigger_instruction = (
                f" Always refer to the main subject as '{lora_trigger}'."
                f" Start the caption with '{lora_trigger}'."
            )

        effective_style = "multi_character" if confirmed_cast else caption_style
        system_prompt = CAPTION_PROMPTS.get(effective_style, CAPTION_PROMPTS["subject"]) + trigger_instruction

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
                (name, label_ref_image(b64, name))
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
            sp = format_speaker_display(s.get("speaker", "unknown"))
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


def caption_single_ollama(
    prep: dict, ollama_url: str, ollama_model: str,
    conversation_messages: list[dict] | None,
) -> tuple[str, list[dict], float, int]:
    """Caption a single clip, continuing a persistent multi-turn
    conversation.  Previous turns keep their text (captions +
    thinking context) but images are stripped so only the current
    clip's frames are sent/processed.
    Returns (caption, updated_messages, caption_only_time, token_count).
    caption_only_time excludes thinking — only measures generation."""
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


def caption_dataset_json(
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
    with open(dataset_json_path) as f:
        entries = json.load(f)

    # Resolve any relative media_path to absolute against output_dir.
    # Without this, the artifact-trumps-JSON guard below computes the
    # wrong condition file path (relative_to() throws on relative
    # paths and the fallback flattens the clips/ subdir away), causing
    # the guard to miss every entry and re-caption an entire dataset.
    normalize_loaded_entries(entries, dataset_json_path.parent)

    # Check if all entries already have captions (and none are marked mismatch)
    uncaptioned = [i for i, e in enumerate(entries) if not e.get("caption")]
    if not uncaptioned:
        logger.info("All clips already captioned, skipping")
        return

    # Of the uncaptioned entries, how many already have a condition file
    # on disk? Those are the ones the artifact-trumps-JSON guard will
    # skip — no captioning, no encoding. Surface the real number up
    # front so the user sees we're not redoing the entire dataset.
    output_dir_for_count = dataset_json_path.parent
    actual_work = sum(
        1 for i in uncaptioned
        if not condition_path_for_clip(
            output_dir_for_count, Path(entries[i]["media_path"])
        ).exists()
    )
    skipped_due_to_existing_cond = len(uncaptioned) - actual_work
    if skipped_due_to_existing_cond:
        logger.info(
            f"Captioning: {actual_work} clip(s) need work "
            f"(JSON says {len(uncaptioned)} lack captions, but "
            f"{skipped_due_to_existing_cond} already have an encoded "
            f"condition file on disk — JSON ignored for those, "
            f"encoded artifact wins)"
        )
    else:
        logger.info(
            f"{len(uncaptioned)} clips need captioning, "
            f"{len(entries) - len(uncaptioned)} already done"
        )

    # Make sure the requested Ollama model is installed before we
    # start the loop.  If it's not, pull it (streaming progress to
    # the log).  Without this, every clip would 404 and get rejected.
    #
    # Skip when there's no actual captioning work — the artifact-trumps-
    # JSON guard above may have eliminated every uncaptioned entry by
    # finding their condition files already on disk. In that case
    # contacting Ollama is pointless and would falsely fail the run if
    # the user has Ollama stopped (which is reasonable when they're
    # only filling missing encodes).
    if caption_mode == "ollama" and actual_work > 0:
        ensure_ollama_model(ollama_url, ollama_model)
    elif caption_mode == "ollama":
        logger.info(
            "All uncaptioned entries already have encoded conditions on "
            "disk — skipping Ollama startup check (nothing to caption)."
        )
        return

    removed_count = 0
    # Persistent multi-turn conversation state for ollama captions.
    # The model's thinking from earlier clips carries forward.
    ollama_messages: list[dict] | None = None
    caption_first_time: float = 0.0  # gen time of first caption in session
    caption_gen_times: list[float] = []  # all gen times in session for averaging

    # The encoded condition file is the source of truth. If it exists
    # on disk, this entry has already been captioned + encoded —
    # regardless of whether dataset.json still has the caption text
    # (e.g., a wiped or partial JSON). Skipping here avoids re-doing
    # work the artifacts already prove was done.
    output_dir_for_skip = dataset_json_path.parent

    i = 0
    while i < len(entries):
        entry = entries[i]
        if entry.get("caption"):
            i += 1
            continue

        vf = Path(entry["media_path"])

        # Artifact-trumps-JSON guard. If the condition file is already
        # there, this clip is encoded; the missing caption in JSON is
        # cosmetic and should not trigger captioning + re-encode.
        existing_cond = condition_path_for_clip(output_dir_for_skip, vf)
        if existing_cond.exists():
            logger.info(
                f"[{i+1}/{len(entries)}] Skip {vf.name}: condition exists "
                f"on disk (caption missing in JSON is cosmetic only)"
            )
            i += 1
            continue

        if caption_mode == "ollama":
            logger.info(f"[{i+1}/{len(entries)}] Captioning: {vf.name}")
            prep = prepare_clip_for_caption(
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
                reject_entry(i, entries, dataset_json_path, "llm_mismatch")
                removed_count += 1
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

            # Caption via persistent multi-turn conversation, with a
            # length-check + re-roll loop. Probabilistic VLMs occasionally
            # emit page-long captions that, post-encoding, balloon
            # attention memory at training time and cause C-level aborts.
            # On overshoot we wipe Ollama's conversation context (the
            # runaway turn would poison subsequent rolls if carried
            # forward) and try again. After MAX_CAPTION_RETRIES failures
            # the clip is fully rejected: removed from dataset.json,
            # logged in rejected.json, and its clip / latents / audio
            # latents files are deleted so it can't be re-introduced.
            attempt = 0
            caption = ""
            cap_gen_time = 0.0
            token_count = 0
            while True:
                caption, ollama_messages, cap_gen_time, token_count = caption_single_ollama(
                    prep, ollama_url, ollama_model, ollama_messages,
                )
                if len(caption) <= MAX_CAPTION_CHARS:
                    break
                attempt += 1
                logger.warning(
                    f"  Caption {len(caption)} chars > {MAX_CAPTION_CHARS} "
                    f"(attempt {attempt}/{MAX_CAPTION_RETRIES}) — wiping context, re-rolling. "
                    f"First 200 chars: {caption[:200]!r}"
                )
                # Wipe context AND gen-time tracking; the slow / runaway
                # turn distorts both, and the next session should start
                # clean.
                ollama_messages = None
                caption_first_time = 0.0
                caption_gen_times.clear()
                if attempt >= MAX_CAPTION_RETRIES:
                    break

            if len(caption) > MAX_CAPTION_CHARS:
                # Persistently runaway — clean reject and purge artifacts
                # so the clip can't be re-introduced.
                logger.error(
                    f"  Caption stayed over {MAX_CAPTION_CHARS} chars after "
                    f"{MAX_CAPTION_RETRIES} attempts — rejecting {vf.name}"
                )
                reject_entry(
                    i, entries, dataset_json_path,
                    "caption_too_long", purge_artifacts=True,
                )
                removed_count += 1
                continue

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
