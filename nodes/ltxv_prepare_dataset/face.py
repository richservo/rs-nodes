"""Face detection, embedding, matching and crop computation.

Wraps InsightFace (antelopev2 / SCRFD + ArcFace) and a lightweight
DNN fallback. Holds the loaded face-app + per-frame analysis caches
in module globals. Call `unload_face_models()` to release them.
Crop helpers (`compute_face_crop`, `compute_pan_and_scan`) take
`_FACE_PADDING` from the module global; orchestrator calls
`set_face_padding(...)` once at the start of prepare().
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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

# Cache the actual frame (strong ref), not id(frame): CPython recycles
# object ids immediately once an array is freed, so a freshly-decoded
# video frame can land at the same heap slot a reference image just
# vacated, hit a stale cache, and return faces whose bboxes don't exist
# in the new frame — observed as "no face detected" until the address
# happens to differ.
_last_analysis_frame: "np.ndarray | None" = None
_last_analysis_faces: list = []


def set_face_padding(value: float) -> None:
    """Update the module-level face padding used by crop helpers.
    Replaces direct `global _FACE_PADDING; _FACE_PADDING = value` patterns
    so the orchestrator doesn't need to reach into module internals."""
    global _FACE_PADDING
    _FACE_PADDING = value


def get_face_app():
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


def analyze_frame(frame: np.ndarray) -> list:
    """Run InsightFace on frame; memoize by id(frame) so repeat calls are
    free. Drops low-confidence detections (cactus/tiki/poster/cartoon
    patterns) so they don't pollute hit counts or unknown-face counts."""
    global _last_analysis_frame, _last_analysis_faces
    app = get_face_app()
    if app is None:
        _last_analysis_frame = frame
        _last_analysis_faces = []
        return []
    if frame is _last_analysis_frame:
        return _last_analysis_faces
    raw = app.get(frame)
    # Filter by detector confidence — non-human face-like patterns
    # typically score below _FACE_DET_MIN_CONFIDENCE.
    faces = [
        f for f in raw
        if float(getattr(f, "det_score", 1.0)) >= _FACE_DET_MIN_CONFIDENCE
    ]
    _last_analysis_frame = frame
    _last_analysis_faces = faces
    return faces


def detect_face_dnn(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest face bbox (x, y, w, h) or None."""
    faces = detect_all_faces_dnn(frame)
    if not faces:
        return None
    return max(faces, key=lambda r: r[2] * r[3])


def detect_all_faces_dnn(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """All face bboxes as (x, y, w, h)."""
    out: list[tuple[int, int, int, int]] = []
    for f in analyze_frame(frame):
        x1, y1, x2, y2 = f.bbox.astype(int).tolist()
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        out.append((x1, y1, w, h))
    return out


def get_face_embedding(frame: np.ndarray, face_rect: tuple[int, int, int, int]) -> np.ndarray | None:
    """Return the 512-d L2-normalized ArcFace embedding for the face at face_rect.
    Matches face_rect against the cached analysis by IoU."""
    faces = analyze_frame(frame)
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


def match_face(embedding: np.ndarray, target_embedding: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embeddings. Range [-1, 1]; ~0.5+ = same person for ArcFace."""
    return float(np.dot(embedding, target_embedding))


def has_unknown_face(
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
    for f in analyze_frame(frame):
        score = float(getattr(f, "det_score", 1.0))
        if score < _FACE_UNKNOWN_MIN_CONFIDENCE:
            # Low-confidence face-like pattern (prop/mask/poster/cartoon).
            # Don't flag as an unknown human extra.
            continue
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            continue
        emb = np.asarray(emb, dtype=np.float32)
        best_sim = max(match_face(emb, ref) for ref in face_refs)
        if best_sim < threshold:
            return True
    return False


def compute_face_crop(
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


def compute_pan_and_scan(
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


def unload_face_models() -> list[str]:
    """Free InsightFace state and reset all face module globals.
    Returns a list of names freed (e.g. ["InsightFace"]) so the
    orchestrator can include them in its consolidated freed-models log.
    Safe to call even if nothing was loaded.
    """
    global _face_app, _face_app_checked
    global _last_analysis_frame, _last_analysis_faces

    freed: list[str] = []
    if _face_app is not None:
        try:
            del _face_app
        except Exception:
            pass
        _face_app = None
        _face_app_checked = False
        freed.append("InsightFace")

    # Drop frame memoization (holds a reference to the most recent frame)
    _last_analysis_frame = None
    _last_analysis_faces = []

    return freed
