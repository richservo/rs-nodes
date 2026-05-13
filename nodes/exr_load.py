"""EXR image loader with standard ComfyUI upload-widget UX.

Drop-in replacement for the built-in LoadImage when the file is .exr:
same upload button, same dropdown of files in ComfyUI's input/ folder,
no path-typing. Single-file or whole-sequence modes; outputs a normal
IMAGE tensor (no clamping, so HDR / linear values above 1.0 survive).

Backend: OpenEXR Python bindings (matches RSEXRSequenceSave). Reads
RGB or RGBA depending on what's in the file. Multi-layer EXRs (with
extra named channels like Z / N / albedo) load the main RGB(A) layer
only — extra channels are ignored.
"""

import logging
import os
import re

import numpy as np
import torch

import folder_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Match a trailing zero-padded frame number, e.g.
#   "render_00042.exr"  ->  prefix="render_"  digits="00042"  ext=".exr"
#   "shot.0173.exr"     ->  prefix="shot."    digits="0173"   ext=".exr"
# Greedy prefix so a name like "v002_00010.exr" matches the trailing
# 00010 rather than the embedded 002.
_SEQ_RE = re.compile(r"^(.*?)(\d+)(\.exr)$", re.IGNORECASE)


def _list_exrs_in_input() -> list[str]:
    """All .exr files (top level) of ComfyUI's input dir, sorted."""
    input_dir = folder_paths.get_input_directory()
    if not os.path.isdir(input_dir):
        return []
    out = []
    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if os.path.isfile(full) and name.lower().endswith(".exr"):
            out.append(name)
    return sorted(out)


def _detect_sequence(start_path: str) -> list[str]:
    """Given an absolute path to one EXR, find every sibling EXR
    sharing the same stem-with-trailing-digits and return them sorted
    by frame number. Returns [start_path] alone if the filename
    doesn't end in digits (single-frame export from RSEXRSequenceSave
    always does, but a hand-named file might not)."""
    folder = os.path.dirname(start_path)
    base = os.path.basename(start_path)
    m = _SEQ_RE.match(base)
    if not m:
        return [start_path]

    prefix, _digits, ext = m.group(1), m.group(2), m.group(3)
    pattern = re.compile(
        r"^" + re.escape(prefix) + r"(\d+)" + re.escape(ext) + r"$",
        re.IGNORECASE,
    )
    matches: list[tuple[int, str]] = []
    try:
        for name in os.listdir(folder):
            mm = pattern.match(name)
            if mm:
                matches.append((int(mm.group(1)), os.path.join(folder, name)))
    except OSError:
        return [start_path]
    if not matches:
        return [start_path]
    matches.sort(key=lambda x: x[0])
    return [p for _n, p in matches]


def _read_exr_to_numpy(path: str) -> np.ndarray:
    """Read one EXR -> float32 HxWxC ndarray in RGB(A) channel order.

    Channels we care about: R, G, B, and optionally A. We always
    request FLOAT pixel type so half-float files get expanded
    automatically. Linear / HDR values above 1.0 are preserved
    untouched.
    """
    try:
        import OpenEXR
        import Imath
    except ImportError as e:
        raise RuntimeError(
            "RSEXRLoad requires the OpenEXR Python lib. Install it in your "
            "ComfyUI venv:  pip install OpenEXR Imath\n"
            f"(Underlying error: {e})"
        ) from e

    f = OpenEXR.InputFile(path)
    try:
        header = f.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        channel_names = list(header["channels"].keys())
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        def _ch(name: str) -> np.ndarray | None:
            if name not in channel_names:
                return None
            raw = f.channel(name, pt)
            return np.frombuffer(raw, dtype=np.float32).reshape(h, w)

        # Standard RGB(A) layout. If the file is single-channel
        # ("Y" / "luminance") expand it across RGB so downstream
        # IMAGE-typed nodes accept it.
        r = _ch("R")
        g = _ch("G")
        b = _ch("B")
        a = _ch("A")

        if r is None and g is None and b is None:
            y = _ch("Y") or _ch("Z")
            if y is None:
                raise RuntimeError(
                    f"EXR has no R/G/B/Y/Z channels (only: {channel_names}). "
                    f"Path: {path}"
                )
            r = g = b = y

        # Fill in any missing channel with zeros (rare; mostly defensive).
        z = np.zeros((h, w), dtype=np.float32)
        r = r if r is not None else z
        g = g if g is not None else z
        b = b if b is not None else z

        if a is not None:
            return np.stack([r, g, b, a], axis=-1)
        return np.stack([r, g, b], axis=-1)
    finally:
        f.close()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class RSEXRLoad:
    """Load a single EXR or an auto-detected EXR sequence.

    Single mode: just the picked file. Output shape = [1, H, W, 3 or 4].
    Sequence mode: every file sharing the trailing-digit stem
    (e.g. picking `render_00042.exr` loads `render_*.exr` in
    numeric order). Output shape = [N, H, W, C]. All frames must
    share the same dimensions / channel count.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_exrs_in_input()
        # image_upload=True is the standard ComfyUI marker that
        # turns the dropdown into an upload widget (same UX as the
        # built-in LoadImage).
        return {
            "required": {
                "exr": (files, {"image_upload": True}),
                "mode": (
                    ["single", "sequence"],
                    {
                        "default": "single",
                        "tooltip": (
                            "single = just the picked file. "
                            "sequence = every EXR in the same folder sharing "
                            "the trailing-digit stem (e.g. picking "
                            "render_00042.exr loads render_*.exr in order)."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load"
    CATEGORY = "rs-nodes"

    @classmethod
    def IS_CHANGED(cls, exr, mode):
        # Re-run when the chosen file's mtime changes. For sequence
        # mode this is a lower bound — re-running the workflow after
        # adding extra frames at the end won't bust the cache, but
        # editing the picked frame will. Good enough for the common
        # "tweak a frame, re-run" loop without making the loader
        # always-dirty.
        try:
            path = folder_paths.get_annotated_filepath(exr)
            if os.path.isfile(path):
                return os.path.getmtime(path)
        except Exception:
            pass
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, exr, mode):
        if not exr:
            return "no EXR file selected"
        try:
            path = folder_paths.get_annotated_filepath(exr)
        except Exception as e:
            return f"could not resolve {exr!r}: {e}"
        if not os.path.isfile(path):
            return f"file not found: {path}"
        return True

    def load(self, exr: str, mode: str):
        # get_annotated_filepath handles ComfyUI's "input [subfolder/file]"
        # encoding from the upload widget. Falls through to a plain
        # input-dir lookup for legacy paths.
        path = folder_paths.get_annotated_filepath(exr)
        if not os.path.isfile(path):
            raise RuntimeError(f"RSEXRLoad: file not found: {path}")

        if mode == "sequence":
            paths = _detect_sequence(path)
            logger.info(f"RSEXRLoad: sequence mode — {len(paths)} frame(s) starting at {os.path.basename(path)}")
        else:
            paths = [path]
            logger.info(f"RSEXRLoad: single frame — {os.path.basename(path)}")

        frames = []
        ref_shape: tuple[int, ...] | None = None
        for p in paths:
            arr = _read_exr_to_numpy(p)
            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                raise RuntimeError(
                    f"RSEXRLoad: frame shape mismatch in sequence — "
                    f"{os.path.basename(paths[0])} is {ref_shape} but "
                    f"{os.path.basename(p)} is {arr.shape}. All frames "
                    f"in a sequence must share the same dimensions and "
                    f"channel count."
                )
            frames.append(arr)

        stacked = np.stack(frames, axis=0)  # [N, H, W, C]
        tensor = torch.from_numpy(stacked.copy())  # float32
        return (tensor,)
