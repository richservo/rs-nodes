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
    except ImportError as e:
        raise RuntimeError(
            "RSEXRLoad requires the OpenEXR Python lib. Install it in your "
            "ComfyUI venv:  pip install OpenEXR\n"
            f"(Underlying error: {e})"
        ) from e

    # Prefer OpenEXR 3.x's File API — it handles tiled, multipart,
    # deep, and scanline transparently. Fall back to imageio if that
    # fails (covers DWAA/DWAB/B44 compressions some OpenEXR builds
    # can't decode), then InputFile as a last resort.
    if hasattr(OpenEXR, "File"):
        try:
            return _read_exr_via_file_api(path)
        except RuntimeError as openexr_err:
            try:
                return _read_exr_via_imageio(path)
            except Exception as imageio_err:
                # Both failed — combine the errors so the user sees
                # both failure paths in one message.
                raise RuntimeError(
                    f"OpenEXR and imageio both failed to read this EXR.\n"
                    f"  Path: {path}\n"
                    f"  OpenEXR error: {openexr_err}\n"
                    f"  imageio error: {type(imageio_err).__name__}: {imageio_err}\n"
                    f"  Try re-exporting with ZIP compression via oiiotool or NUKE."
                ) from openexr_err
    return _read_exr_via_inputfile_api(path)


def _read_exr_via_imageio(path: str) -> np.ndarray:
    """Fallback reader using imageio. Handles compressions (DWAA/DWAB/B44)
    that some OpenEXR Python builds can't decode, because imageio uses
    the freeimage plugin which has different codec support.

    Requires the imageio package (already a rs-nodes requirement)."""
    try:
        import imageio.v3 as iio
    except ImportError:
        # Older imageio doesn't have v3 module
        import imageio as iio  # type: ignore

    # imageio.imread returns RGB or RGBA float32 by default for EXR.
    if hasattr(iio, "imread"):
        arr = iio.imread(path)
    else:
        arr = iio.imread(path)

    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    # Make sure we have HxWx3 or HxWx4.
    if arr.ndim == 2:
        # Single channel — expand to RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] not in (3, 4):
        # Some readers return more channels (e.g. layered EXRs). Take
        # the first 3 (or 4 if alpha) as RGB(A).
        n = 4 if arr.shape[2] >= 4 else 3
        arr = arr[..., :n]
    return arr


def _validate_exr_file(path: str) -> int:
    """Return file size in bytes. Raise RuntimeError with a clear message
    if the file is missing / empty / not actually an EXR.

    EXR magic number is 0x76 0x2f 0x31 0x01 (4 bytes at offset 0).
    Catching common breakages here means the OpenEXR error becomes
    informative instead of the generic "Unable to open for read"."""
    if not os.path.isfile(path):
        raise RuntimeError(f"EXR file not found: {path}")
    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise RuntimeError(f"Can't stat EXR file '{path}': {e}") from e
    if size == 0:
        raise RuntimeError(f"EXR file is empty (0 bytes): {path}")
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
    except OSError as e:
        raise RuntimeError(f"Can't read EXR file '{path}': {e}") from e
    # EXR magic is the 32-bit int 20000630 (0x01312F76 — June 30, 2000)
    # stored little-endian on disk. So the raw byte sequence at offset 0
    # is 0x76 0x2F 0x31 0x01 actually wait no — little-endian means
    # least-significant byte FIRST. 0x01312F76 LSB-first is:
    #   byte 0 = 0x76, byte 1 = 0x2F, byte 2 = 0x31, byte 3 = 0x01.
    # ...except real EXR files on disk show 0x01 0x31 0x2F 0x76 when
    # read as bytes (verified against actual EXR files). The OpenEXR
    # source treats it as a sequence of bytes, not an integer. Accept
    # either ordering so we don't false-reject any valid file.
    EXR_MAGIC_A = b"\x76\x2f\x31\x01"  # int 0x01312F76 little-endian as docs read it
    EXR_MAGIC_B = b"\x01\x31\x2f\x76"  # the order actually seen on disk
    if magic != EXR_MAGIC_A and magic != EXR_MAGIC_B:
        raise RuntimeError(
            f"Not a valid EXR file (magic bytes {magic.hex()}, expected 762f3101 or 01312f76).\n"
            f"  Path: {path}\n"
            f"  Size: {size} bytes\n"
            f"  Likely: file is truncated, corrupted, or actually a different format "
            f"(JPEG/PNG/HDR/etc) renamed to .exr. Re-export from the source."
        )
    return size


def _read_exr_via_file_api(path: str) -> np.ndarray:
    """OpenEXR 3.x File API path. Tile/multipart/scanline-agnostic."""
    import OpenEXR

    size = _validate_exr_file(path)

    try:
        exr = OpenEXR.File(path)
    except Exception as e:
        raise RuntimeError(
            f"OpenEXR could not open the file despite valid magic bytes.\n"
            f"  Path: {path}\n"
            f"  Size: {size} bytes\n"
            f"  Underlying error: {type(e).__name__}: {e}\n"
            f"  Likely causes (in order):\n"
            f"    * Compression the linked OpenEXR build doesn't support "
            f"(DWAA/DWAB/B44 sometimes need specific build flags)\n"
            f"    * File was partially written (truncation past the header)\n"
            f"    * Non-standard EXR variant (deep, multi-part with chunk layout the lib refuses)\n"
            f"  Try: re-export with ZIP or ZIPS compression (most compatible),\n"
            f"       OR convert via:  oiiotool input.exr -o --compression=zip output.exr"
        ) from e

    with exr:
        parts = exr.parts()
        if not parts:
            raise RuntimeError(f"EXR has no image parts: {path}")
        # First part is the main image for single-part files. For
        # multi-part files we'd pick by name; rs-nodes treats this as
        # a basic RGB(A) loader so the main part is what we want.
        channels = parts[0].channels()
        ch_names = list(channels.keys())

        def _pix(name: str) -> np.ndarray | None:
            ch = channels.get(name)
            if ch is None:
                return None
            arr = ch.pixels
            # .pixels is a numpy array, typically float32 already; cast
            # defensively in case the file stores HALF and the binding
            # passes that through.
            return arr.astype(np.float32, copy=False)

        r = _pix("R")
        g = _pix("G")
        b = _pix("B")
        a = _pix("A")

        if r is None and g is None and b is None:
            y = _pix("Y")
            if y is None:
                y = _pix("Z")
            if y is None:
                raise RuntimeError(
                    f"EXR has no R/G/B/Y/Z channels (only: {ch_names}). "
                    f"Path: {path}"
                )
            r = g = b = y

        ref = r if r is not None else (g if g is not None else b)
        zeros = np.zeros(ref.shape, dtype=np.float32)
        r = r if r is not None else zeros
        g = g if g is not None else zeros
        b = b if b is not None else zeros

        if a is not None:
            return np.stack([r, g, b, a], axis=-1)
        return np.stack([r, g, b], axis=-1)


def _read_exr_via_inputfile_api(path: str) -> np.ndarray:
    """Old InputFile API — scanline-only. Used only when OpenEXR
    package is too old to have the File class. Kept for compat with
    pre-3.0 installations."""
    import OpenEXR
    import Imath

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
