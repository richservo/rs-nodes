import json
import logging
import os
import time

import folder_paths

logger = logging.getLogger(__name__)

_STATE_FILENAME = "video_batch_state.json"
_DEFAULT_EXTS = ("mp4", "mov", "mkv", "webm", "avi", "m4v")


def _state_path() -> str:
    return os.path.join(folder_paths.get_output_directory(), _STATE_FILENAME)


def _load_state() -> dict:
    path = _state_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"RSVideoBatchLoader: could not read state file, starting fresh ({e})")
    return {}


def _save_state(state: dict) -> None:
    with open(_state_path(), "w") as f:
        json.dump(state, f, indent=2)


class RSVideoBatchLoader:
    """Iterates through videos in a folder by index. Auto-advances each run.

    Pair the video_path output with VHS_LoadVideoPath (or any path-based
    video loader) — this node deals with file selection, not decoding.

    State is per-folder, keyed by absolute path, persisted in
    video_batch_state.json in ComfyUI's output directory. Each run reads
    the persisted next-index, picks that file, then increments. Wraps
    around at the end of the listing when wrap=True.

    `index` widget is used as the starting value on the very first run
    for a folder, AND on every run when auto_increment=False (manual
    selection mode).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to folder containing videos",
                }),
                "index": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Manual index (used when auto_increment=False, or as start value on first run)",
                }),
                "extensions": ("STRING", {
                    "default": "mp4,mov,mkv,webm,avi,m4v",
                    "tooltip": "Comma-separated video extensions to include",
                }),
                "auto_increment": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Advance index by 1 on every run (persisted per-folder)",
                }),
                "wrap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Wrap to 0 after last file (else clamp at last)",
                }),
                "reset": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force index back to 0 on this run",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("video_path", "filename", "current_index", "total_count")
    FUNCTION = "execute"
    CATEGORY = "rs-nodes/io"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute so auto-increment advances every queue.
        return time.time()

    def execute(self, folder_path: str, index: int, extensions: str,
                auto_increment: bool, wrap: bool, reset: bool):
        folder_path = folder_path.strip().strip('"').strip("'")
        if not folder_path:
            raise ValueError("folder_path is empty")
        if not os.path.isdir(folder_path):
            raise ValueError(f"folder_path is not a directory: {folder_path!r}")
        folder_abs = os.path.abspath(folder_path)

        exts = tuple(
            f".{e.strip().lstrip('.').lower()}"
            for e in extensions.split(",") if e.strip()
        ) or tuple(f".{e}" for e in _DEFAULT_EXTS)

        videos = sorted(
            f for f in os.listdir(folder_abs)
            if f.lower().endswith(exts)
            and os.path.isfile(os.path.join(folder_abs, f))
        )
        total = len(videos)
        if total == 0:
            raise ValueError(
                f"No videos found in {folder_abs!r} matching extensions {list(exts)}"
            )

        state = _load_state()
        if auto_increment and not reset:
            current = state.get(folder_abs, index)
        else:
            current = 0 if reset else index

        if wrap:
            current = current % total
        else:
            current = max(0, min(current, total - 1))

        if auto_increment:
            next_idx = current + 1
            if wrap:
                next_idx = next_idx % total
            else:
                next_idx = min(next_idx, total - 1)
            state[folder_abs] = next_idx
            _save_state(state)

        filename = videos[current]
        video_path = os.path.join(folder_abs, filename)

        logger.info(
            f"RSVideoBatchLoader: {filename} (index {current + 1}/{total} in {folder_abs})"
        )

        return (video_path, filename, current, total)
