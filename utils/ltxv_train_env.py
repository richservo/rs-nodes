import atexit
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

import comfy.model_management as mm

logger = logging.getLogger(__name__)

# Track running training processes for cleanup
_running_processes: list[subprocess.Popen] = []


def _cleanup_processes():
    for proc in _running_processes:
        if proc.poll() is None:
            logger.warning(f"Killing orphaned training process {proc.pid}")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


atexit.register(_cleanup_processes)


def get_submodule_root() -> Path:
    return Path(__file__).parent.parent / "LTX-2"


def validate_submodule():
    root = get_submodule_root()
    trainer_scripts = root / "packages" / "ltx-trainer" / "scripts"
    if not trainer_scripts.exists():
        raise RuntimeError(
            "LTX-2 submodule not initialized. Run:\n"
            "  cd custom_nodes/rs-nodes\n"
            "  git submodule update --init --recursive"
        )


_DEFAULT_TEXT_ENCODER_REPO = "google/gemma-3-12b-it"


def get_text_encoder_download_dir() -> Path:
    """Get the directory where text encoders should be downloaded to."""
    import folder_paths
    for folder_name in ("clip", "text_encoders"):
        try:
            paths = folder_paths.get_folder_paths(folder_name)
            if paths:
                p = Path(paths[0])
                p.mkdir(parents=True, exist_ok=True)
                return p
        except Exception:
            pass
    raise RuntimeError("Could not find a clip or text_encoders model directory")


def ensure_text_encoder(path: str) -> str:
    """Validate text encoder path. If it doesn't exist or is the auto-download
    sentinel, download google/gemma-3-12b-it to the clip model folder.
    Returns the validated path."""
    p = Path(path)

    # Auto-download if path is empty, doesn't exist, or is the sentinel value
    if not path or path.startswith("auto") or not p.exists():
        return _auto_download_text_encoder()

    # Validate it has tokenizer.model AND model weights. The previous check
    # only verified tokenizer.model, which meant a partial / interrupted
    # download (where config + tokenizer arrive first but the multi-GB
    # safetensors shards never finish) passed validation, then the
    # subprocess crashed deep in gemma_8bit._find_gemma_subpath looking for
    # `model*.safetensors`. Catching it here gives a recoverable error
    # path: missing weights → re-download.
    tokenizer = p / "tokenizer.model"
    if not tokenizer.exists():
        found = list(p.rglob("tokenizer.model"))
        if not found:
            raise ValueError(
                f"Text encoder directory must contain tokenizer.model: {path}\n"
                "Expected a full HF model directory (e.g. google/gemma-3-12b-it)"
            )

    weights = list(p.glob("model*.safetensors")) + list(p.rglob("model*.safetensors"))
    if not weights:
        logger.warning(
            f"Text encoder at {path} is missing model*.safetensors weights "
            f"(likely an interrupted download). Re-downloading..."
        )
        return _auto_download_text_encoder()

    return str(p)


def _auto_download_text_encoder() -> str:
    """Download google/gemma-3-12b-it to the clip model folder."""
    dest = get_text_encoder_download_dir() / "gemma-3-12b-it"

    # Already downloaded? Require BOTH tokenizer.model AND at least one
    # model*.safetensors weight shard. Without the weights check, an
    # interrupted download (config + tokenizer arrive first, multi-GB
    # safetensors shards never finish) is treated as complete; the
    # subprocess then crashes deep in gemma_8bit._find_gemma_subpath.
    # Detecting it here lets snapshot_download below resume the partial
    # download cleanly.
    if dest.exists() and (dest / "tokenizer.model").exists():
        weights = list(dest.glob("model*.safetensors"))
        if weights:
            logger.info(f"Text encoder already downloaded: {dest}")
            return str(dest)
        logger.warning(
            f"Text encoder at {dest} has tokenizer but no model*.safetensors "
            f"weights — likely an interrupted download. Resuming..."
        )

    logger.info(f"Downloading {_DEFAULT_TEXT_ENCODER_REPO} to {dest} (first-time setup)...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            _DEFAULT_TEXT_ENCODER_REPO,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to auto-download the text encoder.\n"
            "Install it with: pip install huggingface_hub\n"
            "Or manually download google/gemma-3-12b-it to your clip/ model folder."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {_DEFAULT_TEXT_ENCODER_REPO}: {e}\n"
            "You may need to run: huggingface-cli login\n"
            "Gemma-3 requires accepting the license at https://huggingface.co/google/gemma-3-12b-it"
        )

    if not (dest / "tokenizer.model").exists():
        raise RuntimeError(f"Download completed but tokenizer.model not found in {dest}")

    logger.info(f"Text encoder downloaded successfully: {dest}")
    return str(dest)


def validate_text_encoder_path(path: str) -> str:
    """Validate or auto-download text encoder. Returns the resolved path."""
    return ensure_text_encoder(path)


def get_trainer_env() -> dict:
    root = get_submodule_root()
    env = os.environ.copy()

    # Add ltx-core and ltx-trainer src to PYTHONPATH
    ltx_core_src = str(root / "packages" / "ltx-core" / "src")
    ltx_trainer_src = str(root / "packages" / "ltx-trainer" / "src")
    existing = env.get("PYTHONPATH", "")
    parts = [ltx_trainer_src, ltx_core_src]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    # CUDA memory config for training
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_MODULE_LOADING"] = "LAZY"

    # Force UTF-8 so Rich console doesn't crash on Windows cp1252
    env["PYTHONIOENCODING"] = "utf-8"

    return env


def get_script_path(script_name: str) -> str:
    path = get_submodule_root() / "packages" / "ltx-trainer" / "scripts" / script_name
    if not path.exists():
        raise FileNotFoundError(f"Training script not found: {path}")
    return str(path)


def free_vram():
    mm.unload_all_models()
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_training_subprocess(
    cmd: list[str],
    env: dict,
    progress_bar=None,
    total_steps: int = 0,
    cancel_check=None,
) -> int:
    """Run a subprocess with progress tracking and cancellation support.

    Args:
        cmd: Command to run
        env: Environment variables
        progress_bar: comfy.utils.ProgressBar instance
        total_steps: Total steps for progress tracking
        cancel_check: Callable that raises if processing should be interrupted

    Returns:
        Return code of the subprocess
    """
    logger.info(f"Running: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    _running_processes.append(proc)

    current_step = 0
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"[train] {line}")

            # Parse progress from trainer output
            # The trainer logs lines like "Step 100/2000" or progress bar output
            if progress_bar and total_steps > 0:
                step = _parse_step(line, total_steps)
                if step is not None and step > current_step:
                    if progress_bar:
                        progress_bar.update_absolute(step, total_steps)
                    current_step = step

            # Check for cancellation
            if cancel_check:
                try:
                    cancel_check()
                except Exception:
                    logger.info("Training cancelled by user")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise

        proc.wait()
    finally:
        if proc in _running_processes:
            _running_processes.remove(proc)

    return proc.returncode


def _parse_step(line: str, total_steps: int) -> int | None:
    """Parse current step from trainer output lines.

    Matches patterns like:
    - "Step 100/2000"
    - "step=100"
    - "steps: 100/2000"
    - Progress bar: " 50%|...| 100/200"
    """
    import re

    # "Step N/M" or "step N/M"
    m = re.search(r'[Ss]tep\s*[:=]?\s*(\d+)', line)
    if m:
        return int(m.group(1))

    # tqdm-style progress: " 50%|...| 100/200"
    m = re.search(r'\|\s*(\d+)/(\d+)', line)
    if m:
        return int(m.group(1))

    return None
