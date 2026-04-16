"""In-process LTX-2 LoRA trainer.

Runs training directly inside ComfyUI without launching a subprocess, allowing
the already-loaded transformer and VAE to be reused instead of reloading from
disk. Only the embeddings processor (~50 MB) is loaded fresh from the checkpoint.

Key design decisions vs the subprocess approach:
- No Accelerate / DDP wrappers — single-GPU only.
- No wandb, no Rich progress bars — uses ComfyUI's ProgressBar instead.
- Gradient checkpointing and quantization are supported.
- Validation reuses the same VAE decoder that was passed in (kept on CPU, moved
  to GPU only during validation, then moved back).
- LoRA weights are saved in ComfyUI-compatible format (diffusion_model. prefix).
"""

import gc
import logging
import shutil
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

# Build a tuple of OOM-like exception types.  PyTorch 2.10+ raises
# torch.AcceleratorError for CUDA OOMs instead of torch.cuda.OutOfMemoryError,
# so we catch whichever classes exist on the installed version.
_OOM_EXCEPTIONS: tuple = (torch.cuda.OutOfMemoryError,)
if hasattr(torch, "AcceleratorError"):
    _OOM_EXCEPTIONS = _OOM_EXCEPTIONS + (torch.AcceleratorError,)

logger = logging.getLogger(__name__)


def _setup_ltx_paths() -> None:
    """Add ltx-core and ltx-trainer src directories to sys.path if not already present."""
    import sys

    rs_nodes_root = Path(__file__).parent.parent
    ltx2_root = rs_nodes_root / "LTX-2"
    paths_to_add = [
        str(ltx2_root / "packages" / "ltx-trainer" / "src"),
        str(ltx2_root / "packages" / "ltx-core" / "src"),
    ]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


class InProcessTrainer:
    """Trains a LoRA adapter for LTX-2 entirely in-process (no subprocess).

    The caller is responsible for:
    - Extracting raw model objects from ComfyUI wrappers before passing them.
    - Calling cleanup() after training (even if training raises).
    - Freeing VRAM before calling train() so there is headroom for optimizer
      states and activations.

    Args:
        transformer: Raw LTXModel diffusion transformer (from model.model.diffusion_model).
            Must NOT already have LoRA applied. The trainer wraps it with PEFT internally.
        embeddings_processor: EmbeddingsProcessor loaded from the LTX checkpoint.
            Provides feature-extractor connectors used in each training step.
            Kept on CUDA throughout training.
        vae_decoder: VideoDecoder kept on CPU. Moved to CUDA only during validation.
        scheduler: LTX2Scheduler (stateless, used only in validation).
        dataset: PrecomputedDataset with latents + conditions.
        training_strategy: TextToVideoStrategy (or video_to_video variant).
        timestep_sampler: Timestep sampler for flow matching.
        output_dir: Directory to write checkpoints and validation samples to.
        loras_dir: ComfyUI loras folder — final LoRA is copied here on completion.
        lora_name: Filename (without .safetensors) for the output LoRA file.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        target_modules: List of module name patterns to apply LoRA to.
        learning_rate: AdamW learning rate.
        total_steps: Total training steps.
        optimizer_type: "adamw8bit" | "adamw".
        scheduler_type: LR scheduler type ("linear" | "constant" | "cosine" | ...).
        max_grad_norm: Gradient clipping norm.
        gradient_checkpointing: Enable activation checkpointing on the transformer.
        quantization: Quantization level, e.g. "fp8-quanto". None = disabled.
        keep_last_n: How many intermediate checkpoints to keep (-1 = keep all).
        checkpoint_interval: Save a checkpoint every N steps (0 = disabled).
        validation_config: Dict with keys for validation (prompt, width, height,
            num_frames, frame_rate, num_inference_steps, guidance_scale, seed,
            stg_scale, stg_blocks, stg_mode, generate_audio, interval). None = skip.
        cached_validation_embeddings: Pre-computed CachedPromptEmbeddings for the
            validation prompt. None = skip validation.
        audio_vae_decoder: AudioDecoder for audio generation in validation. None = no audio.
        vocoder: Vocoder for audio generation in validation. None = no audio.
        seed: Global random seed.
        resume_checkpoint: Path to a .safetensors LoRA checkpoint to resume from.
    """

    def __init__(
        self,
        *,
        transformer,
        embeddings_processor,
        vae_decoder,
        scheduler,
        dataset,
        training_strategy,
        timestep_sampler,
        output_dir: str | Path,
        loras_dir: str | Path,
        lora_name: str,
        rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: list[str] | None = None,
        learning_rate: float = 1e-4,
        total_steps: int = 2000,
        optimizer_type: str = "adamw8bit",
        scheduler_type: str = "linear",
        lr_cycle_steps: int = 0,
        max_grad_norm: float = 1.0,
        gradient_checkpointing: bool = True,
        quantization: str | None = "fp8-quanto",
        keep_last_n: int = 2,
        checkpoint_interval: int = 500,
        validation_config: dict | None = None,
        cached_validation_embeddings=None,
        audio_vae_decoder=None,
        vocoder=None,
        seed: int = 42,
        resume_checkpoint: str = "",
        layer_offloading: bool = True,
        node_id: str = "",
        diverge_detect_steps: int = 150,
        diverge_stop_steps: int = 300,
        diverge_threshold: float = 0.0001,
        ffn_chunks: int = 0,
        auto_stop: bool = False,
    ):
        _setup_ltx_paths()

        self._transformer = transformer
        self._embeddings_processor = embeddings_processor
        self._vae_decoder = vae_decoder
        self._scheduler = scheduler
        self._dataset = dataset
        self._training_strategy = training_strategy
        self._timestep_sampler = timestep_sampler
        self._output_dir = Path(output_dir)
        self._loras_dir = Path(loras_dir)
        self._lora_name = lora_name

        self._rank = rank
        self._alpha = alpha
        self._dropout = dropout
        self._target_modules = target_modules or ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"]
        self._learning_rate = learning_rate
        self._total_steps = len(dataset) if auto_stop else total_steps
        self._optimizer_type = optimizer_type
        self._scheduler_type = scheduler_type
        self._max_grad_norm = max_grad_norm
        self._gradient_checkpointing = gradient_checkpointing
        self._quantization = quantization if quantization and quantization != "none" else None
        self._keep_last_n = keep_last_n
        self._checkpoint_interval = checkpoint_interval
        self._validation_config = validation_config
        self._cached_validation_embeddings = cached_validation_embeddings
        self._audio_vae_decoder = audio_vae_decoder
        self._vocoder = vocoder
        self._seed = seed
        self._resume_checkpoint = resume_checkpoint
        self._layer_offloading = layer_offloading
        self._node_id = node_id

        # Runtime state set during train()
        self._optimizer = None
        self._lr_scheduler = None
        self._pending_optimizer_state: Path | None = None
        self._global_step = 0
        self._checkpoint_paths: list[Path] = []
        self._lora_applied = False

        # Divergence detection state
        self._ema_loss = None           # EMA-smoothed loss
        self._diverge_detect_steps = diverge_detect_steps
        self._diverge_stop_steps = diverge_stop_steps
        self._ffn_chunks = ffn_chunks
        self._auto_stop = auto_stop
        # Epoch = one full pass through the dataset (for logging, auto_stop, etc.)
        self._step_epoch = len(dataset)
        # LR cycle length (independent of epoch). 0 = match epoch length.
        self._lr_cycle = lr_cycle_steps if lr_cycle_steps > 0 else len(dataset)
        self._ema_alpha = 0.02          # EMA decay (smooth over ~50 steps)
        self._diverge_threshold = diverge_threshold  # Minimum slope to trigger detection
        self._ema_history: list[tuple[int, float]] = []  # (step, ema_loss) for slope computation
        self._diverge_detect_step = 0   # Step when upward slope first detected
        self._diverge_monitoring = False # Currently in monitoring mode
        self._pre_diverge_ckpt = None   # Path to checkpoint saved at detection
        self._diverge_final_ckpt = None # Set when divergence stops training — used by _save_final_lora
        self._diverge_lr_boosted = False # Whether LR boost was already attempted during divergence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        progress_bar=None,
        cancel_check: Callable | None = None,
    ) -> str:
        """Run the full training loop.

        Args:
            progress_bar: comfy.utils.ProgressBar instance (optional).
            cancel_check: Callable that raises when the user cancels (optional).
                Typically comfy.model_management.throw_exception_if_processing_interrupted.

        Returns:
            Absolute path to the final LoRA .safetensors file.
        """
        torch.manual_seed(self._seed)
        device = torch.device("cuda")

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: apply LoRA FIRST (on standard nn.Linear modules).
        # Must happen before quantization because PEFT doesn't recognise
        # quanto's quantized module types.
        self._apply_lora()

        # Step 2: quantize the base model weights (skipping LoRA params).
        if self._quantization is not None:
            self._quantize_base_model()

        # Step 2c: ensure ONLY LoRA params are trainable.
        # Quantization may create new parameters with requires_grad=True.
        self._freeze_non_lora_params()

        # Step 2b: enable input require_grads — CRITICAL for gradient checkpointing
        # with quantized + LoRA models.  PyTorch's checkpoint() needs inputs with
        # requires_grad=True to build the recomputation graph during backward.
        # Without this, all checkpointed block outputs lose requires_grad and the
        # loss ends up detached (no grad_fn).
        # NOTE: not needed with layer offloading — our custom autograd Function
        # handles gradient tracking directly.
        if not self._layer_offloading:
            self._enable_input_require_grads()

        # Step 3: optionally load resume checkpoint into LoRA weights
        if self._resume_checkpoint:
            self._load_resume_checkpoint()

        # Step 4: enable gradient checkpointing
        # With layer offloading, our custom autograd Function already does
        # checkpointing+offloading combined.  Only enable the model's built-in
        # checkpointing for the non-offloaded path.
        if self._gradient_checkpointing and not self._layer_offloading:
            base = self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
            if hasattr(base, "set_gradient_checkpointing"):
                base.set_gradient_checkpointing(True)
                logger.debug("Gradient checkpointing enabled")

        # Step 4b: FFN chunking — split feed-forward layers along sequence dim
        # to reduce peak activation memory (same technique as inference)
        if self._ffn_chunks > 0:
            self._apply_ffn_chunking()

        # Step 5: move model to GPU (or set up layer offloading)
        if self._layer_offloading:
            from .ltxv_layer_offload import setup_layer_offloading
            logger.info("Setting up layer offloading (blocks stream CPU↔GPU one at a time)...")
            self._transformer.train()
            setup_layer_offloading(self._transformer, device)
            self._embeddings_processor.to(device)
            self._embeddings_processor.eval()
        else:
            logger.info(f"Moving full transformer to GPU (quantization={self._quantization or 'none'})...")
            self._transformer.to(device)
            self._transformer.train()
            self._embeddings_processor.to(device)
            self._embeddings_processor.eval()
        logger.info(
            f"VRAM after model setup: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated, "
            f"{torch.cuda.memory_reserved() / 1024**3:.1f} GB reserved"
        )

        # Step 5b: de-inference-ify all parameters.
        # ComfyUI wraps execution in torch.inference_mode(), so all model
        # parameters are "inference tensors" that cannot participate in
        # autograd.  We must replace Parameter objects entirely (not just
        # .data) because the inference flag lives on the Parameter wrapper.
        with torch.inference_mode(False):
            for root in [self._transformer, self._embeddings_processor]:
                for module in root.modules():
                    # Replace parameters (non-recursive — .modules() handles recursion)
                    for key, param in list(module._parameters.items()):
                        if param is not None:
                            new_param = torch.nn.Parameter(
                                param.data.clone(), requires_grad=param.requires_grad
                            )
                            module._parameters[key] = new_param
                    # Replace buffers
                    for key, buf in list(module._buffers.items()):
                        if buf is not None:
                            module._buffers[key] = buf.data.clone()
        logger.debug("Cloned all parameters/buffers out of inference mode")

        # Step 6: build optimizer + lr scheduler
        trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in trainable_params)
        logger.info(f"Training {param_count:,} LoRA parameters ({len(trainable_params)} tensors)")

        self._optimizer = self._create_optimizer(trainable_params)
        self._lr_scheduler = self._create_lr_scheduler(self._optimizer)

        # On resume: restore optimizer state (Adam momentum, etc.) from the
        # sidecar .pt saved alongside the LoRA weights.  Without this, Adam's
        # first and second moments are fresh, producing a brief burst of
        # outsized parameter updates that destabilise the resumed weights.
        if self._pending_optimizer_state is not None:
            try:
                resume_state = torch.load(
                    self._pending_optimizer_state, map_location="cpu", weights_only=False
                )
                saved_opt_type = resume_state.get("optimizer_type")
                if saved_opt_type != self._optimizer_type:
                    logger.warning(
                        f"Optimizer type changed ({saved_opt_type} → "
                        f"{self._optimizer_type}) — skipping optimizer state restore"
                    )
                else:
                    self._optimizer.load_state_dict(resume_state["optimizer"])
                    if resume_state.get("torch_rng") is not None:
                        torch.set_rng_state(resume_state["torch_rng"])
                    if (
                        resume_state.get("cuda_rng") is not None
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
                    logger.info(
                        f"Restored optimizer + RNG state from "
                        f"{self._pending_optimizer_state.name}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to restore optimizer state ({e}) — continuing with "
                    f"fresh optimizer"
                )

        # On resume with auto_stop: extend total_steps to cover the next epoch
        if self._global_step > 0 and self._auto_stop:
            epochs_done = self._global_step // self._step_epoch
            self._total_steps = (epochs_done + 1) * self._step_epoch

        # On resume: reset base LR and rebuild scheduler for the CURRENT LR cycle,
        # then fast-forward to position within the cycle.
        if self._global_step > 0:
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._learning_rate
                pg["initial_lr"] = self._learning_rate
            self._lr_scheduler = self._create_lr_scheduler(self._optimizer)
            # Fast-forward within current LR cycle only
            steps_into_cycle = self._global_step % self._lr_cycle
            if self._lr_scheduler is not None:
                for _ in range(steps_into_cycle):
                    self._lr_scheduler.step()
            resumed_lr = self._optimizer.param_groups[0]["lr"]
            epoch_num = self._global_step // self._step_epoch + 1
            logger.info(
                f"Resumed at epoch {epoch_num} ({self._step_epoch} steps/epoch), "
                f"LR cycle={self._lr_cycle}, fast-forwarded {steps_into_cycle} steps "
                f"(resumed LR = {resumed_lr:.2e})"
            )

        # Step 7: build DataLoader (num_workers=0 required on Windows)
        # Use a Generator seeded from current time so each run (and resume)
        # gets a different shuffle order instead of repeating the same sequence.
        dl_generator = torch.Generator()
        dl_generator.manual_seed(int(torch.seed()) ^ self._global_step)
        dataloader = DataLoader(
            self._dataset,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            generator=dl_generator,
        )

        data_iter = iter(dataloader)
        start_step = self._global_step

        # Suppress sage attention warnings during training (fp8 quantized weights
        # produce dtypes sage doesn't support — falls back to pytorch attention).
        # Patch the root logger AND all its handlers to filter the message.
        class _SageFilter(logging.Filter):
            def filter(self, record):
                return "sage attention" not in record.getMessage()
        _sage_filter = _SageFilter()
        _root_logger = logging.getLogger()
        _root_logger.addFilter(_sage_filter)
        for _h in _root_logger.handlers:
            _h.addFilter(_sage_filter)

        logger.info(f"Starting training: {start_step} → {self._total_steps} steps"
                    f"{f' (auto_stop, epoch={self._step_epoch} samples)' if self._auto_stop else ''}")

        step = start_step
        while step < self._total_steps:
            step += 1
            # Cancellation check
            if cancel_check is not None:
                cancel_check()

            # Fetch next batch, cycling the dataset
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device
            batch = self._move_batch_to_device(batch, device)

            # Forward + backward.
            # NOTE: inference_mode is exited at the top level (ltxv_train_lora.py)
            # so all operations here have full autograd support.
            oom_skipped = False
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self._training_step(batch, device)

                if not loss.requires_grad:
                    raise RuntimeError(
                        "Loss has no grad_fn — the forward pass did not create a computation graph. "
                        "This usually means PEFT LoRA layers are not active or quanto quantization "
                        "is incompatible with this PEFT version. Try quantization='none'."
                    )
                loss.backward()
                loss_val = loss.item()
                del loss  # Release computation graph immediately
            except _OOM_EXCEPTIONS:
                # OOM on a hard sample — double FFN chunks and retry once
                loss = None
                gc.collect()
                torch.cuda.empty_cache()
                self._optimizer.zero_grad(set_to_none=True)

                original_chunks = self._ffn_chunks
                self._ffn_chunks = max(original_chunks * 2, 8)
                print(f"Step {step}: OOM — retrying with ffn_chunks={self._ffn_chunks} (was {original_chunks})")

                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = self._training_step(batch, device)
                    loss.backward()
                    loss_val = loss.item()
                    del loss
                except _OOM_EXCEPTIONS:
                    # Still OOM even with more chunks — skip this step
                    print(f"Step {step}: OOM even with ffn_chunks={self._ffn_chunks} — skipping step")
                    loss = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    self._optimizer.zero_grad(set_to_none=True)
                    oom_skipped = True
                finally:
                    self._ffn_chunks = original_chunks

            del batch  # Release batch tensors
            gc.collect()
            torch.cuda.empty_cache()

            if oom_skipped:
                continue

            # With layer offloading, LoRA grads were moved to CPU by the
            # backward hooks.  Ensure param/grad device agreement for the
            # optimizer (handles both offloaded and non-offloaded cases).
            if self._layer_offloading:
                for p in trainable_params:
                    if p.grad is not None and p.grad.device != p.device:
                        p.grad = p.grad.to(p.device)

            if self._max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, self._max_grad_norm)

            self._optimizer.step()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
            self._optimizer.zero_grad(set_to_none=True)

            self._global_step = step

            # Progress
            if progress_bar is not None:
                progress_bar.update_absolute(step, self._total_steps)
            lr = self._optimizer.param_groups[0]["lr"]
            if step % 10 == 0:
                vram_gb = torch.cuda.memory_allocated() / 1024**3
                vram_peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Step {step}/{self._total_steps}  loss={loss_val:.4f}  lr={lr:.2e}  vram={vram_gb:.1f}G  peak={vram_peak:.1f}G")
                torch.cuda.reset_peak_memory_stats()

            # Live chart update via websocket
            if self._node_id:
                try:
                    from server import PromptServer
                    epoch = (step - 1) // self._step_epoch + 1
                    PromptServer.instance.send_sync("rs-training-update", {
                        "node_id": self._node_id,
                        "step": step,
                        "total_steps": self._total_steps,
                        "loss": loss_val,
                        "lr": lr,
                        "ema_loss": self._ema_loss,
                        "monitoring": self._diverge_monitoring,
                        "checkpoint_interval": self._checkpoint_interval,
                        "epoch": epoch,
                        "step_epoch": self._step_epoch,
                    })
                except Exception:
                    pass

            # --- Divergence detection ---
            # Track EMA loss and detect divergence via trendline slope.
            # Skip the first 25% of steps (warmup).
            # Uses a sliding window linear regression (same as the JS graph trendline).
            # If slope is positive (upward) for detect_steps: enter monitoring.
            # If slope turns negative (downward) during monitoring: recovered, cancel.
            # If slope stays positive for stop_steps: stop and keep pre-divergence checkpoint.
            warmup_end = start_step + max(50, int((self._total_steps - start_step) * 0.25))
            if self._ema_loss is None:
                self._ema_loss = loss_val
            else:
                self._ema_loss = (1 - self._ema_alpha) * self._ema_loss + self._ema_alpha * loss_val

            if step >= warmup_end:
                self._ema_history.append((step, self._ema_loss))

                # Compute slope over a trailing window (matching JS graph: 75% of history, min 50 pts)
                min_pts = 50
                if len(self._ema_history) >= min_pts:
                    window_size = max(min_pts, int(len(self._ema_history) * 0.75))
                    window = self._ema_history[-window_size:]
                    n = len(window)
                    sum_x = sum(s for s, _ in window)
                    sum_y = sum(v for _, v in window)
                    sum_xy = sum(s * v for s, v in window)
                    sum_xx = sum(s * s for s, _ in window)
                    denom = n * sum_xx - sum_x * sum_x
                    slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0.0

                    # Compute predicted % increase over the detection window
                    # slope is loss/step, so slope * window_size = predicted loss change
                    # Convert to percentage relative to current EMA loss
                    predicted_change = slope * self._diverge_detect_steps
                    pct_change = (predicted_change / self._ema_loss * 100) if self._ema_loss > 0 else 0.0

                    # Only trigger if predicted increase exceeds threshold %
                    if not self._diverge_monitoring and pct_change > self._diverge_threshold:
                        # Count consecutive above-threshold steps
                        if self._diverge_detect_step == 0:
                            self._diverge_detect_step = step
                        steps_positive = step - self._diverge_detect_step

                        if steps_positive >= self._diverge_detect_steps:
                            self._diverge_monitoring = True
                            self._pre_diverge_ckpt = self._save_checkpoint(step)
                            print(
                                f"[divergence] Upward trend detected at step {step} — "
                                f"predicted +{pct_change:.1f}% (threshold={self._diverge_threshold:.0f}%), EMA={self._ema_loss:.4f}. "
                                f"Monitoring for {self._diverge_stop_steps} steps."
                            )
                    elif not self._diverge_monitoring and pct_change <= self._diverge_threshold:
                        # Reset counter — trend is below threshold
                        self._diverge_detect_step = 0

                    # In monitoring mode
                    if self._diverge_monitoring:
                        if slope <= 0:
                            # Slope turned negative — recovered
                            print(f"[divergence] Recovered at step {step} — slope turned negative ({slope:.6f}), training continues")
                            self._diverge_monitoring = False
                            self._diverge_detect_step = 0
                            self._pre_diverge_ckpt = None
                            self._diverge_lr_boosted = False
                            self._cleanup_old_checkpoints()
                        else:
                            steps_in_monitoring = step - self._diverge_detect_step
                            if steps_in_monitoring > 0 and steps_in_monitoring % 100 == 0:
                                self._save_checkpoint(step)

                            if steps_in_monitoring >= self._diverge_stop_steps:
                                if not self._diverge_lr_boosted:
                                    # First attempt: full LR reset to break out of plateau
                                    for pg in self._optimizer.param_groups:
                                        pg["lr"] = self._learning_rate
                                        pg["initial_lr"] = self._learning_rate
                                    self._lr_scheduler = self._create_lr_scheduler(self._optimizer)
                                    self._diverge_lr_boosted = True
                                    self._diverge_detect_step = step  # reset monitoring window
                                    print(
                                        f"[divergence] LR reset at step {step} — "
                                        f"boosting to {self._learning_rate:.2e} to attempt recovery. "
                                        f"Monitoring for another {self._diverge_stop_steps} steps."
                                    )
                                else:
                                    # Already tried LR boost — give up
                                    print(
                                        f"[divergence] No recovery after LR boost + {self._diverge_stop_steps} steps — stopping training. "
                                        f"Slope still positive ({slope:.6f}). "
                                        f"Rewinding to pre-divergence checkpoint: {self._pre_diverge_ckpt}"
                                    )
                                    self._diverge_final_ckpt = self._pre_diverge_ckpt
                                    break

            # Validation (save checkpoint first so progress is safe)
            if (
                self._validation_config is not None
                and self._cached_validation_embeddings is not None
                and self._validation_config.get("interval", 0) > 0
                and step % self._validation_config["interval"] == 0
            ):
                self._save_checkpoint(step)
                self._run_validation(step, device)

            # Checkpoint
            if self._checkpoint_interval > 0 and step % self._checkpoint_interval == 0:
                self._save_checkpoint(step)

            # Epoch boundary: log epoch completion, extend for auto_stop.
            if step > 0 and step % self._step_epoch == 0:
                epoch_num = step // self._step_epoch
                if self._auto_stop:
                    self._total_steps = step + self._step_epoch
                print(
                    f"[epoch {epoch_num} complete]"
                    f"{f' extending to step {self._total_steps}' if self._auto_stop else ''}"
                )

            # LR cycle boundary: reset learning rate and rebuild scheduler.
            if step > 0 and step % self._lr_cycle == 0:
                for pg in self._optimizer.param_groups:
                    pg["lr"] = self._learning_rate
                    pg["initial_lr"] = self._learning_rate
                self._lr_scheduler = self._create_lr_scheduler(self._optimizer)
                new_lr = self._optimizer.param_groups[0]["lr"]
                print(f"LR schedule reset (lr={new_lr:.2e}, cycle={self._lr_cycle})")

        # Restore sage attention logging
        _root_logger.removeFilter(_sage_filter)
        for _h in _root_logger.handlers:
            _h.removeFilter(_sage_filter)

        # Final save
        final_path = self._save_final_lora()
        logger.info(f"Training complete. LoRA saved to {final_path}")
        print(f"Training complete. LoRA saved to {final_path}")
        return str(final_path)

    def cleanup(self) -> None:
        """Release GPU memory and remove LoRA wrapping from the transformer.

        Called by the node after train() completes (or raises), so ComfyUI can
        reload/reuse the model normally.
        """
        # Tear down layer offloading if active
        if self._layer_offloading:
            from .ltxv_layer_offload import teardown_layer_offloading
            teardown_layer_offloading(self._transformer)

        # Move everything back to CPU
        try:
            self._transformer.to("cpu")
        except Exception:
            pass
        try:
            self._embeddings_processor.to("cpu")
        except Exception:
            pass
        if self._vae_decoder is not None:
            try:
                self._vae_decoder.to("cpu")
            except Exception:
                pass

        # Remove LoRA wrapping so the transformer is clean for future use.
        # PeftModel.base_model is the LoraModel tuner, which has .unload() —
        # it walks every submodule and replaces each LoRA layer with its
        # original base_layer in place on the underlying transformer.
        if self._lora_applied:
            try:
                if hasattr(self._transformer, "base_model"):
                    self._transformer.base_model.unload()
            except Exception as e:
                logger.warning(f"Could not cleanly unload LoRA adapter: {e}")
            self._lora_applied = False

        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_ffn_chunking(self) -> None:
        """Monkeypatch FFN layers to process in chunks along the sequence dim.

        Reduces peak activation memory during training by splitting the large
        FFN intermediate tensors into smaller chunks.  Same technique as the
        inference node but applied directly to the raw transformer (no ComfyUI
        model clone / add_object_patch).

        The chunk count is read from self._ffn_chunks at call time, so it can
        be changed dynamically (e.g. doubled on OOM, then restored).
        """
        base = (
            self._transformer.get_base_model()
            if hasattr(self._transformer, "get_base_model")
            else self._transformer
        )
        try:
            blocks = base.transformer_blocks
        except AttributeError:
            logger.warning("Could not find transformer_blocks for FFN chunking")
            return

        patched = 0
        trainer = self  # capture for closure

        def _make_chunked_forward(original_forward):
            def chunked_forward(x, *args, **kwargs):
                chunks = trainer._ffn_chunks
                if chunks <= 0 or x.shape[1] <= chunks:
                    return original_forward(x, *args, **kwargs)
                chunk_size = (x.shape[1] + chunks - 1) // chunks
                output_chunks = []
                for i in range(0, x.shape[1], chunk_size):
                    chunk = x[:, i : i + chunk_size]
                    output_chunks.append(original_forward(chunk, *args, **kwargs))
                return torch.cat(output_chunks, dim=1)

            return chunked_forward

        for block in blocks:
            if hasattr(block, "ff"):
                block.ff.forward = _make_chunked_forward(block.ff.forward)
                patched += 1

        logger.info(f"FFN chunking enabled: {self._ffn_chunks} chunks across {patched} blocks")

    def _enable_input_require_grads(self) -> None:
        """Force the first patchify layer's output to require grad.

        Gradient checkpointing (torch.utils.checkpoint) only builds a backward
        graph if its *input* tensors have requires_grad=True.  When the base
        model is fully frozen (quantized), no parameter produces gradients in the
        forward pass, so checkpoint blocks silently produce outputs with
        requires_grad=False — even though LoRA params DO require grad.

        The standard fix (used by HF ``prepare_model_for_kbit_training``) is to
        register a forward hook on the first embedding / patchify layer that
        calls ``output.requires_grad_(True)``.  This seeds the gradient chain so
        every subsequent checkpoint block sees requires_grad inputs.
        """
        # Try PEFT/HF built-in first
        if hasattr(self._transformer, "enable_input_require_grads"):
            self._transformer.enable_input_require_grads()
            logger.debug("enable_input_require_grads (built-in)")
            return

        # Manual fallback: hook the patchify projection (first layer).
        base = (
            self._transformer.get_base_model()
            if hasattr(self._transformer, "get_base_model")
            else self._transformer
        )
        target = getattr(base, "patchify_proj", None) or getattr(base, "patch_embed", None)
        if target is not None:
            def _hook(_mod, _inp, output):
                if isinstance(output, torch.Tensor) and output.is_floating_point():
                    output.requires_grad_(True)
                return output
            target.register_forward_hook(_hook)
            logger.debug(f"enable_input_require_grads hooked on {type(target).__name__}")
        else:
            # Last resort: hook the transformer itself
            def _hook(_mod, args):
                return tuple(
                    a.requires_grad_(True) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                    for a in args
                )
            base.register_forward_pre_hook(_hook)
            logger.debug("enable_input_require_grads hooked on transformer (pre-hook)")

    def _freeze_non_lora_params(self) -> None:
        """Ensure only LoRA parameters are trainable after quantization.

        Quantization (quanto) may replace modules with new ones whose params
        default to requires_grad=True.  This explicitly freezes everything
        that isn't a LoRA adapter weight.
        """
        frozen = 0
        for name, param in self._transformer.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)
                frozen += 1
        lora_count = sum(1 for n, p in self._transformer.named_parameters() if p.requires_grad)
        logger.debug(f"Froze {frozen} non-LoRA params, {lora_count} LoRA params remain trainable")

    @staticmethod
    def _unwrap_comfy_ops(module: torch.nn.Module) -> None:
        """Replace ComfyUI's custom Linear wrappers with standard nn.Linear.

        ComfyUI loads models with its own ops (e.g. ``comfy.ops.manual_cast.Linear``,
        ``comfy.ops.disable_weight_init.Linear``) for device management.  These
        subclass or duck-type ``nn.Linear`` but PEFT doesn't recognise them.

        Since we're taking over the model for training (not using ComfyUI's
        inference pipeline), it's safe to replace them with standard modules.
        """
        for name, child in list(module.named_children()):
            # Recurse first (depth-first)
            InProcessTrainer._unwrap_comfy_ops(child)

            # Check: is it Linear-like but NOT exactly torch.nn.Linear?
            if (
                type(child) is not torch.nn.Linear
                and child.__class__.__name__ == "Linear"
                and hasattr(child, "weight")
                and hasattr(child, "in_features")
            ):
                replacement = torch.nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                replacement.weight = torch.nn.Parameter(child.weight) if not isinstance(child.weight, torch.nn.Parameter) else child.weight
                if child.bias is not None:
                    replacement.bias = torch.nn.Parameter(child.bias) if not isinstance(child.bias, torch.nn.Parameter) else child.bias
                setattr(module, name, replacement)

    def _patch_rope_for_training(self) -> None:
        """Replace LTX-2's in-place RoPE with an autograd-safe version.

        apply_split_rotary_emb uses addcmul_ which breaks autograd when
        recomputing forward during gradient-checkpointed backward.
        """
        import ltx_core.model.transformer.rope as rope_module
        import ltx_core.model.transformer.attention as attn_module
        from einops import rearrange as _rearrange

        self._orig_apply_split_rotary_emb = rope_module.apply_split_rotary_emb

        def _apply_split_rotary_emb_no_inplace(input_tensor, cos_freqs, sin_freqs):
            needs_reshape = False
            if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
                b, h, t, _ = cos_freqs.shape
                input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
                needs_reshape = True

            split_input = _rearrange(input_tensor, "... (d r) -> ... d r", d=2)
            first_half_input = split_input[..., :1, :]
            second_half_input = split_input[..., 1:, :]

            cos_out = split_input * cos_freqs.unsqueeze(-2)
            # Out-of-place addcmul equivalents
            first_half = cos_out[..., :1, :] + (-sin_freqs.unsqueeze(-2)) * second_half_input
            second_half = cos_out[..., 1:, :] + sin_freqs.unsqueeze(-2) * first_half_input

            output = torch.cat([first_half, second_half], dim=-2)
            output = _rearrange(output, "... d r -> ... (d r)")
            if needs_reshape:
                output = output.swapaxes(1, 2).reshape(b, t, -1)
            return output

        # Patch both the module-level function and the dispatch in apply_rotary_emb
        rope_module.apply_split_rotary_emb = _apply_split_rotary_emb_no_inplace

        # Also patch the reference in attention.py if it imported directly
        if hasattr(attn_module, 'apply_rotary_emb'):
            orig_apply = attn_module.apply_rotary_emb
            self._orig_apply_rotary_emb = orig_apply
            # apply_rotary_emb dispatches to apply_split_rotary_emb internally,
            # so patching rope_module is sufficient.  But if attention.py cached
            # a direct reference, patch that too.

        logger.debug("Patched RoPE for training (in-place ops removed)")

    def _unpatch_rope(self) -> None:
        """Restore original RoPE function."""
        if hasattr(self, '_orig_apply_split_rotary_emb'):
            import ltx_core.model.transformer.rope as rope_module
            rope_module.apply_split_rotary_emb = self._orig_apply_split_rotary_emb
            logger.debug("Restored original RoPE function")

    def _apply_lora(self) -> None:
        """Wrap transformer with PEFT LoRA config."""
        from peft import LoraConfig, get_peft_model

        # ComfyUI loads models with custom ops (manual_cast.Linear, etc.)
        # that PEFT doesn't recognise.  Unwrap them first.
        self._unwrap_comfy_ops(self._transformer)
        logger.debug("Unwrapped ComfyUI ops → standard nn.Linear")

        # exclude_modules (regex string) prevents PEFT from matching the same
        # attn1/attn2/ff suffix patterns inside audio_embeddings_connector.
        # Without this, ~8% of LoRA rank gets wasted on audio connector
        # submodules that do nothing for video generation quality.
        lora_config = LoraConfig(
            r=self._rank,
            lora_alpha=self._alpha,
            target_modules=self._target_modules,
            exclude_modules=r".*audio_embeddings_connector.*",
            lora_dropout=self._dropout,
            init_lora_weights=True,
        )
        self._transformer = get_peft_model(self._transformer, lora_config)
        self._lora_applied = True

        # Verify LoRA layers were actually injected (can fail silently if PEFT
        # doesn't recognise quanto-quantized module types)
        trainable = [p for p in self._transformer.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in trainable)
        if param_count == 0:
            raise RuntimeError(
                "LoRA injection produced 0 trainable parameters. PEFT could not find "
                f"target modules matching {self._target_modules} in the (possibly quantized) "
                "model. Try: 1) pip install -U peft  2) set quantization to 'none'."
            )
        logger.info(f"LoRA applied: {param_count:,} trainable params, rank={self._rank}, alpha={self._alpha}")

    def _quantize_base_model(self) -> None:
        """Quantize the frozen base model weights, preserving LoRA params.

        Called AFTER LoRA is applied.  Uses ltx-trainer's block-by-block
        quantization but adds ``lora_*`` to the exclusion list so that the
        small trainable LoRA matrices stay in float32/bf16.
        """
        import ltx_trainer.quantization as ltx_quant
        from optimum.quanto import freeze, quantize

        precision = self._quantization
        logger.info(f"Quantizing base model with {precision} (LoRA weights excluded)...")

        # Map precision string to quanto weight type
        PRECISION_MAP = {
            "int8-quanto": "qint8",
            "int4-quanto": "qint4",
            "int2-quanto": "qint2",
            "fp8-quanto": "qfloat8",
            "fp8uz-quanto": "qfloat8_e4m3fnuz",
        }
        weight_qtype_name = PRECISION_MAP.get(precision)
        if weight_qtype_name is None:
            raise ValueError(f"Unknown quantization precision: {precision!r}")

        from optimum.quanto import qfloat8, qint2, qint4, qint8
        QTYPE_MAP = {"qint8": qint8, "qint4": qint4, "qint2": qint2, "qfloat8": qfloat8}
        # Handle fp8uz if available
        try:
            from optimum.quanto import qfloat8_e4m3fnuz
            QTYPE_MAP["qfloat8_e4m3fnuz"] = qfloat8_e4m3fnuz
        except ImportError:
            pass
        weight_qtype = QTYPE_MAP[weight_qtype_name]

        # Exclusion patterns: original ltx-trainer patterns + LoRA modules
        exclude = list(getattr(ltx_quant, "EXCLUDE_PATTERNS", [
            "patchify_proj", "proj_out", "audio_patchify_proj", "audio_proj_out",
            "time_proj", "timestep_embedder", "*adaln*", "*norm*",
            "caption_projection*", "audio_caption_projection*",
        ]))
        exclude.extend(["*lora_A*", "*lora_B*", "*lora_embedding*"])

        # Get the base model through PEFT wrapper
        base = (
            self._transformer.get_base_model()
            if hasattr(self._transformer, "get_base_model")
            else self._transformer
        )

        # Block-by-block quantization (memory efficient — only one block on GPU at a time)
        if hasattr(base, "transformer_blocks"):
            blocks = base.transformer_blocks
            logger.info(f"Block-by-block quantization: {len(blocks)} blocks")
            for i, block in enumerate(blocks):
                block.to("cuda")
                quantize(block, weights=weight_qtype, exclude=exclude)
                freeze(block)
                block.to("cpu")
                if (i + 1) % 10 == 0:
                    logger.debug(f"  Quantized {i + 1}/{len(blocks)} blocks")

            # Quantize remaining non-block modules (skip LoRA)
            for name, module in base.named_children():
                if name != "transformer_blocks":
                    skip = any(
                        (p.endswith("*") and p[:-1] in name) or
                        (p.startswith("*") and p[1:] in name) or
                        (p == name)
                        for p in exclude
                    )
                    if not skip and sum(1 for _ in module.parameters()) > 0:
                        module.to("cuda")
                        quantize(module, weights=weight_qtype, exclude=exclude)
                        freeze(module)
                        module.to("cpu")
        else:
            # Fallback: quantize entire model at once
            quantize(base, weights=weight_qtype, exclude=exclude)
            freeze(base)

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Base model quantized (LoRA weights preserved in float)")

    def _load_resume_checkpoint(self) -> None:
        """Load LoRA weights from a resume checkpoint into the current PEFT model.

        Strips ComfyUI-compatibility additions (the "diffusion_model." prefix
        and any alpha keys) before handing the state_dict to PEFT.  Also
        stashes the path to the optimizer-state sidecar for later restore
        (the optimizer doesn't exist yet at this point in train()).
        """
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        ckpt_path = Path(self._resume_checkpoint)
        if not ckpt_path.is_file():
            logger.warning(f"Resume checkpoint not found: {ckpt_path} — starting from scratch")
            return

        state_dict = load_file(str(ckpt_path))
        # Strip ComfyUI "diffusion_model." prefix that was added on save
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}
        # Strip alpha keys — they are metadata for ComfyUI's LoRA loader, not
        # part of PEFT's adapter state dict (alpha is in LoraConfig, not weights).
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".alpha")}
        base_model = self._transformer.get_base_model()
        set_peft_model_state_dict(base_model, state_dict)

        # Restore step counter from checkpoint filename or metadata
        import re
        step_match = re.search(r"step_(\d+)", ckpt_path.stem)
        if step_match:
            self._global_step = int(step_match.group(1))
        elif "step" in (load_file(str(ckpt_path), metadata=True) or {}):
            self._global_step = int(load_file(str(ckpt_path), metadata=True)["step"])

        # Stash optimizer sidecar path — loaded later once optimizer exists.
        sidecar = ckpt_path.with_suffix(".state.pt")
        self._pending_optimizer_state = sidecar if sidecar.is_file() else None
        if self._pending_optimizer_state is None:
            logger.warning(
                f"No optimizer sidecar found at {sidecar.name} — optimizer "
                f"momentum will be fresh on resume (may cause brief instability)"
            )

        logger.info(f"Resumed from checkpoint: {ckpt_path} (step {self._global_step})")
        print(f"Resumed from checkpoint: {ckpt_path} (step {self._global_step})")

    def _training_step(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Run one forward pass and return scalar loss tensor.

        Uses ComfyUI's model interface directly (not ltx-core's Modality
        interface).  The flow:
        1. Run embedding connectors on pre-computed text features.
        2. Get clean latents from dataset, sample sigma, add noise.
        3. Call ComfyUI model: model(x, timestep, context, ...).
        4. Compute velocity-prediction MSE loss.
        """
        from ltx_core.text_encoders.gemma import convert_to_additive_mask

        # ---- 1. Text embeddings ----
        conditions = batch["conditions"]

        if "video_prompt_embeds" in conditions:
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            video_features = conditions["prompt_embeds"]
            audio_features = conditions["prompt_embeds"]

        video_features = video_features.to(dtype=torch.bfloat16)
        if audio_features is not None:
            audio_features = audio_features.to(dtype=torch.bfloat16)

        mask = conditions["prompt_attention_mask"]
        additive_mask = convert_to_additive_mask(mask, video_features.dtype)

        # Embeddings processor weights are inference tensors (created under
        # ComfyUI's inference_mode).  Run it in inference mode, then clone
        # outputs to get normal tensors that can participate in autograd.
        video_embeds, audio_embeds, emb_mask = self._embeddings_processor.create_embeddings(
            video_features, audio_features, additive_mask
        )
        video_embeds = video_embeds.clone()
        audio_embeds = audio_embeds.clone()
        emb_mask = emb_mask.clone()

        # ComfyUI expects context = cat([video_embeds, audio_embeds], dim=-1)
        # The model internally splits along the last dim.
        context = torch.cat([video_embeds, audio_embeds], dim=-1)

        # ---- 2. Latents + noise (flow matching velocity prediction) ----
        video_latents = batch["latents"]["latents"]  # [B, C, T, H, W]
        audio_data = batch.get("audio_latents")
        audio_latents = audio_data["latents"] if audio_data is not None else None

        B = video_latents.shape[0]

        # Sample sigma using the shifted logit-normal sampler.
        # sample() needs seq_length — estimate from latent spatial dims.
        # (T/1) * (H/1) * (W/1) since latents are already VAE-compressed.
        _, C, T, H, W = video_latents.shape
        seq_length = T * H * W
        sigmas = self._timestep_sampler.sample(B, seq_length, device=device)  # [B,]

        # Add noise: noisy = (1 - sigma) * clean + sigma * noise
        video_noise = torch.randn_like(video_latents)
        sigma_v = sigmas.view(B, 1, 1, 1, 1)
        noisy_video = (1 - sigma_v) * video_latents + sigma_v * video_noise
        video_target = video_noise - video_latents  # velocity target

        has_audio = audio_latents is not None and audio_latents.numel() > 0

        if has_audio:
            audio_noise = torch.randn_like(audio_latents)
            # audio latents: [B, C_a, F_a, T_a] (4D) or [B, C_a, T_a] (3D)
            sigma_a = sigmas.view(B, *([1] * (audio_latents.ndim - 1)))
            noisy_audio = (1 - sigma_a) * audio_latents + sigma_a * audio_noise
            audio_target = audio_noise - audio_latents
            x = [noisy_video, noisy_audio]
            timestep_input = (sigmas, sigmas)
        else:
            # Pass single-element list — model creates empty audio internally
            x = [noisy_video]
            timestep_input = sigmas

        # Ensure noisy latents require grad for autograd graph
        if isinstance(x, list):
            x[0] = x[0].requires_grad_(True)
        else:
            x = x.requires_grad_(True)

        # ---- 3. Forward through ComfyUI model ----
        # inference_mode is already exited at the top level (ltxv_train_lora.py).
        pred = self._transformer(
            x=x,
            timestep=timestep_input,
            context=context,
            attention_mask=emb_mask,
            frame_rate=25,
            transformer_options={},
        )

        # Debug: check gradient chain
        if isinstance(pred, (list, tuple)):
            _dbg = pred[0]
        else:
            _dbg = pred
        if not _dbg.requires_grad:
            logger.warning(
                f"Model output has no grad! shape={_dbg.shape}, dtype={_dbg.dtype}, "
                f"input x[0].requires_grad={x[0].requires_grad if isinstance(x, list) else x.requires_grad}"
            )

        # ---- 4. Loss ----
        if isinstance(pred, (list, tuple)) and len(pred) >= 2:
            video_pred = pred[0]
            audio_pred = pred[1]
            loss = (video_pred - video_target).pow(2).mean()
            if has_audio and audio_pred.numel() > 0:
                loss = loss + (audio_pred - audio_target).pow(2).mean()
        else:
            loss = (pred - video_target).pow(2).mean()
        return loss

    @staticmethod
    def _move_batch_to_device(batch: dict, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> dict:
        """Recursively move all tensors in a nested dict to device and cast to dtype."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # Cast float tensors to model dtype; leave bool/int masks untouched
                t_dtype = dtype if v.is_floating_point() else None
                result[k] = v.to(device=device, dtype=t_dtype, non_blocking=True)
            elif isinstance(v, dict):
                result[k] = InProcessTrainer._move_batch_to_device(v, device)
            else:
                result[k] = v
        return result

    def _create_optimizer(self, params: list) -> torch.optim.Optimizer:
        opt_type = self._optimizer_type
        # bitsandbytes AdamW8bit doesn't support CPU tensors — fall back to
        # regular AdamW when layer offloading keeps LoRA params on CPU.
        if opt_type == "adamw8bit" and self._layer_offloading:
            logger.info("Layer offloading active → using AdamW (CPU) instead of AdamW8bit (GPU-only)")
            opt_type = "adamw"

        if opt_type == "adamw8bit":
            try:
                from bitsandbytes.optim import AdamW8bit
                return AdamW8bit(params, lr=self._learning_rate)
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to AdamW")
                return torch.optim.AdamW(params, lr=self._learning_rate)
        elif opt_type == "adamw":
            return torch.optim.AdamW(params, lr=self._learning_rate)
        else:
            raise ValueError(f"Unknown optimizer_type: {self._optimizer_type!r}")

    def _create_lr_scheduler(self, optimizer):
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            CosineAnnealingWarmRestarts,
            LinearLR,
            PolynomialLR,
        )

        t = self._scheduler_type
        steps = self._lr_cycle  # schedule per LR cycle, not total training

        if t is None or t == "constant":
            return None
        elif t == "linear":
            return LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=steps)
        elif t == "cosine":
            return CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)
        elif t == "cosine_with_restarts":
            return CosineAnnealingWarmRestarts(optimizer, T_0=steps)
        elif t == "polynomial":
            return PolynomialLR(optimizer, total_iters=steps, power=1.0)
        else:
            logger.warning(f"Unknown scheduler_type {t!r}, using constant LR")
            return None

    def _save_checkpoint(self, step: int) -> Path:
        """Save intermediate LoRA checkpoint and prune old ones.

        Also saves a sidecar .pt file with optimizer + RNG state so resume
        can continue with proper Adam momentum and LR progression rather
        than starting from scratch (which destabilises trained weights).
        """
        from safetensors.torch import save_file

        ckpt_dir = self._output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_path = ckpt_dir / f"lora_weights_step_{step:05d}.safetensors"
        state_dict = self._extract_lora_state_dict()
        save_file(state_dict, save_path, metadata={"step": str(step)})
        logger.info(f"Checkpoint saved: {save_path.name}")

        # Sidecar: optimizer state + RNG state for proper resume
        try:
            sidecar_path = save_path.with_suffix(".state.pt")
            resume_state = {
                "step": step,
                "optimizer_type": self._optimizer_type,
                "optimizer": self._optimizer.state_dict() if self._optimizer else None,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            }
            torch.save(resume_state, sidecar_path)
        except Exception as e:
            logger.warning(f"Failed to save optimizer sidecar ({e}) — resume will use fresh optimizer")

        self._checkpoint_paths.append(save_path)
        # Don't delete old checkpoints during monitoring — keep them all for rewind
        if not self._diverge_monitoring:
            self._cleanup_old_checkpoints()

        return save_path

    def _save_final_lora(self) -> Path:
        """Save the final LoRA weights to the ComfyUI loras folder.

        If training was stopped by divergence detection, copies the
        pre-divergence checkpoint instead of the current (diverged) weights.
        """
        self._loras_dir.mkdir(parents=True, exist_ok=True)
        dest = self._loras_dir / f"{self._lora_name}.safetensors"

        # If divergence stopped training, use the pre-divergence checkpoint
        diverge_ckpt = getattr(self, "_diverge_final_ckpt", None)
        if diverge_ckpt and Path(diverge_ckpt).exists():
            print(f"[divergence] Saving pre-divergence checkpoint as final LoRA: {diverge_ckpt}")
            shutil.copy2(diverge_ckpt, dest)
            # Also copy to output dir for consistency
            shutil.copy2(diverge_ckpt, self._output_dir / f"{self._lora_name}.safetensors")
            return dest

        from safetensors.torch import save_file

        # Write to output_dir first
        final_in_output = self._output_dir / f"{self._lora_name}.safetensors"
        state_dict = self._extract_lora_state_dict()
        save_file(state_dict, final_in_output, metadata={"steps": str(self._total_steps)})

        # Copy to loras folder
        shutil.copy2(final_in_output, dest)

        return dest

    def _extract_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract LoRA-only weights and convert to ComfyUI-compatible format."""
        from peft import get_peft_model_state_dict

        # get_peft_model_state_dict extracts only the adapter weights
        state_dict = get_peft_model_state_dict(self._transformer)

        # Remove PEFT's "base_model.model." prefix
        state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}

        # ComfyUI's LoRAAdapter supports diffusers2 format natively
        # (lora_A.weight / lora_B.weight), so we keep PEFT's native names and
        # only add the "diffusion_model." prefix + alpha keys.
        result = {}
        for k, v in state_dict.items():
            full_key = f"diffusion_model.{k}"
            result[full_key] = v.to(torch.bfloat16)
            # Add alpha key per LoRA pair — ComfyUI uses alpha/rank for scaling
            if k.endswith("lora_A.weight"):
                alpha_key = full_key.replace("lora_A.weight", "alpha")
                result[alpha_key] = torch.tensor(float(self._alpha))

        return result

    def _cleanup_old_checkpoints(self) -> None:
        if self._keep_last_n <= 0:
            return
        import re
        ckpt_dir = self._output_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            return
        all_ckpts = sorted(
            ckpt_dir.glob("lora_weights_step_*.safetensors"),
            key=lambda p: int(m.group(1)) if (m := re.search(r"step_(\d+)", p.stem)) else 0,
        )
        while len(all_ckpts) > self._keep_last_n:
            old = all_ckpts.pop(0)
            old.unlink()
            # Also remove the optimizer state sidecar
            sidecar = old.with_suffix(".state.pt")
            if sidecar.exists():
                sidecar.unlink()
            logger.debug(f"Removed old checkpoint: {old.name}")

    def run_validation_only(self, device: torch.device) -> None:
        """Run a single validation with full training setup (LoRA, quantization, etc).

        Mirrors the exact same model state as during training.
        """
        # Full setup identical to train()
        self._apply_lora()
        if self._quantization is not None:
            self._quantize_base_model()
        self._freeze_non_lora_params()
        if self._resume_checkpoint:
            self._load_resume_checkpoint()

        if self._layer_offloading:
            from .ltxv_layer_offload import setup_layer_offloading
            self._transformer.eval()
            setup_layer_offloading(self._transformer, device)
        else:
            self._transformer.to(device)
            self._transformer.eval()

        self._run_validation(self._global_step, device)

    def _run_validation(self, step: int, device: torch.device) -> None:
        """Generate a validation sample using the layer-offloaded model.

        Does NOT use ltx_trainer's ValidationSampler (which tries to load the
        full model to GPU).  Instead runs a simple Euler flow-matching loop
        through our monkey-patched forward that streams blocks one at a time.
        """
        cfg = self._validation_config
        cached = self._cached_validation_embeddings

        samples_dir = self._output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        height = cfg.get("height", 576)
        width = cfg.get("width", 576)
        num_frames = cfg.get("num_frames", 49)
        frame_rate = cfg.get("frame_rate", 25.0)
        num_steps = cfg.get("num_inference_steps", 30)
        guidance_scale = cfg.get("guidance_scale", 4.0)
        seed = cfg.get("seed", 42)

        try:
            self._transformer.eval()

            # Build context from cached embeddings (already on CPU)
            if isinstance(cached, dict):
                v_ctx = cached["video_context_positive"]
                a_ctx = cached["audio_context_positive"]
            elif isinstance(cached, (list, tuple)):
                v_ctx = cached[0]
                a_ctx = cached[1]
            else:
                v_ctx = cached.video_context_positive
                a_ctx = cached.audio_context_positive

            v_ctx = v_ctx.to(device=device, dtype=torch.bfloat16)
            a_ctx = a_ctx.to(device=device, dtype=torch.bfloat16)
            context_pos = torch.cat([v_ctx, a_ctx], dim=-1)

            # Negative embeddings for CFG
            if isinstance(cached, dict):
                v_neg = cached["video_context_negative"]
                a_neg = cached["audio_context_negative"]
            elif isinstance(cached, (list, tuple)):
                v_neg = cached[2]
                a_neg = cached[3]
            else:
                v_neg = cached.video_context_negative
                a_neg = cached.audio_context_negative
            v_neg = v_neg.to(device=device, dtype=torch.bfloat16)
            a_neg = a_neg.to(device=device, dtype=torch.bfloat16)
            context_neg = torch.cat([v_neg, a_neg], dim=-1)

            # Attention mask: all ones
            emb_mask = torch.ones(
                v_ctx.shape[0], v_ctx.shape[1], device=device, dtype=torch.bfloat16
            )

            # Latent dimensions
            latent_h, latent_w = height // 32, width // 32
            latent_t = (num_frames - 1) // 8 + 1  # temporal compression
            C = 128  # LTXV latent channels

            # Start from pure noise
            generator = torch.Generator(device=device).manual_seed(seed)
            x_noisy = torch.randn(1, C, latent_t, latent_h, latent_w,
                                  device=device, dtype=torch.bfloat16,
                                  generator=generator)

            # Euler flow-matching: step from t=1 (noise) to t=0 (clean)
            # Linear sigma schedule
            sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(num_steps):
                    sigma = sigmas[i]
                    sigma_next = sigmas[i + 1]
                    dt = sigma_next - sigma  # negative step
                    t = sigma.unsqueeze(0)

                    # Conditional prediction
                    pred_cond = self._transformer(
                        x=[x_noisy.clone()],
                        timestep=t,
                        context=context_pos,
                        attention_mask=emb_mask,
                        frame_rate=frame_rate,
                        transformer_options={},
                    )
                    if isinstance(pred_cond, (list, tuple)):
                        pred_cond = pred_cond[0]

                    # Unconditional prediction for CFG
                    if guidance_scale > 1.0:
                        pred_uncond = self._transformer(
                            x=[x_noisy.clone()],
                            timestep=t,
                            context=context_neg,
                            attention_mask=emb_mask,
                            frame_rate=frame_rate,
                            transformer_options={},
                        )
                        if isinstance(pred_uncond, (list, tuple)):
                            pred_uncond = pred_uncond[0]
                        # CFG: pred = uncond + scale * (cond - uncond)
                        v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                    else:
                        v_pred = pred_cond

                    # Euler step: x = x + dt * velocity
                    x_noisy = x_noisy + dt * v_pred

            # Decode latents on CUDA (VAE conv layers require it)
            # Free VRAM by offloading transformer first
            print(f"Validation step {step}: decoding latents...")
            latents = x_noisy.to(dtype=torch.float32)
            del x_noisy
            self._transformer.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            self._vae_decoder.to(device)
            # Match VAE weight dtype (usually bfloat16)
            vae_dtype = next(self._vae_decoder.parameters()).dtype
            latents = latents.to(dtype=vae_dtype)
            with torch.no_grad():
                pixels = self._vae_decoder(latents)
            self._vae_decoder.to("cpu")
            del latents
            gc.collect()
            torch.cuda.empty_cache()

            # Save as video or image
            ext = "png" if num_frames == 1 else "mp4"
            out_path = samples_dir / f"step_{step:06d}.{ext}"

            if num_frames == 1 and pixels.ndim >= 4:
                from torchvision.utils import save_image as tv_save_image
                # pixels: [B, C, H, W] or [B, C, 1, H, W]
                if pixels.ndim == 5:
                    pixels = pixels[:, :, 0]
                tv_save_image(pixels.clamp(0, 1), str(out_path))
            else:
                # Save video using imageio
                import imageio
                # pixels: [B, C, T, H, W] → [T, H, W, C] uint8
                vid = pixels[0]  # remove batch
                if vid.ndim == 4:  # [C, T, H, W]
                    vid = vid.permute(1, 2, 3, 0)  # [T, H, W, C]
                vid = vid.clamp(0, 1).mul(255).byte().cpu().numpy()
                imageio.mimwrite(str(out_path), vid, fps=frame_rate)

            print(f"Validation sample saved: {out_path.name}")

        except Exception as e:
            logger.warning(f"Validation failed at step {step}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore device layout for training
            if self._layer_offloading:
                # Non-block params to GPU, blocks stay on CPU
                base = self._transformer
                if hasattr(base, "get_base_model"):
                    base = base.get_base_model()
                for name, child in base.named_children():
                    if name != "transformer_blocks":
                        child.to(device)
            else:
                self._transformer.to(device)
            self._transformer.train()
            gc.collect()
            torch.cuda.empty_cache()
