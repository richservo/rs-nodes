import gc
import logging
import sys
from pathlib import Path

import torch
import comfy.model_management as mm
import comfy.utils
import folder_paths

from ..utils.ltxv_train_env import (
    get_submodule_root,
    validate_submodule,
)

logger = logging.getLogger(__name__)

# Track active training output dir for the loss history API endpoint
_active_training_dir: Path | None = None

try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/rs-nodes/loss-history")
    async def _serve_loss_history(request):
        """Serve loss_history.json for the training monitor page."""
        if _active_training_dir is None:
            return web.json_response({"steps": []})
        history_path = _active_training_dir / "loss_history.json"
        if not history_path.exists():
            return web.json_response({"steps": []})
        return web.FileResponse(history_path)
except Exception:
    pass

# Module group -> target_modules mapping for the LTX-2 transformer.
# PEFT target_modules with a list uses suffix matching (key.endswith(f".{x}")),
# NOT regex. To prevent matching audio_embeddings_connector submodules that
# share the same attn1/attn2/ff naming, we use EXCLUDE_MODULES (see below).
MODULE_GROUPS = {
    "video_self_attention": ["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0", "attn1.to_gate_logits"],
    "video_cross_attention": ["attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0", "attn2.to_gate_logits"],
    "video_feed_forward": ["ff.net.0.proj", "ff.net.2"],
    "audio_self_attention": ["audio_attn1.to_k", "audio_attn1.to_q", "audio_attn1.to_v", "audio_attn1.to_out.0", "audio_attn1.to_gate_logits"],
    "audio_cross_attention": ["audio_attn2.to_k", "audio_attn2.to_q", "audio_attn2.to_v", "audio_attn2.to_out.0", "audio_attn2.to_gate_logits"],
    "audio_feed_forward": ["audio_ff.net.0.proj", "audio_ff.net.2"],
    "video_attends_to_audio": [
        "audio_to_video_attn.to_k", "audio_to_video_attn.to_q",
        "audio_to_video_attn.to_v", "audio_to_video_attn.to_out.0",
        "audio_to_video_attn.to_gate_logits",
    ],
    "audio_attends_to_video": [
        "video_to_audio_attn.to_k", "video_to_audio_attn.to_q",
        "video_to_audio_attn.to_v", "video_to_audio_attn.to_out.0",
        "video_to_audio_attn.to_gate_logits",
    ],
}


def _setup_ltx_paths() -> None:
    """Add ltx-core and ltx-trainer src directories to sys.path."""
    root = get_submodule_root()
    paths = [
        str(root / "packages" / "ltx-trainer" / "src"),
        str(root / "packages" / "ltx-core" / "src"),
    ]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


class _ComfyVAEDecoderAdapter(torch.nn.Module):
    """Wraps ComfyUI's CausalVideoAutoencoder to act as a standalone video decoder.

    The ValidationSampler in ltx-trainer calls the decoder as:
        decoded = self._vae_decoder(latents)          # __call__ / forward
        self._vae_decoder.to(device)                  # device management

    ComfyUI's CausalVideoAutoencoder uses explicit .decode(x) which un-normalizes
    per-channel statistics before running the raw decoder. This adapter routes
    forward() through that path.
    """

    def __init__(self, autoencoder):
        super().__init__()
        self._autoencoder = autoencoder

    def forward(self, x, **kwargs):
        return self._autoencoder.decode(x)

    def to(self, *args, **kwargs):
        self._autoencoder.to(*args, **kwargs)
        return self


class RSLTXVTrainLoRA:
    """Train a LoRA adapter for LTX-2 in-process, reusing the already-loaded
    transformer from ComfyUI's MODEL wrapper.

    Eliminates the 5+ minute startup cost of the subprocess approach because
    the 22B transformer is already resident in memory and does not need to be
    reloaded from safetensors.

    Flow:
    1. Extract raw LTXModel from ComfyUI MODEL wrapper.
    2. Load EmbeddingsProcessor (~50 MB connector weights) from checkpoint.
    3. Optionally encode validation prompt using the CLIP text encoder.
    4. Free all other VRAM (ComfyUI managed models, clip).
    5. Load VideoDecoder from checkpoint for validation (CPU, moved to GPU only
       during validation passes).
    6. Build PrecomputedDataset + TextToVideoStrategy + timestep sampler.
    7. Run InProcessTrainer.
    8. Cleanup: remove LoRA, free VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "LTX-2 model loaded via CheckpointLoaderSimple"}),
                "output_name": ("STRING", {"default": "my_lora", "tooltip": "Name for the output LoRA file (without extension)"}),
                "preprocessed_data_root": ("STRING", {"default": "", "tooltip": "Path output by RSLTXVPrepareDataset"}),
                "model_path": (checkpoints, {"tooltip": "LTX-2 safetensors checkpoint (used to load EmbeddingsProcessor and VAE)"}),
            },
            "optional": {
                # ComfyUI models for in-process decode
                "vae": ("VAE", {"tooltip": "VAE (from CheckpointLoaderSimple). When connected, uses ComfyUI's decoder for validation instead of loading from checkpoint."}),
                # Preset
                "preset": (
                    ["custom", "subject", "style", "motion", "subject + style", "all video", "audio + video"],
                    {"default": "subject", "tooltip": "Preset module configurations. Selecting a preset updates the toggles and rank below."},
                ),
                # LoRA config
                "lora_rank":    ("INT",   {"default": 16,  "min": 1,    "max": 256,   "step": 1}),
                "lora_alpha":   ("INT",   {"default": 16,  "min": 1,    "max": 256,   "step": 1}),
                "lora_dropout": ("FLOAT", {"default": 0.0, "min": 0.0,  "max": 1.0,   "step": 0.01}),
                # Module toggles
                "video_self_attention":   ("BOOLEAN", {"default": True}),
                "video_cross_attention":  ("BOOLEAN", {"default": False}),
                "video_feed_forward":     ("BOOLEAN", {"default": False}),
                "audio_self_attention":   ("BOOLEAN", {"default": False}),
                "audio_cross_attention":  ("BOOLEAN", {"default": False}),
                "audio_feed_forward":     ("BOOLEAN", {"default": False}),
                "video_attends_to_audio": ("BOOLEAN", {"default": False}),
                "audio_attends_to_video": ("BOOLEAN", {"default": False}),
                # Training
                "learning_rate":          ("FLOAT", {"default": 1e-4,  "min": 1e-7, "max": 1e-1, "step": 1e-5}),
                "lr_end":                 ("FLOAT", {"default": 0.0,   "min": 0.0,  "max": 1e-1, "step": 1e-6, "tooltip": "Minimum LR at end of schedule (0=scheduler default). For cosine: eta_min, for linear: end_factor ratio."}),
                "epochs":                 ("INT",   {"default": 3,     "min": 1,    "max": 1000, "tooltip": "Number of passes through the full dataset. Each epoch = len(dataset) steps."}),
                "auto_stop":              ("BOOLEAN", {"default": False, "tooltip": "Ignore epoch count — train until divergence detection stops it"}),
                "optimizer":              (["adamw8bit", "adamw", "rose"], {"default": "adamw8bit"}),
                "rose_stabilize":         ("BOOLEAN", {"default": True, "tooltip": "ROSE only: CV Trust Gating. Smooths noisy gradient ranges. Creator suggests False may help for some conditions (e.g. pretraining)."}),
                "rose_weight_decay":      ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 1e-1, "step": 1e-5, "tooltip": "ROSE only: Decoupled weight decay strength. Regularizes by shrinking weights toward zero. 0 = disabled."}),
                "rose_wd_schedule":       ("BOOLEAN", {"default": False, "tooltip": "ROSE only: Scale weight decay with LR schedule so decay weakens as LR drops. Recommended when using a scheduler."}),
                "scheduler":              (["linear", "constant", "cosine", "cosine_with_restarts", "polynomial"],),
                "lr_cycle_steps":         ("INT",   {"default": 0, "min": 0, "max": 100000, "step": 100,
                                                     "tooltip": "LR schedule cycle length in steps. 0 = one cycle per epoch (dataset size). Set to e.g. 1000 for faster LR resets on large datasets."}),
                "lr_cycle_decay":         ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01,
                                                     "tooltip": "Multiply LR by this factor each cycle reset. 1.0 = no decay, 0.8 = 20% reduction per cycle."}),
                "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                "ffn_chunks":             ("INT",     {"default": 0, "min": 0, "max": 16, "step": 1, "tooltip": "Split FFN layers into N chunks along sequence dim to reduce peak VRAM. 0 = disabled, 4 = good default. Trades speed for memory."}),
                # Quantization
                "quantization": (["fp8-quanto", "int8-quanto", "int4-quanto", "none"], {"default": "fp8-quanto", "tooltip": "fp8 recommended (no C++ build tools needed). int4/int2 require 'pip install ninja' + C++ compiler."}),
                # Strategy
                "strategy":                  (["text_to_video", "video_to_video"], {"default": "text_to_video"}),
                "first_frame_conditioning_p": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "with_audio":                ("BOOLEAN", {"default": False}),
                # Validation
                "clip": ("CLIP", {"tooltip": "Text encoder for encoding the validation prompt. Connect the same CLIP used for generation."}),
                "validation_prompt":   ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt for validation video generation during training"}),
                "validation_interval": ("INT",   {"default": 0, "min": 0, "max": 10000, "tooltip": "0 = disabled"}),
                "validation_width":    ("INT",   {"default": 576, "min": 64, "max": 8192, "step": 32}),
                "validation_height":   ("INT",   {"default": 576, "min": 64, "max": 8192, "step": 32}),
                "validation_frames":   ("INT",   {"default": 49,  "min": 1,  "max": 257,  "step": 8}),
                # Checkpoints
                "checkpoint_interval": ("INT",   {"default": 500, "min": 1,  "max": 10000}),
                "keep_last_n":         ("INT",   {"default": 2,   "min": -1, "max": 100,
                                                  "tooltip": "-1 = keep all"}),
                # Divergence detection
                "diverge_detect_steps": ("INT", {"default": 150, "min": 10, "max": 5000, "step": 10,
                                                 "tooltip": "Steps above threshold before entering monitoring mode"}),
                "diverge_stop_steps":   ("INT", {"default": 300, "min": 10, "max": 5000, "step": 10,
                                                 "tooltip": "Steps in monitoring without recovery before stopping training"}),
                "diverge_threshold":    ("FLOAT", {"default": 15.0, "min": 0.0, "max": 200.0, "step": 1.0,
                                                   "tooltip": "% above lowest EMA loss to trigger divergence detection. 15 = trigger if loss rises 15% above best."}),
                # Resume
                "resume": ("BOOLEAN", {"default": False, "tooltip": "Resume from latest checkpoint in output directory"}),
                # Debug
                "validation_only": ("BOOLEAN", {"default": False, "tooltip": "TEMP: Skip training, just run one validation step using latest checkpoint"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "lora_path")
    FUNCTION = "train"
    CATEGORY = "RS Nodes"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute — training is never cacheable
        return float("nan")

    @staticmethod
    def _find_latest_checkpoint(output_dir: Path) -> str:
        """Find the checkpoint with the highest step number."""
        import re
        ckpt_dir = output_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            return ""
        best_step, best_path = -1, ""
        for f in ckpt_dir.glob("lora_weights_step_*.safetensors"):
            m = re.search(r"step_(\d+)", f.stem)
            if m and int(m.group(1)) > best_step:
                best_step = int(m.group(1))
                best_path = str(f)
        if best_path:
            print(f"Auto-resume: found latest checkpoint at step {best_step}")
        return best_path

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(
        self,
        model,
        output_name: str,
        preprocessed_data_root: str,
        model_path: str,
        preset: str = "subject",
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        video_self_attention: bool = True,
        video_cross_attention: bool = False,
        video_feed_forward: bool = False,
        audio_self_attention: bool = False,
        audio_cross_attention: bool = False,
        audio_feed_forward: bool = False,
        video_attends_to_audio: bool = False,
        audio_attends_to_video: bool = False,
        learning_rate: float = 1e-4,
        lr_end: float = 0.0,
        epochs: int = 3,
        auto_stop: bool = False,
        optimizer: str = "adamw8bit",
        rose_stabilize: bool = True,
        rose_weight_decay: float = 1e-4,
        rose_wd_schedule: bool = False,
        scheduler: str = "linear",
        lr_cycle_steps: int = 0,
        lr_cycle_decay: float = 1.0,
        gradient_checkpointing: bool = True,
        ffn_chunks: int = 0,
        quantization: str = "fp8-quanto",
        strategy: str = "text_to_video",
        first_frame_conditioning_p: float = 0.5,
        with_audio: bool = False,
        clip=None,
        validation_prompt: str = "",
        validation_interval: int = 250,
        validation_width: int = 576,
        validation_height: int = 576,
        validation_frames: int = 49,
        checkpoint_interval: int = 500,
        keep_last_n: int = 2,
        diverge_detect_steps: int = 150,
        diverge_stop_steps: int = 300,
        diverge_threshold: float = 0.0001,
        resume: bool = False,
        validation_only: bool = False,
        vae=None,
        unique_id=None,
    ):
        validate_submodule()

        data_root = Path(preprocessed_data_root)
        if not data_root.exists():
            raise ValueError(f"Preprocessed data root does not exist: {data_root}")

        model_full_path = folder_paths.get_full_path("checkpoints", model_path)
        if not model_full_path:
            raise ValueError(f"Checkpoint not found in ComfyUI checkpoints: {model_path}")

        # Build target_modules from toggles
        target_modules = self._build_target_modules(
            video_self_attention, video_cross_attention, video_feed_forward,
            audio_self_attention, audio_cross_attention, audio_feed_forward,
            video_attends_to_audio, audio_attends_to_video,
        )
        if not target_modules:
            raise ValueError("At least one module group must be enabled for LoRA training")

        # Output directories — inside the model's own folder, not the shared parent
        output_dir = data_root / "training_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        global _active_training_dir
        _active_training_dir = output_dir

        # Use the loras folder that already has files (extra_model_paths), fall back to first
        lora_paths = folder_paths.get_folder_paths("loras")
        loras_dir = Path(lora_paths[0])
        for p in lora_paths:
            pp = Path(p)
            if pp.is_dir() and any(pp.iterdir()):
                loras_dir = pp
                break
        loras_dir.mkdir(parents=True, exist_ok=True)

        lora_name = output_name.strip() or data_root.parent.name

        # Make sure ltx packages are importable
        _setup_ltx_paths()

        use_quantization = quantization != "none"

        # ---- Step 1: get transformer ----
        # Always use ComfyUI's already-loaded transformer to avoid having TWO 22B
        # models in memory (which OOMs on ≤24 GB GPUs).
        #
        # When quantization is requested, quanto modifies the model IN PLACE.
        # This corrupts ComfyUI's copy — the user must reload the checkpoint after
        # training.  But it's the only way to fit in VRAM.
        raw_transformer = model.model.diffusion_model
        raw_transformer.requires_grad_(False)

        if use_quantization:
            logger.info(
                "Quantization requested — will quantize ComfyUI's model in-place. "
                "You will need to reload the checkpoint after training."
            )
        else:
            logger.info("Reusing already-loaded transformer from ComfyUI MODEL wrapper.")

        # ---- Step 2: load EmbeddingsProcessor from checkpoint ----
        logger.info("Loading EmbeddingsProcessor from checkpoint...")
        from ltx_trainer.model_loader import load_embeddings_processor
        embeddings_processor = load_embeddings_processor(
            checkpoint_path=model_full_path,
            device="cuda",
            dtype=torch.bfloat16,
        )
        # Drop the feature_extractor — only connectors are needed during training
        embeddings_processor.feature_extractor = None
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Step 3: cache validation embeddings if CLIP + prompt provided ----
        cached_validation_embeddings = None
        validation_config = None

        if clip is not None and validation_prompt.strip() and validation_interval > 0:
            logger.info("Encoding validation prompt...")
            try:
                cached_validation_embeddings = self._encode_validation_prompt(
                    clip=clip,
                    embeddings_processor=embeddings_processor,
                    validation_prompt=validation_prompt,
                    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                    device=torch.device("cuda"),
                )
                validation_config = {
                    "prompt": validation_prompt,
                    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
                    "width": validation_width,
                    "height": validation_height,
                    "num_frames": validation_frames,
                    "frame_rate": 25.0,
                    "num_inference_steps": 30,
                    "guidance_scale": 4.0,
                    "seed": 42,
                    "interval": validation_interval,
                    "stg_scale": 1.0,
                    "stg_blocks": [29],
                    "stg_mode": "stg_av" if with_audio else "stg_v",
                    "generate_audio": with_audio,
                }
                logger.info("Validation prompt encoded.")
            except Exception as e:
                logger.warning(f"Failed to encode validation prompt — validation disabled: {e}")
                cached_validation_embeddings = None
                validation_config = None

        # ---- Step 4: free all other VRAM and RAM so training has headroom ----
        # Aggressively unload everything from GPU. The transformer will be moved
        # back to GPU by the InProcessTrainer after quantization + LoRA setup.
        mm.unload_all_models()

        # Drop local CLIP reference so it's not held during training.
        # We intentionally do NOT move the CLIP to "meta" — ComfyUI caches the
        # clip object across workflow runs, and destroying its weights leaves
        # it in a state that can't be recovered without a server restart.
        # After mm.unload_all_models() above, the CLIP is already off the GPU;
        # its CPU/pinned-RAM copy is kept for reuse at inference time.
        if clip is not None:
            clip = None

        # Explicitly move ComfyUI components to CPU
        if vae is not None:
            try:
                vae.first_stage_model.to("cpu")
            except Exception:
                pass
        try:
            raw_transformer.to("cpu")
        except Exception:
            pass
        try:
            embeddings_processor.to("cpu")
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"VRAM freed. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        logger.info(f"RAM usage: {__import__('psutil').Process().memory_info().rss / 1024**3:.1f} GB")

        # ---- Step 5: load VAE decoder for validation (skip if disabled) ----
        vae_decoder = None
        audio_vae_decoder = None
        vocoder = None
        if validation_config is not None:
            if vae is not None:
                logger.info("Preparing ComfyUI VAE decoder for validation...")
                vae_model = vae.first_stage_model
                with torch.inference_mode(False):
                    clean_sd = {k: v.clone() for k, v in vae_model.state_dict().items()}
                vae_model.load_state_dict(clean_sd, assign=True)
                del clean_sd
                vae_decoder = _ComfyVAEDecoderAdapter(vae_model)
                vae_decoder.requires_grad_(False)
            else:
                logger.info("Loading VideoDecoder from checkpoint for validation...")
                _setup_ltx_paths()
                from ltx_trainer.model_loader import load_video_vae_decoder
                vae_decoder = load_video_vae_decoder(
                    checkpoint_path=model_full_path,
                    device="cpu",
                    dtype=torch.bfloat16,
                )
                vae_decoder.requires_grad_(False)

            if with_audio:
                logger.info("Loading audio VAE + vocoder for validation...")
                from ltx_trainer.model_loader import load_audio_vae_decoder, load_vocoder
                audio_vae_decoder = load_audio_vae_decoder(
                    checkpoint_path=model_full_path, device="cpu", dtype=torch.bfloat16
                )
                audio_vae_decoder.requires_grad_(False)
                vocoder = load_vocoder(
                    checkpoint_path=model_full_path, device="cpu", dtype=torch.bfloat16
                )
                vocoder.requires_grad_(False)
        else:
            logger.info("Validation disabled — skipping VAE decoder load")

        # ---- Step 6: build training strategy + dataset + timestep sampler ----
        from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
        from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig, VideoToVideoStrategy
        from ltx_trainer.datasets import PrecomputedDataset
        from ltx_trainer.timestep_samplers import ShiftedLogitNormalTimestepSampler

        if strategy == "text_to_video":
            strategy_config = TextToVideoConfig(
                first_frame_conditioning_p=first_frame_conditioning_p,
                with_audio=with_audio,
            )
            training_strategy = TextToVideoStrategy(strategy_config)
        elif strategy == "video_to_video":
            strategy_config = VideoToVideoConfig(
                first_frame_conditioning_p=first_frame_conditioning_p,
            )
            training_strategy = VideoToVideoStrategy(strategy_config)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        data_sources = training_strategy.get_data_sources()
        dataset = PrecomputedDataset(
            data_root=str(data_root),
            data_sources=data_sources,
        )
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        total_steps = epochs * len(dataset)
        if auto_stop:
            logger.info(f"Training plan: {len(dataset)} samples/epoch — epochs will continue until divergence detection stops it")
        else:
            logger.info(f"Training plan: {epochs} epochs × {len(dataset)} samples = {total_steps} steps")

        # Default: shifted logit-normal sampler (matches official config)
        timestep_sampler = ShiftedLogitNormalTimestepSampler()

        # ---- Step 7: build and run InProcessTrainer ----
        from ..utils.ltxv_inprocess_trainer import InProcessTrainer

        trainer = InProcessTrainer(
            transformer=raw_transformer,
            embeddings_processor=embeddings_processor,
            vae_decoder=vae_decoder,
            scheduler=None,  # Not needed for training — only used in validation via ValidationSampler
            dataset=dataset,
            training_strategy=training_strategy,
            timestep_sampler=timestep_sampler,
            output_dir=output_dir,
            loras_dir=loras_dir,
            lora_name=lora_name,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
            learning_rate=learning_rate,
            lr_end=lr_end,
            total_steps=total_steps,
            auto_stop=auto_stop,
            optimizer_type=optimizer,
            rose_stabilize=rose_stabilize,
            rose_weight_decay=rose_weight_decay,
            rose_wd_schedule=rose_wd_schedule,
            scheduler_type=scheduler,
            lr_cycle_steps=lr_cycle_steps,
            lr_cycle_decay=lr_cycle_decay,
            max_grad_norm=1.0,
            gradient_checkpointing=gradient_checkpointing,
            quantization=quantization if quantization != "none" else None,
            keep_last_n=keep_last_n,
            checkpoint_interval=checkpoint_interval,
            diverge_detect_steps=diverge_detect_steps,
            diverge_stop_steps=diverge_stop_steps,
            diverge_threshold=diverge_threshold,
            validation_config=validation_config,
            cached_validation_embeddings=cached_validation_embeddings,
            audio_vae_decoder=audio_vae_decoder,
            vocoder=vocoder,
            seed=42,
            resume_checkpoint=self._find_latest_checkpoint(output_dir) if resume else "",
            layer_offloading=True,
            node_id=str(unique_id) if unique_id is not None else "",
            ffn_chunks=ffn_chunks,
        )

        pbar = comfy.utils.ProgressBar(total_steps)
        cancel_check = mm.throw_exception_if_processing_interrupted

        lora_path = ""
        status = ""
        try:
            # CRITICAL: ComfyUI wraps all node execution in torch.inference_mode().
            # Training requires autograd, so we must fully exit inference mode for
            # the entire training run.  This is the outermost exit point — all
            # tensors created inside will be normal (non-inference) tensors.
            if validation_only:
                device = mm.get_torch_device()
                trainer.run_validation_only(device)
                status = "Validation-only run complete."
            else:
                with torch.inference_mode(False):
                    lora_path = trainer.train(progress_bar=pbar, cancel_check=cancel_check)
                status = f"Training complete! {total_steps} steps ({epochs} epochs). LoRA saved to: {lora_path}"
        except Exception as e:
            status = f"Training failed: {e}"
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            trainer.cleanup()
            # Clean up local copies of loaded components
            del embeddings_processor
            del vae_decoder
            if audio_vae_decoder is not None:
                del audio_vae_decoder
            if vocoder is not None:
                del vocoder
            gc.collect()
            torch.cuda.empty_cache()

        return (status, lora_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_target_modules(self, *toggles) -> list[str]:
        """Build target_modules list from the 8 boolean toggles."""
        group_names = list(MODULE_GROUPS.keys())
        modules = []
        for toggle, name in zip(toggles, group_names):
            if toggle:
                modules.extend(MODULE_GROUPS[name])
        return modules

    def _encode_validation_prompt(
        self,
        clip,
        embeddings_processor,
        validation_prompt: str,
        negative_prompt: str,
        device: torch.device,
    ):
        """Encode validation prompt via ComfyUI's CLIP + EmbeddingsProcessor connectors.

        ComfyUI's LTX-2 AV CLIP (LTXAVTEModel) runs Gemma + feature projection
        (blocks 1+2 of the text encoder pipeline).  The EmbeddingsProcessor runs
        the connectors (block 3).  Together they produce CachedPromptEmbeddings
        identical to those produced by the original trainer's
        _load_text_encoder_and_cache_embeddings().

        For LTX-2   (single_linear): cond is [B, seq, 3840]; same features used
                                      for both video and audio connectors.
        For LTX-2.3 (dual_linear):   cond is [B, seq, 4096+2048]; split to get
                                      separate video/audio features.

        Returns a CachedPromptEmbeddings with all four embedding tensors on CPU.
        """
        from ltx_core.text_encoders.gemma import convert_to_additive_mask
        from ltx_trainer.validation_sampler import CachedPromptEmbeddings

        embeddings_processor.to(device)

        def _encode_single(prompt: str):
            # Use ComfyUI's CLIP to run Gemma + feature projection
            # encode_from_tokens handles model loading internally via the patcher
            tokens = clip.tokenize(prompt)
            # return_dict=True so we get any extra keys (e.g. unprocessed_ltxav_embeds)
            result = clip.encode_from_tokens(tokens, return_dict=True)
            cond = result["cond"]  # [1, seq_len, proj_dim]

            cond = cond.to(device=device, dtype=torch.bfloat16)
            seq_len = cond.shape[1]

            # Detect model variant by projection output dimension
            # LTX-2 single_linear -> proj_dim=3840; LTX-2.3 dual_linear -> 4096+2048=6144
            proj_dim = cond.shape[2]

            if proj_dim == 3840:
                # LTX-2: same features for both modalities
                video_features = cond
                audio_features = cond
            elif proj_dim >= 6144:
                # LTX-2.3: split at 4096
                video_features = cond[:, :, :4096]
                audio_features = cond[:, :, 4096:]
            else:
                # Unknown split — use same for both (fallback)
                video_features = cond
                audio_features = cond

            # EmbeddingsProcessor.create_embeddings requires seq_len divisible
            # by num_learnable_registers (128). Left-pad to match Gemma convention.
            pad_to = 128
            pad_len = (pad_to - seq_len % pad_to) % pad_to
            if pad_len > 0:
                video_features = torch.nn.functional.pad(video_features, (0, 0, pad_len, 0))
                if audio_features is not video_features:
                    audio_features = torch.nn.functional.pad(audio_features, (0, 0, pad_len, 0))
                seq_len = seq_len + pad_len

            # Attention mask: False for left-padding, True for real tokens
            attention_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
            if pad_len > 0:
                attention_mask[:, :pad_len] = False
            additive_mask = convert_to_additive_mask(attention_mask, video_features.dtype)

            with torch.inference_mode():
                video_enc, audio_enc, _ = embeddings_processor.create_embeddings(
                    video_features, audio_features, additive_mask
                )

            return video_enc.cpu(), audio_enc.cpu() if audio_enc is not None else None

        # Encode positive and negative prompts
        v_pos, a_pos = _encode_single(validation_prompt)
        v_neg, a_neg = _encode_single(negative_prompt)

        # Soft cache clear after encoding — ComfyUI manages CLIP's lifetime
        mm.soft_empty_cache()

        return CachedPromptEmbeddings(
            video_context_positive=v_pos,
            audio_context_positive=a_pos if a_pos is not None else torch.zeros_like(v_pos),
            video_context_negative=v_neg,
            audio_context_negative=a_neg if a_neg is not None else torch.zeros_like(v_neg),
        )
