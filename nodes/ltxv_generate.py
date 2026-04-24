import gc
import logging
import math
import random
import uuid

import torch
import comfy.model_management as mm
import comfy.model_sampling
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.nested_tensor
import folder_paths
import node_helpers
import latent_preview

logger = logging.getLogger(__name__)


class RSLTXVGenerate:
    """
    All-in-one LTXV video generation node. Handles conditioning, frame injection,
    optional audio latents, scheduling, sampling, optional upscaling, and decoding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("MODEL",),
                "positive":   ("CONDITIONING",),
                "negative":   ("CONDITIONING",),
                "vae":        ("VAE",),
            },
            "optional": {
                # Generation
                "width":      ("INT",   {"default": 768,  "min": 64,  "max": 8192, "step": 32}),
                "height":     ("INT",   {"default": 512,  "min": 64,  "max": 8192, "step": 32}),
                "num_frames": ("INT",   {"default": 97,   "min": 9,   "max": 8192, "step": 8}),
                "steps":      ("INT",   {"default": 20,   "min": 1,   "max": 10000}),
                "cfg":        ("FLOAT", {"default": 3.0,  "min": 0.0, "max": 100.0, "step": 0.1}),
                "noise_seed": ("INT",   {"default": 0,    "min": 0,   "max": 0xffffffffffffffff}),
                "seed_mode":  (["random", "fixed", "increment", "decrement"],),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                # Frame injection
                "first_image":       ("IMAGE",),
                "middle_image":      ("IMAGE",),
                "last_image":        ("IMAGE",),
                "guide_video":       ("IMAGE", {"tooltip": "Video-to-video: inject every 8th frame as a guide. When guide_mask is also connected, acts as init latent for inpainting instead."}),
                "guide_mask":        ("IMAGE", {"tooltip": "Inpaint mask video (white=regenerate, black=preserve). Requires guide_video. Changes guide_video from frame injection to inpaint mode."}),
                "first_strength":    ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "middle_strength":   ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "last_strength":     ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "guide_strength":    ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength for guide_video frames."}),
                "guide_every_nth":   ("INT",   {"default": 8,   "min": 1,   "max": 64,  "step": 1,    "tooltip": "Inject a guide every N frames. Match this to the splitter's every_nth."}),
                "guide_index_list":  ("STRING", {"default": "",  "tooltip": "Comma-separated frame indices to inject as guides (e.g. '0,8,24'). Overrides guide_every_nth when set."}),
                "crf":               ("INT",   {"default": 35,   "min": 0,   "max": 100}),
                # Audio
                "audio":             ("AUDIO",),
                "audio_vae":         ("VAE",),
                # Multimodal guidance (used when audio_vae is connected)
                "audio_cfg":         ("FLOAT", {"default": 7.0,  "min": 0.0, "max": 100.0, "step": 0.1}),
                "stg_scale":         ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 10.0,  "step": 0.1}),
                "stg_perturbation":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                                "tooltip": "Attention perturbation strength. 1.0 = full skip (original STG), <1.0 = soft blend."}),
                "audio_stg_scale":   ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0, "step": 0.1, "tooltip": "Audio STG scale (-1 = use video stg_scale)"}),
                "cfg_end":           ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
                "stg_end":           ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0,  "step": 0.1}),
                "stg_blocks":        ("STRING", {"default": "29"}),
                "rescale":           ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0,   "step": 0.01}),
                "video_modality_scale": ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Video modality isolation (0 = match official default)"}),
                "audio_modality_scale": ("FLOAT", {"default": 3.0,  "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Audio modality isolation (3 = match official default)"}),
                "cfg_star_rescale":  ("BOOLEAN", {"default": True, "tooltip": "CFG-Zero*: project negative prediction onto positive to prevent garbage at high sigma. Recommended on."}),
                "skip_sigma":        ("FLOAT",  {"default": 0.0,  "min": 0.0, "max": 1.0,  "step": 0.001, "tooltip": "Skip guidance when sigma > this value (CFG-Zero init). 0 = disabled, 0.997 = official default."}),
                # Efficiency
                "attention_mode":    (["auto", "default", "sage"],),
                "ffn_chunks":        ("INT",   {"default": 4, "min": 0, "max": 16, "step": 1}),
                "video_attn_scale":  ("FLOAT", {"default": 1.03, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Video attention scale (1.03 recommended). Also enables VRAM-efficient block forward for longer generation."}),
                # Upscale
                "upscale":           ("BOOLEAN", {"default": False}),
                "upscale_model":     ("LATENT_UPSCALE_MODEL",),
                "temporal_upscale_model": ("LATENT_UPSCALE_MODEL",),
                "upscale_lora":      (["none"] + folder_paths.get_filename_list("loras"),),
                "upscale_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "upscale_steps":     ("INT",   {"default": 4,   "min": 1,   "max": 10000}),
                "upscale_cfg":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "upscale_denoise":   ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,   "step": 0.01}),
                "upscale_fallback":  ("BOOLEAN", {"default": False}),
                "upscale_tiling":    ("BOOLEAN", {"default": False, "tooltip": "Enable temporal tiling for upscale. Reduces VRAM but will cause temporal instability in fine details. Use ffn_chunks instead if possible."}),
                "upscale_tile_t":    ("INT",     {"default": 4, "min": 0, "max": 256, "step": 1, "tooltip": "Temporal tile size (latent frames) when upscale_tiling is enabled (0 = auto). Reduce if OOM during upscale."}),
                "rediffusion_mask":  ("MASK", {"tooltip": "Spatial mask for rediffusion. 1=rediffuse (subject), 0=preserve (background). Use RMBG to generate."}),
                "rediffusion_mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of the rediffusion mask. 0=ignore mask (full rediffusion everywhere), 1=full mask effect."}),
                # Output
                "decode":            ("BOOLEAN", {"default": True}),
                "tile_t":            ("INT",     {"default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "Temporal tile size for VAE decode (0 = auto). Lower values reduce VRAM but may cause seams."}),
                # Overrides
                "guider":            ("GUIDER",),
                "sampler":           ("SAMPLER",),
                "sigmas":            ("SIGMAS",),
                # Scheduler (ignored if sigmas provided)
                "max_shift":         ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift":        ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES  = ("LATENT", "LATENT", "IMAGE", "AUDIO")
    RETURN_NAMES  = ("latent", "audio_latent", "images", "audio_output")
    OUTPUT_NODE   = True
    FUNCTION      = "generate"
    CATEGORY      = "rs-nodes"

    def __init__(self):
        self.loaded_lora = None
        self._last_seed = None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return uuid.uuid4().hex

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        model,
        positive,
        negative,
        vae,
        # Generation
        width=768,
        height=512,
        num_frames=97,
        steps=20,
        cfg=3.0,
        noise_seed=0,
        seed_mode="random",
        frame_rate=25.0,
        # Frame injection
        first_image=None,
        middle_image=None,
        last_image=None,
        guide_video=None,
        guide_mask=None,
        first_strength=1.0,
        middle_strength=1.0,
        last_strength=1.0,
        guide_strength=0.7,
        guide_every_nth=8,
        guide_index_list="",
        crf=35,
        # Audio
        audio=None,
        audio_vae=None,
        # Multimodal guidance
        audio_cfg=7.0,
        stg_scale=0.0,
        stg_perturbation=1.0,
        audio_stg_scale=-1.0,
        cfg_end=-1.0,
        stg_end=-1.0,
        stg_blocks="29",
        rescale=0.7,
        video_modality_scale=0.0,
        audio_modality_scale=3.0,
        cfg_star_rescale=True,
        skip_sigma=0.0,
        # Efficiency
        attention_mode="auto",
        ffn_chunks=4,
        video_attn_scale=1.03,
        # Upscale
        upscale=False,
        upscale_model=None,
        temporal_upscale_model=None,
        upscale_lora="none",
        upscale_lora_strength=1.0,
        upscale_steps=4,
        upscale_cfg=1.0,
        upscale_denoise=0.5,
        upscale_fallback=False,
        upscale_tiling=False,
        upscale_tile_t=4,
        rediffusion_mask=None,
        rediffusion_mask_strength=1.0,
        # Output
        decode=False,
        tile_t=0,
        # Overrides
        guider=None,
        sampler=None,
        sigmas=None,
        # Scheduler
        max_shift=2.05,
        base_shift=0.95,
        **kwargs,
    ):
        # Resolve seed based on mode (replaces broken control_after_generate)
        seed = noise_seed
        if seed_mode == "random":
            seed = random.randint(0, 0xffffffffffffffff)
        elif seed_mode == "increment" and self._last_seed is not None:
            seed = (self._last_seed + 1) % (0xffffffffffffffff + 1)
        elif seed_mode == "decrement" and self._last_seed is not None:
            seed = (self._last_seed - 1) % (0xffffffffffffffff + 1)
        self._last_seed = seed
        logger.info(f"Starting generation (seed={seed}, mode={seed_mode})")
        try:
            result = self._generate_impl(
                model, positive, negative, vae,
                width=width, height=height, num_frames=num_frames,
                steps=steps, cfg=cfg, seed=seed, frame_rate=frame_rate,
                first_image=first_image, middle_image=middle_image, last_image=last_image,
                guide_video=guide_video, guide_mask=guide_mask,
                first_strength=first_strength, middle_strength=middle_strength,
                last_strength=last_strength, guide_strength=guide_strength,
                guide_every_nth=guide_every_nth, guide_index_list=guide_index_list, crf=crf,
                audio=audio, audio_vae=audio_vae,
                audio_cfg=audio_cfg, stg_scale=stg_scale,
                stg_perturbation=stg_perturbation,
                audio_stg_scale=audio_stg_scale,
                cfg_end=cfg_end, stg_end=stg_end,
                stg_blocks=stg_blocks,
                rescale=rescale,
                video_modality_scale=video_modality_scale,
                audio_modality_scale=audio_modality_scale,
                cfg_star_rescale=cfg_star_rescale, skip_sigma=skip_sigma,
                attention_mode=attention_mode, ffn_chunks=ffn_chunks,
                video_attn_scale=video_attn_scale,
                upscale=upscale, upscale_model=upscale_model,
                temporal_upscale_model=temporal_upscale_model,
                upscale_lora=upscale_lora, upscale_lora_strength=upscale_lora_strength,
                upscale_steps=upscale_steps, upscale_cfg=upscale_cfg,
                upscale_denoise=upscale_denoise,
                upscale_fallback=upscale_fallback,
                upscale_tiling=upscale_tiling,
                upscale_tile_t=upscale_tile_t,
                rediffusion_mask=rediffusion_mask,
                rediffusion_mask_strength=rediffusion_mask_strength,
                decode=decode, tile_t=tile_t,
                guider=guider, sampler=sampler, sigmas=sigmas,
                max_shift=max_shift, base_shift=base_shift,
            )
            # Write resolved seed back to the widget so the user can see/reuse it
            return {"ui": {"noise_seed": [seed]}, "result": result}
        except Exception:
            logger.info("Error during generation, cleaning up VRAM")
            raise
        finally:
            self._free_vram()

    def _generate_impl(
        self,
        model, positive, negative, vae,
        width, height, num_frames, steps, cfg, seed, frame_rate,
        first_image, middle_image, last_image, guide_video, guide_mask,
        first_strength, middle_strength, last_strength, guide_strength, guide_every_nth, guide_index_list, crf,
        audio, audio_vae,
        audio_cfg, stg_scale, stg_perturbation, audio_stg_scale, cfg_end, stg_end,
        stg_blocks, rescale, video_modality_scale, audio_modality_scale,
        cfg_star_rescale, skip_sigma,
        attention_mode, ffn_chunks, video_attn_scale,
        upscale, upscale_model, temporal_upscale_model, upscale_lora, upscale_lora_strength,
        upscale_steps, upscale_cfg, upscale_denoise, upscale_fallback,
        upscale_tiling, upscale_tile_t,
        rediffusion_mask, rediffusion_mask_strength,
        decode, tile_t,
        guider, sampler, sigmas,
        max_shift, base_shift,
    ):
        # ----------------------------------------------------------------
        # 1. SETUP
        # ----------------------------------------------------------------

        m = model.clone()

        # Attention override
        attn_func = None
        if attention_mode != "default":
            if attention_mode == "sage":
                from comfy.ldm.modules.attention import attention_sage
                attn_func = attention_sage
            elif attention_mode == "auto":
                from comfy.ldm.modules.attention import SAGE_ATTENTION_IS_AVAILABLE, attention_sage
                if SAGE_ATTENTION_IS_AVAILABLE:
                    attn_func = attention_sage

        if attn_func is not None:
            m.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = lambda func, *args, **kwargs: attn_func(*args, **kwargs)

        # FFN chunking
        if ffn_chunks > 0:
            self._apply_ffn_chunking(m, ffn_chunks)

        # When audio is provided, derive video length from audio duration
        if audio is not None and audio_vae is not None:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            audio_duration = waveform.shape[-1] / sample_rate
            audio_num_frames = int(audio_duration * frame_rate) + 1
            # Round up to nearest multiple of 8 + 1 (LTXV temporal stride)
            audio_num_frames = ((audio_num_frames - 1 + 7) // 8) * 8 + 1
            if audio_num_frames != num_frames:
                logger.info(f"Audio duration {audio_duration:.2f}s → overriding num_frames: {num_frames} → {audio_num_frames}")
                num_frames = audio_num_frames

        # When upscaling, generate at half resolution — the 2x latent upscaler
        # brings it to the target width x height afterwards.
        # Round to 64-align so latent dims are even (required for IC-LoRA dsf=2)
        gen_width = (width + 63) // 64 * 64
        gen_height = (height + 63) // 64 * 64
        if gen_width != width or gen_height != height:
            logger.info(f"Resolution 64-aligned: {width}x{height} → {gen_width}x{gen_height}")
        do_upscale = upscale and upscale_model is not None
        if upscale and upscale_model is None:
            logger.info("WARNING: upscale=True but no upscale_model connected — generating at full resolution")
        if do_upscale:
            gen_width = (width // 2 + 63) // 64 * 64
            gen_height = (height // 2 + 63) // 64 * 64
            logger.info(f"Upscale enabled: generating at {gen_width}x{gen_height}, target {width}x{height}")

        # T2V first-frame bootstrap: generate a short clip at full resolution,
        # decode the first frame, and use it as I2V guidance for the main generation.
        # This converts T2V→I2V so the upscale path only needs one re-diffusion pass.
        # DISABLED: testing natural first pass generation
        if False and do_upscale and first_image is None and last_image is None and middle_image is None:
            logger.info(f"T2V bootstrap: generating 9 frames at {width}x{height} for first frame")
            bootstrap_frames = 9
            bootstrap_latent = torch.zeros(
                [1, 128, ((bootstrap_frames - 1) // 8) + 1, height // 32, width // 32],
                device=mm.intermediate_device(),
            )

            # Build shift + sigmas for bootstrap resolution
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST

            class _BootstrapSampling(sampling_base, sampling_type):
                pass

            boot_tokens = math.prod(bootstrap_latent.shape[2:])
            x1, x2 = 1024, 4096
            boot_mm = (max_shift - base_shift) / (x2 - x1)
            boot_b = base_shift - boot_mm * x1
            boot_shift = boot_tokens * boot_mm + boot_b

            boot_model = m.clone()
            boot_ms = _BootstrapSampling(boot_model.model.model_config)
            boot_ms.set_parameters(shift=boot_shift)
            boot_model.add_object_patch("model_sampling", boot_ms)

            boot_steps = 30  # Fixed step count for sharp bootstrap frame
            boot_sig = torch.linspace(1.0, 0.0, boot_steps + 1, dtype=torch.float64)
            boot_sig = torch.where(
                boot_sig != 0,
                math.exp(boot_shift) / (math.exp(boot_shift) + (1.0 / boot_sig - 1.0) ** 1),
                torch.zeros_like(boot_sig),
            )
            nz = boot_sig != 0
            omz = 1.0 - boot_sig[nz]
            boot_sig[nz] = 1.0 - (omz / (omz[-1] / (1.0 - 0.1)))
            boot_sig = boot_sig.float()

            # Stamp full frame rate for bootstrap
            boot_pos = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
            boot_neg = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

            from ..utils.multimodal_guider import MultimodalGuider
            boot_guider = MultimodalGuider(
                boot_model, boot_pos, boot_neg,
                video_cfg=cfg, audio_cfg=audio_cfg,
                stg_scale=stg_scale,
                stg_blocks=[int(s.strip()) for s in stg_blocks.split(",")],
                rescale=rescale,
                video_cfg_end=cfg_end if cfg_end >= 0 else None,
                stg_scale_end=stg_end if stg_end >= 0 else None,
                audio_stg_scale=audio_stg_scale if audio_stg_scale >= 0 else None,
                video_modality_scale=video_modality_scale,
                audio_modality_scale=audio_modality_scale,
                cfg_star_rescale=cfg_star_rescale,
                skip_sigma_threshold=skip_sigma,
                video_attn_scale=video_attn_scale,
            )

            boot_latent = comfy.sample.fix_empty_latent_channels(boot_guider.model_patcher, bootstrap_latent)
            boot_noise = comfy.sample.prepare_noise(boot_latent, seed)
            boot_sampler = getattr(guider, 'ic_lora_sampler', None) or comfy.samplers.sampler_object("euler_ancestral")
            boot_cb = latent_preview.prepare_callback(boot_guider.model_patcher, boot_sig.shape[-1] - 1)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

            logger.info("T2V bootstrap: sampling first frame...")
            boot_out = boot_guider.sample(
                boot_noise, boot_latent, boot_sampler, boot_sig,
                denoise_mask=None, callback=boot_cb,
                disable_pbar=disable_pbar, seed=seed,
            )
            boot_out = boot_out.to(mm.intermediate_device())

            # Decode first frame
            first_pixels = vae.decode(boot_out[:, :, :1])
            if len(first_pixels.shape) == 5:
                first_pixels = first_pixels.reshape(
                    -1, first_pixels.shape[-3], first_pixels.shape[-2], first_pixels.shape[-1]
                )
            first_image = first_pixels
            first_strength = 1.0
            logger.info(f"T2V bootstrap: first frame {first_image.shape[2]}x{first_image.shape[1]} ready")

            del boot_out, boot_model, boot_guider
            self._free_vram()

        # Temporal upscale: generate at half frame count, then 2x temporal upscale
        do_temporal_upscale = do_upscale and temporal_upscale_model is not None
        gen_num_frames = num_frames
        if do_temporal_upscale:
            target_latent_T = ((num_frames - 1) // 8) + 1
            half_latent_T = target_latent_T // 2 + 1
            gen_num_frames = (half_latent_T - 1) * 8 + 1
            logger.info(f"Temporal upscale enabled: generating {gen_num_frames} frames at {frame_rate / 2:.1f}fps (target {num_frames} frames at {frame_rate}fps)")

        # Create empty latent: [B, C, T, H, W] — LTXV latent space
        latent = torch.zeros(
            [1, 128, ((gen_num_frames - 1) // 8) + 1, gen_height // 32, gen_width // 32],
            device=mm.intermediate_device(),
        )

        # Stamp frame rate onto conditioning — halved for temporal upscale so the
        # model generates motion at the correct speed for the full target duration
        gen_frame_rate = frame_rate / 2 if do_temporal_upscale else frame_rate
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": gen_frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": gen_frame_rate})

        # ----------------------------------------------------------------
        # 2. FRAME INJECTION (all guides use LTXVAddGuide append + keyframe_idxs)
        # ----------------------------------------------------------------

        from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask

        latent_dict = {"samples": latent}
        noise_mask = get_noise_mask(latent_dict)

        guides = []
        guide_video_latent = None  # Single VAE encode for entire guide video
        inpaint_mode = guide_video is not None and guide_mask is not None
        if inpaint_mode:
            # INPAINT MODE: guide_video becomes the init latent, guide_mask controls
            # which regions get denoised (white=1=regenerate, black=0=preserve)
            import torch.nn.functional as F

            num_video_frames = guide_video.shape[0]
            scale_factors = vae.downscale_index_formula
            _, _, _, latent_height, latent_width = latent_dict["samples"].shape

            # Encode guide video → use as starting latent
            _, guide_video_latent = LTXVAddGuide.encode(
                vae, latent_width, latent_height, guide_video, scale_factors
            )
            logger.info(f"Inpaint mode: encoded {num_video_frames} guide frames → latent {list(guide_video_latent.shape)}")

            # Replace the empty latent with the encoded guide video
            latent = guide_video_latent
            latent_dict = {"samples": latent}

            # Build noise_mask from guide_mask video
            # guide_mask is [F, H, W, C] IMAGE — extract luminance → [F, H, W]
            mask_frames = guide_mask[..., 0]  # [F, H, W] — take first channel

            # Downsample spatially to latent dims
            # [F, H, W] → [1, F, H, W] (F treated as channels) → interpolate → [1, F, lH, lW]
            mask_4d = mask_frames.unsqueeze(0).float()
            mask_latent = F.interpolate(mask_4d, size=(latent_height, latent_width), mode="nearest")
            # → [1, F, lH, lW]

            # Downsample temporally: average within each latent frame's receptive field
            time_sf = scale_factors[0]
            latent_T = latent.shape[2]
            mask_t = torch.zeros(1, 1, latent_T, latent_height, latent_width,
                                 device=latent.device, dtype=latent.dtype)
            for t in range(latent_T):
                if t == 0:
                    # First latent frame maps to pixel frame 0
                    pix_start, pix_end = 0, 1
                else:
                    pix_start = 1 + (t - 1) * time_sf
                    pix_end = min(1 + t * time_sf, mask_frames.shape[0])
                if pix_start < mask_latent.shape[1]:
                    chunk = mask_latent[0, pix_start:pix_end]  # [chunk_len, lH, lW]
                    mask_t[0, 0, t] = chunk.mean(dim=0)

            # Threshold: anything > 0.5 gets denoised
            noise_mask = (mask_t > 0.5).float()

            # Blank out masked region so the model generates fresh content there
            latent = latent * (1.0 - noise_mask)
            latent_dict = {"samples": latent}

            latent_dict["noise_mask"] = noise_mask
            logger.info(f"Inpaint mask: latent shape {list(noise_mask.shape)}, "
                        f"denoise ratio: {noise_mask.mean().item():.1%}")
        elif guide_video is not None:
            num_video_frames = guide_video.shape[0]
            # Parse index list if provided, otherwise use every_nth
            if guide_index_list and guide_index_list.strip():
                indices = [int(x.strip()) for x in guide_index_list.split(",") if x.strip()]
                indices = [i if i >= 0 else num_video_frames + i for i in indices]
                indices = [i for i in indices if 0 <= i < num_video_frames]
            else:
                indices = list(range(0, num_video_frames, guide_every_nth))

            # Encode entire guide video in one VAE pass
            scale_factors = vae.downscale_index_formula
            time_sf = scale_factors[0]
            _, _, _, latent_height, latent_width = latent_dict["samples"].shape
            _, guide_video_latent = LTXVAddGuide.encode(
                vae, latent_width, latent_height, guide_video, scale_factors
            )
            logger.info(f"Video guide: encoded {num_video_frames} frames → latent {list(guide_video_latent.shape)}")

            # Map pixel indices to latent frame indices and slice
            for i in indices:
                if i == 0:
                    lat_idx = 0
                else:
                    lat_idx = (i - 1) // time_sf + 1
                lat_idx = min(lat_idx, guide_video_latent.shape[2] - 1)
                guide_slice = guide_video_latent[:, :, lat_idx:lat_idx+1, :, :]
                guides.append((guide_slice, i, guide_strength, f"v2v_{i}"))
            logger.info(f"Video-to-video: {len(guides)} guide latent slices at pixel indices {indices}")
        else:
            if first_image is not None:
                guides.append((first_image, 0, first_strength, "first"))
            if middle_image is not None:
                mid_idx = (gen_num_frames - 1) // 2
                mid_idx = max(0, (mid_idx // 8) * 8)
                if mid_idx == 0 and gen_num_frames > 8:
                    mid_idx = 8
                guides.append((middle_image, mid_idx, middle_strength, "middle"))
            if last_image is not None:
                guides.append((last_image, -1, last_strength, "last"))

        if guides:
            scale_factors = vae.downscale_index_formula
            for guide_entry, frame_idx, strength_val, label in guides:
                _, _, latent_length, latent_height, latent_width = latent_dict["samples"].shape

                if guide_video_latent is not None:
                    # Already encoded — guide_entry is a latent slice
                    t = guide_entry
                else:
                    # Individual image — encode per-frame
                    _, t = LTXVAddGuide.encode(vae, latent_width, latent_height, guide_entry, scale_factors)

                # guide_length: for pre-encoded latent slices use 1 pixel frame,
                # for individual images use the actual pixel frame count
                guide_length = 1 if guide_video_latent is not None else len(guide_entry)
                frame_idx_actual, latent_idx = LTXVAddGuide.get_latent_index(
                    positive, latent_length, guide_length, frame_idx, scale_factors
                )

                logger.info(f"Guide {label}: frame_idx={frame_idx_actual}, latent_idx={latent_idx}, strength={strength_val}")

                positive, negative, latent_samples, noise_mask = LTXVAddGuide.append_keyframe(
                    positive, negative, frame_idx_actual,
                    latent_dict["samples"], noise_mask,
                    t, strength_val, scale_factors,
                )
                latent_dict = {"samples": latent_samples, "noise_mask": noise_mask}

        # Propagate keyframe_idxs to external guider so it knows about
        # guide frame positions (RoPE mapping).
        if guides and guider is not None:
            from comfy_extras.nodes_lt import get_keyframe_idxs
            kf_idxs, _ = get_keyframe_idxs(positive)
            if kf_idxs is not None:
                guider_pos = guider._get_positive()
                guider_neg = guider._get_negative()
                guider_pos = node_helpers.conditioning_set_values(
                    guider_pos, {"keyframe_idxs": kf_idxs})
                guider_neg = node_helpers.conditioning_set_values(
                    guider_neg, {"keyframe_idxs": kf_idxs})
                guider.inner_set_conds(
                    {"positive": guider_pos, "negative": guider_neg})
                # Also update the guider's reset baseline so
                # _encode_and_inject_guide doesn't wipe these on re-run
                guider._orig_positive = guider_pos
                guider._orig_negative = guider_neg
                logger.info(f"Propagated keyframe_idxs to guider")

        # ----------------------------------------------------------------
        # 3. AUDIO (requires audio_vae for the AV dual-tower model)
        # ----------------------------------------------------------------

        has_audio = audio_vae is not None
        audio_is_input = has_audio and audio is not None  # True = user provided audio, pass through
        if has_audio:
            if audio is not None:
                # Encode provided audio into latent space
                logger.info("Encoding input audio latents")
                audio_samples = audio_vae.encode(audio)
            else:
                # Create empty audio latents — the AV model generates audio from scratch
                logger.info("Creating empty audio latents for generation")
                audio_samples = torch.zeros(
                    (1, audio_vae.latent_channels,
                     audio_vae.num_of_latents_from_frames(num_frames, frame_rate),
                     audio_vae.latent_frequency_bins),
                    device=mm.intermediate_device(),
                )

            video_samples = latent_dict["samples"]
            combined_samples = comfy.nested_tensor.NestedTensor((video_samples, audio_samples))

            video_noise_mask = latent_dict.get("noise_mask", None)
            if video_noise_mask is None:
                video_noise_mask = torch.ones(
                    video_samples.shape[0], 1, video_samples.shape[2], 1, 1,
                    device=video_samples.device, dtype=video_samples.dtype,
                )
            # Audio mask: 0 = preserve input audio, 1 = generate from scratch
            if audio is not None:
                audio_noise_mask = torch.zeros(
                    audio_samples.shape[0], 1, *audio_samples.shape[2:],
                    device=audio_samples.device, dtype=audio_samples.dtype,
                )
            else:
                audio_noise_mask = torch.ones(
                    audio_samples.shape[0], 1, *audio_samples.shape[2:],
                    device=audio_samples.device, dtype=audio_samples.dtype,
                )
            combined_noise_mask = comfy.nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))

            latent_dict = {"samples": combined_samples, "noise_mask": combined_noise_mask}

        # ----------------------------------------------------------------
        # 4. MODEL SAMPLING + SCHEDULING
        # ----------------------------------------------------------------

        # Build a combined ModelSamplingFlux + CONST class for LTXV shift scheduling
        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        # Compute token count for shift calculation
        latent_samples_ref = latent_dict["samples"]
        if latent_samples_ref.is_nested:
            tokens = math.prod(latent_samples_ref.unbind()[0].shape[2:])
        else:
            tokens = math.prod(latent_samples_ref.shape[2:])

        x1, x2 = 1024, 4096
        # Clamp tokens to prevent schedule collapse at high resolutions.
        # The linear shift formula is calibrated for x1–x2; beyond ~8k tokens
        # exp(shift) dominates and sigmas compress to ~1.0 (no denoising).
        tokens = min(tokens, x2 * 2)
        mm_shift = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm_shift * x1
        shift = tokens * mm_shift + b

        model_sampling_obj = ModelSamplingAdvanced(m.model.model_config)
        model_sampling_obj.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling_obj)

        # Build sigmas if not provided
        if sigmas is None:
            # Use float64 to avoid precision loss at high shift values
            # (shift > ~20 causes exp(shift) to swallow small terms in float32)
            sig = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
            sig = torch.where(
                sig != 0,
                math.exp(shift) / (math.exp(shift) + (1.0 / sig - 1.0) ** 1),
                torch.zeros_like(sig),
            )
            # Stretch so the terminal sigma = 0.1
            non_zero_mask = sig != 0
            non_zero_sigmas = sig[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - 0.1)
            sig[non_zero_mask] = 1.0 - (one_minus_z / scale_factor)
            sigmas = sig.float()  # back to float32 for sampler

        # Build sampler if not provided
        if sampler is None:
            sampler = getattr(guider, 'ic_lora_sampler', None) if guider is not None else None
            if sampler is None:
                sampler = comfy.samplers.sampler_object("euler_ancestral")

        # Build guider if not provided
        if guider is None:
            from ..utils.multimodal_guider import MultimodalGuider
            guider = MultimodalGuider(
                m, positive, negative,
                video_cfg=cfg, audio_cfg=audio_cfg,
                stg_scale=stg_scale,
                stg_perturbation=stg_perturbation,
                stg_blocks=[int(s.strip()) for s in stg_blocks.split(",")],
                rescale=rescale,
                video_cfg_end=cfg_end if cfg_end >= 0 else None,
                stg_scale_end=stg_end if stg_end >= 0 else None,
                audio_stg_scale=audio_stg_scale if audio_stg_scale >= 0 else None,
                video_modality_scale=video_modality_scale,
                audio_modality_scale=audio_modality_scale,
                cfg_star_rescale=cfg_star_rescale,
                skip_sigma_threshold=skip_sigma,
                video_attn_scale=video_attn_scale,
            )
            logger.info("Using MultimodalGuider")

        # ----------------------------------------------------------------
        # 5. SAMPLE
        # ----------------------------------------------------------------

        latent_image = latent_dict["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent_dict["samples"] = latent_image

        noise_mask_for_sampling = latent_dict.get("noise_mask", None)

        noise = comfy.sample.prepare_noise(latent_image, seed)

        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        logger.info("Sampling...")
        samples = guider.sample(
            noise, latent_image, sampler, sigmas,
            denoise_mask=noise_mask_for_sampling,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        samples = samples.to(mm.intermediate_device())

        output_latent = latent_dict.copy()
        output_latent["samples"] = samples

        # ----------------------------------------------------------------
        # 6. POST-SAMPLE: separate AV, then crop appended guide frames
        # ----------------------------------------------------------------

        from comfy_extras.nodes_lt import get_keyframe_idxs

        sampled = output_latent["samples"]

        # Separate video from audio FIRST (NestedTensor doesn't support clone/slicing)
        audio_latent_out = None
        if has_audio and sampled.is_nested:
            parts = sampled.unbind()
            sampled = parts[0]
            audio_latent_out = parts[1] if len(parts) > 1 else None
            # Replace with video-only so get_noise_mask gets a plain tensor
            noise_mask_raw = output_latent.get("noise_mask", None)
            if noise_mask_raw is not None and noise_mask_raw.is_nested:
                output_latent["noise_mask"] = noise_mask_raw.unbind()[0]
            output_latent["samples"] = sampled

        # Now safe to call get_noise_mask (plain tensors only)
        noise_mask_out = get_noise_mask(output_latent)
        latent_image_out = sampled.clone()

        _, num_keyframes = get_keyframe_idxs(positive)
        if num_keyframes > 0:
            latent_image_out = latent_image_out[:, :, :-num_keyframes]
            noise_mask_out = noise_mask_out[:, :, :-num_keyframes]
            positive = node_helpers.conditioning_set_values(positive, {"keyframe_idxs": None})
            negative = node_helpers.conditioning_set_values(negative, {"keyframe_idxs": None})

        output_latent = {"samples": latent_image_out, "noise_mask": noise_mask_out}

        # ----------------------------------------------------------------
        # 6b. IC-LoRA REDIFFUSION (runs regardless of upscale)
        # ----------------------------------------------------------------

        _iclora_info = getattr(guider, 'control_info', None) if guider is not None else None
        _iclora_has_control = _iclora_info is not None and _iclora_info.get("control_image") is not None
        if not do_upscale and _iclora_has_control:
            _rediff_passes = _iclora_info.get('_rediffusion_passes', 1)
            if _rediff_passes > 0 and upscale_steps > 0 and upscale_denoise > 0:
                from comfy_extras.nodes_lt import LTXVAddGuide as RdGuide, get_noise_mask as rd_get_noise_mask, get_keyframe_idxs as rd_get_keyframe_idxs

                for _rediff_i in range(_rediff_passes):
                    if hasattr(guider, 'control_info'):
                        guider._current_rediff_pass = _rediff_i
                        guider._total_rediff_passes = _rediff_passes
                    logger.info(f"=== IC-LoRA re-diffusion pass {_rediff_i + 1}/{_rediff_passes} ===")

                    rd_latent = output_latent["samples"]
                    _, _, rd_lt, rd_lh, rd_lw = rd_latent.shape

                    # Re-inject guide frames
                    rd_scale_factors = vae.downscale_index_formula
                    rd_guides = []
                    rd_guide_video_latent = None
                    if guide_video is not None and not inpaint_mode:
                        num_video_frames = guide_video.shape[0]
                        if guide_index_list and guide_index_list.strip():
                            rd_indices = [int(x.strip()) for x in guide_index_list.split(",") if x.strip()]
                            rd_indices = [i if i >= 0 else num_video_frames + i for i in rd_indices]
                            rd_indices = [i for i in rd_indices if 0 <= i < num_video_frames]
                        else:
                            rd_indices = list(range(0, num_video_frames, guide_every_nth))

                        _, rd_guide_video_latent = RdGuide.encode(
                            vae, rd_lw, rd_lh, guide_video, rd_scale_factors
                        )
                        time_sf = rd_scale_factors[0]
                        for i in rd_indices:
                            lat_idx = 0 if i == 0 else (i - 1) // time_sf + 1
                            lat_idx = min(lat_idx, rd_guide_video_latent.shape[2] - 1)
                            guide_slice = rd_guide_video_latent[:, :, lat_idx:lat_idx+1, :, :]
                            rd_guides.append((guide_slice, i, guide_strength, f"v2v_{i}"))
                    else:
                        if first_image is not None:
                            rd_guides.append((first_image, 0, first_strength, "first"))
                        if middle_image is not None:
                            mid_idx = (num_frames - 1) // 2
                            mid_idx = max(0, (mid_idx // 8) * 8)
                            if mid_idx == 0 and num_frames > 8:
                                mid_idx = 8
                            rd_guides.append((middle_image, mid_idx, middle_strength, "middle"))
                        if last_image is not None:
                            rd_guides.append((last_image, -1, last_strength, "last"))

                    rd_noise_mask = None
                    if rd_guides:
                        rd_latent_dict = {"samples": rd_latent}
                        rd_noise_mask = rd_get_noise_mask(rd_latent_dict)
                        for guide_entry, frame_idx, strength_val, label in rd_guides:
                            _, _, latent_length, latent_height, latent_width = rd_latent_dict["samples"].shape
                            if rd_guide_video_latent is not None:
                                rd_t = guide_entry
                            else:
                                _, rd_t = RdGuide.encode(vae, latent_width, latent_height, guide_entry, rd_scale_factors)
                            guide_length = 1 if rd_guide_video_latent is not None else len(guide_entry)
                            frame_idx_actual, _ = RdGuide.get_latent_index(
                                positive, latent_length, guide_length, frame_idx, rd_scale_factors
                            )
                            logger.info(f"Rediffusion re-inject {label}: frame_idx={frame_idx_actual}, strength={strength_val}")
                            positive, negative, rd_latent_samples, rd_noise_mask = RdGuide.append_keyframe(
                                positive, negative, frame_idx_actual,
                                rd_latent_dict["samples"], rd_noise_mask,
                                rd_t, strength_val, rd_scale_factors,
                            )
                            rd_latent_dict = {"samples": rd_latent_samples, "noise_mask": rd_noise_mask}
                        rd_latent = rd_latent_dict["samples"]

                    # Apply rediffusion mask if present
                    if rediffusion_mask is not None and rd_noise_mask is not None:
                        rd_noise_mask = self._apply_rediffusion_mask(rd_noise_mask, rediffusion_mask, rediffusion_mask_strength, rd_lh, rd_lw)

                    # Build model + sigmas
                    rd_model = m.clone()
                    if upscale_lora and upscale_lora != "none" and upscale_lora_strength != 0:
                        lora_path = folder_paths.get_full_path_or_raise("loras", upscale_lora)
                        lora = None
                        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
                            lora = self.loaded_lora[1]
                        if lora is None:
                            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                            self.loaded_lora = (lora_path, lora)
                        rd_model, _ = comfy.sd.load_lora_for_models(rd_model, None, lora, upscale_lora_strength, 0)
                        logger.info(f"Applied rediffusion LoRA: {upscale_lora} (strength={upscale_lora_strength})")

                    rd_tokens = min(math.prod(rd_latent.shape[2:]), x2 * 2)
                    rd_shift = rd_tokens * mm_shift + b
                    logger.info(f"Rediffusion shift: tokens={rd_tokens}, shift={rd_shift:.3f}")

                    rd_sampling = ModelSamplingAdvanced(rd_model.model.model_config)
                    rd_sampling.set_parameters(shift=rd_shift)
                    rd_model.add_object_patch("model_sampling", rd_sampling)

                    exp_shift = math.exp(rd_shift)
                    t_sched = torch.linspace(1.0, 0.0, upscale_steps + 1, dtype=torch.float64)
                    non_zero = t_sched != 0
                    inv_t_m1 = torch.where(non_zero, 1.0 / t_sched - 1.0, torch.zeros_like(t_sched))
                    omz = torch.where(non_zero, inv_t_m1 / (exp_shift + inv_t_m1), torch.ones_like(t_sched))
                    nz_omz = omz[non_zero]
                    sf = nz_omz[-1] / (1.0 - 0.1)
                    omz[non_zero] = nz_omz / sf
                    rd_sig = torch.where(non_zero, 1.0 - omz, torch.zeros_like(omz)).float()
                    start_step = int((1.0 - upscale_denoise) * upscale_steps)
                    rd_sig = rd_sig[start_step:]
                    logger.info(f"Rediffusion sigmas ({len(rd_sig)}): {rd_sig.tolist()}")

                    # Build IC-LoRA guider for rediffusion
                    rd_guider = self._rebuild_iclora_guider(
                        rd_model, positive, negative, vae,
                        _iclora_info, upscale_cfg,
                        latent_h=rd_lh, latent_w=rd_lw, latent_t=rd_lt,
                    )
                    rd_guider._current_rediff_pass = _rediff_i
                    rd_guider._total_rediff_passes = _rediff_passes

                    # Recombine AV if needed
                    rd_combined = rd_latent
                    if has_audio and audio_latent_out is not None:
                        rd_combined = comfy.nested_tensor.NestedTensor((rd_latent, audio_latent_out))
                        audio_mask_val = 0.0 if audio_is_input else 1.0
                        if rd_noise_mask is not None:
                            audio_mask = torch.full_like(audio_latent_out[:, :1], audio_mask_val)
                            rd_noise_mask = comfy.nested_tensor.NestedTensor((rd_noise_mask, audio_mask))

                    rd_latent_image = comfy.sample.fix_empty_latent_channels(rd_guider.model_patcher, rd_combined)
                    rd_noise = comfy.sample.prepare_noise(rd_latent_image, seed + 1 + _rediff_i)
                    rd_sampler = getattr(guider, 'ic_lora_sampler', None) or comfy.samplers.sampler_object("euler_ancestral")
                    rd_callback = latent_preview.prepare_callback(rd_guider.model_patcher, rd_sig.shape[-1] - 1)

                    self._free_vram()
                    rd_samples = rd_guider.sample(
                        rd_noise, rd_latent_image, rd_sampler, rd_sig,
                        denoise_mask=rd_noise_mask,
                        callback=rd_callback,
                        disable_pbar=disable_pbar,
                        seed=seed + 1 + _rediff_i,
                    )
                    rd_samples = rd_samples.to(mm.intermediate_device())

                    # Separate AV
                    if rd_samples.is_nested:
                        rd_parts = rd_samples.unbind()
                        output_latent = {"samples": rd_parts[0]}
                        audio_latent_out = rd_parts[1] if len(rd_parts) > 1 else audio_latent_out
                    else:
                        output_latent = {"samples": rd_samples}

                    # Crop guide keyframes
                    _, rd_num_keyframes = rd_get_keyframe_idxs(positive)
                    if rd_num_keyframes > 0:
                        output_latent["samples"] = output_latent["samples"][:, :, :-rd_num_keyframes]
                        positive = node_helpers.conditioning_set_values(positive, {"keyframe_idxs": None})
                        negative = node_helpers.conditioning_set_values(negative, {"keyframe_idxs": None})

                    del rd_model, rd_guider
                    self._free_vram()

                logger.info("IC-LoRA rediffusion complete")

        # ----------------------------------------------------------------
        # 7. UPSCALE (optional)
        # ----------------------------------------------------------------

        if do_upscale:
            if upscale_fallback:
                pre_upscale_latent = {"samples": output_latent["samples"].detach().cpu().clone()}
            try:
                device = mm.get_torch_device()
                input_dtype = output_latent["samples"].dtype

                # Temporal upscale first (2x frame count at half resolution = cheap)
                if do_temporal_upscale:
                    logger.info("Upscaling video latents (2x temporal at half res)")
                    self._free_vram()

                    temporal_dtype = next(temporal_upscale_model.parameters()).dtype
                    temporal_upscale_model.to(device)
                    try:
                        t_latents = output_latent["samples"].to(dtype=temporal_dtype, device=device)
                        t_latents = vae.first_stage_model.per_channel_statistics.un_normalize(t_latents)
                        t_upsampled = temporal_upscale_model(t_latents)
                    finally:
                        temporal_upscale_model.cpu()

                    t_upsampled = vae.first_stage_model.per_channel_statistics.normalize(t_upsampled)
                    t_upsampled = t_upsampled.to(dtype=input_dtype, device=mm.intermediate_device())

                    target_latent_T = ((num_frames - 1) // 8) + 1
                    if t_upsampled.shape[2] > target_latent_T:
                        t_upsampled = t_upsampled[:, :, :target_latent_T]
                        logger.info(f"Trimmed temporal output to {target_latent_T} latent frames")

                    output_latent = {"samples": t_upsampled}
                    logger.info(f"After temporal upscale: latent shape {list(t_upsampled.shape)}")

                    # Restore full frame rate (was halved for temporal upscale generation)
                    positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
                    negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

                # IC-LoRA: first pass skips spatial upscale (half-res re-diffusion),
                # then runs again with full spatial upscale + normal settings.
                _iclora_pre = getattr(guider, 'control_info', None)
                _iclora_needs_normal_upscale = _iclora_pre is not None and _iclora_pre.get("control_image") is not None
                is_iclora = _iclora_needs_normal_upscale

                if is_iclora:
                    logger.info("IC-LoRA: skipping spatial upscale (half-res re-diffusion)")
                    upsampled = output_latent["samples"]
                else:
                    upscale_label = "2x temporal + 2x spatial" if do_temporal_upscale else "2x spatial"
                    logger.info(f"Upscaling video latents ({upscale_label})")
                    self._free_vram()

                    model_dtype = next(upscale_model.parameters()).dtype
                    up_latents = output_latent["samples"]

                    memory_required = mm.module_size(upscale_model)
                    memory_required += math.prod(up_latents.shape) * 3000.0
                    mm.free_memory(memory_required, device)

                    try:
                        upscale_model.to(device)
                        if upscale_tiling:
                            # Temporal chunking with overlap context for temporal convolutions
                            latent_t = up_latents.shape[2]
                            us_chunk_t = upscale_tile_t if upscale_tile_t > 0 else latent_t
                            us_overlap = min(2, us_chunk_t // 2) if us_chunk_t < latent_t else 0
                            if us_chunk_t < latent_t:
                                logger.info(f"Upscale model: chunking {latent_t} latent frames, chunk_t={us_chunk_t}, overlap={us_overlap}")
                            chunks_out = []
                            t_pos = 0
                            while t_pos < latent_t:
                                ctx_start = max(0, t_pos - us_overlap)
                                ctx_end = min(t_pos + us_chunk_t + us_overlap, latent_t)
                                if ctx_end < latent_t and (latent_t - ctx_end) < us_overlap + 1:
                                    ctx_end = latent_t
                                chunk = up_latents[:, :, ctx_start:ctx_end]
                                chunk = chunk.to(dtype=model_dtype, device=device)
                                chunk = vae.first_stage_model.per_channel_statistics.un_normalize(chunk)
                                out = upscale_model(chunk).to(device=mm.intermediate_device())
                                trim_start = t_pos - ctx_start
                                keep_end = min(t_pos + us_chunk_t, latent_t) - ctx_start
                                if ctx_end == latent_t and t_pos + us_chunk_t < latent_t:
                                    keep_end = out.shape[2]
                                chunks_out.append(out[:, :, trim_start:keep_end])
                                t_pos = ctx_start + keep_end
                            upsampled = torch.cat(chunks_out, dim=2)
                            del chunks_out
                        else:
                            # Original single-pass
                            up_latents = up_latents.to(dtype=model_dtype, device=device)
                            up_latents = vae.first_stage_model.per_channel_statistics.un_normalize(up_latents)
                            upsampled = upscale_model(up_latents)
                    finally:
                        upscale_model.cpu()

                    upsampled = vae.first_stage_model.per_channel_statistics.normalize(upsampled)
                    upsampled = upsampled.to(dtype=input_dtype, device=mm.intermediate_device())
                output_latent = {"samples": upsampled}

                _rediff_passes = getattr(guider, 'control_info', {}).get('_rediffusion_passes', 1) if is_iclora else 1
                for _rediff_i in range(_rediff_passes):
                    # Tell the guider which pass we're on so it can trim sigmas
                    if hasattr(guider, 'control_info'):
                        guider._current_rediff_pass = _rediff_i
                        guider._total_rediff_passes = _rediff_passes
                    if _rediff_passes > 1:
                        logger.info(f"=== Half-res re-diffusion pass {_rediff_i + 1}/{_rediff_passes} ===")
                    upsampled = output_latent["samples"]

                    # Re-inject guide images at full resolution into upscaled latent
                    # using LTXVAddGuide crop guides (same as inference stage)
                    from comfy_extras.nodes_lt import LTXVAddGuide as UpGuide, get_noise_mask as up_get_noise_mask, get_keyframe_idxs as up_get_keyframe_idxs

                    up_scale_factors = vae.downscale_index_formula

                    # Re-inject all guides at upscaled resolution (crop guides handle
                    # positioning correctly, so multiple guides no longer cause snapping)
                    up_guides = []
                    up_guide_video_latent = None
                    if guide_video is not None:
                        num_video_frames = guide_video.shape[0]
                        if guide_index_list and guide_index_list.strip():
                            up_indices = [int(x.strip()) for x in guide_index_list.split(",") if x.strip()]
                            up_indices = [i if i >= 0 else num_video_frames + i for i in up_indices]
                            up_indices = [i for i in up_indices if 0 <= i < num_video_frames]
                        else:
                            up_indices = list(range(0, num_video_frames, guide_every_nth))

                        # Encode entire guide video in one VAE pass at upscale resolution
                        _, _, _, up_lat_h, up_lat_w = upsampled.shape
                        _, up_guide_video_latent = UpGuide.encode(
                            vae, up_lat_w, up_lat_h, guide_video, up_scale_factors
                        )
                        time_sf = up_scale_factors[0]
                        for i in up_indices:
                            if i == 0:
                                lat_idx = 0
                            else:
                                lat_idx = (i - 1) // time_sf + 1
                            lat_idx = min(lat_idx, up_guide_video_latent.shape[2] - 1)
                            guide_slice = up_guide_video_latent[:, :, lat_idx:lat_idx+1, :, :]
                            up_guides.append((guide_slice, i, guide_strength, f"v2v_{i}"))
                    else:
                        if first_image is not None:
                            up_guides.append((first_image, 0, first_strength, "first"))
                        if middle_image is not None:
                            mid_idx = (num_frames - 1) // 2
                            mid_idx = max(0, (mid_idx // 8) * 8)
                            if mid_idx == 0 and num_frames > 8:
                                mid_idx = 8
                            up_guides.append((middle_image, mid_idx, middle_strength, "middle"))
                        if last_image is not None:
                            up_guides.append((last_image, -1, last_strength, "last"))

                    up_noise_mask = None

                    if up_guides:
                        up_latent_dict = {"samples": upsampled}
                        up_noise_mask = up_get_noise_mask(up_latent_dict)

                        for guide_entry, frame_idx, strength_val, label in up_guides:
                            _, _, latent_length, latent_height, latent_width = up_latent_dict["samples"].shape

                            if up_guide_video_latent is not None:
                                up_t = guide_entry
                            else:
                                _, up_t = UpGuide.encode(vae, latent_width, latent_height, guide_entry, up_scale_factors)

                            guide_length = 1 if up_guide_video_latent is not None else len(guide_entry)
                            frame_idx_actual, latent_idx = UpGuide.get_latent_index(
                                positive, latent_length, guide_length, frame_idx, up_scale_factors
                            )

                            logger.info(f"Upscale re-inject {label}: frame_idx={frame_idx_actual}, latent_idx={latent_idx}, strength={strength_val}")

                            positive, negative, up_latent_samples, up_noise_mask = UpGuide.append_keyframe(
                                positive, negative, frame_idx_actual,
                                up_latent_dict["samples"], up_noise_mask,
                                up_t, strength_val, up_scale_factors,
                            )
                            up_latent_dict = {"samples": up_latent_samples, "noise_mask": up_noise_mask}

                        upsampled = up_latent_dict["samples"]
                        output_latent = {"samples": upsampled}

                    # Re-diffusion at upscaled resolution
                    has_image_guides = len(up_guides) > 0
                    do_rediffusion = has_image_guides and upscale_denoise > 0 and upscale_steps > 0
                    if has_image_guides:
                        rediffusion_label = f"{upscale_steps} steps, cfg={upscale_cfg}"
                    else:
                        rediffusion_label = f"3 steps (manual sigmas), cfg={upscale_cfg}"
                        do_rediffusion = True  # T2V always re-diffuses with manual sigmas
                    if do_rediffusion:
                        logger.info(f"Re-diffusing at upscaled resolution ({rediffusion_label})")
                        self._free_vram()

                        # Recombine video + audio for the AV model
                        # Input audio: mask=0 (preserve for guidance), generated: mask=1 (regenerate at full res)
                        if has_audio and audio_latent_out is not None:
                            up_combined = comfy.nested_tensor.NestedTensor((upsampled, audio_latent_out))
                            audio_mask_val = 0.0 if audio_is_input else 1.0
                            if up_noise_mask is not None:
                                audio_mask = torch.full_like(audio_latent_out[:, :1], audio_mask_val)
                                up_noise_mask = comfy.nested_tensor.NestedTensor((up_noise_mask, audio_mask))
                        else:
                            up_combined = upsampled

                        # Apply upscale LoRA to a fresh clone
                        up_model = m.clone()
                        if upscale_lora and upscale_lora != "none" and upscale_lora_strength != 0:
                            lora_path = folder_paths.get_full_path_or_raise("loras", upscale_lora)
                            lora = None
                            if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
                                lora = self.loaded_lora[1]
                            if lora is None:
                                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                                self.loaded_lora = (lora_path, lora)
                            up_model, _ = comfy.sd.load_lora_for_models(up_model, None, lora, upscale_lora_strength, 0)
                            logger.info(f"Applied upscale LoRA: {upscale_lora} (strength={upscale_lora_strength})")

                        if has_image_guides:
                            # I2V: compute shift from upscaled latent tokens
                            up_tokens = min(math.prod(upsampled.shape[2:]), x2 * 2)
                            up_shift = up_tokens * mm_shift + b
                            logger.info(f"Upscale shift: tokens={up_tokens}, shift={up_shift:.3f}")

                            up_model_sampling = ModelSamplingAdvanced(up_model.model.model_config)
                            up_model_sampling.set_parameters(shift=up_shift)
                            up_model.add_object_patch("model_sampling", up_model_sampling)

                            # Build sigma schedule with numerically stable computation.
                            exp_shift = math.exp(up_shift)
                            t = torch.linspace(1.0, 0.0, upscale_steps + 1, dtype=torch.float64)
                            non_zero = t != 0
                            inv_t_m1 = torch.where(non_zero, 1.0 / t - 1.0, torch.zeros_like(t))
                            omz = torch.where(non_zero, inv_t_m1 / (exp_shift + inv_t_m1), torch.ones_like(t))
                            nz_omz = omz[non_zero]
                            sf = nz_omz[-1] / (1.0 - 0.1)
                            omz[non_zero] = nz_omz / sf
                            up_sig = torch.where(non_zero, 1.0 - omz, torch.zeros_like(omz)).float()

                            # Trim to denoise strength
                            start_step = int((1.0 - upscale_denoise) * upscale_steps)
                            up_sig = up_sig[start_step:]
                        else:
                            # T2V: same computed sigma schedule as I2V, with correct
                            # shift from actual token count.
                            up_tokens = min(math.prod(upsampled.shape[2:]), x2 * 2)
                            up_shift = up_tokens * mm_shift + b
                            logger.info(f"T2V upscale: tokens={up_tokens}, shift={up_shift:.3f}")

                            up_model_sampling = ModelSamplingAdvanced(up_model.model.model_config)
                            up_model_sampling.set_parameters(shift=up_shift)
                            up_model.add_object_patch("model_sampling", up_model_sampling)

                            exp_shift = math.exp(up_shift)
                            t = torch.linspace(1.0, 0.0, upscale_steps + 1, dtype=torch.float64)
                            non_zero = t != 0
                            inv_t_m1 = torch.where(non_zero, 1.0 / t - 1.0, torch.zeros_like(t))
                            omz = torch.where(non_zero, inv_t_m1 / (exp_shift + inv_t_m1), torch.ones_like(t))
                            nz_omz = omz[non_zero]
                            sf = nz_omz[-1] / (1.0 - 0.1)
                            omz[non_zero] = nz_omz / sf
                            up_sig = torch.where(non_zero, 1.0 - omz, torch.zeros_like(omz)).float()

                            # Trim to denoise strength
                            start_step = int((1.0 - upscale_denoise) * upscale_steps)
                            up_sig = up_sig[start_step:]

                        logger.info(f"Rediffusion sigmas ({len(up_sig)}): {up_sig.tolist()}")

                        # Use IC-LoRA guider if original guider had control_info
                        _iclora_info = getattr(guider, 'control_info', None)
                        if _iclora_info and _iclora_info.get("control_image") is not None:
                            _, _, up_lt, up_lh, up_lw = upsampled.shape
                            up_guider = self._rebuild_iclora_guider(
                                up_model, positive, negative, vae,
                                _iclora_info, upscale_cfg,
                                latent_h=up_lh, latent_w=up_lw, latent_t=up_lt,
                            )
                            up_guider._current_rediff_pass = _rediff_i
                            up_guider._total_rediff_passes = _rediff_passes
                        else:
                            up_guider = comfy.samplers.CFGGuider(up_model)
                            up_guider.set_conds(positive, negative)
                            up_guider.set_cfg(upscale_cfg)

                        up_latent_image = comfy.sample.fix_empty_latent_channels(up_guider.model_patcher, up_combined)
                        up_noise = comfy.sample.prepare_noise(up_latent_image, seed + 1)
                        up_sampler = getattr(guider, 'ic_lora_sampler', None) or comfy.samplers.sampler_object("euler_ancestral")

                        # Apply rediffusion mask: modulate noise_mask so masked-out
                        # regions (background) are preserved from the first pass.
                        if rediffusion_mask is not None:
                            _, _, _, up_lat_h, up_lat_w = upsampled.shape
                            # Create noise_mask if none exists from guides
                            if up_noise_mask is None:
                                up_noise_mask = torch.ones(
                                    upsampled.shape[0], 1, upsampled.shape[2], up_lat_h, up_lat_w,
                                    device=upsampled.device, dtype=upsampled.dtype,
                                )
                            # Handle NestedTensor (AV model) — apply mask to video portion only
                            if up_noise_mask.is_nested:
                                vid_nm, aud_nm = up_noise_mask.unbind()
                                vid_nm = self._apply_rediffusion_mask(vid_nm, rediffusion_mask, rediffusion_mask_strength, up_lat_h, up_lat_w)
                                up_noise_mask = comfy.nested_tensor.NestedTensor((vid_nm, aud_nm))
                            else:
                                up_noise_mask = self._apply_rediffusion_mask(up_noise_mask, rediffusion_mask, rediffusion_mask_strength, up_lat_h, up_lat_w)
                            logger.info(f"Applied rediffusion mask (strength={rediffusion_mask_strength})")

                        # Determine temporal tiling for re-diffusion
                        video_t = upsampled.shape[2]
                        if not upscale_tiling:
                            rd_chunk_t = video_t  # single pass (original behavior)
                        elif upscale_tile_t > 0:
                            rd_chunk_t = min(upscale_tile_t, video_t)
                        else:
                            # Auto: estimate from free VRAM
                            try:
                                free_mem = mm.get_free_memory(device)
                                _, _, _, up_h, up_w = upsampled.shape
                                per_frame_bytes = 128 * up_h * up_w * 4 * 40
                                estimated_tiles = max(2, int(free_mem * 0.6 / per_frame_bytes))
                                rd_chunk_t = min(estimated_tiles, video_t)
                            except Exception:
                                rd_chunk_t = min(4, video_t)

                        if rd_chunk_t >= video_t:
                            # No chunking needed — run as single pass
                            up_callback = latent_preview.prepare_callback(up_guider.model_patcher, up_sig.shape[-1] - 1)
                            up_samples = up_guider.sample(
                                up_noise, up_latent_image, up_sampler, up_sig,
                                denoise_mask=up_noise_mask,
                                callback=up_callback,
                                disable_pbar=disable_pbar,
                                seed=seed + 1,
                            )
                            up_samples = up_samples.to(mm.intermediate_device())
                        else:
                            # Temporal tiling: process with overlap context, trim to core
                            rd_overlap = min(2, rd_chunk_t // 2)
                            logger.info(f"Re-diffusion tiling: {video_t} latent frames, chunk_t={rd_chunk_t}, overlap={rd_overlap}")

                            # Decompose AV for video-only tiling
                            is_av = up_latent_image.is_nested
                            if is_av:
                                vid_latent, aud_latent = up_latent_image.unbind()
                                vid_noise, aud_noise = up_noise.unbind()
                                vid_mask = up_noise_mask.unbind()[0] if up_noise_mask is not None and up_noise_mask.is_nested else None
                                aud_mask = up_noise_mask.unbind()[1] if up_noise_mask is not None and up_noise_mask.is_nested else None
                            else:
                                vid_latent = up_latent_image
                                vid_noise = up_noise
                                vid_mask = up_noise_mask

                            chunks_out = []
                            chunk_idx = 0
                            t_pos = 0
                            while t_pos < video_t:
                                # Expand window to include overlap context
                                ctx_start = max(0, t_pos - rd_overlap) if chunk_idx > 0 else 0
                                ctx_end = min(t_pos + rd_chunk_t + rd_overlap, video_t)
                                if ctx_end < video_t and (video_t - ctx_end) < rd_overlap + 1:
                                    ctx_end = video_t

                                # Slice video tensors for this chunk
                                chunk_vid = vid_latent[:, :, ctx_start:ctx_end]
                                chunk_noise = vid_noise[:, :, ctx_start:ctx_end]
                                chunk_mask = vid_mask[:, :, ctx_start:ctx_end] if vid_mask is not None else None

                                # Recombine with full audio for AV model
                                if is_av:
                                    chunk_input = comfy.nested_tensor.NestedTensor((chunk_vid, aud_latent))
                                    chunk_n = comfy.nested_tensor.NestedTensor((chunk_noise, aud_noise))
                                    chunk_dm = comfy.nested_tensor.NestedTensor((chunk_mask, aud_mask)) if chunk_mask is not None else None
                                else:
                                    chunk_input = chunk_vid
                                    chunk_n = chunk_noise
                                    chunk_dm = chunk_mask

                                logger.info(f"  Chunk {chunk_idx}: ctx=[{ctx_start}:{ctx_end}], core=[{t_pos}:{min(t_pos + rd_chunk_t, video_t)}]")
                                up_callback = latent_preview.prepare_callback(up_guider.model_patcher, up_sig.shape[-1] - 1)
                                chunk_out = up_guider.sample(
                                    chunk_n, chunk_input, up_sampler, up_sig,
                                    denoise_mask=chunk_dm,
                                    callback=up_callback,
                                    disable_pbar=disable_pbar,
                                    seed=seed + 1,
                                )
                                chunk_out = chunk_out.to(mm.intermediate_device())

                                # Extract video portion
                                if chunk_out.is_nested:
                                    chunk_vid_out = chunk_out.unbind()[0]
                                    audio_latent_out = chunk_out.unbind()[1] if len(chunk_out.unbind()) > 1 else audio_latent_out
                                else:
                                    chunk_vid_out = chunk_out

                                logger.info(f"  Chunk {chunk_idx} output: shape={chunk_vid_out.shape}, min={chunk_vid_out.min():.4f}, max={chunk_vid_out.max():.4f}, mean={chunk_vid_out.mean():.4f}")

                                # Trim overlap — keep only the core frames
                                trim_start = t_pos - ctx_start
                                keep_end = min(t_pos + rd_chunk_t, video_t) - ctx_start
                                if ctx_end == video_t and t_pos + rd_chunk_t < video_t:
                                    keep_end = chunk_vid_out.shape[2]
                                chunks_out.append(chunk_vid_out[:, :, trim_start:keep_end])

                                t_pos = ctx_start + keep_end
                                chunk_idx += 1
                                self._free_vram()

                            up_samples = torch.cat(chunks_out, dim=2)
                            logger.info(f"Re-diffusion result: shape={up_samples.shape}, min={up_samples.min():.4f}, max={up_samples.max():.4f}, mean={up_samples.mean():.4f}")
                            del chunks_out

                            # Re-wrap as nested if AV
                            if is_av:
                                up_samples = comfy.nested_tensor.NestedTensor((up_samples, audio_latent_out))

                        # Separate AV again after re-diffusion
                        if up_samples.is_nested:
                            up_parts = up_samples.unbind()
                            output_latent = {"samples": up_parts[0]}
                            audio_latent_out = up_parts[1] if len(up_parts) > 1 else audio_latent_out
                        else:
                            output_latent = {"samples": up_samples}

                        # Crop guide keyframes out of the re-diffusion output
                        _, up_num_keyframes = up_get_keyframe_idxs(positive)
                        if up_num_keyframes > 0:
                            output_latent["samples"] = output_latent["samples"][:, :, :-up_num_keyframes]
                            positive = node_helpers.conditioning_set_values(positive, {"keyframe_idxs": None})
                            negative = node_helpers.conditioning_set_values(negative, {"keyframe_idxs": None})

                        # Free rediffusion model to reclaim VRAM for VAE decode
                        del up_model, up_guider
                        self._free_vram()

                # IC-LoRA pass 3: normal spatial upscale + re-diffusion
                # (passes 1+2 produced a refined half-res latent, now upscale it properly)
                if _iclora_needs_normal_upscale:
                    is_iclora = False  # Disable IC-LoRA skip for this pass
                    _iclora_needs_normal_upscale = False

                    # --- Spatial upscale ---
                    logger.info("IC-LoRA pass 3: spatial 2x upscale + normal re-diffusion")
                    # Aggressive VRAM cleanup — unload diffusion model from pass 2
                    mm.unload_all_models()
                    gc.collect()
                    torch.cuda.empty_cache()
                    mm.soft_empty_cache()

                    model_dtype = next(upscale_model.parameters()).dtype
                    up_latents = output_latent["samples"]

                    memory_required = mm.module_size(upscale_model)
                    memory_required += math.prod(up_latents.shape) * 3000.0
                    mm.free_memory(memory_required, device)

                    try:
                        upscale_model.to(device)
                        if upscale_tiling:
                            latent_t = up_latents.shape[2]
                            us_chunk_t = upscale_tile_t if upscale_tile_t > 0 else latent_t
                            us_overlap = min(2, us_chunk_t // 2) if us_chunk_t < latent_t else 0
                            if us_chunk_t < latent_t:
                                logger.info(f"Upscale model: chunking {latent_t} latent frames, chunk_t={us_chunk_t}, overlap={us_overlap}")
                            chunks_out = []
                            t_pos = 0
                            while t_pos < latent_t:
                                ctx_start = max(0, t_pos - us_overlap)
                                ctx_end = min(t_pos + us_chunk_t + us_overlap, latent_t)
                                if ctx_end < latent_t and (latent_t - ctx_end) < us_overlap + 1:
                                    ctx_end = latent_t
                                chunk = up_latents[:, :, ctx_start:ctx_end]
                                chunk = chunk.to(dtype=model_dtype, device=device)
                                chunk = vae.first_stage_model.per_channel_statistics.un_normalize(chunk)
                                out = upscale_model(chunk).to(device=mm.intermediate_device())
                                trim_start = t_pos - ctx_start
                                keep_end = min(t_pos + us_chunk_t, latent_t) - ctx_start
                                if ctx_end == latent_t and t_pos + us_chunk_t < latent_t:
                                    keep_end = out.shape[2]
                                chunks_out.append(out[:, :, trim_start:keep_end])
                                t_pos = ctx_start + keep_end
                            upsampled = torch.cat(chunks_out, dim=2)
                            del chunks_out
                        else:
                            up_latents = up_latents.to(dtype=model_dtype, device=device)
                            up_latents = vae.first_stage_model.per_channel_statistics.un_normalize(up_latents)
                            upsampled = upscale_model(up_latents)
                    finally:
                        upscale_model.cpu()

                    upsampled = vae.first_stage_model.per_channel_statistics.normalize(upsampled)
                    upsampled = upsampled.to(dtype=input_dtype, device=mm.intermediate_device())
                    output_latent = {"samples": upsampled}

                    # --- Re-inject guides at full resolution ---
                    up_scale_factors = vae.downscale_index_formula
                    up_guides = []
                    if first_image is not None:
                        up_guides.append((first_image, 0, first_strength, "first"))
                    if middle_image is not None:
                        mid_idx = (num_frames - 1) // 2
                        mid_idx = max(0, (mid_idx // 8) * 8)
                        if mid_idx == 0 and num_frames > 8:
                            mid_idx = 8
                        up_guides.append((middle_image, mid_idx, middle_strength, "middle"))
                    if last_image is not None:
                        up_guides.append((last_image, -1, last_strength, "last"))

                    up_noise_mask = None
                    if up_guides:
                        up_latent_dict = {"samples": upsampled}
                        up_noise_mask = up_get_noise_mask(up_latent_dict)
                        for img, frame_idx, strength_val, label in up_guides:
                            _, _, latent_length, latent_height, latent_width = up_latent_dict["samples"].shape
                            _, up_t = UpGuide.encode(vae, latent_width, latent_height, img, up_scale_factors)
                            frame_idx_actual, latent_idx = UpGuide.get_latent_index(
                                positive, latent_length, len(img), frame_idx, up_scale_factors
                            )
                            logger.info(f"Pass 3 re-inject {label}: frame_idx={frame_idx_actual}, latent_idx={latent_idx}, strength={strength_val}")
                            positive, negative, up_latent_samples, up_noise_mask = UpGuide.append_keyframe(
                                positive, negative, frame_idx_actual,
                                up_latent_dict["samples"], up_noise_mask,
                                up_t, strength_val, up_scale_factors,
                            )
                            up_latent_dict = {"samples": up_latent_samples, "noise_mask": up_noise_mask}
                        upsampled = up_latent_dict["samples"]
                        output_latent = {"samples": upsampled}

                    # --- Normal re-diffusion ---
                    has_image_guides = len(up_guides) > 0
                    do_rediffusion = has_image_guides and upscale_denoise > 0 and upscale_steps > 0
                    if do_rediffusion:
                        logger.info(f"Pass 3 re-diffusion ({upscale_steps} steps, cfg={upscale_cfg})")
                        # Free spatial upscale model VRAM before loading diffusion model
                        mm.unload_all_models()
                        gc.collect()
                        torch.cuda.empty_cache()
                        mm.soft_empty_cache()

                        up_combined = upsampled
                        up_model = m.clone()
                        if upscale_lora and upscale_lora != "none" and upscale_lora_strength != 0:
                            lora_path = folder_paths.get_full_path_or_raise("loras", upscale_lora)
                            lora = None
                            if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
                                lora = self.loaded_lora[1]
                            if lora is None:
                                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                                self.loaded_lora = (lora_path, lora)
                            up_model, _ = comfy.sd.load_lora_for_models(up_model, None, lora, upscale_lora_strength, 0)
                            logger.info(f"Applied upscale LoRA: {upscale_lora} (strength={upscale_lora_strength})")

                        up_tokens = min(math.prod(upsampled.shape[2:]), x2 * 2)
                        up_shift = up_tokens * mm_shift + b
                        logger.info(f"Pass 3 shift: tokens={up_tokens}, shift={up_shift:.3f}")

                        up_model_sampling = ModelSamplingAdvanced(up_model.model.model_config)
                        up_model_sampling.set_parameters(shift=up_shift)
                        up_model.add_object_patch("model_sampling", up_model_sampling)

                        exp_shift = math.exp(up_shift)
                        t = torch.linspace(1.0, 0.0, upscale_steps + 1, dtype=torch.float64)
                        non_zero = t != 0
                        inv_t_m1 = torch.where(non_zero, 1.0 / t - 1.0, torch.zeros_like(t))
                        omz = torch.where(non_zero, inv_t_m1 / (exp_shift + inv_t_m1), torch.ones_like(t))
                        nz_omz = omz[non_zero]
                        sf = nz_omz[-1] / (1.0 - 0.1)
                        omz[non_zero] = nz_omz / sf
                        up_sig = torch.where(non_zero, 1.0 - omz, torch.zeros_like(omz)).float()
                        start_step = int((1.0 - upscale_denoise) * upscale_steps)
                        up_sig = up_sig[start_step:]

                        logger.info(f"Pass 3 sigmas ({len(up_sig)}): {up_sig.tolist()}")

                        # Standard CFGGuider — no IC-LoRA, no distilled
                        up_guider = comfy.samplers.CFGGuider(up_model)
                        up_guider.set_conds(positive, negative)
                        up_guider.set_cfg(upscale_cfg)

                        up_latent_image = comfy.sample.fix_empty_latent_channels(up_guider.model_patcher, up_combined)
                        up_noise = comfy.sample.prepare_noise(up_latent_image, seed + 1)
                        up_sampler = sampler if sampler is not None else comfy.samplers.sampler_object("euler_ancestral")

                        up_callback = latent_preview.prepare_callback(up_guider.model_patcher, up_sig.shape[-1] - 1)
                        up_samples = up_guider.sample(
                            up_noise, up_latent_image, up_sampler, up_sig,
                            denoise_mask=up_noise_mask,
                            callback=up_callback,
                            disable_pbar=disable_pbar,
                            seed=seed + 1,
                        )
                        up_samples = up_samples.to(mm.intermediate_device())

                        if up_samples.is_nested:
                            up_parts = up_samples.unbind()
                            output_latent = {"samples": up_parts[0]}
                        else:
                            output_latent = {"samples": up_samples}

                        _, up_num_keyframes = up_get_keyframe_idxs(positive)
                        if up_num_keyframes > 0:
                            output_latent["samples"] = output_latent["samples"][:, :, :-up_num_keyframes]
                            positive = node_helpers.conditioning_set_values(positive, {"keyframe_idxs": None})
                            negative = node_helpers.conditioning_set_values(negative, {"keyframe_idxs": None})

                        del up_model, up_guider
                        self._free_vram()

                if False:  # T2V pass 2 removed — single rediffusion with MultimodalGuider is sufficient
                    logger.info("T2V pass 2: decoding first frame for I2V guidance")
                    self._free_vram()

                    video_latent = output_latent["samples"]

                    # Decode just the first latent frame
                    first_pixels = vae.decode(video_latent[:, :, :1])
                    if len(first_pixels.shape) == 5:
                        first_pixels = first_pixels.reshape(
                            -1, first_pixels.shape[-3], first_pixels.shape[-2], first_pixels.shape[-1]
                        )

                    # Sharpen + grain to enhance detail before using as anchor
                    # Encode first frame and inject at position 0
                    _, height_sf, width_sf = vae.downscale_index_formula
                    _, _, _, p2_lh, p2_lw = video_latent.shape
                    p2_tw, p2_th = p2_lw * width_sf, p2_lh * height_sf
                    if first_pixels.shape[1] != p2_th or first_pixels.shape[2] != p2_tw:
                        first_pixels = comfy.utils.common_upscale(
                            first_pixels.movedim(-1, 1), p2_tw, p2_th, "bilinear", "center"
                        ).movedim(1, -1)
                    first_encoded = vae.encode(first_pixels[:, :, :, :3])

                    video_latent = video_latent.clone()
                    video_latent[:, :, :first_encoded.shape[2]] = first_encoded

                    # Denoise mask: preserve first frame (0.0), denoise rest (1.0)
                    p2_denoise_mask = torch.ones(
                        (video_latent.shape[0], 1, video_latent.shape[2], 1, 1),
                        dtype=video_latent.dtype, device=video_latent.device,
                    )
                    p2_denoise_mask[:, :, :first_encoded.shape[2]] = 0.0

                    # Input audio: preserve (mask=0), generated: regenerate at full res (mask=1)
                    if has_audio and audio_latent_out is not None:
                        p2_combined = comfy.nested_tensor.NestedTensor((video_latent, audio_latent_out))
                        p2_audio_mask_val = 0.0 if audio_is_input else 1.0
                        audio_mask = torch.full_like(audio_latent_out[:, :1], p2_audio_mask_val)
                        p2_denoise_mask = comfy.nested_tensor.NestedTensor((p2_denoise_mask, audio_mask))
                    else:
                        p2_combined = video_latent

                    # I2V-style model + sigmas (computed from actual upscaled tokens)
                    p2_model = m.clone()
                    if upscale_lora and upscale_lora != "none" and upscale_lora_strength != 0:
                        lora_path = folder_paths.get_full_path_or_raise("loras", upscale_lora)
                        lora = self.loaded_lora[1] if self.loaded_lora and self.loaded_lora[0] == lora_path else None
                        if lora is None:
                            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                            self.loaded_lora = (lora_path, lora)
                        p2_model, _ = comfy.sd.load_lora_for_models(p2_model, None, lora, upscale_lora_strength, 0)

                    p2_tokens = min(math.prod(video_latent.shape[2:]), x2 * 2)
                    p2_shift = p2_tokens * mm_shift + b
                    p2_ms = ModelSamplingAdvanced(p2_model.model.model_config)
                    p2_ms.set_parameters(shift=p2_shift)
                    p2_model.add_object_patch("model_sampling", p2_ms)

                    exp_s = math.exp(p2_shift)
                    t = torch.linspace(1.0, 0.0, upscale_steps + 1, dtype=torch.float64)
                    nz = t != 0
                    inv = torch.where(nz, 1.0 / t - 1.0, torch.zeros_like(t))
                    omz = torch.where(nz, inv / (exp_s + inv), torch.ones_like(t))
                    nz_omz = omz[nz]
                    omz[nz] = nz_omz / (nz_omz[-1] / (1.0 - 0.1))
                    p2_sig = torch.where(nz, 1.0 - omz, torch.zeros_like(omz)).float()
                    p2_sig = p2_sig[int((1.0 - upscale_denoise) * upscale_steps):]

                    logger.info(f"T2V pass 2: {upscale_steps} steps, denoise={upscale_denoise}, shift={p2_shift:.3f}")

                    # Use IC-LoRA guider if original guider had control_info
                    _iclora_info = getattr(guider, 'control_info', None)
                    if _iclora_info and _iclora_info.get("control_image") is not None:
                        _, _, p2_lt, p2_lh2, p2_lw2 = video_latent.shape
                        p2_guider = self._rebuild_iclora_guider(
                            p2_model, positive, negative, vae,
                            _iclora_info, upscale_cfg,
                            latent_h=p2_lh2, latent_w=p2_lw2, latent_t=p2_lt,
                        )
                    else:
                        from ..utils.multimodal_guider import MultimodalGuider
                        p2_guider = MultimodalGuider(
                            p2_model, positive, negative,
                            video_cfg=upscale_cfg, audio_cfg=audio_cfg,
                            stg_scale=stg_scale,
                            stg_perturbation=stg_perturbation,
                            stg_blocks=[int(s.strip()) for s in stg_blocks.split(",")],
                            rescale=rescale,
                            video_cfg_end=cfg_end if cfg_end >= 0 else None,
                            stg_scale_end=stg_end if stg_end >= 0 else None,
                            audio_stg_scale=audio_stg_scale if audio_stg_scale >= 0 else None,
                            video_modality_scale=video_modality_scale,
                            audio_modality_scale=audio_modality_scale,
                            cfg_star_rescale=cfg_star_rescale,
                            skip_sigma_threshold=skip_sigma,
                            video_attn_scale=video_attn_scale,
                        )

                    p2_latent = comfy.sample.fix_empty_latent_channels(p2_guider.model_patcher, p2_combined)
                    p2_noise = comfy.sample.prepare_noise(p2_latent, seed + 2)
                    p2_sampler = getattr(guider, 'ic_lora_sampler', None) or comfy.samplers.sampler_object("euler_ancestral")

                    p2_cb = latent_preview.prepare_callback(p2_guider.model_patcher, p2_sig.shape[-1] - 1)
                    p2_out = p2_guider.sample(
                        p2_noise, p2_latent, p2_sampler, p2_sig,
                        denoise_mask=p2_denoise_mask,
                        callback=p2_cb,
                        disable_pbar=disable_pbar,
                        seed=seed + 2,
                    )
                    p2_out = p2_out.to(mm.intermediate_device())

                    if p2_out.is_nested:
                        p2_parts = p2_out.unbind()
                        output_latent = {"samples": p2_parts[0]}
                        audio_latent_out = p2_parts[1] if len(p2_parts) > 1 else audio_latent_out
                    else:
                        output_latent = {"samples": p2_out}
                    logger.info("T2V pass 2 complete")

                if upscale_fallback:
                    del pre_upscale_latent  # free system RAM backup

            except Exception as e:
                if not upscale_fallback:
                    raise
                logger.info(f"WARNING: Upscale failed ({type(e).__name__}: {e})")
                logger.info("Falling back to half-resolution decode")
                self._free_vram()
                output_latent = pre_upscale_latent
                decode = True

        # ----------------------------------------------------------------
        # 7b. MASKED REDIFFUSION (no upscale — full-res first pass + selective rediffusion)
        # ----------------------------------------------------------------

        if not do_upscale and rediffusion_mask is not None and upscale_denoise > 0 and upscale_steps > 0:
            logger.info(f"Masked rediffusion: {upscale_steps} steps, denoise={upscale_denoise}, cfg={upscale_cfg}")
            self._free_vram()

            rd_latent = output_latent["samples"]

            # Build noise_mask: ones everywhere (all regions eligible for denoise),
            # then modulate by the rediffusion mask
            rd_noise_mask = torch.ones(
                rd_latent.shape[0], 1, rd_latent.shape[2], rd_latent.shape[3], rd_latent.shape[4],
                device=rd_latent.device, dtype=rd_latent.dtype,
            )
            rd_noise_mask = self._apply_rediffusion_mask(rd_noise_mask, rediffusion_mask, rediffusion_mask_strength)

            # Compute shift + sigmas from current latent tokens
            rd_tokens = min(math.prod(rd_latent.shape[2:]), x2 * 2)
            rd_shift = rd_tokens * mm_shift + b
            logger.info(f"Masked rediffusion shift: tokens={rd_tokens}, shift={rd_shift:.3f}")

            rd_model = m.clone()

            # Apply upscale LoRA if specified
            if upscale_lora and upscale_lora != "none" and upscale_lora_strength != 0:
                lora_path = folder_paths.get_full_path_or_raise("loras", upscale_lora)
                lora = None
                if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                if lora is None:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    self.loaded_lora = (lora_path, lora)
                rd_model, _ = comfy.sd.load_lora_for_models(rd_model, None, lora, upscale_lora_strength, 0)
                logger.info(f"Applied rediffusion LoRA: {upscale_lora} (strength={upscale_lora_strength})")

            rd_model_sampling = ModelSamplingAdvanced(rd_model.model.model_config)
            rd_model_sampling.set_parameters(shift=rd_shift)
            rd_model.add_object_patch("model_sampling", rd_model_sampling)

            # Build sigma schedule
            exp_shift = math.exp(rd_shift)
            t = torch.linspace(1.0, 0.0, upscale_steps + 1, dtype=torch.float64)
            non_zero = t != 0
            inv_t_m1 = torch.where(non_zero, 1.0 / t - 1.0, torch.zeros_like(t))
            omz = torch.where(non_zero, inv_t_m1 / (exp_shift + inv_t_m1), torch.ones_like(t))
            nz_omz = omz[non_zero]
            sf = nz_omz[-1] / (1.0 - 0.1)
            omz[non_zero] = nz_omz / sf
            rd_sig = torch.where(non_zero, 1.0 - omz, torch.zeros_like(omz)).float()

            # Trim to denoise strength
            start_step = int((1.0 - upscale_denoise) * upscale_steps)
            rd_sig = rd_sig[start_step:]
            logger.info(f"Masked rediffusion sigmas ({len(rd_sig)}): {rd_sig.tolist()}")

            # Re-inject guides for the rediffusion pass
            from comfy_extras.nodes_lt import LTXVAddGuide as RdGuide, get_noise_mask as rd_get_noise_mask

            rd_guides = []
            rd_guide_video_latent = None
            if guide_video is not None:
                num_video_frames = guide_video.shape[0]
                if guide_index_list and guide_index_list.strip():
                    rd_indices = [int(x.strip()) for x in guide_index_list.split(",") if x.strip()]
                    rd_indices = [i if i >= 0 else num_video_frames + i for i in rd_indices]
                    rd_indices = [i for i in rd_indices if 0 <= i < num_video_frames]
                else:
                    rd_indices = list(range(0, num_video_frames, guide_every_nth))

                rd_scale_factors = vae.downscale_index_formula
                _, _, _, rd_lat_h, rd_lat_w = rd_latent.shape
                _, rd_guide_video_latent = RdGuide.encode(
                    vae, rd_lat_w, rd_lat_h, guide_video, rd_scale_factors
                )
                time_sf = rd_scale_factors[0]
                for i in rd_indices:
                    if i == 0:
                        lat_idx = 0
                    else:
                        lat_idx = (i - 1) // time_sf + 1
                    lat_idx = min(lat_idx, rd_guide_video_latent.shape[2] - 1)
                    guide_slice = rd_guide_video_latent[:, :, lat_idx:lat_idx+1, :, :]
                    rd_guides.append((guide_slice, i, guide_strength, f"v2v_{i}"))
            else:
                if first_image is not None:
                    rd_guides.append((first_image, 0, first_strength, "first"))
                if middle_image is not None:
                    mid_idx = (num_frames - 1) // 2
                    mid_idx = max(0, (mid_idx // 8) * 8)
                    if mid_idx == 0 and num_frames > 8:
                        mid_idx = 8
                    rd_guides.append((middle_image, mid_idx, middle_strength, "middle"))
                if last_image is not None:
                    rd_guides.append((last_image, -1, last_strength, "last"))

            if rd_guides:
                rd_scale_factors = vae.downscale_index_formula
                rd_latent_dict = {"samples": rd_latent}
                rd_guide_noise_mask = rd_get_noise_mask(rd_latent_dict)

                for guide_entry, frame_idx, strength_val, label in rd_guides:
                    _, _, latent_length, latent_height, latent_width = rd_latent_dict["samples"].shape

                    if rd_guide_video_latent is not None:
                        rd_t = guide_entry
                    else:
                        _, rd_t = RdGuide.encode(vae, latent_width, latent_height, guide_entry, rd_scale_factors)

                    guide_length = 1 if rd_guide_video_latent is not None else len(guide_entry)
                    frame_idx_actual, latent_idx = RdGuide.get_latent_index(
                        positive, latent_length, guide_length, frame_idx, rd_scale_factors
                    )

                    logger.info(f"Masked rediff guide {label}: frame_idx={frame_idx_actual}, latent_idx={latent_idx}")

                    positive, negative, rd_latent_samples, rd_guide_noise_mask = RdGuide.append_keyframe(
                        positive, negative, frame_idx_actual,
                        rd_latent_dict["samples"], rd_guide_noise_mask,
                        rd_t, strength_val, rd_scale_factors,
                    )
                    rd_latent_dict = {"samples": rd_latent_samples, "noise_mask": rd_guide_noise_mask}

                rd_latent = rd_latent_dict["samples"]
                # Apply rediffusion mask to the guide noise mask
                _, _, _, rd_lat_h, rd_lat_w = rd_latent.shape
                rd_noise_mask = self._apply_rediffusion_mask(rd_guide_noise_mask, rediffusion_mask, rediffusion_mask_strength, rd_lat_h, rd_lat_w)

            # Build guider
            rd_guider = comfy.samplers.CFGGuider(rd_model)
            rd_guider.set_conds(positive, negative)
            rd_guider.set_cfg(upscale_cfg)

            rd_latent_image = comfy.sample.fix_empty_latent_channels(rd_guider.model_patcher, rd_latent)
            rd_noise = comfy.sample.prepare_noise(rd_latent_image, seed + 1)
            rd_sampler = comfy.samplers.sampler_object("euler_ancestral")

            rd_callback = latent_preview.prepare_callback(rd_guider.model_patcher, rd_sig.shape[-1] - 1)
            rd_samples = rd_guider.sample(
                rd_noise, rd_latent_image, rd_sampler, rd_sig,
                denoise_mask=rd_noise_mask,
                callback=rd_callback,
                disable_pbar=disable_pbar,
                seed=seed + 1,
            )
            rd_samples = rd_samples.to(mm.intermediate_device())

            # Crop appended guide frames
            _, rd_num_keyframes = get_keyframe_idxs(positive)
            if rd_num_keyframes > 0:
                rd_samples = rd_samples[:, :, :-rd_num_keyframes]
                positive = node_helpers.conditioning_set_values(positive, {"keyframe_idxs": None})
                negative = node_helpers.conditioning_set_values(negative, {"keyframe_idxs": None})

            output_latent = {"samples": rd_samples}
            logger.info("Masked rediffusion complete")

        # Free model clones to reclaim CPU RAM from offloaded weights
        del m
        self._free_vram()

        # ----------------------------------------------------------------
        # 8. DECODE (optional)
        # ----------------------------------------------------------------

        if decode:
            logger.info("Decoding latents to images (tiled)")
            compression = vae.spacial_compression_decode()
            tile_size = 512 // compression
            # Scale overlap with resolution: ~25% of tile_size, clamped to [4, 8] latent pixels
            lat_max_dim = max(output_latent["samples"].shape[3], output_latent["samples"].shape[4])
            overlap = min(8, max(4, lat_max_dim // 8))
            temporal_compression = vae.temporal_compression_decode()
            if tile_t > 0:
                # User override
                decode_tile_t = tile_t
                overlap_t = max(2, decode_tile_t // 2)
            elif temporal_compression is not None:
                decode_tile_t = max(2, 64 // temporal_compression)
                overlap_t = max(2, decode_tile_t // 2)
            else:
                decode_tile_t = None
                overlap_t = None

            # Add tiny noise to break up tile boundary artifacts, scaled with resolution
            samples = output_latent["samples"]
            noise_scale = 1e-5 + (lat_max_dim / 60.0) * 3e-4  # ~1e-5 at 540p, ~3e-4 at 1080p
            samples = samples + torch.randn_like(samples) * noise_scale

            images = vae.decode_tiled(
                samples,
                tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                tile_t=decode_tile_t, overlap_t=overlap_t,
            )
            # Video VAE returns 5D [B, T, H, W, C] — flatten to standard 4D IMAGE [B*T, H, W, C]
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            # Return a 1-frame black placeholder so downstream nodes don't crash on None
            images = torch.zeros(1, height // 8, width // 8, 3)

        # ----------------------------------------------------------------
        # 9. AUDIO DECODE (if audio was used)
        # ----------------------------------------------------------------

        audio_output = None
        if has_audio and audio_latent_out is not None and audio_vae is not None:
            if audio_is_input:
                # Pass through original audio — skip VAE decode roundtrip to
                # ensure 1:1 fidelity with the source audio
                logger.info("Audio passthrough (input audio preserved)")
                audio_output = audio
            else:
                logger.info("Decoding generated audio latents")
                decoded_audio = audio_vae.decode(audio_latent_out).to(audio_latent_out.device)
                audio_output = {
                    "waveform": decoded_audio,
                    "sample_rate": int(audio_vae.output_sample_rate),
                }

        # Build audio latent output
        audio_latent_dict = None
        if audio_latent_out is not None:
            audio_latent_dict = {"samples": audio_latent_out}

        logger.info("Done")
        return (output_latent, audio_latent_dict, images, audio_output)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _rebuild_iclora_guider(up_model, positive, negative, vae,
                               control_info, upscale_cfg, latent_h, latent_w,
                               latent_t):
        """Rebuild an ICLoRAGuider at upscaled resolution.

        Uses ICLoRAGuider with deferred encoding — the control image is
        stored and re-encoded at sample() time at the actual upscaled
        latent dimensions, following the official LTXVAddGuide approach.
        """
        from ..utils.multimodal_guider import ICLoRAGuider
        from comfy_extras.nodes_lt import preprocess as ltxv_preprocess

        ci = control_info
        lora_path = folder_paths.get_full_path_or_raise("loras", ci["lora_name"])
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        if ci["lora_strength"] != 0:
            up_model, _ = comfy.sd.load_lora_for_models(
                up_model, None, lora, ci["lora_strength"], 0
            )
            logger.info(f"IC-LoRA re-applied: {ci['lora_name']} (strength={ci['lora_strength']})")
        del lora

        # Re-apply optional second stacked LoRA (must run on every pass to match pass 1)
        lora_name_2 = ci.get("lora_name_2", "none")
        lora_strength_2 = ci.get("lora_strength_2", 0.0)
        if lora_name_2 and lora_name_2 != "none" and lora_strength_2 != 0:
            lora_path_2 = folder_paths.get_full_path_or_raise("loras", lora_name_2)
            lora_2 = comfy.utils.load_torch_file(lora_path_2, safe_load=True)
            up_model, _ = comfy.sd.load_lora_for_models(
                up_model, None, lora_2, lora_strength_2, 0
            )
            logger.info(f"Second LoRA re-applied: {lora_name_2} (strength={lora_strength_2})")
            del lora_2

        # Attention override
        attn_func = None
        attn_mode = ci.get("attention_mode", "auto")
        if attn_mode != "default":
            if attn_mode == "sage":
                from comfy.ldm.modules.attention import attention_sage
                attn_func = attention_sage
            elif attn_mode == "auto":
                from comfy.ldm.modules.attention import SAGE_ATTENTION_IS_AVAILABLE, attention_sage
                if SAGE_ATTENTION_IS_AVAILABLE:
                    attn_func = attention_sage
        if attn_func is not None:
            up_model.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = (
                lambda func, *args, **kwargs: attn_func(*args, **kwargs)
            )

        # CRF preprocess control image
        control_image = ci["control_image"]
        crf = ci.get("crf", 35)
        if crf > 0:
            processed_frames = []
            for i in range(control_image.shape[0]):
                processed_frames.append(ltxv_preprocess(control_image[i], crf))
            control_image = torch.stack(processed_frames)

        # Stamp frame rate
        fr = ci.get("frame_rate", 25.0)
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": fr})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": fr})

        stg_blocks_str = ci.get("stg_blocks", "29")
        up_guider = ICLoRAGuider(
            up_model, positive, negative,
            control_pixels=control_image,
            vae=vae,
            downscale_factor=ci["downscale_factor"],
            guide_strength=ci["control_strength"],
            guide_frame_idx=ci.get("guide_frame_idx", 0),
            max_shift=ci.get("max_shift", 2.2),
            base_shift=ci.get("base_shift", 0.95),
            video_cfg=upscale_cfg,
            audio_cfg=ci.get("audio_cfg", 7.0),
            stg_scale=ci.get("stg_scale", 0.0),
            stg_perturbation=ci.get("stg_perturbation", 1.0),
            stg_blocks=[int(s.strip()) for s in stg_blocks_str.split(",")],
            stg_indexes=[int(s.strip()) for s in ci.get("stg_indexes", "0").split(",")],
            rescale=ci.get("rescale", 0.7),
            video_cfg_end=ci["cfg_end"] if ci.get("cfg_end", -1) >= 0 else None,
            stg_scale_end=ci["stg_end"] if ci.get("stg_end", -1) >= 0 else None,
            cfg_star_rescale=ci.get("cfg_star_rescale", True),
            skip_sigma_threshold=ci.get("skip_sigma", 0.0),
            audio_stg_scale=ci["audio_stg_scale"] if ci.get("audio_stg_scale", -1) >= 0 else None,
            video_modality_scale=ci.get("video_modality_scale", 0.0),
            audio_modality_scale=ci.get("audio_modality_scale", 3.0),
            video_attn_scale=ci.get("video_attn_scale", 1.03),
        )
        up_guider.control_info = ci
        up_guider.ic_lora_sampler = ci.get("_ic_lora_sampler", None)

        # Propagate distilled LoRA for distilled sigma passes
        dl_name = ci.get("_distilled_lora", "none")
        dl_strength = ci.get("_distilled_lora_strength", 1.0)
        if dl_name and dl_name != "none" and dl_strength != 0:
            dl_path = folder_paths.get_full_path_or_raise("loras", dl_name)
            dl_data = comfy.utils.load_torch_file(dl_path, safe_load=True)
            up_guider._distilled_lora = (dl_data, dl_strength, dl_name)

        logger.info("IC-LoRA guider rebuilt for upscale (deferred encoding)")
        return up_guider

    @staticmethod
    def _apply_rediffusion_mask(noise_mask, mask, strength=1.0, latent_h=None, latent_w=None):
        """Multiply noise_mask by a spatial mask resized to latent dims.

        noise_mask: [B, 1, T, H, W] — existing rediffusion mask (may have 1x1 spatial)
        mask: [H, W] or [B, H, W] — pixel-space subject mask (1=rediffuse, 0=preserve)
        strength: 0.0 = ignore mask entirely, 1.0 = full mask effect
        latent_h/latent_w: actual latent spatial dims (required when noise_mask is 1x1)
        """
        import torch.nn.functional as F

        # Ensure 2D → [1, 1, H, W] for interpolation
        if mask.ndim == 2:
            m = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            m = mask[:1].unsqueeze(0)  # take first batch, add channel dim
        else:
            m = mask

        # Use explicit latent dims if provided, otherwise read from noise_mask
        if latent_h is None or latent_w is None:
            _, _, _, latent_h, latent_w = noise_mask.shape
        m = F.interpolate(m.float(), size=(latent_h, latent_w), mode="bilinear", align_corners=False)

        # Blend: strength=1 → full mask, strength=0 → all ones (no masking)
        m = m * strength + (1.0 - strength)

        # Broadcast: [1, 1, 1, H, W] — applies uniformly across time and channels
        m = m.unsqueeze(2).to(device=noise_mask.device, dtype=noise_mask.dtype)

        return noise_mask * m

    def _free_vram(self):
        """Unload models and flush all VRAM/RAM caches."""
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()


    @staticmethod
    def _apply_ffn_chunking(model_clone, num_chunks):
        """Apply FFN chunking to reduce VRAM usage by processing FFN in chunks along the sequence dim."""
        try:
            blocks = model_clone.model.diffusion_model.transformer_blocks
        except AttributeError:
            logger.info("Warning: Could not find transformer_blocks for FFN chunking")
            return

        def make_chunked_forward(ff_module, chunks):
            original_forward = ff_module.forward

            def chunked_forward(x, *args, **kwargs):
                if x.shape[1] <= chunks:
                    return original_forward(x, *args, **kwargs)
                chunk_size = (x.shape[1] + chunks - 1) // chunks
                output_chunks = []
                for i in range(0, x.shape[1], chunk_size):
                    chunk = x[:, i:i + chunk_size]
                    output_chunks.append(original_forward(chunk, *args, **kwargs))
                return torch.cat(output_chunks, dim=1)

            return chunked_forward

        for idx in range(len(blocks)):
            original_ff = blocks[idx].ff
            model_clone.add_object_patch(
                f"diffusion_model.transformer_blocks.{idx}.ff.forward",
                make_chunked_forward(original_ff, num_chunks),
            )
