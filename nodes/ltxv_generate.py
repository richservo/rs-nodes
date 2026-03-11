import gc
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
                "first_strength":    ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "middle_strength":   ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "last_strength":     ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "crf":               ("INT",   {"default": 35,   "min": 0,   "max": 100}),
                # Audio
                "audio":             ("AUDIO",),
                "audio_vae":         ("VAE",),
                # Multimodal guidance (used when audio_vae is connected)
                "audio_cfg":         ("FLOAT", {"default": 7.0,  "min": 0.0, "max": 100.0, "step": 0.1}),
                "stg_scale":         ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 10.0,  "step": 0.1}),
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
        first_strength=1.0,
        middle_strength=1.0,
        last_strength=1.0,
        crf=35,
        # Audio
        audio=None,
        audio_vae=None,
        # Multimodal guidance
        audio_cfg=7.0,
        stg_scale=0.0,
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
        print(f"[RSLTXVGenerate] Starting generation (seed={seed}, mode={seed_mode})")
        try:
            result = self._generate_impl(
                model, positive, negative, vae,
                width=width, height=height, num_frames=num_frames,
                steps=steps, cfg=cfg, seed=seed, frame_rate=frame_rate,
                first_image=first_image, middle_image=middle_image, last_image=last_image,
                first_strength=first_strength, middle_strength=middle_strength,
                last_strength=last_strength, crf=crf,
                audio=audio, audio_vae=audio_vae,
                audio_cfg=audio_cfg, stg_scale=stg_scale,
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
                decode=decode, tile_t=tile_t,
                guider=guider, sampler=sampler, sigmas=sigmas,
                max_shift=max_shift, base_shift=base_shift,
            )
            # Write resolved seed back to the widget so the user can see/reuse it
            return {"ui": {"noise_seed": [seed]}, "result": result}
        except Exception:
            print("[RSLTXVGenerate] Error during generation, cleaning up VRAM")
            raise
        finally:
            self._free_vram()

    def _generate_impl(
        self,
        model, positive, negative, vae,
        width, height, num_frames, steps, cfg, seed, frame_rate,
        first_image, middle_image, last_image,
        first_strength, middle_strength, last_strength, crf,
        audio, audio_vae,
        audio_cfg, stg_scale, audio_stg_scale, cfg_end, stg_end,
        stg_blocks, rescale, video_modality_scale, audio_modality_scale,
        cfg_star_rescale, skip_sigma,
        attention_mode, ffn_chunks, video_attn_scale,
        upscale, upscale_model, temporal_upscale_model, upscale_lora, upscale_lora_strength,
        upscale_steps, upscale_cfg, upscale_denoise, upscale_fallback,
        upscale_tiling, upscale_tile_t,
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
                print(f"[RSLTXVGenerate] Audio duration {audio_duration:.2f}s → overriding num_frames: {num_frames} → {audio_num_frames}")
                num_frames = audio_num_frames

        # When upscaling, generate at half resolution — the 2x latent upscaler
        # brings it to the target width x height afterwards.
        gen_width = width
        gen_height = height
        do_upscale = upscale and upscale_model is not None
        if upscale and upscale_model is None:
            print("[RSLTXVGenerate] WARNING: upscale=True but no upscale_model connected — generating at full resolution")
        if do_upscale:
            gen_width = width // 2
            gen_height = height // 2
            print(f"[RSLTXVGenerate] Upscale enabled: generating at {gen_width}x{gen_height}, target {width}x{height}")

        # T2V first-frame bootstrap: generate a short clip at full resolution,
        # decode the first frame, and use it as I2V guidance for the main generation.
        # This converts T2V→I2V so the upscale path only needs one re-diffusion pass.
        if do_upscale and first_image is None and last_image is None and middle_image is None:
            print(f"[RSLTXVGenerate] T2V bootstrap: generating 9 frames at {width}x{height} for first frame")
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

            boot_sig = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
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
            boot_sampler = comfy.samplers.sampler_object("euler_ancestral")
            boot_cb = latent_preview.prepare_callback(boot_guider.model_patcher, boot_sig.shape[-1] - 1)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

            print("[RSLTXVGenerate] T2V bootstrap: sampling first frame...")
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
            print(f"[RSLTXVGenerate] T2V bootstrap: first frame {first_image.shape[2]}x{first_image.shape[1]} ready")

            del boot_out, boot_model, boot_guider
            self._free_vram()

        # Temporal upscale: generate at half frame count, then 2x temporal upscale
        do_temporal_upscale = do_upscale and temporal_upscale_model is not None
        gen_num_frames = num_frames
        if do_temporal_upscale:
            target_latent_T = ((num_frames - 1) // 8) + 1
            half_latent_T = target_latent_T // 2 + 1
            gen_num_frames = (half_latent_T - 1) * 8 + 1
            print(f"[RSLTXVGenerate] Temporal upscale enabled: generating {gen_num_frames} frames at {frame_rate / 2:.1f}fps (target {num_frames} frames at {frame_rate}fps)")

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
            for img, frame_idx, strength_val, label in guides:
                _, _, latent_length, latent_height, latent_width = latent_dict["samples"].shape

                # Use LTXVAddGuide.encode directly (resize + VAE encode, no CRF)
                _, t = LTXVAddGuide.encode(vae, latent_width, latent_height, img, scale_factors)

                frame_idx_actual, latent_idx = LTXVAddGuide.get_latent_index(
                    positive, latent_length, len(img), frame_idx, scale_factors
                )

                print(f"[RSLTXVGenerate] Guide {label}: frame_idx={frame_idx_actual}, latent_idx={latent_idx}, strength={strength_val}")

                positive, negative, latent_samples, noise_mask = LTXVAddGuide.append_keyframe(
                    positive, negative, frame_idx_actual,
                    latent_dict["samples"], noise_mask,
                    t, strength_val, scale_factors,
                )
                latent_dict = {"samples": latent_samples, "noise_mask": noise_mask}

        # ----------------------------------------------------------------
        # 3. AUDIO (requires audio_vae for the AV dual-tower model)
        # ----------------------------------------------------------------

        has_audio = audio_vae is not None
        audio_is_input = has_audio and audio is not None  # True = user provided audio, pass through
        if has_audio:
            if audio is not None:
                # Encode provided audio into latent space
                print("[RSLTXVGenerate] Encoding input audio latents")
                audio_samples = audio_vae.encode(audio)
            else:
                # Create empty audio latents — the AV model generates audio from scratch
                print("[RSLTXVGenerate] Creating empty audio latents for generation")
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
                video_noise_mask = torch.ones_like(video_samples)
            # Audio mask: 0 = preserve input audio, 1 = generate from scratch
            if audio is not None:
                audio_noise_mask = torch.zeros_like(audio_samples)
            else:
                audio_noise_mask = torch.ones_like(audio_samples)
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
            sampler = comfy.samplers.sampler_object("euler_ancestral")

        # Build guider if not provided
        if guider is None:
            from ..utils.multimodal_guider import MultimodalGuider
            guider = MultimodalGuider(
                m, positive, negative,
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
            print("[RSLTXVGenerate] Using MultimodalGuider")

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

        print("[RSLTXVGenerate] Sampling...")
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
                    print("[RSLTXVGenerate] Upscaling video latents (2x temporal at half res)")
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
                        print(f"[RSLTXVGenerate] Trimmed temporal output to {target_latent_T} latent frames")

                    output_latent = {"samples": t_upsampled}
                    print(f"[RSLTXVGenerate] After temporal upscale: latent shape {list(t_upsampled.shape)}")

                    # Restore full frame rate (was halved for temporal upscale generation)
                    positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
                    negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

                # Spatial upscale (2x resolution)
                upscale_label = "2x temporal + 2x spatial" if do_temporal_upscale else "2x spatial"
                print(f"[RSLTXVGenerate] Upscaling video latents ({upscale_label})")
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
                            print(f"[RSLTXVGenerate] Upscale model: chunking {latent_t} latent frames, chunk_t={us_chunk_t}, overlap={us_overlap}")
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

                # Re-inject guide images at full resolution into upscaled latent
                # using LTXVAddGuide crop guides (same as inference stage)
                from comfy_extras.nodes_lt import LTXVAddGuide as UpGuide, get_noise_mask as up_get_noise_mask, get_keyframe_idxs as up_get_keyframe_idxs

                up_scale_factors = vae.downscale_index_formula

                # Re-inject all guides at upscaled resolution (crop guides handle
                # positioning correctly, so multiple guides no longer cause snapping)
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

                        print(f"[RSLTXVGenerate] Upscale re-inject {label}: frame_idx={frame_idx_actual}, latent_idx={latent_idx}, strength={strength_val}")

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
                    print(f"[RSLTXVGenerate] Re-diffusing at upscaled resolution ({rediffusion_label})")
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
                        print(f"[RSLTXVGenerate] Applied upscale LoRA: {upscale_lora} (strength={upscale_lora_strength})")

                    if has_image_guides:
                        # I2V: compute shift from upscaled latent tokens
                        up_tokens = math.prod(upsampled.shape[2:])
                        up_shift = up_tokens * mm_shift + b
                        print(f"[RSLTXVGenerate] Upscale shift: tokens={up_tokens}, shift={up_shift:.3f}")

                        up_model_sampling = ModelSamplingAdvanced(up_model.model.model_config)
                        up_model_sampling.set_parameters(shift=up_shift)
                        up_model.add_object_patch("model_sampling", up_model_sampling)

                        # Build sigma schedule with numerically stable computation.
                        # sig = exp(s) / (exp(s) + (1/t - 1)) and 1-sig = (1/t - 1) / (exp(s) + (1/t - 1))
                        # Computing 1-sig directly avoids catastrophic cancellation when sig ≈ 1.0
                        # (at high shift, sig is so close to 1.0 that 1.0-sig rounds to 0 in float64)
                        exp_shift = math.exp(up_shift)
                        t = torch.linspace(1.0, 0.0, upscale_steps + 1, dtype=torch.float64)
                        non_zero = t != 0
                        inv_t_m1 = torch.where(non_zero, 1.0 / t - 1.0, torch.zeros_like(t))
                        # omz = 1 - sigma, computed directly
                        omz = torch.where(non_zero, inv_t_m1 / (exp_shift + inv_t_m1), torch.ones_like(t))
                        # Stretch so terminal sigma = 0.1 (i.e., terminal omz = 0.9)
                        nz_omz = omz[non_zero]
                        sf = nz_omz[-1] / (1.0 - 0.1)
                        omz[non_zero] = nz_omz / sf
                        # sigma = 1 - omz; t=0 maps to sigma=0
                        up_sig = torch.where(non_zero, 1.0 - omz, torch.zeros_like(omz)).float()

                        # Trim to denoise strength
                        start_step = int((1.0 - upscale_denoise) * upscale_steps)
                        up_sig = up_sig[start_step:]
                    else:
                        # T2V: use manual sigmas from the official LTX 2.3 upscale workflow.
                        # No image anchors, so computed schedules produce artifacts (blobs).
                        # These hand-tuned values with default shift give clean results.
                        up_sig = torch.tensor([0.909375, 0.725, 0.421875, 0.0])
                        print(f"[RSLTXVGenerate] T2V upscale: manual sigmas {up_sig.tolist()}")

                        up_shift = 4096 * mm_shift + b  # default shift (tokens=4096)
                        up_model_sampling = ModelSamplingAdvanced(up_model.model.model_config)
                        up_model_sampling.set_parameters(shift=up_shift)
                        up_model.add_object_patch("model_sampling", up_model_sampling)

                    up_guider = comfy.samplers.CFGGuider(up_model)
                    up_guider.set_conds(positive, negative)
                    up_guider.set_cfg(upscale_cfg)

                    up_latent_image = comfy.sample.fix_empty_latent_channels(up_guider.model_patcher, up_combined)
                    up_noise = comfy.sample.prepare_noise(up_latent_image, seed + 1)
                    up_sampler = comfy.samplers.sampler_object("euler_ancestral")

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
                        print(f"[RSLTXVGenerate] Re-diffusion tiling: {video_t} latent frames, chunk_t={rd_chunk_t}, overlap={rd_overlap}")

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

                            print(f"[RSLTXVGenerate]   Chunk {chunk_idx}: ctx=[{ctx_start}:{ctx_end}], core=[{t_pos}:{min(t_pos + rd_chunk_t, video_t)}]")
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

                            print(f"[RSLTXVGenerate]   Chunk {chunk_idx} output: shape={chunk_vid_out.shape}, min={chunk_vid_out.min():.4f}, max={chunk_vid_out.max():.4f}, mean={chunk_vid_out.mean():.4f}")

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
                        print(f"[RSLTXVGenerate] Re-diffusion result: shape={up_samples.shape}, min={up_samples.min():.4f}, max={up_samples.max():.4f}, mean={up_samples.mean():.4f}")
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

                # T2V pass 2: decode first frame from pass 1, sharpen, use as
                # I2V-style guidance for a second re-diffusion with audio context.
                # Runs after temporal upscale to fix any artifacts and improve lip sync.
                if not has_image_guides and do_rediffusion and upscale_denoise > 0 and upscale_steps > 0:
                    print("[RSLTXVGenerate] T2V pass 2: decoding first frame for I2V guidance")
                    self._free_vram()

                    video_latent = output_latent["samples"]

                    # Decode just the first latent frame
                    first_pixels = vae.decode(video_latent[:, :, :1])
                    if len(first_pixels.shape) == 5:
                        first_pixels = first_pixels.reshape(
                            -1, first_pixels.shape[-3], first_pixels.shape[-2], first_pixels.shape[-1]
                        )

                    # Sharpen + grain to enhance detail before using as anchor
                    first_pixels = self._sharpen_and_grain(first_pixels)

                    # Encode sharpened frame and inject at position 0
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

                    p2_tokens = math.prod(video_latent.shape[2:])
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

                    print(f"[RSLTXVGenerate] T2V pass 2: {upscale_steps} steps, denoise={upscale_denoise}, shift={p2_shift:.3f}")

                    p2_guider = comfy.samplers.CFGGuider(p2_model)
                    p2_guider.set_conds(positive, negative)
                    p2_guider.set_cfg(upscale_cfg)

                    p2_latent = comfy.sample.fix_empty_latent_channels(p2_guider.model_patcher, p2_combined)
                    p2_noise = comfy.sample.prepare_noise(p2_latent, seed + 2)
                    p2_sampler = comfy.samplers.sampler_object("euler_ancestral")

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
                    print("[RSLTXVGenerate] T2V pass 2 complete")

                if upscale_fallback:
                    del pre_upscale_latent  # free system RAM backup

            except Exception as e:
                if not upscale_fallback:
                    raise
                print(f"[RSLTXVGenerate] WARNING: Upscale failed ({type(e).__name__}: {e})")
                print("[RSLTXVGenerate] Falling back to half-resolution decode")
                self._free_vram()
                output_latent = pre_upscale_latent
                decode = True

        # Free model clones to reclaim CPU RAM from offloaded weights
        del m
        self._free_vram()

        # ----------------------------------------------------------------
        # 8. DECODE (optional)
        # ----------------------------------------------------------------

        if decode:
            print("[RSLTXVGenerate] Decoding latents to images (tiled)")
            compression = vae.spacial_compression_decode()
            tile_size = 512 // compression
            overlap = 128 // compression  # larger overlap to reduce tile seams
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

            # Add tiny noise to break up tile boundary artifacts
            samples = output_latent["samples"]
            samples = samples + torch.randn_like(samples) * 1e-5

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
                print("[RSLTXVGenerate] Audio passthrough (input audio preserved)")
                audio_output = audio
            else:
                print("[RSLTXVGenerate] Decoding generated audio latents")
                decoded_audio = audio_vae.decode(audio_latent_out).to(audio_latent_out.device)
                audio_output = {
                    "waveform": decoded_audio,
                    "sample_rate": int(audio_vae.output_sample_rate),
                }

        # Build audio latent output
        audio_latent_dict = None
        if audio_latent_out is not None:
            audio_latent_dict = {"samples": audio_latent_out}

        print("[RSLTXVGenerate] Done")
        return (output_latent, audio_latent_dict, images, audio_output)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _free_vram(self):
        """Unload models and flush all VRAM/RAM caches."""
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()

    @staticmethod
    def _sharpen_and_grain(image, sharpen_amount=0.15,
                           grain_intensity=0.05, grain_size=1.2,
                           grain_color_amount=0.30, grain_highlight_protection=0.50):
        """Subtle sharpen + film grain for T2V upscale guidance frame.
        image: [B, H, W, C] float 0-1."""
        device = image.device

        # Unsharp mask
        x = image.movedim(-1, 1)  # BCHW
        blurred = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = (x + sharpen_amount * (x - blurred)).clamp(0, 1)
        image = x.movedim(1, -1)  # BHWC
        del x, blurred

        # Film grain (from RSFilmGrain, single-frame fast path)
        if grain_intensity > 0:
            B, H, W, C = image.shape
            noise_h = max(1, round(H / grain_size))
            noise_w = max(1, round(W / grain_size))
            need_upscale = noise_h != H or noise_w != W

            lum_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device)
            luminance = (image * lum_weights).sum(dim=-1)
            midtone_mask = 4.0 * luminance * (1.0 - luminance)
            mask = (1.0 - grain_highlight_protection + grain_highlight_protection * midtone_mask).unsqueeze(-1)

            for i in range(B):
                if need_upscale:
                    small_mono = torch.randn(1, 1, noise_h, noise_w, device=device)
                    mono = torch.nn.functional.interpolate(
                        small_mono, size=(H, W), mode="bilinear", align_corners=False
                    ).permute(0, 2, 3, 1).squeeze(0)
                    small_color = torch.randn(1, C, noise_h, noise_w, device=device)
                    color = torch.nn.functional.interpolate(
                        small_color, size=(H, W), mode="bilinear", align_corners=False
                    ).permute(0, 2, 3, 1).squeeze(0)
                else:
                    mono = torch.randn(H, W, 1, device=device)
                    color = torch.randn(H, W, C, device=device)

                noise = mono.lerp(color, grain_color_amount)
                image[i].add_(noise.mul_(grain_intensity).mul_(mask[i]))

            image = image.clamp(0, 1)

        return image

    @staticmethod
    def _apply_ffn_chunking(model_clone, num_chunks):
        """Apply FFN chunking to reduce VRAM usage by processing FFN in chunks along the sequence dim."""
        try:
            blocks = model_clone.model.diffusion_model.transformer_blocks
        except AttributeError:
            print("[RSLTXVGenerate] Warning: Could not find transformer_blocks for FFN chunking")
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
