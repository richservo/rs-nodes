import gc
import math

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
                "seed":       ("INT",   {"default": 0,    "min": 0,   "max": 0xffffffffffffffff}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                # Frame injection
                "first_image":       ("IMAGE",),
                "middle_image":      ("IMAGE",),
                "last_image":        ("IMAGE",),
                "first_strength":    ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "middle_strength":   ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "last_strength":     ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "crf":               ("INT",   {"default": 35,   "min": 0,   "max": 100}),
                # Audio
                "audio":             ("AUDIO",),
                "audio_vae":         ("VAE",),
                # Multimodal guidance (used when audio_vae is connected)
                "audio_cfg":         ("FLOAT", {"default": 7.0,  "min": 0.0, "max": 100.0, "step": 0.1}),
                "stg_scale":         ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 10.0,  "step": 0.1}),
                "stg_blocks":        ("STRING", {"default": "29"}),
                "rescale":           ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0,   "step": 0.01}),
                "modality_scale":    ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 100.0, "step": 0.1}),
                # Efficiency
                "attention_mode":    (["auto", "default", "sage"],),
                "ffn_chunks":        ("INT",   {"default": 0, "min": 0, "max": 16, "step": 1}),
                # Upscale
                "upscale":           ("BOOLEAN", {"default": False}),
                "upscale_model":     ("LATENT_UPSCALE_MODEL",),
                "upscale_lora":      (["none"] + folder_paths.get_filename_list("loras"),),
                "upscale_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "upscale_steps":     ("INT",   {"default": 4,   "min": 1,   "max": 10000}),
                "upscale_cfg":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "upscale_denoise":   ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,   "step": 0.01}),
                # Output
                "decode":            ("BOOLEAN", {"default": True}),
                # Overrides
                "guider":            ("GUIDER",),
                "sampler":           ("SAMPLER",),
                "sigmas":            ("SIGMAS",),
                # Scheduler (ignored if sigmas provided)
                "max_shift":         ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift":        ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES  = ("LATENT", "IMAGE", "AUDIO")
    RETURN_NAMES  = ("latent", "images", "audio_output")
    FUNCTION      = "generate"
    CATEGORY      = "rs-nodes"

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

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
        seed=0,
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
        stg_blocks="29",
        rescale=0.7,
        modality_scale=1.0,
        # Efficiency
        attention_mode="auto",
        ffn_chunks=0,
        # Upscale
        upscale=False,
        upscale_model=None,
        upscale_lora="none",
        upscale_lora_strength=1.0,
        upscale_steps=4,
        upscale_cfg=1.0,
        upscale_denoise=0.5,
        # Output
        decode=False,
        # Overrides
        guider=None,
        sampler=None,
        sigmas=None,
        # Scheduler
        max_shift=2.05,
        base_shift=0.95,
        **kwargs,
    ):
        print("[RSLTXVGenerate] Starting generation")
        try:
            return self._generate_impl(
                model, positive, negative, vae,
                width=width, height=height, num_frames=num_frames,
                steps=steps, cfg=cfg, seed=seed, frame_rate=frame_rate,
                first_image=first_image, middle_image=middle_image, last_image=last_image,
                first_strength=first_strength, middle_strength=middle_strength,
                last_strength=last_strength, crf=crf,
                audio=audio, audio_vae=audio_vae,
                audio_cfg=audio_cfg, stg_scale=stg_scale, stg_blocks=stg_blocks,
                rescale=rescale, modality_scale=modality_scale,
                attention_mode=attention_mode, ffn_chunks=ffn_chunks,
                upscale=upscale, upscale_model=upscale_model,
                upscale_lora=upscale_lora, upscale_lora_strength=upscale_lora_strength,
                upscale_steps=upscale_steps, upscale_cfg=upscale_cfg,
                upscale_denoise=upscale_denoise,
                decode=decode,
                guider=guider, sampler=sampler, sigmas=sigmas,
                max_shift=max_shift, base_shift=base_shift,
            )
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
        audio_cfg, stg_scale, stg_blocks, rescale, modality_scale,
        attention_mode, ffn_chunks,
        upscale, upscale_model, upscale_lora, upscale_lora_strength,
        upscale_steps, upscale_cfg, upscale_denoise,
        decode,
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

        # Create empty latent: [B, C, T, H, W] — LTXV latent space
        latent = torch.zeros(
            [1, 128, ((num_frames - 1) // 8) + 1, gen_height // 32, gen_width // 32],
            device=mm.intermediate_device(),
        )

        # Stamp frame rate onto conditioning
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        # ----------------------------------------------------------------
        # 2. FRAME INJECTION
        # ----------------------------------------------------------------

        from comfy_extras.nodes_lt import preprocess as ltxv_preprocess, LTXVAddGuide, get_noise_mask

        latent_dict = {"samples": latent}
        noise_mask = get_noise_mask(latent_dict)

        # 2a. First image — inplace injection (matches LTXVImgToVideoInplace).
        #     Encodes raw pixels (no CRF), places at position 0, noise_mask
        #     controls preservation: strength=1.0 → mask=0.0 → pixel-perfect.
        if first_image is not None:
            _, height_sf, width_sf = vae.downscale_index_formula
            _, _, _, lh, lw = latent_dict["samples"].shape
            target_w, target_h = lw * width_sf, lh * height_sf

            if first_image.shape[1] != target_h or first_image.shape[2] != target_w:
                pixels = comfy.utils.common_upscale(
                    first_image.movedim(-1, 1), target_w, target_h, "bilinear", "center"
                ).movedim(1, -1)
            else:
                pixels = first_image
            t = vae.encode(pixels[:, :, :, :3])

            latent_dict["samples"][:, :, :t.shape[2]] = t
            noise_mask[:, :, :t.shape[2]] = 1.0 - first_strength
            latent_dict["noise_mask"] = noise_mask

        # 2b. Middle/last images — guide conditioning via LTXVAddGuide
        guides = []
        if middle_image is not None:
            mid_idx = (num_frames - 1) // 2
            mid_idx = max(0, (mid_idx // 8) * 8)
            if mid_idx == 0 and num_frames > 8:
                mid_idx = 8
            guides.append((middle_image, mid_idx, middle_strength))
        if last_image is not None:
            guides.append((last_image, -1, last_strength))

        if guides:
            scale_factors = vae.downscale_index_formula
            for img, frame_idx, strength_val in guides:
                _, _, latent_length, latent_height, latent_width = latent_dict["samples"].shape

                processed_frames = []
                for i in range(img.shape[0]):
                    processed_frames.append(ltxv_preprocess(img[i], crf))
                processed = torch.stack(processed_frames)

                _, t = LTXVAddGuide.encode(vae, latent_width, latent_height, processed, scale_factors)

                frame_idx_actual, latent_idx = LTXVAddGuide.get_latent_index(
                    positive, latent_length, len(processed), frame_idx, scale_factors
                )

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
            sig = torch.linspace(1.0, 0.0, steps + 1)
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
            sigmas = sig

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
                rescale=rescale, modality_scale=modality_scale,
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
            # Spatial upscale is video-only (AV already separated in section 6)
            print("[RSLTXVGenerate] Upscaling video latents (2x spatial)")
            self._free_vram()

            device = mm.get_torch_device()
            model_dtype = next(upscale_model.parameters()).dtype
            up_latents = output_latent["samples"]
            input_dtype = up_latents.dtype

            memory_required = mm.module_size(upscale_model)
            memory_required += math.prod(up_latents.shape) * 3000.0
            mm.free_memory(memory_required, device)

            try:
                upscale_model.to(device)
                up_latents = up_latents.to(dtype=model_dtype, device=device)
                up_latents = vae.first_stage_model.per_channel_statistics.un_normalize(up_latents)
                upsampled = upscale_model(up_latents)
            finally:
                upscale_model.cpu()

            upsampled = vae.first_stage_model.per_channel_statistics.normalize(upsampled)
            upsampled = upsampled.to(dtype=input_dtype, device=mm.intermediate_device())
            output_latent = {"samples": upsampled}

            # Re-diffusion at upscaled resolution
            if upscale_denoise > 0 and upscale_steps > 0:
                print(f"[RSLTXVGenerate] Re-diffusing at upscaled resolution ({upscale_steps} steps, cfg={upscale_cfg})")
                self._free_vram()

                # Recombine video + audio for the AV model
                if has_audio and audio_latent_out is not None:
                    up_combined = comfy.nested_tensor.NestedTensor((upsampled, audio_latent_out))
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

                up_tokens = math.prod(upsampled.shape[2:])
                up_shift = up_tokens * mm_shift + b

                up_model_sampling = ModelSamplingAdvanced(up_model.model.model_config)
                up_model_sampling.set_parameters(shift=up_shift)
                up_model.add_object_patch("model_sampling", up_model_sampling)

                up_sig = torch.linspace(1.0, 0.0, upscale_steps + 1)
                up_sig = torch.where(
                    up_sig != 0,
                    math.exp(up_shift) / (math.exp(up_shift) + (1.0 / up_sig - 1.0) ** 1),
                    torch.zeros_like(up_sig),
                )
                non_zero = up_sig != 0
                nz_sigmas = up_sig[non_zero]
                omz = 1.0 - nz_sigmas
                sf = omz[-1] / (1.0 - 0.1)
                up_sig[non_zero] = 1.0 - (omz / sf)

                # Trim to denoise strength
                start_step = int((1.0 - upscale_denoise) * upscale_steps)
                up_sig = up_sig[start_step:]

                up_guider = comfy.samplers.CFGGuider(up_model)
                up_guider.set_conds(positive, negative)
                up_guider.set_cfg(upscale_cfg)

                up_latent_image = comfy.sample.fix_empty_latent_channels(up_guider.model_patcher, up_combined)
                up_noise = comfy.sample.prepare_noise(up_latent_image, seed + 1)
                up_sampler = comfy.samplers.sampler_object("euler_ancestral")
                up_callback = latent_preview.prepare_callback(up_guider.model_patcher, up_sig.shape[-1] - 1)

                up_samples = up_guider.sample(
                    up_noise, up_latent_image, up_sampler, up_sig,
                    callback=up_callback,
                    disable_pbar=disable_pbar,
                    seed=seed + 1,
                )
                up_samples = up_samples.to(mm.intermediate_device())

                # Separate AV again after re-diffusion
                if up_samples.is_nested:
                    up_parts = up_samples.unbind()
                    output_latent = {"samples": up_parts[0]}
                    audio_latent_out = up_parts[1] if len(up_parts) > 1 else audio_latent_out
                else:
                    output_latent = {"samples": up_samples}

        # ----------------------------------------------------------------
        # 8. DECODE (optional)
        # ----------------------------------------------------------------

        if decode:
            print("[RSLTXVGenerate] Decoding latents to images (tiled)")
            self._free_vram()
            # Tiled decode — LTXV VAE has 32x spatial compression
            compression = vae.spacial_compression_decode()
            tile_size = 512 // compression
            overlap = 64 // compression
            temporal_compression = vae.temporal_compression_decode()
            if temporal_compression is not None:
                tile_t = max(2, 64 // temporal_compression)
                overlap_t = max(1, min(tile_t // 2, 8 // temporal_compression))
            else:
                tile_t = None
                overlap_t = None
            images = vae.decode_tiled(
                output_latent["samples"],
                tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                tile_t=tile_t, overlap_t=overlap_t,
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
            print("[RSLTXVGenerate] Decoding audio latents")
            decoded_audio = audio_vae.decode(audio_latent_out).to(audio_latent_out.device)
            audio_output = {
                "waveform": decoded_audio,
                "sample_rate": int(audio_vae.output_sample_rate),
            }

        print("[RSLTXVGenerate] Done")
        return (output_latent, images, audio_output)

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
