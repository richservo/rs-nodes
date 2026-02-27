import gc
import math

import torch
import comfy.model_management as mm
import comfy.model_sampling
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.nested_tensor
import node_helpers
import latent_preview


class RSLTXVExtend:
    """
    Extends an existing LTXV video latent by generating new frames with
    overlap blending for seamless continuation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":          ("MODEL",),
                "positive":       ("CONDITIONING",),
                "negative":       ("CONDITIONING",),
                "vae":            ("VAE",),
                "latent":         ("LATENT",),
            },
            "optional": {
                # Extension
                "num_new_frames": ("INT",   {"default": 80,   "min": 8,   "max": 8192, "step": 8}),
                "overlap_frames": ("INT",   {"default": 16,   "min": 8,   "max": 256,  "step": 8}),
                "steps":          ("INT",   {"default": 20,   "min": 1,   "max": 10000}),
                "cfg":            ("FLOAT", {"default": 3.0,  "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed":           ("INT",   {"default": 0,    "min": 0,   "max": 0xffffffffffffffff}),
                "frame_rate":     ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                # Frame injection for end of extension
                "last_image":        ("IMAGE",),
                "last_strength":     ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "overlap_strength":  ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "crf":               ("INT",   {"default": 35,   "min": 0,   "max": 100}),
                # Audio
                "audio":             ("AUDIO",),
                "audio_vae":         ("VAE",),
                # Efficiency
                "attention_mode":    (["auto", "default", "sage"],),
                "ffn_chunks":        ("INT",   {"default": 0, "min": 0, "max": 16, "step": 1}),
                # Upscale
                "upscale":           ("BOOLEAN", {"default": False}),
                "upscale_model":     ("LATENT_UPSCALE_MODEL",),
                "upscale_steps":     ("INT",   {"default": 10,  "min": 1,   "max": 10000}),
                "upscale_cfg":       ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "upscale_denoise":   ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,   "step": 0.01}),
                "upscale_fallback":  ("BOOLEAN", {"default": False}),
                # Output
                "decode":            ("BOOLEAN", {"default": False}),
                # Overrides
                "guider":            ("GUIDER",),
                "sampler":           ("SAMPLER",),
                "sigmas":            ("SIGMAS",),
                # Scheduler (ignored if sigmas provided)
                "max_shift":         ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift":        ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "AUDIO")
    RETURN_NAMES = ("latent", "images", "audio_output")
    FUNCTION     = "extend"
    CATEGORY     = "rs-nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extend(
        self,
        model,
        positive,
        negative,
        vae,
        latent,
        # Extension
        num_new_frames=80,
        overlap_frames=16,
        steps=20,
        cfg=3.0,
        seed=0,
        frame_rate=25.0,
        # Frame injection
        last_image=None,
        last_strength=1.0,
        overlap_strength=1.0,
        crf=35,
        # Audio
        audio=None,
        audio_vae=None,
        # Efficiency
        attention_mode="auto",
        ffn_chunks=0,
        # Upscale
        upscale=False,
        upscale_model=None,
        upscale_steps=10,
        upscale_cfg=3.0,
        upscale_denoise=0.5,
        upscale_fallback=False,
        # Output
        decode=False,
        # Overrides
        guider=None,
        sampler=None,
        sigmas=None,
        # Scheduler
        max_shift=2.05,
        base_shift=0.95,
    ):
        print("[RSLTXVExtend] Starting extension")
        try:
            return self._extend_impl(
                model, positive, negative, vae, latent,
                num_new_frames=num_new_frames, overlap_frames=overlap_frames,
                steps=steps, cfg=cfg, seed=seed, frame_rate=frame_rate,
                last_image=last_image, last_strength=last_strength,
                overlap_strength=overlap_strength, crf=crf,
                audio=audio, audio_vae=audio_vae,
                attention_mode=attention_mode, ffn_chunks=ffn_chunks,
                upscale=upscale, upscale_model=upscale_model,
                upscale_steps=upscale_steps, upscale_cfg=upscale_cfg,
                upscale_denoise=upscale_denoise,
                upscale_fallback=upscale_fallback,
                decode=decode,
                guider=guider, sampler=sampler, sigmas=sigmas,
                max_shift=max_shift, base_shift=base_shift,
            )
        except Exception:
            print("[RSLTXVExtend] Error during extension, cleaning up VRAM")
            raise
        finally:
            self._free_vram()

    def _extend_impl(
        self,
        model, positive, negative, vae, latent,
        num_new_frames, overlap_frames, steps, cfg, seed, frame_rate,
        last_image, last_strength, overlap_strength, crf,
        audio, audio_vae,
        attention_mode, ffn_chunks,
        upscale, upscale_model, upscale_steps, upscale_cfg, upscale_denoise, upscale_fallback,
        decode,
        guider, sampler, sigmas,
        max_shift, base_shift,
    ):
        from comfy_extras.nodes_lt import (
            preprocess as ltxv_preprocess,
            LTXVAddGuide,
            get_noise_mask,
            get_keyframe_idxs,
        )

        # ----------------------------------------------------------------
        # 1. SETUP
        # ----------------------------------------------------------------

        input_samples = latent["samples"]
        batch, channels, input_latent_frames, latent_height, latent_width = input_samples.shape

        # Convert overlap/new pixel frames to latent frames
        overlap_latent_frames = ((overlap_frames - 1) // 8) + 1
        overlap_latent_frames = min(overlap_latent_frames, input_latent_frames)
        new_latent_frames = ((num_new_frames - 1) // 8) + 1
        extension_latent_frames = overlap_latent_frames + new_latent_frames

        print(f"[RSLTXVExtend] Input: {input_latent_frames} latent frames, "
              f"overlap: {overlap_latent_frames}, new: {new_latent_frames}, "
              f"extension chunk: {extension_latent_frames}")

        m = model.clone()

        # ----------------------------------------------------------------
        # 2. EFFICIENCY: Attention + FFN chunking
        # ----------------------------------------------------------------

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
            m.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = attn_func

        if ffn_chunks > 0:
            self._apply_ffn_chunking(m, ffn_chunks)

        # ----------------------------------------------------------------
        # 3. EXTRACT OVERLAP + CREATE EXTENSION LATENT
        # ----------------------------------------------------------------

        overlap_guide = input_samples[:, :, -overlap_latent_frames:].clone()

        ext_latent = torch.zeros(
            [batch, 128, extension_latent_frames, latent_height, latent_width],
            device=mm.intermediate_device(),
        )

        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        # ----------------------------------------------------------------
        # 4. INJECT OVERLAP AS GUIDE AT POSITION 0
        # ----------------------------------------------------------------

        ext_latent_dict = {"samples": ext_latent}
        noise_mask = get_noise_mask(ext_latent_dict)
        scale_factors = vae.downscale_index_formula

        positive, negative, ext_samples, noise_mask = LTXVAddGuide.append_keyframe(
            positive, negative, 0,
            ext_latent, noise_mask,
            overlap_guide, overlap_strength, scale_factors,
        )
        ext_latent_dict = {"samples": ext_samples, "noise_mask": noise_mask}

        # ----------------------------------------------------------------
        # 5. INJECT LAST_IMAGE GUIDE (optional)
        # ----------------------------------------------------------------

        if last_image is not None:
            processed_frames = []
            for i in range(last_image.shape[0]):
                processed_frames.append(ltxv_preprocess(last_image[i], crf))
            processed = torch.stack(processed_frames)

            _, _, ext_length, ext_h, ext_w = ext_latent_dict["samples"].shape
            _, t = LTXVAddGuide.encode(vae, ext_w, ext_h, processed, scale_factors)

            frame_idx_actual, latent_idx = LTXVAddGuide.get_latent_index(
                positive, ext_length, len(processed), -1, scale_factors
            )

            positive, negative, ext_samples, noise_mask = LTXVAddGuide.append_keyframe(
                positive, negative, frame_idx_actual,
                ext_latent_dict["samples"], noise_mask,
                t, last_strength, scale_factors,
            )
            ext_latent_dict = {"samples": ext_samples, "noise_mask": noise_mask}

        # ----------------------------------------------------------------
        # 6. AUDIO (optional)
        # ----------------------------------------------------------------

        has_audio = audio is not None and audio_vae is not None
        if has_audio:
            print("[RSLTXVExtend] Encoding audio latents")
            audio_latents = audio_vae.encode(audio)

            video_samples = ext_latent_dict["samples"]
            audio_samples = audio_latents

            combined_samples = comfy.nested_tensor.NestedTensor((video_samples, audio_samples))

            video_noise_mask = ext_latent_dict.get("noise_mask", None)
            if video_noise_mask is None:
                video_noise_mask = torch.ones_like(video_samples)
            audio_noise_mask = torch.ones_like(audio_samples)
            combined_noise_mask = comfy.nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))

            ext_latent_dict = {"samples": combined_samples, "noise_mask": combined_noise_mask}

        # ----------------------------------------------------------------
        # 7. MODEL SAMPLING + SCHEDULING
        # ----------------------------------------------------------------

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        latent_samples_ref = ext_latent_dict["samples"]
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

        if sigmas is None:
            sig = torch.linspace(1.0, 0.0, steps + 1)
            sig = torch.where(
                sig != 0,
                math.exp(shift) / (math.exp(shift) + (1.0 / sig - 1.0) ** 1),
                torch.zeros_like(sig),
            )
            non_zero_mask = sig != 0
            non_zero_sigmas = sig[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor_sig = one_minus_z[-1] / (1.0 - 0.1)
            sig[non_zero_mask] = 1.0 - (one_minus_z / scale_factor_sig)
            sigmas = sig

        if sampler is None:
            sampler = comfy.samplers.sampler_object("euler_ancestral")

        if guider is None:
            guider = comfy.samplers.CFGGuider(m)
            guider.set_conds(positive, negative)
            guider.set_cfg(cfg)

        # ----------------------------------------------------------------
        # 8. SAMPLE
        # ----------------------------------------------------------------

        latent_image = ext_latent_dict["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        ext_latent_dict["samples"] = latent_image

        noise = comfy.sample.prepare_noise(latent_image, seed)
        noise_mask_for_sampling = ext_latent_dict.get("noise_mask", None)

        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        print("[RSLTXVExtend] Sampling extension chunk...")
        samples = guider.sample(
            noise, latent_image, sampler, sigmas,
            denoise_mask=noise_mask_for_sampling,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        samples = samples.to(mm.intermediate_device())

        ext_output = ext_latent_dict.copy()
        ext_output["samples"] = samples

        # ----------------------------------------------------------------
        # 9. POST-SAMPLE: Crop guides + separate AV
        # ----------------------------------------------------------------

        ext_samples_out = ext_output["samples"].clone()
        ext_noise_mask = get_noise_mask(ext_output)

        _, num_keyframes = get_keyframe_idxs(positive)
        if num_keyframes > 0:
            ext_samples_out = ext_samples_out[:, :, :-num_keyframes]
            ext_noise_mask = ext_noise_mask[:, :, :-num_keyframes]

        ext_output = {"samples": ext_samples_out}

        audio_latent_out = None
        if has_audio and ext_output["samples"].is_nested:
            latents_list = ext_output["samples"].unbind()
            ext_output["samples"] = latents_list[0]
            audio_latent_out = latents_list[1] if len(latents_list) > 1 else None

        # ----------------------------------------------------------------
        # 10. BLEND + CONCATENATE
        # ----------------------------------------------------------------

        ext_video = ext_output["samples"]

        # Linear blend in overlap region
        for i in range(overlap_latent_frames):
            alpha = i / max(overlap_latent_frames - 1, 1)
            ext_video[:, :, i] = (
                (1 - alpha) * input_samples[:, :, -(overlap_latent_frames - i)]
                + alpha * ext_video[:, :, i]
            )

        # Concatenate: original (minus overlap) + blended extension
        original_trimmed = input_samples[:, :, :-overlap_latent_frames]
        final_video = torch.cat([original_trimmed, ext_video], dim=2)

        output_latent = {"samples": final_video}
        print(f"[RSLTXVExtend] Extended: {input_latent_frames} → {final_video.shape[2]} latent frames")

        # ----------------------------------------------------------------
        # 11. UPSCALE (optional)
        # ----------------------------------------------------------------

        if upscale and upscale_model is not None:
            if upscale_fallback:
                pre_upscale_latent = {"samples": output_latent["samples"].detach().cpu().clone()}
            try:
                print("[RSLTXVExtend] Upscaling latents")
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

                # Optional re-diffusion at upscaled resolution
                if upscale_denoise > 0 and upscale_steps > 0:
                    print("[RSLTXVExtend] Re-sampling at upscaled resolution")
                    self._free_vram()

                    up_tokens = math.prod(upsampled.shape[2:])
                    up_shift = up_tokens * mm_shift + b

                    up_model_sampling = ModelSamplingAdvanced(m.model.model_config)
                    up_model_sampling.set_parameters(shift=up_shift)
                    m.add_object_patch("model_sampling", up_model_sampling)

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

                    start_step = int((1.0 - upscale_denoise) * upscale_steps)
                    up_sig = up_sig[start_step:]

                    up_guider = comfy.samplers.CFGGuider(m)
                    up_guider.set_conds(positive, negative)
                    up_guider.set_cfg(upscale_cfg)

                    up_noise = comfy.sample.prepare_noise(upsampled, seed + 1)
                    up_sampler = comfy.samplers.sampler_object("euler")
                    up_callback = latent_preview.prepare_callback(up_guider.model_patcher, up_sig.shape[-1] - 1)

                    up_samples = up_guider.sample(
                        up_noise, upsampled, up_sampler, up_sig,
                        callback=up_callback,
                        disable_pbar=disable_pbar,
                        seed=seed + 1,
                    )
                    up_samples = up_samples.to(mm.intermediate_device())
                    output_latent = {"samples": up_samples}

                if upscale_fallback:
                    del pre_upscale_latent  # free system RAM backup

            except Exception as e:
                if not upscale_fallback:
                    raise
                print(f"[RSLTXVExtend] WARNING: Upscale failed ({type(e).__name__}: {e})")
                print("[RSLTXVExtend] Falling back to half-resolution decode")
                self._free_vram()
                output_latent = pre_upscale_latent
                decode = True

        # ----------------------------------------------------------------
        # 12. DECODE (optional)
        # ----------------------------------------------------------------

        images = None
        if decode:
            print("[RSLTXVExtend] Decoding latents to images (tiled)")
            self._free_vram()
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

        # ----------------------------------------------------------------
        # 13. AUDIO DECODE (if audio was used)
        # ----------------------------------------------------------------

        audio_output = None
        if has_audio and audio_latent_out is not None and audio_vae is not None:
            print("[RSLTXVExtend] Decoding audio latents")
            decoded_audio = audio_vae.decode(audio_latent_out).to(audio_latent_out.device)
            audio_output = {
                "waveform": decoded_audio,
                "sample_rate": int(audio_vae.output_sample_rate),
            }

        print("[RSLTXVExtend] Done")
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
            print("[RSLTXVExtend] Warning: Could not find transformer_blocks for FFN chunking")
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
