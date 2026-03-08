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


class RSLTXVUpscale:
    """
    Standalone LTXV 2x video upscaler. Encodes input video to latent space,
    optional temporal 2x, spatial 2x via upscale model, then re-diffuses with
    first-frame I2V guidance to add real detail.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("MODEL",),
                "positive":      ("CONDITIONING",),
                "negative":      ("CONDITIONING",),
                "vae":           ("VAE",),
                "images":        ("IMAGE",),
            },
            "optional": {
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "frame_rate":    ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                # Audio
                "audio":         ("AUDIO",),
                "audio_vae":     ("VAE",),
                # Temporal upscale
                "temporal_upscale_model": ("LATENT_UPSCALE_MODEL",),
                # Re-diffusion
                "upscale_lora":          (["none"] + folder_paths.get_filename_list("loras"),),
                "upscale_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "upscale_steps":         ("INT",   {"default": 4,   "min": 1,   "max": 10000}),
                "upscale_cfg":           ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "upscale_denoise":       ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,   "step": 0.01}),
                # Scheduler
                "max_shift":     ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift":    ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
                # Output
                "decode":        ("BOOLEAN", {"default": True}),
                "tile_t":        ("INT",     {"default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "Temporal tile size for VAE decode (0 = auto). Lower values reduce VRAM but may cause seams."}),
                # Efficiency
                "attention_mode":   (["auto", "default", "sage"],),
                "ffn_chunks":       ("INT",   {"default": 4, "min": 0, "max": 16, "step": 1}),
                "video_attn_scale": ("FLOAT", {"default": 1.03, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Video attention scale (1.03 recommended). Also enables VRAM-efficient block forward."}),
            },
        }

    RETURN_TYPES  = ("LATENT", "IMAGE", "AUDIO")
    RETURN_NAMES  = ("latent", "images", "audio_output")
    OUTPUT_NODE   = True
    FUNCTION      = "upscale"
    CATEGORY      = "rs-nodes"

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def upscale(
        self,
        model,
        positive,
        negative,
        vae,
        images,
        upscale_model=None,
        frame_rate=25.0,
        audio=None,
        audio_vae=None,
        temporal_upscale_model=None,
        upscale_lora="none",
        upscale_lora_strength=1.0,
        upscale_steps=4,
        upscale_cfg=1.0,
        upscale_denoise=0.5,
        max_shift=2.05,
        base_shift=0.95,
        decode=True,
        tile_t=0,
        attention_mode="auto",
        ffn_chunks=4,
        video_attn_scale=1.03,
        **kwargs,
    ):
        do_spatial = upscale_model is not None
        do_temporal = temporal_upscale_model is not None
        temporal_label = " + 2x temporal" if do_temporal else ""
        if do_spatial:
            print(f"[RSLTXVUpscale] Starting: {images.shape[0]} frames, {images.shape[2]}x{images.shape[1]} → {images.shape[2]*2}x{images.shape[1]*2}{temporal_label}")
        elif do_temporal:
            print(f"[RSLTXVUpscale] Starting 2x temporal: {images.shape[0]} frames, {images.shape[2]}x{images.shape[1]}")
        else:
            print("[RSLTXVUpscale] WARNING: No upscale models connected")
        try:
            result = self._upscale_impl(
                model, positive, negative, vae, images,
                upscale_model=upscale_model,
                frame_rate=frame_rate, audio=audio, audio_vae=audio_vae,
                temporal_upscale_model=temporal_upscale_model,
                upscale_lora=upscale_lora, upscale_lora_strength=upscale_lora_strength,
                upscale_steps=upscale_steps, upscale_cfg=upscale_cfg,
                upscale_denoise=upscale_denoise,
                max_shift=max_shift, base_shift=base_shift,
                decode=decode, tile_t=tile_t,
                attention_mode=attention_mode, ffn_chunks=ffn_chunks,
                video_attn_scale=video_attn_scale,
            )
            return {"result": result}
        except Exception:
            print("[RSLTXVUpscale] Error during upscale, cleaning up VRAM")
            raise
        finally:
            self._free_vram()

    def _upscale_impl(
        self,
        model, positive, negative, vae, images,
        upscale_model,
        frame_rate, audio, audio_vae, temporal_upscale_model,
        upscale_lora, upscale_lora_strength,
        upscale_steps, upscale_cfg, upscale_denoise,
        max_shift, base_shift,
        decode, tile_t,
        attention_mode, ffn_chunks, video_attn_scale,
    ):
        device = mm.get_torch_device()
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        do_spatial = upscale_model is not None

        # ----------------------------------------------------------------
        # 1. ENCODE INPUT VIDEO TO LATENT
        # ----------------------------------------------------------------

        print("[RSLTXVUpscale] Encoding input video to latent space")
        latent = vae.encode_tiled(images[:, :, :, :3],
                                  tile_x=512, tile_y=512, overlap=128)
        input_dtype = latent.dtype
        print(f"[RSLTXVUpscale] Input latent shape: {list(latent.shape)}")

        # ----------------------------------------------------------------
        # 2. TEMPORAL UPSCALE (optional — 2x frame count at input resolution)
        # ----------------------------------------------------------------

        output_frame_rate = frame_rate
        if temporal_upscale_model is not None:
            print("[RSLTXVUpscale] Temporal upscaling latent 2x")
            self._free_vram()

            temporal_dtype = next(temporal_upscale_model.parameters()).dtype
            temporal_upscale_model.to(device)
            try:
                t_latents = latent.to(dtype=temporal_dtype, device=device)
                t_latents = vae.first_stage_model.per_channel_statistics.un_normalize(t_latents)
                t_upsampled = temporal_upscale_model(t_latents)
            finally:
                temporal_upscale_model.cpu()

            t_upsampled = vae.first_stage_model.per_channel_statistics.normalize(t_upsampled)
            latent = t_upsampled.to(dtype=input_dtype, device=mm.intermediate_device())

            output_frame_rate = frame_rate * 2
            print(f"[RSLTXVUpscale] After temporal upscale: latent shape {list(latent.shape)}, output fps={output_frame_rate:.1f}")

        # ----------------------------------------------------------------
        # 3. SPATIAL UPSCALE LATENT 2x (optional)
        # ----------------------------------------------------------------

        if do_spatial:
            print("[RSLTXVUpscale] Spatial upscaling latent 2x")
            self._free_vram()

            model_dtype = next(upscale_model.parameters()).dtype

            memory_required = mm.module_size(upscale_model)
            memory_required += math.prod(latent.shape) * 3000.0
            mm.free_memory(memory_required, device)

            try:
                upscale_model.to(device)
                up_latents = latent.to(dtype=model_dtype, device=device)
                up_latents = vae.first_stage_model.per_channel_statistics.un_normalize(up_latents)
                upsampled = upscale_model(up_latents)
            finally:
                upscale_model.cpu()

            upsampled = vae.first_stage_model.per_channel_statistics.normalize(upsampled)
            latent = upsampled.to(dtype=input_dtype, device=mm.intermediate_device())
            print(f"[RSLTXVUpscale] Upscaled latent shape: {list(latent.shape)}")
            del up_latents, upsampled

        # ----------------------------------------------------------------
        # 4–6. RE-DIFFUSE (skip for temporal-only)
        # ----------------------------------------------------------------

        video_latent = latent
        audio_latent_out = None
        audio_is_input = False
        do_rediffuse = do_spatial or upscale_denoise > 0

        if do_rediffuse:
            # 4. PREPARE FIRST FRAME + INJECT AT POSITION 0
            _, height_sf, width_sf = vae.downscale_index_formula
            _, _, _, lh, lw = latent.shape
            encode_w, encode_h = lw * width_sf, lh * height_sf

            guide_frame = images[:1]

            # When spatially upscaling, pixel-upscale the guide frame to 2x
            if do_spatial:
                src_h, src_w = guide_frame.shape[1], guide_frame.shape[2]
                guide_frame = comfy.utils.common_upscale(
                    guide_frame.movedim(-1, 1), src_w * 2, src_h * 2, "bilinear", "center"
                ).movedim(1, -1)

            if guide_frame.shape[1] != encode_h or guide_frame.shape[2] != encode_w:
                guide_frame = comfy.utils.common_upscale(
                    guide_frame.movedim(-1, 1), encode_w, encode_h, "bilinear", "center"
                ).movedim(1, -1)

            first_encoded = vae.encode(guide_frame[:, :, :, :3])
            latent = latent.clone()
            latent[:, :, :first_encoded.shape[2]] = first_encoded

            # Denoise mask: preserve first frame (0.0), denoise rest (1.0)
            denoise_mask = torch.ones(
                (latent.shape[0], 1, latent.shape[2], 1, 1),
                dtype=latent.dtype, device=latent.device,
            )
            denoise_mask[:, :, :first_encoded.shape[2]] = 0.0
            print(f"[RSLTXVUpscale] First frame injected ({encode_w}x{encode_h})")

            # 5. AUDIO LATENTS (optional — requires audio_vae for AV model)
            has_audio = audio_vae is not None
            audio_is_input = has_audio and audio is not None
            if has_audio:
                if audio is not None:
                    # Encode provided audio — mask=0 to preserve
                    print("[RSLTXVUpscale] Encoding input audio latents")
                    audio_samples = audio_vae.encode(audio)
                    audio_noise_mask = torch.zeros_like(audio_samples)
                else:
                    # Empty audio latents — mask=1 to generate from scratch
                    # Use post-upscale video latent T to derive frame count
                    ref = latent.unbind()[0] if hasattr(latent, 'is_nested') and latent.is_nested else latent
                    num_frames = (ref.shape[2] - 1) * 8 + 1
                    print(f"[RSLTXVUpscale] Creating empty audio latents for generation ({num_frames} frames at {output_frame_rate}fps)")
                    audio_samples = torch.zeros(
                        (1, audio_vae.latent_channels,
                         audio_vae.num_of_latents_from_frames(num_frames, output_frame_rate),
                         audio_vae.latent_frequency_bins),
                        device=mm.intermediate_device(),
                    )
                    audio_noise_mask = torch.ones_like(audio_samples)

                latent = comfy.nested_tensor.NestedTensor((latent, audio_samples))
                denoise_mask = comfy.nested_tensor.NestedTensor((denoise_mask, audio_noise_mask))

            # 6. RE-DIFFUSE
            print(f"[RSLTXVUpscale] Re-diffusing: {upscale_steps} steps, cfg={upscale_cfg}, denoise={upscale_denoise}")
            self._free_vram()

            positive = node_helpers.conditioning_set_values(positive, {"frame_rate": output_frame_rate})
            negative = node_helpers.conditioning_set_values(negative, {"frame_rate": output_frame_rate})

            m = model.clone()

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

            if ffn_chunks > 0:
                self._apply_ffn_chunking(m, ffn_chunks)

            if upscale_lora and upscale_lora != "none" and upscale_lora_strength != 0:
                lora_path = folder_paths.get_full_path_or_raise("loras", upscale_lora)
                lora = None
                if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                if lora is None:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    self.loaded_lora = (lora_path, lora)
                m, _ = comfy.sd.load_lora_for_models(m, None, lora, upscale_lora_strength, 0)
                print(f"[RSLTXVUpscale] Applied upscale LoRA: {upscale_lora} (strength={upscale_lora_strength})")

            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST

            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass

            # Token count from the upscaled video latent (latent may be NestedTensor if audio was added)
            ref_latent = latent.unbind()[0] if hasattr(latent, 'is_nested') and latent.is_nested else latent
            up_tokens = math.prod(ref_latent.shape[2:])
            x1, x2 = 1024, 4096
            mm_shift = (max_shift - base_shift) / (x2 - x1)
            b = base_shift - mm_shift * x1
            shift = up_tokens * mm_shift + b
            print(f"[RSLTXVUpscale] Shift: tokens={up_tokens}, shift={shift:.3f}")

            model_sampling_obj = ModelSamplingAdvanced(m.model.model_config)
            model_sampling_obj.set_parameters(shift=shift)
            m.add_object_patch("model_sampling", model_sampling_obj)

            exp_shift = math.exp(shift)
            t = torch.linspace(1.0, 0.0, upscale_steps + 1, dtype=torch.float64)
            non_zero = t != 0
            inv_t_m1 = torch.where(non_zero, 1.0 / t - 1.0, torch.zeros_like(t))
            omz = torch.where(non_zero, inv_t_m1 / (exp_shift + inv_t_m1), torch.ones_like(t))
            nz_omz = omz[non_zero]
            sf = nz_omz[-1] / (1.0 - 0.1)
            omz[non_zero] = nz_omz / sf
            sigmas = torch.where(non_zero, 1.0 - omz, torch.zeros_like(omz)).float()

            start_step = int((1.0 - upscale_denoise) * upscale_steps)
            sigmas = sigmas[start_step:]

            up_guider = comfy.samplers.CFGGuider(m)
            up_guider.set_conds(positive, negative)
            up_guider.set_cfg(upscale_cfg)

            up_latent = comfy.sample.fix_empty_latent_channels(up_guider.model_patcher, latent)
            up_noise = comfy.sample.prepare_noise(up_latent, 0)
            up_sampler = comfy.samplers.sampler_object("euler_ancestral")
            callback = latent_preview.prepare_callback(up_guider.model_patcher, sigmas.shape[-1] - 1)

            print("[RSLTXVUpscale] Sampling...")
            samples = up_guider.sample(
                up_noise, up_latent, up_sampler, sigmas,
                denoise_mask=denoise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=0,
            )
            samples = samples.to(mm.intermediate_device())

            if samples.is_nested:
                parts = samples.unbind()
                video_latent = parts[0]
                audio_latent_out = parts[1] if len(parts) > 1 else None
            else:
                video_latent = samples

            del m
            self._free_vram()
        else:
            print("[RSLTXVUpscale] Temporal-only mode, skipping re-diffusion")

        output_latent = {"samples": video_latent}

        # ----------------------------------------------------------------
        # 7. DECODE (optional)
        # ----------------------------------------------------------------

        if decode:
            print("[RSLTXVUpscale] Decoding latents to images (tiled)")
            compression = vae.spacial_compression_decode()
            tile_size = 512 // compression
            overlap = 128 // compression
            temporal_compression = vae.temporal_compression_decode()
            if tile_t > 0:
                decode_tile_t = tile_t
                decode_overlap_t = max(2, decode_tile_t // 2)
            elif temporal_compression is not None:
                decode_tile_t = max(2, 64 // temporal_compression)
                decode_overlap_t = max(2, decode_tile_t // 2)
            else:
                decode_tile_t = None
                decode_overlap_t = None

            dec_samples = video_latent + torch.randn_like(video_latent) * 1e-5

            out_images = vae.decode_tiled(
                dec_samples,
                tile_x=tile_size, tile_y=tile_size, overlap=overlap,
                tile_t=decode_tile_t, overlap_t=decode_overlap_t,
            )
            if len(out_images.shape) == 5:
                out_images = out_images.reshape(-1, out_images.shape[-3], out_images.shape[-2], out_images.shape[-1])
        else:
            out_images = torch.zeros(1, 64, 64, 3)

        # ----------------------------------------------------------------
        # 8. AUDIO OUTPUT
        # ----------------------------------------------------------------

        audio_output = None
        if audio_latent_out is not None and audio_vae is not None:
            if audio_is_input:
                # Pass through original audio — skip VAE decode roundtrip
                print("[RSLTXVUpscale] Audio passthrough (input audio preserved)")
                audio_output = audio
            else:
                # Decode generated audio
                print("[RSLTXVUpscale] Decoding generated audio latents")
                decoded_audio = audio_vae.decode(audio_latent_out).to(audio_latent_out.device)
                audio_output = {
                    "waveform": decoded_audio,
                    "sample_rate": int(audio_vae.output_sample_rate),
                }
        elif audio is not None:
            audio_output = audio

        print("[RSLTXVUpscale] Done")
        return (output_latent, out_images, audio_output)

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
        """Apply FFN chunking to reduce VRAM usage."""
        try:
            blocks = model_clone.model.diffusion_model.transformer_blocks
        except AttributeError:
            print("[RSLTXVUpscale] Warning: Could not find transformer_blocks for FFN chunking")
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
