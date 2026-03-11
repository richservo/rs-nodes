import gc
import math
import random
import uuid

import torch
import comfy.model_management as mm
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
import latent_preview


PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def generalized_time_snr_shift(t, mu: float, sigma: float):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class RSFlux2Generate:
    """
    All-in-one Flux2.dev generation node. Handles text encoding, reference images,
    scheduling, sampling, and VAE decode internally.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("MODEL",),
                "clip":     ("CLIP",),
                "vae":      ("VAE",),
                "prompt":   ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width":    ("INT",   {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "height":   ("INT",   {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "steps":    ("INT",   {"default": 50,   "min": 1,  "max": 10000}),
                "guidance": ("FLOAT", {"default": 4.0,  "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed":     ("INT",   {"default": 0,    "min": 0,   "max": 0xffffffffffffffff}),
            },
            "optional": {
                "ref_image_1":    ("IMAGE",),
                "ref_image_2":    ("IMAGE",),
                "ref_image_3":    ("IMAGE",),
                "ref_image_4":    ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "cfg":            ("FLOAT", {"default": 1.0,  "min": 0.0,   "max": 100.0, "step": 0.1}),
                "denoise":        ("FLOAT", {"default": 1.0,  "min": 0.0,   "max": 1.0,   "step": 0.01}),
                "seed_mode":      (["random", "fixed", "increment", "decrement"],),
                "lora":           (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_strength":  ("FLOAT", {"default": 1.0,  "min": -10.0, "max": 10.0,  "step": 0.01}),
                "attention_mode": (["auto", "default", "sage"],),
                "ffn_chunks":     ("INT",   {"default": 4,    "min": 0,     "max": 16,    "step": 1}),
                "latent_upscale": (["off", "on"],),
                "upscale_denoise": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "Denoise strength for the second pass at full resolution"}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("images",)
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
        clip,
        vae,
        prompt,
        width=1024,
        height=1024,
        steps=50,
        guidance=4.0,
        seed=0,
        # Optional
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
        ref_image_4=None,
        negative_prompt="",
        cfg=1.0,
        denoise=1.0,
        seed_mode="random",
        lora="none",
        lora_strength=1.0,
        attention_mode="auto",
        ffn_chunks=4,
        latent_upscale="off",
        upscale_denoise=0.45,
        **kwargs,
    ):
        # Resolve seed
        actual_seed = seed
        if seed_mode == "random":
            actual_seed = random.randint(0, 0xffffffffffffffff)
        elif seed_mode == "increment" and self._last_seed is not None:
            actual_seed = (self._last_seed + 1) % (0xffffffffffffffff + 1)
        elif seed_mode == "decrement" and self._last_seed is not None:
            actual_seed = (self._last_seed - 1) % (0xffffffffffffffff + 1)
        self._last_seed = actual_seed
        print(f"[RSFlux2Generate] Starting generation (seed={actual_seed}, mode={seed_mode})")

        try:
            images = self._generate_impl(
                model, clip, vae, prompt,
                width=width, height=height, steps=steps,
                guidance=guidance, seed=actual_seed,
                ref_image_1=ref_image_1, ref_image_2=ref_image_2,
                ref_image_3=ref_image_3, ref_image_4=ref_image_4,
                negative_prompt=negative_prompt or "",
                cfg=cfg, denoise=denoise,
                lora=lora, lora_strength=lora_strength,
                attention_mode=attention_mode, ffn_chunks=ffn_chunks,
                latent_upscale=latent_upscale, upscale_denoise=upscale_denoise,
            )
            return {"ui": {"seed": [actual_seed]}, "result": (images,)}
        except Exception:
            print("[RSFlux2Generate] Error during generation, cleaning up VRAM")
            raise
        finally:
            self._free_vram()

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _generate_impl(
        self,
        model, clip, vae, prompt,
        width, height, steps, guidance, seed,
        ref_image_1, ref_image_2, ref_image_3, ref_image_4,
        negative_prompt, cfg, denoise,
        lora, lora_strength,
        attention_mode, ffn_chunks,
        latent_upscale="off", upscale_denoise=0.45,
    ):
        # ----------------------------------------------------------------
        # 1. TEXT ENCODE
        # ----------------------------------------------------------------

        tokens = clip.tokenize(prompt)
        positive = clip.encode_from_tokens_scheduled(
            tokens, add_dict={"guidance": guidance}
        )

        neg_tokens = clip.tokenize(negative_prompt)
        negative = clip.encode_from_tokens_scheduled(
            neg_tokens, add_dict={"guidance": guidance}
        )

        # ----------------------------------------------------------------
        # 2. REFERENCE IMAGES
        # ----------------------------------------------------------------

        ref_images = [
            img for img in [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
            if img is not None
        ]
        if ref_images:
            for i, ref_img in enumerate(ref_images):
                img_w = ref_img.shape[2]
                img_h = ref_img.shape[1]
                aspect_ratio = img_w / img_h
                _, target_w, target_h = min(
                    (abs(aspect_ratio - w / h), w, h)
                    for w, h in PREFERED_KONTEXT_RESOLUTIONS
                )
                scaled = comfy.utils.common_upscale(
                    ref_img.movedim(-1, 1), target_w, target_h, "lanczos", "center"
                ).movedim(1, -1)

                ref_latent = vae.encode(scaled[:, :, :, :3])
                print(f"[RSFlux2Generate] Reference image {i+1}: {img_w}x{img_h} → {target_w}x{target_h}")

                positive = node_helpers.conditioning_set_values(
                    positive, {"reference_latents": [ref_latent]}, append=True
                )

        # ----------------------------------------------------------------
        # 3. MODEL SETUP
        # ----------------------------------------------------------------

        m = model.clone()

        # LoRA
        if lora and lora != "none" and lora_strength != 0:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora)
            lora_data = None
            if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
                lora_data = self.loaded_lora[1]
            if lora_data is None:
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora_data)
            m, _ = comfy.sd.load_lora_for_models(m, None, lora_data, lora_strength, 0)
            print(f"[RSFlux2Generate] Applied LoRA: {lora} (strength={lora_strength})")

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
            m.model_options.setdefault("transformer_options", {})[
                "optimized_attention_override"
            ] = lambda func, *args, **kwargs: attn_func(*args, **kwargs)
            print("[RSFlux2Generate] Using SageAttention")
        else:
            print("[RSFlux2Generate] Using default attention")

        # FFN chunking (double blocks only — single blocks have fused MLP)
        if ffn_chunks > 0:
            self._apply_ffn_chunking(m, ffn_chunks)

        # ----------------------------------------------------------------
        # 4. FIRST PASS (half res if upscaling, otherwise full res)
        # ----------------------------------------------------------------

        if latent_upscale == "on":
            first_w = ((width // 2) // 16) * 16
            first_h = ((height // 2) // 16) * 16
            print(f"[RSFlux2Generate] Latent upscale: first pass {first_w}x{first_h} → {width}x{height}")
        else:
            first_w = width
            first_h = height

        samples = self._sample_pass(
            m, positive, negative, cfg, seed,
            first_w, first_h, steps, denoise,
        )

        # ----------------------------------------------------------------
        # 5. SECOND PASS (upscale latent + re-diffuse)
        # ----------------------------------------------------------------

        if latent_upscale == "on" and upscale_denoise > 0:
            # Bilinear upscale latent to full resolution
            samples = torch.nn.functional.interpolate(
                samples, size=(height // 8, width // 8),
                mode="bilinear", align_corners=False,
            )
            print(f"[RSFlux2Generate] Latent upscaled, re-diffusing at denoise={upscale_denoise}")

            samples = self._sample_pass(
                m, positive, negative, cfg, seed + 1,
                width, height, steps, upscale_denoise,
                init_latent=samples,
            )

        samples = samples.to(mm.intermediate_device())

        # Free model clones before decode
        del m
        self._free_vram()

        # ----------------------------------------------------------------
        # 6. VAE DECODE
        # ----------------------------------------------------------------

        print("[RSFlux2Generate] Decoding latents...")
        pixel_count = samples.shape[2] * 8 * samples.shape[3] * 8
        if pixel_count > 1024 * 1024:
            print(f"[RSFlux2Generate] Large image ({samples.shape[3]*8}x{samples.shape[2]*8}), using tiled decode")
            images = vae.decode_tiled(samples)
        else:
            images = vae.decode(samples)

        print("[RSFlux2Generate] Done")
        return images

    def _sample_pass(self, m, positive, negative, cfg, seed, width, height, steps, denoise, init_latent=None):
        """Run a single sampling pass. If init_latent is provided, use it instead of empty latent."""

        if init_latent is not None:
            latent = init_latent
        else:
            latent = torch.zeros(
                [1, 16, height // 8, width // 8],
                device=mm.intermediate_device(),
            )

        image_seq_len = (width * height) // (16 * 16)
        mu = compute_empirical_mu(image_seq_len, steps)

        sigmas = torch.linspace(1, 0, steps + 1)
        sigmas = generalized_time_snr_shift(sigmas, mu, 1.0)

        if denoise < 1.0:
            start_step = int((1.0 - denoise) * steps)
            sigmas = sigmas[start_step:]

        print(f"[RSFlux2Generate] Schedule: seq_len={image_seq_len}, mu={mu:.4f}, {len(sigmas)-1} steps")

        guider = comfy.samplers.CFGGuider(m)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        sampler = comfy.samplers.sampler_object("euler")

        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent)
        noise = comfy.sample.prepare_noise(latent_image, seed)

        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        print("[RSFlux2Generate] Sampling...")
        samples = guider.sample(
            noise, latent_image, sampler, sigmas,
            denoise_mask=None,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        del guider
        return samples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _free_vram(self):
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()

    @staticmethod
    def _apply_ffn_chunking(model_clone, num_chunks):
        """Apply FFN chunking to double_blocks' img_mlp and txt_mlp."""
        try:
            double_blocks = model_clone.model.diffusion_model.double_blocks
        except AttributeError:
            print("[RSFlux2Generate] Warning: Could not find double_blocks for FFN chunking")
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

        for idx in range(len(double_blocks)):
            block = double_blocks[idx]
            if hasattr(block, "img_mlp"):
                model_clone.add_object_patch(
                    f"diffusion_model.double_blocks.{idx}.img_mlp.forward",
                    make_chunked_forward(block.img_mlp, num_chunks),
                )
            if hasattr(block, "txt_mlp"):
                model_clone.add_object_patch(
                    f"diffusion_model.double_blocks.{idx}.txt_mlp.forward",
                    make_chunked_forward(block.txt_mlp, num_chunks),
                )
