import gc
import random
import uuid

import torch
import comfy.model_management as mm
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview


SYSTEM_PROMPTS = {
    "superior": (
        "You are an assistant designed to generate superior images with the superior "
        "degree of image-text alignment based on textual prompts or user prompts."
    ),
    "alignment": (
        "You are an assistant designed to generate high-quality images with the "
        "highest degree of image-text alignment based on textual prompts."
    ),
}


class RSZImageGenerate:
    """
    All-in-one Z-Image Turbo generation node. Handles text encoding (Qwen3-4B
    with system prompt), RenormCFG, scheduling, sampling, and VAE decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":  ("MODEL",),
                "clip":   ("CLIP",),
                "vae":    ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "width":  ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "steps":  ("INT", {"default": 10,   "min": 1,  "max": 10000}),
                "seed":   ("INT", {"default": 0,    "min": 0,  "max": 0xffffffffffffffff}),
            },
            "optional": {
                "system_prompt":   (list(SYSTEM_PROMPTS.keys()),),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "cfg":             ("FLOAT", {"default": 1.5,   "min": 0.0,  "max": 100.0, "step": 0.1}),
                "cfg_trunc":       ("FLOAT", {"default": 100.0, "min": 0.0,  "max": 100.0, "step": 0.01,
                                              "tooltip": "Timestep below which CFG is applied. 100=always"}),
                "renorm_cfg":      ("FLOAT", {"default": 1.0,   "min": 0.0,  "max": 100.0, "step": 0.01,
                                              "tooltip": "Renormalize CFG output to limit norm growth. 0=off"}),
                "denoise":         ("FLOAT", {"default": 1.0,   "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "seed_mode":       (["random", "fixed", "increment", "decrement"],),
                "lora":            (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_strength":   ("FLOAT", {"default": 1.0,   "min": -10.0, "max": 10.0, "step": 0.01}),
                "attention_mode":  (["auto", "default", "sage"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_NODE  = True
    FUNCTION     = "generate"
    CATEGORY     = "rs-nodes"

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
        steps=4,
        seed=0,
        # Optional
        system_prompt="superior",
        negative_prompt="",
        cfg=1.5,
        cfg_trunc=100.0,
        renorm_cfg=1.0,
        denoise=1.0,
        seed_mode="random",
        lora="none",
        lora_strength=1.0,
        attention_mode="auto",
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
        print(f"[RSZImageGenerate] Starting generation (seed={actual_seed}, mode={seed_mode})")

        try:
            images = self._generate_impl(
                model, clip, vae, prompt,
                width=width, height=height, steps=steps, seed=actual_seed,
                system_prompt=system_prompt,
                negative_prompt=negative_prompt or "",
                cfg=cfg, cfg_trunc=cfg_trunc, renorm_cfg=renorm_cfg,
                denoise=denoise,
                lora=lora, lora_strength=lora_strength,
                attention_mode=attention_mode,
            )
            return {"ui": {"seed": [actual_seed]}, "result": (images,)}
        except Exception:
            print("[RSZImageGenerate] Error during generation, cleaning up VRAM")
            raise
        finally:
            self._free_vram()

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _generate_impl(
        self,
        model, clip, vae, prompt,
        width, height, steps, seed,
        system_prompt, negative_prompt,
        cfg, cfg_trunc, renorm_cfg, denoise,
        lora, lora_strength, attention_mode,
    ):
        # ----------------------------------------------------------------
        # 1. TEXT ENCODE
        # ----------------------------------------------------------------

        full_prompt = f"{SYSTEM_PROMPTS[system_prompt]} <Prompt Start> {prompt}"
        tokens = clip.tokenize(full_prompt)
        positive = clip.encode_from_tokens_scheduled(tokens)

        if negative_prompt:
            full_neg = f"{SYSTEM_PROMPTS[system_prompt]} <Prompt Start> {negative_prompt}"
            neg_tokens = clip.tokenize(full_neg)
            negative = clip.encode_from_tokens_scheduled(neg_tokens)
        else:
            neg_tokens = clip.tokenize("")
            negative = clip.encode_from_tokens_scheduled(neg_tokens)

        # ----------------------------------------------------------------
        # 2. MODEL SETUP
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
            print(f"[RSZImageGenerate] Applied LoRA: {lora} (strength={lora_strength})")

        # RenormCFG
        in_channels = m.model.diffusion_model.in_channels
        _cfg_trunc = cfg_trunc
        _renorm_cfg = renorm_cfg

        def renorm_cfg_func(args):
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            x_orig = args["input"]

            if timestep[0] < _cfg_trunc:
                cond_eps = cond_denoised[:, :in_channels]
                uncond_eps = uncond_denoised[:, :in_channels]
                cond_rest = cond_denoised[:, in_channels:]
                half_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
                half_rest = cond_rest

                if float(_renorm_cfg) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(
                        cond_eps, dim=tuple(range(1, len(cond_eps.shape))), keepdim=True
                    )
                    max_new_norm = ori_pos_norm * float(_renorm_cfg)
                    new_pos_norm = torch.linalg.vector_norm(
                        half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True
                    )
                    if new_pos_norm >= max_new_norm:
                        half_eps = half_eps * (max_new_norm / new_pos_norm)
            else:
                cond_eps = cond_denoised[:, :in_channels]
                cond_rest = cond_denoised[:, in_channels:]
                half_eps = cond_eps
                half_rest = cond_rest

            cfg_result = torch.cat([half_eps, half_rest], dim=1)
            return x_orig - cfg_result

        m.set_model_sampler_cfg_function(renorm_cfg_func)

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

        # ----------------------------------------------------------------
        # 3. EMPTY LATENT
        # ----------------------------------------------------------------

        latent = torch.zeros(
            [1, 16, height // 8, width // 8],
            device=mm.intermediate_device(),
        )

        # ----------------------------------------------------------------
        # 4. SCHEDULE (normal scheduler, model has shift=3.0)
        # ----------------------------------------------------------------

        sigmas = comfy.samplers.normal_scheduler(m.model.model_sampling, steps)

        if denoise < 1.0:
            start_step = int((1.0 - denoise) * steps)
            sigmas = sigmas[start_step:]

        print(f"[RSZImageGenerate] Schedule: {len(sigmas)-1} steps, denoise={denoise}")

        # ----------------------------------------------------------------
        # 5. SAMPLE
        # ----------------------------------------------------------------

        guider = comfy.samplers.CFGGuider(m)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        sampler = comfy.samplers.sampler_object("euler")

        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent)
        noise = comfy.sample.prepare_noise(latent_image, seed)

        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        print("[RSZImageGenerate] Sampling...")
        samples = guider.sample(
            noise, latent_image, sampler, sigmas,
            denoise_mask=None,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        samples = samples.to(mm.intermediate_device())

        # Free model clones before decode
        del m, guider
        self._free_vram()

        # ----------------------------------------------------------------
        # 6. VAE DECODE
        # ----------------------------------------------------------------

        print("[RSZImageGenerate] Decoding latents...")
        images = vae.decode(samples)

        print("[RSZImageGenerate] Done")
        return images

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _free_vram(self):
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()
