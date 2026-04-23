"""IC-LoRA guider for LTXV structural control (canny/depth/pose/motion).

Loads an IC-LoRA, preprocesses control images, and produces an ICLoRAGuider
that plugs into RSLTXVGenerate's guider input.

Follows the official LTXVAddGuide approach: guide encoding and keyframe
injection are deferred to sample() time (inside ICLoRAGuider) where the
actual video latent dimensions are known. This ensures guide latent spatial
dims always match the video latent regardless of resolution or upscale.
"""

import logging

import torch
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers

logger = logging.getLogger(__name__)


class RSLTXVICLoRAGuider:
    """Creates an IC-LoRA guider for LTXV structural control.

    Loads an IC-LoRA (union or individual), preprocesses control images,
    and returns an ICLoRAGuider that handles guide frame injection during
    sampling using the official LTXVAddGuide keyframe approach.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":     ("MODEL",),
                "positive":  ("CONDITIONING",),
                "negative":  ("CONDITIONING",),
                "vae":       ("VAE",),
                "control_image": ("IMAGE",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                "lora_strength":     ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "scene_embed": (["none"] + folder_paths.get_filename_list("loras"),
                                {"default": "none",
                                 "tooltip": "Optional scene / prompt embedding tensor (.safetensors) shipped alongside the LoRA. Concatenated to the positive conditioning to reliably trigger the LoRA's effect (e.g. Lightricks HDR LoRA's LogC scene embed)."}),
                "scene_embed_strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01,
                                                    "tooltip": "Multiplier applied to the scene embed tensor before concat. 1.0 = as released; <1.0 = softer effect; >1.0 = stronger."}),
                "control_strength":  ("FLOAT", {"default": 1.0, "min": 0.0,   "max": 1.0,  "step": 0.01}),
                "guide_frame_idx":   ("INT",   {"default": 0,   "min": -1,    "max": 10000}),
                "crf":               ("INT",   {"default": 35,  "min": 0,     "max": 100,
                                                "tooltip": "CRF compression to match IC-LoRA training distribution. 0 = disabled."}),
                # Guidance
                "cfg":               ("FLOAT", {"default": 1.0, "min": 0.0,   "max": 100.0, "step": 0.1}),
                "audio_cfg":         ("FLOAT", {"default": 1.0, "min": 0.0,   "max": 100.0, "step": 0.1}),
                "stg_scale":         ("FLOAT", {"default": 0.0, "min": 0.0,   "max": 10.0,  "step": 0.1}),
                "stg_perturbation":  ("FLOAT", {"default": 1.0, "min": 0.0,   "max": 1.0,  "step": 0.01,
                                                "tooltip": "Attention perturbation strength. 1.0 = full skip (original STG), <1.0 = soft blend. Lower values preserve guided subjects while still enhancing fine detail."}),
                "stg_blocks":        ("STRING", {"default": "29"}),
                "stg_indexes":       ("STRING", {"default": "0",
                                                 "tooltip": "Attention index to perturb. 0 = self-attention (default STG), 1 = cross-attention (preserves guide frame influence)."}),
                "rescale":           ("FLOAT", {"default": 0.0, "min": 0.0,   "max": 1.0,   "step": 0.01}),
                "cfg_end":           ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
                "stg_end":           ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0,  "step": 0.1}),
                "cfg_star_rescale":  ("BOOLEAN", {"default": False,
                                                  "tooltip": "CFG-Zero*: project negative onto positive."}),
                "skip_sigma":        ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                                "tooltip": "Skip guidance when sigma > this. 0 = disabled."}),
                # Scheduling
                "max_shift":         ("FLOAT", {"default": 2.2,  "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift":        ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
                # Misc
                "frame_rate":        ("FLOAT", {"default": 25.0, "min": 0.0,  "max": 1000.0, "step": 0.01}),
                "attention_mode":    (["auto", "default", "sage"],),
                "upscale":           ("BOOLEAN", {"default": False}),
                "width":             ("INT",   {"default": 768, "min": 64,    "max": 8192, "step": 32}),
                "height":            ("INT",   {"default": 512, "min": 64,    "max": 8192, "step": 32}),
                # Audio/modality
                "audio_stg_scale":   ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0, "step": 0.1,
                                                "tooltip": "Audio STG scale (-1 = use video stg_scale)"}),
                "video_modality_scale": ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 100.0, "step": 0.1,
                                                   "tooltip": "Video modality isolation (0 = match official default)"}),
                "audio_modality_scale": ("FLOAT", {"default": 3.0,  "min": 0.0, "max": 100.0, "step": 0.1,
                                                   "tooltip": "Audio modality isolation (3 = match official default)"}),
                "video_attn_scale":  ("FLOAT", {"default": 1.03, "min": 0.0, "max": 10.0, "step": 0.01,
                                                "tooltip": "Video attention scale (1.03 recommended)"}),
                "sampler":           ("SAMPLER", {"tooltip": "Sampler for IC-LoRA guidance passes. Default: euler."}),
                "rediffusion_passes": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                                               "tooltip": "Number of half-res re-diffusion passes (stage 2)."}),
                "distilled_lora":    (["none"] + folder_paths.get_filename_list("loras"),
                                      {"tooltip": "Distilled LoRA to apply when running distilled sigmas (cfg=1.0)."}),
                "distilled_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("GUIDER", "INT", "INT")
    RETURN_NAMES = ("guider", "width", "height")
    FUNCTION = "create_guider"
    CATEGORY = "rs-nodes"

    def create_guider(
        self, model, positive, negative, vae, control_image, lora_name,
        lora_strength=1.0, control_strength=1.0, guide_frame_idx=0, crf=35,
        cfg=3.0, audio_cfg=7.0, stg_scale=0.0, stg_perturbation=1.0, stg_blocks="29", stg_indexes="0",
        rescale=0.7, cfg_end=-1.0, stg_end=-1.0,
        cfg_star_rescale=True, skip_sigma=0.0,
        max_shift=2.2, base_shift=0.95,
        frame_rate=25.0, attention_mode="auto",
        upscale=False, width=768, height=512,
        audio_stg_scale=-1.0, video_modality_scale=0.0,
        audio_modality_scale=3.0, video_attn_scale=1.03,
        sampler=None, rediffusion_passes=1,
        distilled_lora="none", distilled_lora_strength=1.0,
        scene_embed="none", scene_embed_strength=1.0,
    ):
        from ..utils.multimodal_guider import ICLoRAGuider

        # --- Load IC-LoRA and extract downscale factor ---
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        downscale_factor = self._read_downscale_factor(lora_path)
        logger.info(f"Loading IC-LoRA: {lora_name} (downscale_factor={downscale_factor})")

        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)

        m = model.clone()
        if lora_strength != 0:
            m, _ = comfy.sd.load_lora_for_models(m, None, lora_data, lora_strength, 0)
            logger.info(f"Applied LoRA (strength={lora_strength})")

        # --- Attention override ---
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
            m.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = (
                lambda func, *args, **kwargs: attn_func(*args, **kwargs)
            )

        # --- Output dimensions (for upscale) ---
        out_width, out_height = width, height

        # --- CRF preprocessing (matches IC-LoRA training distribution) ---
        from comfy_extras.nodes_lt import preprocess as ltxv_preprocess

        img = control_image
        if crf > 0:
            processed_frames = []
            for i in range(img.shape[0]):
                processed_frames.append(ltxv_preprocess(img[i], crf))
            img = torch.stack(processed_frames)
            logger.info(f"CRF preprocessed {img.shape[0]} control frame(s) (crf={crf})")

        # --- Optional scene / prompt embedding concat ---
        # Some LoRAs (e.g. Lightricks HDR) ship a pre-computed conditioning
        # tensor that reliably triggers the LoRA's effect. Concatenate it
        # to each positive conditioning along the sequence dimension so
        # it rides along with the user's prompt tokens. Negative is left
        # untouched — the embed is a "what we want" signal, not "what we
        # don't want".
        if scene_embed and scene_embed != "none" and scene_embed_strength != 0:
            embed_path = folder_paths.get_full_path_or_raise("loras", scene_embed)
            embed_data = comfy.utils.load_torch_file(embed_path, safe_load=True)
            if isinstance(embed_data, dict):
                # Prefer conventional keys, else fall back to the first tensor in the file.
                embed_tensor = None
                for _key in (
                    "scene_embed", "scene_embedding",
                    "prompt_embed", "prompt_embedding",
                    "conditioning", "embedding",
                ):
                    if _key in embed_data and torch.is_tensor(embed_data[_key]):
                        embed_tensor = embed_data[_key]
                        logger.info(f"Scene embed: using key '{_key}' from {scene_embed}")
                        break
                if embed_tensor is None:
                    for _k, _v in embed_data.items():
                        if torch.is_tensor(_v):
                            embed_tensor = _v
                            logger.info(f"Scene embed: using first tensor key '{_k}' from {scene_embed}")
                            break
            else:
                embed_tensor = embed_data
            if embed_tensor is not None:
                if embed_tensor.dim() == 2:
                    embed_tensor = embed_tensor.unsqueeze(0)
                elif embed_tensor.dim() == 1:
                    embed_tensor = embed_tensor.unsqueeze(0).unsqueeze(0)
                if scene_embed_strength != 1.0:
                    embed_tensor = embed_tensor * scene_embed_strength
                new_positive = []
                for entry in positive:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        cond_t, meta = entry[0], entry[1]
                    else:
                        cond_t, meta = entry, {}
                    e = embed_tensor.to(cond_t.device, cond_t.dtype)
                    if e.shape[0] != cond_t.shape[0]:
                        e = e.expand(cond_t.shape[0], *e.shape[1:])
                    combined = torch.cat([cond_t, e], dim=1)
                    new_positive.append([combined, meta])
                positive = new_positive
                logger.info(
                    f"Scene embed concatenated to positive conditioning "
                    f"(shape={tuple(embed_tensor.shape)}, strength={scene_embed_strength})"
                )
            else:
                logger.warning(f"Scene embed: no tensor found in {scene_embed}, skipping")

        # --- Stamp frame rate on conditioning ---
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        # --- Build control_info for upscale re-encoding ---
        control_info = {
            "control_image": control_image,
            "downscale_factor": downscale_factor,
            "control_strength": control_strength,
            "guide_frame_idx": guide_frame_idx,
            "crf": crf,
            "lora_name": lora_name,
            "lora_strength": lora_strength,
            "attention_mode": attention_mode,
            "cfg": cfg,
            "audio_cfg": audio_cfg,
            "stg_scale": stg_scale,
            "stg_perturbation": stg_perturbation,
            "stg_blocks": stg_blocks,
            "stg_indexes": stg_indexes,
            "rescale": rescale,
            "cfg_end": cfg_end,
            "stg_end": stg_end,
            "cfg_star_rescale": cfg_star_rescale,
            "skip_sigma": skip_sigma,
            "max_shift": max_shift,
            "base_shift": base_shift,
            "frame_rate": frame_rate,
            "audio_stg_scale": audio_stg_scale,
            "video_modality_scale": video_modality_scale,
            "audio_modality_scale": audio_modality_scale,
            "video_attn_scale": video_attn_scale,
            "_ic_lora_sampler": sampler,
            "_rediffusion_passes": rediffusion_passes,
            "_distilled_lora": distilled_lora,
            "_distilled_lora_strength": distilled_lora_strength,
        }

        # --- Create ICLoRAGuider (deferred encoding at sample() time) ---
        guider = ICLoRAGuider(
            m, positive, negative,
            control_pixels=img,
            vae=vae,
            downscale_factor=downscale_factor,
            guide_strength=control_strength,
            guide_frame_idx=guide_frame_idx,
            max_shift=max_shift,
            base_shift=base_shift,
            video_cfg=cfg,
            audio_cfg=audio_cfg,
            stg_scale=stg_scale,
            stg_perturbation=stg_perturbation,
            stg_blocks=[int(s.strip()) for s in stg_blocks.split(",")],
            stg_indexes=[int(s.strip()) for s in stg_indexes.split(",")],
            rescale=rescale,
            video_cfg_end=cfg_end if cfg_end >= 0 else None,
            stg_scale_end=stg_end if stg_end >= 0 else None,
            cfg_star_rescale=cfg_star_rescale,
            skip_sigma_threshold=skip_sigma,
            audio_stg_scale=audio_stg_scale if audio_stg_scale >= 0 else None,
            video_modality_scale=video_modality_scale,
            audio_modality_scale=audio_modality_scale,
            video_attn_scale=video_attn_scale,
        )

        # Attach control_info for upscale rebuild
        guider.control_info = control_info
        guider.ic_lora_sampler = sampler

        # Load distilled LoRA data for use when running distilled sigmas
        if distilled_lora and distilled_lora != "none" and distilled_lora_strength != 0:
            dl_path = folder_paths.get_full_path_or_raise("loras", distilled_lora)
            dl_data = comfy.utils.load_torch_file(dl_path, safe_load=True)
            guider._distilled_lora = (dl_data, distilled_lora_strength, distilled_lora)
            logger.info(f"Distilled LoRA loaded: {distilled_lora} (strength={distilled_lora_strength})")

        logger.info(f"Guider created (cfg={cfg}, stg={stg_scale}, rescale={rescale})")
        return (guider, out_width, out_height)

    @staticmethod
    def _read_downscale_factor(lora_path):
        """Read reference_downscale_factor from LoRA safetensors metadata."""
        try:
            from safetensors import safe_open
            with safe_open(lora_path, framework="pt") as f:
                metadata = f.metadata() or {}
                return int(metadata.get("reference_downscale_factor", 1))
        except Exception as e:
            logger.warning(f"Could not read LoRA metadata ({e}), defaulting downscale_factor=1")
            return 1
