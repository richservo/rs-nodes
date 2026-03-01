"""RSICLoRAGuider — IC-LoRA (In-Context LoRA) guider node for Union IC-LoRA
control (depth/canny/pose) with LTXV models.

Creates an ICLoRAGuider that plugs into RSLTXVGenerate's guider override
input. Handles LoRA loading, control image preprocessing, and model
sampling shift — all self-contained so the generate node needs no
modifications.

The actual VAE encoding and guide keyframe injection is deferred to the
guider's sample() method, where the target video latent dimensions are
known.
"""

import torch
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers


class RSICLoRAGuider:
    """IC-LoRA guider for Union IC-LoRA control (depth, canny, pose).

    Loads an IC-LoRA safetensors file, applies it to a model clone, prepares
    the control image, and creates an ICLoRAGuider that manages guide frame
    injection during sampling.

    The control image should be preprocessed BEFORE this node (e.g. via a
    depth/canny/pose preprocessor node). The Union LoRA auto-detects the
    control type from the input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("MODEL",),
                "positive":      ("CONDITIONING",),
                "negative":      ("CONDITIONING",),
                "vae":           ("VAE",),
                "control_image": ("IMAGE",),
                "ic_lora":       (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                # IC-LoRA
                "lora_strength":   ("FLOAT",  {"default": 1.0,  "min": 0.0,  "max": 2.0,   "step": 0.01}),
                # Control
                "guide_strength":  ("FLOAT",  {"default": 1.0,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "guide_frame_idx": ("INT",    {"default": 0,    "min": -1,   "max": 10000}),
                "crf":             ("INT",    {"default": 35,   "min": 0,    "max": 100}),
                # Guidance
                "video_cfg":       ("FLOAT",  {"default": 3.0,  "min": 0.0,  "max": 100.0, "step": 0.1}),
                "audio_cfg":       ("FLOAT",  {"default": 7.0,  "min": 0.0,  "max": 100.0, "step": 0.1}),
                "stg_scale":       ("FLOAT",  {"default": 0.0,  "min": 0.0,  "max": 10.0,  "step": 0.1}),
                "stg_blocks":      ("STRING", {"default": "29"}),
                "rescale":         ("FLOAT",  {"default": 0.7,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "modality_scale":  ("FLOAT",  {"default": 1.0,  "min": 0.0,  "max": 100.0, "step": 0.1}),
                "cfg_end":         ("FLOAT",  {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
                "stg_end":         ("FLOAT",  {"default": -1.0, "min": -1.0, "max": 10.0,  "step": 0.1}),
                # Scheduler
                "max_shift":       ("FLOAT",  {"default": 2.05, "min": 0.0,  "max": 100.0, "step": 0.01}),
                "base_shift":      ("FLOAT",  {"default": 0.95, "min": 0.0,  "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "rs-nodes"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def create_guider(
        self,
        model,
        positive,
        negative,
        vae,
        control_image,
        ic_lora,
        lora_strength=1.0,
        guide_strength=1.0,
        guide_frame_idx=0,
        crf=35,
        video_cfg=3.0,
        audio_cfg=7.0,
        stg_scale=0.0,
        stg_blocks="29",
        rescale=0.7,
        modality_scale=1.0,
        cfg_end=-1.0,
        stg_end=-1.0,
        max_shift=2.05,
        base_shift=0.95,
    ):
        # 1. Load IC-LoRA and read metadata
        lora_path = folder_paths.get_full_path_or_raise("loras", ic_lora)
        downscale_factor = self._read_downscale_factor(lora_path)
        print(f"[RSICLoRAGuider] Loading IC-LoRA: {ic_lora}")
        print(f"[RSICLoRAGuider] Reference downscale factor: {downscale_factor}")

        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # 2. Apply LoRA to a cloned model
        m = model.clone()
        m, _ = comfy.sd.load_lora_for_models(
            m, None, lora_data, lora_strength, 0
        )
        print(f"[RSICLoRAGuider] Applied IC-LoRA (strength={lora_strength})")

        # 3. Preprocess control image
        #    Encoding is DEFERRED to the guider's sample() method where the
        #    target video latent dimensions are known. Here we only do pixel-
        #    level preprocessing: downscale/upscale for half-res content and
        #    CRF compression.
        from comfy_extras.nodes_lt import preprocess as ltxv_preprocess

        img = control_image
        src_h, src_w = img.shape[1], img.shape[2]

        # Downscale then upscale to preserve half-res content characteristics
        # that the Union IC-LoRA was trained on
        if downscale_factor > 1:
            small_h = src_h // downscale_factor
            small_w = src_w // downscale_factor
            img = comfy.utils.common_upscale(
                img.movedim(-1, 1), small_w, small_h, "bilinear", "center"
            )
            img = comfy.utils.common_upscale(
                img, src_w, src_h, "bilinear", "center"
            ).movedim(1, -1)
            print(f"[RSICLoRAGuider] Control image: {src_w}x{src_h} -> "
                  f"downscaled to {small_w}x{small_h} -> "
                  f"upscaled back to {src_w}x{src_h}")

        # CRF preprocess each frame
        processed_frames = []
        for i in range(img.shape[0]):
            processed_frames.append(ltxv_preprocess(img[i], crf))
        processed = torch.stack(processed_frames)
        print(f"[RSICLoRAGuider] Preprocessed {processed.shape[0]} control frame(s) (crf={crf})")

        # 4. Stamp frame rate onto conditioning
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": 25.0})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": 25.0})

        # 5. Create ICLoRAGuider with deferred encoding
        from ..utils.multimodal_guider import ICLoRAGuider

        stg_blocks_list = [int(s.strip()) for s in stg_blocks.split(",")]

        guider = ICLoRAGuider(
            m, positive, negative,
            # IC-LoRA guide params (encoding deferred to sample())
            control_pixels=processed,
            vae=vae,
            guide_strength=guide_strength,
            guide_frame_idx=guide_frame_idx,
            # Model sampling
            max_shift=max_shift,
            base_shift=base_shift,
            # MultimodalGuider params
            video_cfg=video_cfg,
            audio_cfg=audio_cfg,
            stg_scale=stg_scale,
            stg_blocks=stg_blocks_list,
            rescale=rescale,
            modality_scale=modality_scale,
            video_cfg_end=cfg_end if cfg_end >= 0 else None,
            stg_scale_end=stg_end if stg_end >= 0 else None,
        )

        print(f"[RSICLoRAGuider] Guider created (video_cfg={video_cfg}, "
              f"stg_scale={stg_scale}, rescale={rescale})")

        return (guider,)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_downscale_factor(lora_path):
        """Read reference_downscale_factor from LoRA safetensors metadata."""
        try:
            from safetensors import safe_open
            with safe_open(lora_path, framework="pt") as f:
                metadata = f.metadata() or {}
                factor = int(metadata.get("reference_downscale_factor", 1))
                return factor
        except Exception as e:
            print(f"[RSICLoRAGuider] Warning: could not read LoRA metadata ({e}), "
                  f"defaulting downscale_factor=1")
            return 1
