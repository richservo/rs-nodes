"""
RSGeminiGenerate — Gemini 3 Pro image generation via API key.

Same functionality as the Google Genmedia Gemini3ProImage node but uses
a Gemini API key instead of Vertex AI / GCP project credentials.
"""

import logging
import os
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("rs_nodes.gemini_generate")

# ---------------------------------------------------------------------------
# API key persistence
# ---------------------------------------------------------------------------
_KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemini_api_key.txt")


def _resolve_key(user_key: str) -> str:
    """Return a valid API key — from user input, saved file, or raise."""
    if user_key and len(user_key.strip()) > 10:
        key = user_key.strip()
        try:
            with open(_KEY_FILE, "w") as f:
                f.write(key)
        except Exception as e:
            logger.warning(f"Could not save API key: {e}")
        return key
    if os.path.exists(_KEY_FILE):
        try:
            with open(_KEY_FILE, "r") as f:
                saved = f.read().strip()
            if len(saved) > 10:
                return saved
        except Exception:
            pass
    raise ValueError("No Gemini API key provided. Enter it in the api_key field.")


# ---------------------------------------------------------------------------
# Image tensor helpers
# ---------------------------------------------------------------------------

def _tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a ComfyUI IMAGE tensor [B,H,W,C] or [H,W,C] to PNG bytes."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to ComfyUI tensor [1,H,W,3] float32."""
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Threshold options (mirroring Google Genmedia constants)
# ---------------------------------------------------------------------------
THRESHOLD_OPTIONS = [
    "BLOCK_NONE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE",
]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class RSGeminiGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "A vivid landscape painting of a futuristic city"}),
                "model": ([
                    "gemini-3-pro-image-preview",
                    "gemini-3.1-flash-image-preview",
                ], {"default": "gemini-3-pro-image-preview"}),
                "image_size": (["1K", "2K", "4K"], {"default": "1K"}),
                "aspect_ratio": ([
                    "1:1", "2:3", "3:2", "3:4", "4:3",
                    "4:5", "5:4", "9:16", "16:9", "21:9",
                ], {"default": "16:9"}),
                "output_format": (["PNG", "JPEG"], {"default": "PNG"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 32, "min": 1, "max": 64}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "safety_threshold": (THRESHOLD_OPTIONS, {"default": "BLOCK_MEDIUM_AND_ABOVE"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "RS Nodes"

    def generate(
        self,
        api_key: str,
        prompt: str,
        model: str,
        image_size: str,
        aspect_ratio: str,
        output_format: str,
        temperature: float,
        top_p: float,
        top_k: int,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        image5: Optional[torch.Tensor] = None,
        image6: Optional[torch.Tensor] = None,
        system_instruction: str = "",
        safety_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
    ) -> Tuple[torch.Tensor]:
        from google import genai
        from google.genai import types

        key = _resolve_key(api_key)
        client = genai.Client(api_key=key)

        output_mime = f"image/{output_format.lower()}"

        # Build config — same structure as the Vertex node
        safety_settings = []
        for category in [
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_IMAGE_HATE",
            "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
            "HARM_CATEGORY_IMAGE_HARASSMENT",
            "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
        ]:
            safety_settings.append(
                types.SafetySetting(category=category, threshold=safety_threshold)
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=32768,
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                output_mime_type=output_mime,
            ),
            system_instruction=system_instruction if system_instruction.strip() else None,
            safety_settings=safety_settings,
        )

        # Build contents
        contents = [types.Part.from_text(text=prompt)]
        for i, img_tensor in enumerate([image1, image2, image3, image4, image5, image6]):
            if img_tensor is not None:
                for j in range(img_tensor.shape[0]):
                    png_bytes = _tensor_to_png_bytes(img_tensor[j:j+1])
                    contents.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))
                    logger.info(f"Attached reference image {i+1} (batch {j+1})")

        logger.info(f"Generating: model={model}, size={image_size}, aspect={aspect_ratio}, format={output_format}")
        response = client.models.generate_content(model=model, contents=contents, config=config)

        # Extract images from response
        pil_images: List[Image.Image] = []
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                logger.info(f"Model response text: {part.text[:200]}")
            elif part.inline_data is not None:
                pil_images.append(Image.open(BytesIO(part.inline_data.data)))

        if not pil_images:
            raise RuntimeError("Gemini API returned no images")

        tensors = [_pil_to_tensor(img) for img in pil_images]
        return (torch.cat(tensors, dim=0),)
