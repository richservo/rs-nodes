from .nodes.prompt_parser import RSPromptParser
from .nodes.audio_concat import RSAudioConcat
from .nodes.prompt_formatter import RSPromptFormatter
from .nodes.audio_save import RSAudioSave
from .nodes.free_vram import RSFreeVRAM
from .nodes.ltxv_generate import RSLTXVGenerate
from .nodes.ltxv_extend import RSLTXVExtend
from .nodes.ic_lora_guider import RSICLoRAGuider
from .nodes.ltxv_iclora_guider import RSLTXVICLoRAGuider
from .nodes.canny_preprocessor import RSCannyPreprocessor
from .nodes.video_trim import RSVideoTrim
from .nodes.film_grain import RSFilmGrain
from .nodes.ltxv_upscale import RSLTXVUpscale
from .nodes.flux2_generate import RSFlux2Generate
from .nodes.prompt_formatter_local import RSPromptFormatterLocal
from .nodes.z_image_generate import RSZImageGenerate
from .nodes.counter import RSCounter
from .nodes.frame_splitter import RSFrameSplitter
from .nodes.frame_collector import RSFrameCollector
from .nodes.gemini_generate import RSGeminiGenerate

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "RSPromptParser": RSPromptParser,
    "RSAudioConcat": RSAudioConcat,
    "RSPromptFormatter": RSPromptFormatter,
    "RSAudioSave": RSAudioSave,
    "RSFreeVRAM": RSFreeVRAM,
    "RSLTXVGenerate": RSLTXVGenerate,
    "RSLTXVExtend": RSLTXVExtend,
    "RSICLoRAGuider": RSICLoRAGuider,
    "RSLTXVICLoRAGuider": RSLTXVICLoRAGuider,
    "RSCannyPreprocessor": RSCannyPreprocessor,
    "RSVideoTrim": RSVideoTrim,
    "RSFilmGrain": RSFilmGrain,
    "RSLTXVUpscale": RSLTXVUpscale,
    "RSFlux2Generate": RSFlux2Generate,
    "RSPromptFormatterLocal": RSPromptFormatterLocal,
    "RSZImageGenerate": RSZImageGenerate,
    "RSCounter": RSCounter,
    "RSFrameSplitter": RSFrameSplitter,
    "RSFrameCollector": RSFrameCollector,
    "RSGeminiGenerate": RSGeminiGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RSPromptParser": "RS Prompt Parser",
    "RSAudioConcat": "RS Audio Concat",
    "RSPromptFormatter": "RS Prompt Formatter",
    "RSAudioSave": "RS Audio Save",
    "RSFreeVRAM": "RS Free VRAM",
    "RSLTXVGenerate": "RS LTXV Generate",
    "RSLTXVExtend": "RS LTXV Extend",
    "RSICLoRAGuider": "RS IC-LoRA Guider",
    "RSLTXVICLoRAGuider": "RS LTXV IC-LoRA Guider",
    "RSCannyPreprocessor": "RS Canny Preprocessor",
    "RSVideoTrim": "RS Video Trim",
    "RSFilmGrain": "RS Film Grain",
    "RSLTXVUpscale": "RS LTXV Upscale",
    "RSFlux2Generate": "RS Flux2 Generate",
    "RSPromptFormatterLocal": "RS Prompt Formatter Local",
    "RSZImageGenerate": "RS Z-Image Generate",
    "RSCounter": "RS Counter",
    "RSFrameSplitter": "RS Frame Splitter",
    "RSFrameCollector": "RS Frame Collector",
    "RSGeminiGenerate": "RS Gemini Generate",
}

# MOSS-TTS nodes — only available if dependencies (transformers, huggingface_hub) are installed
try:
    from .nodes.moss_tts_loader import RSMossTTSLoader
    NODE_CLASS_MAPPINGS["RSMossTTSLoader"] = RSMossTTSLoader
    NODE_DISPLAY_NAME_MAPPINGS["RSMossTTSLoader"] = "RS MOSS TTS Loader"
except ImportError:
    pass

try:
    from .nodes.moss_tts_save import RSMossTTSSave
    NODE_CLASS_MAPPINGS["RSMossTTSSave"] = RSMossTTSSave
    NODE_DISPLAY_NAME_MAPPINGS["RSMossTTSSave"] = "RS MOSS TTS Batch Save"
except ImportError:
    pass

# RTX Super Resolution (V3 node) — only available if nvvfx is installed
try:
    from .nodes.rtx_super_resolution import RSRTXSuperResolution
    NODE_CLASS_MAPPINGS["RSRTXSuperResolution"] = RSRTXSuperResolution
    NODE_DISPLAY_NAME_MAPPINGS["RSRTXSuperResolution"] = "RS RTX Super Resolution"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
