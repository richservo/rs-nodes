from .nodes.prompt_parser import RSPromptParser
from .nodes.audio_concat import RSAudioConcat
from .nodes.prompt_formatter import RSPromptFormatter
from .nodes.audio_save import RSAudioSave
from .nodes.free_vram import RSFreeVRAM
from .nodes.ltxv_generate import RSLTXVGenerate
from .nodes.ltxv_extend import RSLTXVExtend
from .nodes.ic_lora_guider import RSICLoRAGuider

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
}

# MOSS-TTS batch node — only available if comfyui-moss-tts is installed
try:
    from .nodes.moss_tts_save import RSMossTTSSave
    NODE_CLASS_MAPPINGS["RSMossTTSSave"] = RSMossTTSSave
    NODE_DISPLAY_NAME_MAPPINGS["RSMossTTSSave"] = "RS MOSS TTS Batch Save"
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
