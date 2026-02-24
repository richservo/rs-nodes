from .nodes.prompt_parser import RSPromptParser
from .nodes.audio_concat import RSAudioConcat
from .nodes.prompt_formatter import RSPromptFormatter

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "RSPromptParser": RSPromptParser,
    "RSAudioConcat": RSAudioConcat,
    "RSPromptFormatter": RSPromptFormatter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RSPromptParser": "RS Prompt Parser",
    "RSAudioConcat": "RS Audio Concat",
    "RSPromptFormatter": "RS Prompt Formatter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
