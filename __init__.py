# OpenCV reads OPENCV_IO_ENABLE_OPENEXR at IMPORT time — setting it
# later (inside the EXR save node's function body) is too late on
# the opencv-python builds that ship with this disabled by default.
# Setting it here, before any node module imports cv2 transitively,
# guarantees the EXR codec is registered when imwrite runs.
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

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
from .nodes.video_batch_loader import RSVideoBatchLoader
from .nodes.frame_splitter import RSFrameSplitter
from .nodes.frame_collector import RSFrameCollector
from .nodes.ltxv_prepare_dataset import RSLTXVPrepareDataset
from .nodes.ltxv_train_lora import RSLTXVTrainLoRA
from .nodes.sigma_scheduler import RSSigmaScheduler
from .nodes.video_save import RSVideoSave
from .nodes.exr_sequence_save import RSEXRSequenceSave
from .nodes.exr_load import RSEXRLoad
from .nodes.logc3_decode import RSLogC3Decode
from .nodes.prompt_relay_encode import RSPromptRelayEncode
from .nodes.prompt_relay_timeline import RSPromptRelayTimeline
from .nodes.runpod_dispatch import RSRunOnRunPod
from .nodes.image_strip_alpha import RSImageStripAlpha

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
    "RSVideoBatchLoader": RSVideoBatchLoader,
    "RSFrameSplitter": RSFrameSplitter,
    "RSFrameCollector": RSFrameCollector,
    "RSLTXVPrepareDataset": RSLTXVPrepareDataset,
    "RSLTXVTrainLoRA": RSLTXVTrainLoRA,
    "RSSigmaScheduler": RSSigmaScheduler,
    "RSVideoSave": RSVideoSave,
    "RSEXRSequenceSave": RSEXRSequenceSave,
    "RSEXRLoad": RSEXRLoad,
    "RSLogC3Decode": RSLogC3Decode,
    "RSPromptRelayEncode": RSPromptRelayEncode,
    "RSPromptRelayTimeline": RSPromptRelayTimeline,
    "RSRunOnRunPod": RSRunOnRunPod,
    "RSImageStripAlpha": RSImageStripAlpha,
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
    "RSVideoBatchLoader": "RS Video Batch Loader",
    "RSFrameSplitter": "RS Frame Splitter",
    "RSFrameCollector": "RS Frame Collector",
    "RSLTXVPrepareDataset": "RS LTXV Prepare Dataset",
    "RSLTXVTrainLoRA": "RS LTXV Train LoRA",
    "RSSigmaScheduler": "RS Sigma Scheduler",
    "RSVideoSave": "RS Video Save (ProRes)",
    "RSEXRSequenceSave": "RS EXR Sequence Save",
    "RSEXRLoad": "RS EXR Load",
    "RSLogC3Decode": "RS LogC3 HDR Decode",
    "RSPromptRelayEncode": "RS Prompt Relay Encode",
    "RSPromptRelayTimeline": "RS Prompt Relay Timeline",
    "RSRunOnRunPod": "RS Run on RunPod",
    "RSImageStripAlpha": "RS Image Strip Alpha",
}

# MOSS-TTS nodes — only available if dependencies (transformers, huggingface_hub) are installed
try:
    from .nodes.moss_tts_loader import RSMossTTSLoader
    NODE_CLASS_MAPPINGS["RSMossTTSLoader"] = RSMossTTSLoader
    NODE_DISPLAY_NAME_MAPPINGS["RSMossTTSLoader"] = "RS MOSS TTS Loader"
except Exception:
    pass

try:
    from .nodes.moss_tts_save import RSMossTTSSave
    NODE_CLASS_MAPPINGS["RSMossTTSSave"] = RSMossTTSSave
    NODE_DISPLAY_NAME_MAPPINGS["RSMossTTSSave"] = "RS MOSS TTS Batch Save"
except Exception:
    pass

# RTX Super Resolution (V3 node) — only available if nvvfx is installed
try:
    from .nodes.rtx_super_resolution import RSRTXSuperResolution
    NODE_CLASS_MAPPINGS["RSRTXSuperResolution"] = RSRTXSuperResolution
    NODE_DISPLAY_NAME_MAPPINGS["RSRTXSuperResolution"] = "RS RTX Super Resolution"
except ImportError:
    pass

# Side-effect import: registers the /rs/uitoapi server route used by
# rs-studio for canonical UI workflow → API prompt conversion.
# Tolerates failures so ComfyUI still loads if aiohttp is missing or
# server.PromptServer.instance isn't ready.
try:
    from .nodes import uitoapi_route  # noqa: F401
except Exception as _err:
    print(f"[rs-nodes] uitoapi route not registered: {_err}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
