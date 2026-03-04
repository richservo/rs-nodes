import hashlib
import importlib.metadata
import importlib.util
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor

import comfy.model_management as mm
import folder_paths

MODEL_VARIANTS = {
    "MOSS-TTS (Delay 8B)": "OpenMOSS-Team/MOSS-TTS",
    "MOSS-TTS (Local 1.7B)": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    "MOSS-TTSD v1.0": "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "MOSS-VoiceGenerator": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "MOSS-SoundEffect": "OpenMOSS-Team/MOSS-SoundEffect",
}

MODEL_ID_TTSD = "OpenMOSS-Team/MOSS-TTSD-v1.0"
MODEL_ID_VOICE_GENERATOR = "OpenMOSS-Team/MOSS-VoiceGenerator"
DEFAULT_CODEC_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
SAMPLE_RATE = 24000

MOSS_MODELS_DIR = os.path.join(folder_paths.models_dir, "moss-tts")
os.makedirs(MOSS_MODELS_DIR, exist_ok=True)

# Track loaded model so we can free VRAM before loading a new one
_current_model = None
_current_processor = None


def _resolve_local_dir(repo_id_or_path):
    if os.path.isdir(repo_id_or_path):
        return repo_id_or_path
    safe_name = repo_id_or_path.replace("/", "--")
    local_dir = os.path.join(MOSS_MODELS_DIR, safe_name)
    return snapshot_download(repo_id_or_path, local_dir=local_dir)


def _flash_attn_available():
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        importlib.metadata.version("flash_attn")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _resolve_attn_implementation(device, dtype):
    if (
        str(device).startswith("cuda")
        and _flash_attn_available()
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if str(device).startswith("cuda"):
        return "sdpa"
    return "eager"


class RSMossTTSLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (list(MODEL_VARIANTS.keys()),),
                "local_model_path": ("STRING", {"default": ""}),
                "codec_local_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MOSS_TTS_PIPE",)
    FUNCTION = "load_model"
    CATEGORY = "rs-nodes"

    def load_model(self, model_variant, local_model_path, codec_local_path):
        global _current_model, _current_processor

        device = mm.get_torch_device()
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        attn_implementation = _resolve_attn_implementation(device, dtype)

        mm.unload_all_models()

        if _current_processor is not None:
            if hasattr(_current_processor, "audio_tokenizer"):
                _current_processor.audio_tokenizer.cpu()
            _current_processor = None
        if _current_model is not None:
            _current_model.cpu()
            del _current_model
            _current_model = None
        torch.cuda.empty_cache()

        model_id = MODEL_VARIANTS[model_variant]
        model_path = local_model_path.strip() if local_model_path.strip() else model_id

        local_dir = _resolve_local_dir(model_path)

        processor_kwargs = {"trust_remote_code": True}
        if model_id == MODEL_ID_TTSD:
            codec_path = codec_local_path.strip() or DEFAULT_CODEC_PATH
            processor_kwargs["codec_path"] = _resolve_local_dir(codec_path)
        elif model_id == MODEL_ID_VOICE_GENERATOR:
            processor_kwargs["normalize_inputs"] = True

        processor = AutoProcessor.from_pretrained(local_dir, **processor_kwargs)
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        model = AutoModel.from_pretrained(
            local_dir,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        _current_model = model
        _current_processor = processor

        return ((model, processor, SAMPLE_RATE, device, model_id),)

    @classmethod
    def IS_CHANGED(cls, model_variant, local_model_path, codec_local_path):
        h = hashlib.md5()
        h.update(model_variant.encode())
        h.update(local_model_path.encode())
        h.update(codec_local_path.encode())
        return h.hexdigest()
