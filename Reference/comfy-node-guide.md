# ComfyUI Custom Node Development Guide — MOSS-TTS Project

This file is the complete reference for any agent working on this project.
Read this ENTIRE file before writing any code.

---

## Project Location

- Project root: `/mnt/e/Python/MOSS-TTS/comfyui-moss-tts/`
- MOSS-TTS upstream repo (read-only reference): `/mnt/e/Python/MOSS-TTS/repo/`

## File Structure

```
comfyui-moss-tts/
    __init__.py              # NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS
    requirements.txt
    reference/               # Documentation (not shipped)
    nodes/
        __init__.py          # Empty
        model_loader.py      # MossTTSModelLoader
        generate.py          # MossTTSGenerate
        voice_design.py      # MossTTSVoiceDesign
        sound_effect.py      # MossTTSSoundEffect
        dialogue.py          # MossTTSDialogue
    utils/
        __init__.py          # Empty
        backend.py           # cuDNN patch + attention resolver
        audio_utils.py       # ComfyUI AUDIO ↔ MOSS tensor conversion
        constants.py         # Model IDs, defaults, constants
```

---

## ComfyUI Node Anatomy

Every ComfyUI node is a Python class with these required attributes:

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "param_name": ("TYPE", {"default": value, ...}),
            },
            "optional": {
                "opt_param": ("TYPE", {"default": value, ...}),
            },
        }

    RETURN_TYPES = ("OUTPUT_TYPE",)        # Tuple of type strings
    RETURN_NAMES = ("output_name",)        # Tuple of display names
    FUNCTION = "method_name"               # Method that ComfyUI calls
    CATEGORY = "audio/MOSS-TTS"           # Where it appears in the menu
    OUTPUT_NODE = False                    # True only for terminal nodes (save, preview)

    def method_name(self, param_name, opt_param=None):
        # Do work
        return (result,)                   # Must return tuple matching RETURN_TYPES
```

### Type Reference

| ComfyUI Type | Python Input Spec | Notes |
|---|---|---|
| INT | `("INT", {"default": 0, "min": 0, "max": 100, "step": 1})` | |
| FLOAT | `("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})` | |
| STRING | `("STRING", {"default": "", "multiline": False})` | `multiline: True` for text areas |
| BOOLEAN | `("BOOLEAN", {"default": False})` | |
| Dropdown | `(["option1", "option2", "option3"],)` | List = dropdown, first item is default |
| AUDIO | `("AUDIO",)` | ComfyUI's native audio type |
| Custom | `("MOSS_TTS_PIPE",)` | Any string = custom type for wiring nodes |

### AUDIO Format

ComfyUI's AUDIO type is a dict:
```python
{
    "waveform": torch.Tensor,  # Shape: [batch, channels, samples] — typically [1, 1, S]
    "sample_rate": int,        # e.g., 24000
}
```

### Caching & IS_CHANGED

ComfyUI caches node outputs. To force re-execution (e.g., when seed changes):

```python
@classmethod
def IS_CHANGED(cls, **kwargs):
    return kwargs.get("seed", 0)  # Different return value = re-execute
```

### ComfyUI Model Management

```python
import comfy.model_management as mm

device = mm.get_torch_device()           # Returns the GPU device
mm.unload_all_models()                   # Free VRAM before loading big models
mm.soft_empty_cache()                    # Gentle cache clear after generation
```

---

## MOSS-TTS API Reference

### cuDNN SDPA Patch (REQUIRED)

Must be applied at import time before any model operations:

```python
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
```

### Attention Implementation Resolver

```python
import importlib.util

def resolve_attn_implementation(device, dtype) -> str:
    if (
        str(device).startswith("cuda")
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if str(device).startswith("cuda"):
        return "sdpa"
    return "eager"
```

### Model Loading

All 5 models use the same pattern:

```python
from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained(
    model_id,                    # e.g. "OpenMOSS-Team/MOSS-TTS"
    trust_remote_code=True,
    # codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer",  # for TTSD only
    # normalize_inputs=True,                              # for VoiceGenerator only
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation=attn_impl,   # from resolver above
    torch_dtype=torch.bfloat16,      # on CUDA; float32 on CPU
).to(device)
model.eval()
```

### HuggingFace Model IDs

| Display Name | Model ID |
|---|---|
| MOSS-TTS (Delay 8B) | `OpenMOSS-Team/MOSS-TTS` |
| MOSS-TTS (Local 1.7B) | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` |
| MOSS-TTSD v1.0 | `OpenMOSS-Team/MOSS-TTSD-v1.0` |
| MOSS-VoiceGenerator | `OpenMOSS-Team/MOSS-VoiceGenerator` |
| MOSS-SoundEffect | `OpenMOSS-Team/MOSS-SoundEffect` |
| Audio Tokenizer (codec) | `OpenMOSS-Team/MOSS-Audio-Tokenizer` |

### Processor API

#### Building Messages

```python
# TTS / Voice Cloning
user_msg = processor.build_user_message(
    text="Hello world.",                    # Text to speak
    reference=None,                         # None or List[str|Tensor] for voice cloning
    tokens=None,                            # Duration control: int, 1 sec ≈ 12.5 tokens
)

# Voice Design (VoiceGenerator model only)
user_msg = processor.build_user_message(
    text="Hello world.",
    instruction="A warm, deep male voice",  # Voice description
)

# Sound Effect (SoundEffect model only)
user_msg = processor.build_user_message(
    ambient_sound="Rain on a tin roof",     # Sound description
    tokens=125,                             # Duration in tokens (10 sec)
)

# Dialogue (TTSD model only) — reference is list with one entry per speaker
user_msg = processor.build_user_message(
    text="[S1] Hello!\n[S2] Hi there!",    # Text with speaker tags
    reference=[s1_codes, s2_codes],         # List[Tensor|None], one per speaker
)

# For continuation mode (prefix audio):
assistant_msg = processor.build_assistant_message(
    audio_codes_list=[audio_codes_tensor],  # Pre-encoded audio
    content="<|audio|>",
)
```

#### Tokenizing

```python
# Generation mode (no prefix audio):
batch = processor([[user_msg]], mode="generation")

# Continuation mode (with prefix audio):
batch = processor([[user_msg, assistant_msg]], mode="continuation")

# Returns BatchFeature with:
#   batch["input_ids"]      — shape (B, T, 1 + n_vq)
#   batch["attention_mask"]  — shape (B, T)
```

#### Audio Encoding (for reference/clone audio)

```python
# From in-memory tensors (what we use for ComfyUI AUDIO input):
codes_list = processor.encode_audios_from_wav(
    wav_list=[tensor_1d],       # List of 1D float32 tensors
    sampling_rate=24000,        # Sample rate of the input
)
# Returns: List[torch.Tensor], each shape (T, n_vq)

# From file paths:
codes_list = processor.encode_audios_from_path(["path/to/file.wav"])
```

#### Decoding Generation Output

```python
messages = processor.decode(outputs)
# outputs: List[Tuple[int, Tensor]] from model.generate()
# Returns: List[AssistantMessage | None]
# message.audio_codes_list[0] → 1D float32 waveform tensor at 24kHz
```

### Model Generate API

#### Delay 8B model:

```python
outputs = model.generate(
    input_ids=batch["input_ids"].to(device),        # (B, T, 1+n_vq)
    attention_mask=batch["attention_mask"].to(device),  # (B, T)
    max_new_tokens=4096,
    audio_temperature=1.7,
    audio_top_p=0.8,
    audio_top_k=25,
    audio_repetition_penalty=1.0,
)
# Returns: List[Tuple[int, Tensor]]
```

#### Local 1.7B model:

Uses HuggingFace GenerationMixin — same generate() signature works, parameters
are passed through generation_config internally. Same call pattern as Delay.

### Recommended Hyperparameters Per Model

| Model ID | temp | top_p | top_k | rep_penalty |
|---|---|---|---|---|
| OpenMOSS-Team/MOSS-TTS | 1.7 | 0.8 | 25 | 1.0 |
| OpenMOSS-Team/MOSS-TTS-Local-Transformer | 1.0 | 0.95 | 50 | 1.1 |
| OpenMOSS-Team/MOSS-TTSD-v1.0 | 1.1 | 0.9 | 50 | 1.1 |
| OpenMOSS-Team/MOSS-VoiceGenerator | 1.5 | 0.6 | 50 | 1.1 |
| OpenMOSS-Team/MOSS-SoundEffect | 1.5 | 0.6 | 50 | 1.2 |

### Key Constants

- **Sample rate:** 24000 Hz (all models)
- **Codec frame rate:** 12.5 tokens/second
- **Duration formula:** `tokens = max(1, int(seconds * 12.5))`
- **Default max_new_tokens:** 4096

---

## MOSS_TTS_PIPE Custom Type

The pipe passed between nodes is a tuple:

```python
MOSS_TTS_PIPE = (model, processor, sample_rate, device, model_id)
# model: the loaded AutoModel
# processor: the loaded AutoProcessor
# sample_rate: int (24000)
# device: torch.device
# model_id: str (e.g. "OpenMOSS-Team/MOSS-TTS")
```

All generation nodes unpack this:
```python
model, processor, sample_rate, device, model_id = moss_pipe
```

---

## Audio Conversion Utilities (utils/audio_utils.py)

```python
def comfyui_audio_to_moss_tensor(audio_dict):
    """Convert ComfyUI AUDIO → (1D float32 mono tensor, sample_rate)."""
    waveform = audio_dict["waveform"]   # [B, C, S]
    sr = audio_dict["sample_rate"]
    # Take first batch, average channels to mono
    wav = waveform[0].mean(dim=0)       # [S]
    return wav, sr

def moss_tensor_to_comfyui_audio(tensor_1d, sample_rate=24000):
    """Convert 1D float32 tensor → ComfyUI AUDIO dict."""
    return {
        "waveform": tensor_1d.unsqueeze(0).unsqueeze(0),  # [1, 1, S]
        "sample_rate": sample_rate,
    }

def resample_if_needed(waveform, orig_sr, target_sr):
    """Resample tensor if sample rates differ."""
    if orig_sr == target_sr:
        return waveform
    import torchaudio
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)
```

---

## Common Generation Pattern (used by all generation nodes)

```python
import torch
import comfy.model_management as mm
from ..utils.audio_utils import moss_tensor_to_comfyui_audio

def generate_audio(self, moss_pipe, text, seed, temperature, top_p, top_k,
                   repetition_penalty, max_new_tokens, **kwargs):
    model, processor, sample_rate, device, model_id = moss_pipe

    torch.manual_seed(seed)

    # 1. Build user message (varies per node)
    user_msg = processor.build_user_message(text=text, ...)

    # 2. Tokenize
    batch = processor([[user_msg]], mode="generation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # 3. Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            audio_temperature=temperature,
            audio_top_p=top_p,
            audio_top_k=top_k,
            audio_repetition_penalty=repetition_penalty,
        )

    # 4. Decode
    messages = processor.decode(outputs)
    if messages[0] is None:
        raise RuntimeError("Generation failed — model returned no audio")
    wav = messages[0].audio_codes_list[0]

    # 5. Convert to ComfyUI AUDIO
    result = moss_tensor_to_comfyui_audio(wav.cpu(), sample_rate)

    mm.soft_empty_cache()
    return (result,)
```

---

## Node Specifications

### 1. MossTTSModelLoader (nodes/model_loader.py)

**Inputs (all required):**
- `model_variant`: dropdown from MODEL_VARIANTS keys
- `local_model_path`: STRING, default ""  (override HF download)
- `codec_local_path`: STRING, default ""  (override codec download, relevant for TTSD)

**Output:** `MOSS_TTS_PIPE`

**Special handling:**
- TTSD: pass `codec_path` to processor (from codec_local_path or DEFAULT_CODEC_PATH)
- VoiceGenerator: pass `normalize_inputs=True` to processor
- Call `mm.unload_all_models()` before loading

### 2. MossTTSGenerate (nodes/generate.py)

**Required inputs:** moss_pipe, text (multiline), seed, temperature (1.7), top_p (0.8), top_k (25), repetition_penalty (1.0), max_new_tokens (4096), enable_duration_control (False), duration_tokens (325)
**Optional inputs:** reference_audio (AUDIO)

**Output:** AUDIO

**Logic:** No reference = direct TTS. With reference = voice cloning (encode ref audio, pass as `reference` list). Duration control via `tokens` param.

### 3. MossTTSVoiceDesign (nodes/voice_design.py)

**Required inputs:** moss_pipe, text (multiline), instruction (multiline), seed, temperature (1.5), top_p (0.6), top_k (50), repetition_penalty (1.1), max_new_tokens (4096)

**Output:** AUDIO

**Logic:** Pass `instruction` to `build_user_message`. Warn (print) if model_id isn't VoiceGenerator.

### 4. MossTTSSoundEffect (nodes/sound_effect.py)

**Required inputs:** moss_pipe, ambient_sound (multiline), duration_seconds (FLOAT, 5.0, range 0.5–60.0), seed, temperature (1.5), top_p (0.6), top_k (50), repetition_penalty (1.2), max_new_tokens (4096)

**Output:** AUDIO

**Logic:** Convert `duration_seconds` to tokens. Pass `ambient_sound` and `tokens` to `build_user_message`. Warn if model_id isn't SoundEffect.

### 5. MossTTSDialogue (nodes/dialogue.py)

**Required inputs:** moss_pipe, dialogue_text (multiline, with [S1]/[S2] tags), speaker_count (INT, 2, range 2–2), normalize_text (BOOLEAN, True), seed, temperature (1.1), top_p (0.9), top_k (50), repetition_penalty (1.1), max_new_tokens (4096)
**Optional inputs:** s1_reference_audio (AUDIO), s1_prompt_text (STRING), s2_reference_audio (AUDIO), s2_prompt_text (STRING)

**Output:** AUDIO

**Logic:**
- Encode each speaker's reference audio if provided (None in list if not)
- If any speaker has reference + prompt_text, use continuation mode:
  - Concatenate prompt audio codes
  - Build assistant_msg with prompt codes
  - mode="continuation"
- Otherwise: mode="generation"
- Warn if model_id isn't TTSD

---

## Top-level __init__.py

```python
from .nodes.model_loader import MossTTSModelLoader
from .nodes.generate import MossTTSGenerate
from .nodes.voice_design import MossTTSVoiceDesign
from .nodes.sound_effect import MossTTSSoundEffect
from .nodes.dialogue import MossTTSDialogue

NODE_CLASS_MAPPINGS = {
    "MossTTSModelLoader": MossTTSModelLoader,
    "MossTTSGenerate": MossTTSGenerate,
    "MossTTSVoiceDesign": MossTTSVoiceDesign,
    "MossTTSSoundEffect": MossTTSSoundEffect,
    "MossTTSDialogue": MossTTSDialogue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSModelLoader": "MOSS-TTS Model Loader",
    "MossTTSGenerate": "MOSS-TTS Generate",
    "MossTTSVoiceDesign": "MOSS-TTS Voice Design",
    "MossTTSSoundEffect": "MOSS-TTS Sound Effect",
    "MossTTSDialogue": "MOSS-TTS Dialogue",
}
```

## requirements.txt

```
transformers>=4.40.0
safetensors
einops
scipy
librosa
tiktoken
```

No torch/torchaudio — ComfyUI provides these.
