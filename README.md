# RS Nodes for ComfyUI

A comprehensive custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) focused on **LTXV audio-video generation**, **LoRA training**, **image generation**, **prompt engineering**, and **post-processing**. Built for real-world production workflows with emphasis on VRAM efficiency and quality control.

---

## Table of Contents

- [Installation](#installation)
- [Node Reference](#node-reference)
  - [Video Generation](#video-generation)
    - [RS LTXV Generate](#rs-ltxv-generate)
    - [RS LTXV Extend](#rs-ltxv-extend)
    - [RS LTXV Upscale](#rs-ltxv-upscale)
  - [Image Generation](#image-generation)
    - [RS Flux2 Generate](#rs-flux2-generate)
    - [RS Z-Image Generate](#rs-z-image-generate)
  - [LoRA Training](#lora-training)
    - [RS LTXV Prepare Dataset](#rs-ltxv-prepare-dataset)
    - [RS LTXV Train LoRA](#rs-ltxv-train-lora)
  - [Guidance & Control](#guidance--control)
    - [RS IC-LoRA Guider](#rs-ic-lora-guider)
    - [RS LTXV TTM Guider](#rs-ltxv-ttm-guider)
    - [RS Canny Preprocessor](#rs-canny-preprocessor)
  - [Prompt Engineering](#prompt-engineering)
    - [RS Prompt Parser](#rs-prompt-parser)
    - [RS Prompt Formatter](#rs-prompt-formatter)
    - [RS Prompt Formatter Local](#rs-prompt-formatter-local)
  - [Audio](#audio)
    - [RS Audio Concat](#rs-audio-concat)
    - [RS Audio Save](#rs-audio-save)
    - [RS MOSS TTS Loader](#rs-moss-tts-loader) *(optional)*
    - [RS MOSS TTS Batch Save](#rs-moss-tts-batch-save) *(optional)*
  - [Post-Processing & Utilities](#post-processing--utilities)
    - [RS Film Grain](#rs-film-grain)
    - [RS Video Trim](#rs-video-trim)
    - [RS Free VRAM](#rs-free-vram)
    - [RS Counter](#rs-counter)
- [Workflow Examples](#workflow-examples)
- [Tips & Troubleshooting](#tips--troubleshooting)

---

## Installation

### Via ComfyUI Manager (Recommended)

Search for **RS Nodes** in the ComfyUI Manager and click Install.

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/richservo/rs-nodes.git rs_nodes
pip install -r rs_nodes/requirements.txt
```

### Requirements

| Dependency | Required | Purpose |
|---|---|---|
| `openai-whisper` | Yes | Word-level alignment for TTS segmentation |
| `transformers` | Optional | MOSS-TTS model loading |
| `peft` | For training | LoRA adapter creation |
| `optimum-quanto` | For training | FP8/INT8 quantization |
| PyTorch + CUDA | Provided by ComfyUI | GPU computation |
| OpenCV (`cv2`) | Provided by ComfyUI | Canny edge detection, face detection |

### External Dependencies

- **[Ollama](https://ollama.com/)** — Required for RS Prompt Formatter. Install and run locally. The node auto-pulls models on first use.
- **LTXV Models** — The LTXV generation/extension nodes require LTXV model checkpoints and VAEs loaded through ComfyUI's standard model loading nodes.
- **LTX-2 Submodule** — Required for LoRA training. Run `git submodule update --init` after cloning.

---

## Node Reference

All nodes appear under the **RS Nodes** category in ComfyUI's node menu.

---

### Video Generation

#### RS LTXV Generate

All-in-one LTXV video generation with optional audio, keyframe injection, multimodal guidance, and integrated 2x latent upscaling.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | LTXV diffusion model |
| `positive` | CONDITIONING | Positive text conditioning |
| `negative` | CONDITIONING | Negative text conditioning |
| `vae` | VAE | Video VAE for encoding/decoding |

**Optional Inputs — Generation:**

| Input | Type | Default | Description |
|---|---|---|---|
| `width` | INT | 768 | Generation width (step: 32) |
| `height` | INT | 512 | Generation height (step: 32) |
| `num_frames` | INT | 97 | Frames to generate (step: 8). Auto-overridden by audio duration when `audio` + `audio_vae` are connected |
| `steps` | INT | 20 | Denoising steps |
| `cfg` | FLOAT | 3.0 | Video classifier-free guidance scale |
| `noise_seed` | INT | 0 | Random seed |
| `seed_mode` | ENUM | random | `random`, `fixed`, `increment`, `decrement` |
| `frame_rate` | FLOAT | 25.0 | Video frame rate |

**Optional Inputs — Frame Injection:**

| Input | Type | Default | Description |
|---|---|---|---|
| `first_image` | IMAGE | — | Keyframe injected at frame 0 |
| `middle_image` | IMAGE | — | Keyframe injected at midpoint |
| `last_image` | IMAGE | — | Keyframe injected at last frame |
| `first_strength` | FLOAT | 1.0 | Preservation strength (1.0 = exact match) |
| `middle_strength` | FLOAT | 1.0 | Preservation strength for middle frame |
| `last_strength` | FLOAT | 1.0 | Preservation strength for last frame |
| `crf` | INT | 35 | LTXV preprocessing quality (0–100) |

**Optional Inputs — Audio (AV Dual-Tower):**

| Input | Type | Default | Description |
|---|---|---|---|
| `audio` | AUDIO | — | Input audio (passthrough to output if provided) |
| `audio_vae` | VAE | — | Audio VAE (enables AV dual-tower mode) |
| `audio_cfg` | FLOAT | 7.0 | Audio CFG scale |
| `stg_scale` | FLOAT | 0.0 | Spatiotemporal guidance scale (0 = disabled) |
| `stg_blocks` | STRING | "29" | Comma-separated transformer block indices for STG |
| `rescale` | FLOAT | 0.7 | CFG rescaling factor |

**Optional Inputs — Efficiency:**

| Input | Type | Default | Description |
|---|---|---|---|
| `attention_mode` | ENUM | auto | `auto` (SAGE if available), `default`, `sage` |
| `ffn_chunks` | INT | 4 | FFN sequence chunking (0 = disabled) |
| `video_attn_scale` | FLOAT | 1.03 | Video attention scaling + VRAM-efficient forward |

**Optional Inputs — Upscale (2x Latent):**

| Input | Type | Default | Description |
|---|---|---|---|
| `upscale` | BOOLEAN | False | Enable 2x spatial upscaling |
| `upscale_model` | LATENT_UPSCALE_MODEL | — | Latent upscale model |
| `upscale_steps` | INT | 4 | Re-diffusion steps at upscaled resolution |
| `upscale_cfg` | FLOAT | 1.0 | CFG during re-diffusion |
| `upscale_denoise` | FLOAT | 0.5 | Denoise strength (0 = no re-diffuse, 1 = full) |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `latent` | LATENT | Video latent `[B, C, T, H, W]` |
| `audio_latent` | LATENT | Audio latent (if audio_vae used) |
| `images` | IMAGE | Decoded video frames `[B*T, H, W, C]` |
| `audio_output` | AUDIO | Output audio (passthrough or decoded) |

---

#### RS LTXV Extend

Extend an existing video with seamless temporal continuation using overlap blending.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | LTXV diffusion model |
| `positive` | CONDITIONING | Positive conditioning |
| `negative` | CONDITIONING | Negative conditioning |
| `vae` | VAE | Video VAE |
| `latent` | LATENT | Input video latent to extend |

**Optional Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `num_new_frames` | INT | 80 | New frames to generate (step: 8) |
| `overlap_frames` | INT | 16 | Frames of overlap for blending (step: 8) |
| `steps` | INT | 20 | Denoising steps |
| `cfg` | FLOAT | 3.0 | Video CFG scale |
| `last_image` | IMAGE | — | Keyframe for end of extension |

Also accepts the same Audio, Efficiency, Upscale, and Scheduler inputs as RS LTXV Generate.

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `latent` | LATENT | Extended video latent (original + new frames) |
| `images` | IMAGE | Decoded extension frames |
| `audio_output` | AUDIO | Decoded audio (if audio_vae used) |

---

#### RS LTXV Upscale

Standalone 2x video upscaler with optional temporal upscaling and first-frame I2V re-diffusion.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | LTXV diffusion model |
| `positive` | CONDITIONING | Positive conditioning |
| `negative` | CONDITIONING | Negative conditioning |
| `vae` | VAE | Video VAE |
| `images` | IMAGE | Input video frames |

**Optional Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `upscale_model` | LATENT_UPSCALE_MODEL | — | Spatial 2x upscale model |
| `temporal_upscale_model` | LATENT_UPSCALE_MODEL | — | Temporal 2x upscale model |
| `upscale_steps` | INT | 4 | Re-diffusion steps |
| `upscale_cfg` | FLOAT | 1.0 | CFG during re-diffusion |
| `upscale_denoise` | FLOAT | 0.5 | Denoise strength |
| `upscale_lora` | ENUM | none | LoRA for re-diffusion pass |
| `attention_mode` | ENUM | auto | Attention optimization |
| `ffn_chunks` | INT | 4 | FFN chunking for memory |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `latent` | LATENT | Upscaled video latent |
| `images` | IMAGE | Decoded upscaled frames |
| `audio_output` | AUDIO | Audio passthrough |

**Key Behaviors:**
- Encodes input, applies spatial 2x via upscale model, re-diffuses with first-frame I2V guidance
- Optional temporal 2x increases frame count before spatial upscaling
- Tiled encoding/decoding for memory efficiency

---

### Image Generation

#### RS Flux2 Generate

All-in-one Flux2.dev image generation with Kontext reference images, empirical mu scheduling, and optional two-pass latent upscaling.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | Flux2 model |
| `clip` | CLIP | Text encoder |
| `vae` | VAE | Image VAE |
| `prompt` | STRING | Generation prompt |

**Optional Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `width` | INT | 1024 | Image width |
| `height` | INT | 1024 | Image height |
| `steps` | INT | 50 | Sampling steps |
| `guidance` | FLOAT | 4.0 | Guidance scale |
| `seed` | INT | 0 | Random seed |
| `ref_image_1..4` | IMAGE | — | Up to 4 Kontext reference images |
| `lora` | ENUM | none | LoRA adapter |
| `attention_mode` | ENUM | auto | Attention optimization |
| `ffn_chunks` | INT | 4 | FFN chunking (double blocks only) |
| `latent_upscale` | ENUM | off | Two-pass latent upscaling |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | Generated image |

**Key Behaviors:**
- Kontext reference images auto-scaled to optimal aspect ratios and appended to conditioning
- Empirical mu scheduling based on image dimensions
- Optional two-pass: half-res first pass, then full-res re-diffusion

---

#### RS Z-Image Generate

All-in-one Z-Image Turbo generation with Qwen3-4B text encoder, RenormCFG, and system prompt support.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | Z-Image Turbo model |
| `clip` | CLIP | Qwen3-4B text encoder |
| `vae` | VAE | Image VAE |
| `prompt` | STRING | Generation prompt |

**Optional Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `width` | INT | 1024 | Image width |
| `height` | INT | 1024 | Image height |
| `steps` | INT | 10 | Sampling steps (4 for turbo) |
| `system_prompt` | ENUM | superior | System prompt variant |
| `cfg` | FLOAT | 1.5 | CFG scale |
| `renorm_cfg` | FLOAT | 1.0 | RenormCFG norm limit (0=off) |
| `lora` | ENUM | none | LoRA adapter |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | Generated image |

---

### LoRA Training

#### RS LTXV Prepare Dataset

Scans a folder of videos/images, optionally detects and crops faces, generates captions via Ollama, and preprocesses latents for LoRA training.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `media_folder` | STRING | Path to folder of videos and/or images |
| `model_path` | CHECKPOINT | LTX-2 checkpoint |
| `text_encoder_path` | STRING | Gemma-3 HF directory (auto-download available) |
| `output_name` | STRING | Name for preprocessed output folder |

**Optional Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `resolution_buckets` | STRING | "576x576x49" | WxHxF resolution buckets (semicolon-separated) |
| `lora_trigger` | STRING | "" | Trigger word prepended to all captions |
| `caption_mode` | ENUM | ollama | `ollama`, `skip`, `auto_filename` |
| `caption_style` | ENUM | subject | `subject`, `subject + style`, `style`, `motion`, `general` |
| `ollama_model` | STRING | "gemma3:27b" | Vision model for captioning |
| `face_detection` | BOOLEAN | True | Enable face detection and cropping |
| `target_face` | IMAGE | — | Reference face for identity matching |
| `face_similarity` | FLOAT | 0.40 | Face match threshold |
| `conditioning_folder` | STRING | "" | IC-LoRA conditioning inputs folder |
| `clip` | CLIP | — | Text encoder for in-process encoding |
| `vae` | VAE | — | VAE for in-process latent encoding |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `preprocessed_path` | STRING | Path to preprocessed data root |
| `dataset_json_path` | STRING | Path to dataset JSON |

**Key Behaviors:**
- Two-phase processing: clip generation then latent preprocessing
- Face detection via OpenCV DNN with optional identity matching
- Incremental: skips already-processed clips, tracks rejections
- Caption styles control what gets described (environment vs subject vs both)

---

#### RS LTXV Train LoRA

In-process LoRA training for LTX-2, reusing ComfyUI's already-loaded transformer. Includes a **live loss chart** rendered directly on the node during training.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | LTX-2 model (from CheckpointLoaderSimple) |
| `output_name` | STRING | Name for the output LoRA file |
| `preprocessed_data_root` | STRING | Path from RS LTXV Prepare Dataset |
| `model_path` | CHECKPOINT | LTX-2 checkpoint (for EmbeddingsProcessor/VAE) |

**Optional Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `vae` | VAE | — | VAE for validation (avoids loading from checkpoint) |
| `preset` | ENUM | subject | `custom`, `subject`, `style`, `motion`, `subject + style`, `all video`, `audio + video` |
| `lora_rank` | INT | 16 | LoRA rank |
| `lora_alpha` | INT | 16 | LoRA alpha |
| 8 module toggles | BOOLEAN | varies | Select which layers to train (self-attn, cross-attn, FFN, audio) |
| `learning_rate` | FLOAT | 1e-4 | Learning rate |
| `steps` | INT | 2000 | Total training steps |
| `optimizer` | ENUM | adamw8bit | `adamw8bit` or `adamw` |
| `scheduler` | ENUM | linear | LR schedule: `linear`, `constant`, `cosine`, `cosine_with_restarts`, `polynomial` |
| `quantization` | ENUM | fp8-quanto | `fp8-quanto`, `int8-quanto`, `int4-quanto`, `none` |
| `clip` | CLIP | — | Text encoder for validation prompt |
| `validation_prompt` | STRING | "" | Prompt for validation video generation |
| `validation_interval` | INT | 250 | Steps between validations |
| `checkpoint_interval` | INT | 500 | Steps between checkpoints |
| `resume` | BOOLEAN | False | Resume from latest checkpoint |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `status` | STRING | Training status message |
| `lora_path` | STRING | Path to saved LoRA file |

**Key Behaviors:**
- **In-process**: reuses the loaded 22B transformer — no reload, no double memory
- **Live loss chart**: streams loss/LR data via websocket to a live-updating Canvas chart on the node, with EMA smoothing and color-coded trend line
- **Layer offloading**: streams transformer blocks CPU↔GPU one at a time (~0.5 GB VRAM instead of ~11 GB)
- **FP8 quantization**: reduces model memory while preserving LoRA weights in float
- **Resume support**: restores optimizer state, RNG, and rebuilds LR schedule for the new total steps
- **Presets**: auto-configure module toggles and rank for common use cases (subject, style, motion)
- Quantization modifies the transformer in-place — reload checkpoint after training

---

### Guidance & Control

#### RS IC-LoRA Guider

Structural control for LTXV using IC-LoRA (In-Context LoRA) with preprocessed control images.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | LTXV diffusion model |
| `positive` | CONDITIONING | Positive conditioning |
| `negative` | CONDITIONING | Negative conditioning |
| `vae` | VAE | Video VAE |
| `control_image` | IMAGE | Preprocessed control map (e.g., canny edges) |
| `ic_lora` | ENUM | IC-LoRA safetensors file |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `guider` | GUIDER | Connect to the `guider` input of RS LTXV Generate |

---

#### RS LTXV TTM Guider

Time-to-Move (TTM) motion control guider for LTXV. Enforces reference motion via dual-clock denoising with mask-based control.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | LTXV diffusion model |
| `positive` | CONDITIONING | Positive conditioning |
| `negative` | CONDITIONING | Negative conditioning |
| `vae` | VAE | Video VAE |
| `reference_video` | IMAGE | Reference video for motion |
| `mask` | IMAGE | Motion mask (1.0 = enforce reference, 0.0 = free generation) |

**Optional Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `ttm_strength` | FLOAT | 0.5 | Fraction of steps to apply TTM |
| `cfg` | FLOAT | 3.0 | CFG scale |
| `stg_scale` | FLOAT | 0.0 | Spatiotemporal guidance |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `guider` | GUIDER | Connect to RS LTXV Generate's guider input |
| `width` | INT | Generation width |
| `height` | INT | Generation height |

---

#### RS Canny Preprocessor

Canny edge detection with automatic LTXV-safe resolution (128-aligned).

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | Input image |
| `width` | INT | 768 | Target width (step: 128) |
| `height` | INT | 512 | Target height (step: 128) |
| `low_threshold` | INT | 100 | Canny low threshold |
| `high_threshold` | INT | 200 | Canny high threshold |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Edge map (grayscale to RGB) |
| `width` | INT | 128-aligned width |
| `height` | INT | 128-aligned height |

---

### Prompt Engineering

#### RS Prompt Parser

Parse structured dialogue scripts with `[s]`tyle, `[a]`ction, and `[d]`ialogue tags into separate video and audio prompts.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `script` | STRING | "" | Multiline script with tags |
| `dialogue_mode` | ENUM | individual | `individual` or `all` |
| `dialogue_index` | INT | 1 | Which dialogue line to select |

**Script Format:**

```
[s] cinematic, natural lighting, handheld camera
[a] a man walks into a room and sits down
[d] Hello, how are you doing today?
```

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `video_prompt` | STRING | Combined visual prompt |
| `audio_prompt` | STRING | TTS text |
| `dialogue_count` | INT | Total dialogue segments |
| `current_index` | INT | Current dialogue index |
| `dialogue_list` | STRING | Numbered list of all dialogue |

---

#### RS Prompt Formatter

AI-powered prompt enhancement using a local Ollama model with reference image support and output caching.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `prompt` | STRING | "" | Raw prompt to enhance |
| `system_prompt` | STRING | *(built-in)* | Instructions for the model |
| `model` | STRING | "gemma3:12b" | Ollama model name |
| `reference_image` | IMAGE | — | *(optional)* Image for visual context |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `formatted_prompt` | STRING | Enhanced prompt text |

**Key Behaviors:**
- Streams responses from Ollama with live token printing
- Auto-pulls missing models, strips `<think>` blocks
- Caches prompt + output as JSON — skips Ollama when input unchanged

**Requires:** [Ollama](https://ollama.com/) running locally.

---

#### RS Prompt Formatter Local

Ollama-free prompt formatter reusing Gemma3 12B from ComfyUI's DualCLIPLoader. Supports optional reference images via vision embeddings.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `text_encoder` | STRING | — | Gemma3 text encoder file |
| `prompt` | STRING | "" | Prompt to format |
| `system_prompt` | STRING | — | System prompt for generation |
| `first_image` | IMAGE | — | *(optional)* Opening image |
| `middle_image` | IMAGE | — | *(optional)* Mid-scene image |
| `last_image` | IMAGE | — | *(optional)* Ending image |
| `max_tokens` | INT | 1024 | Maximum output tokens |
| `temperature` | FLOAT | 0.8 | Sampling temperature |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `formatted_prompt` | STRING | Enhanced prompt text |

**Key Behaviors:**
- No Ollama dependency — loads Gemma3 12B weights directly
- Vision support for up to 3 reference images
- JSON caching (text-only prompts)

---

### Audio

#### RS Audio Concat

Concatenate up to 20 audio clips with per-clip trimming and configurable pauses.

**Inputs:** For each clip: audio file, start/end trim, pause after.

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `audio` | AUDIO | Concatenated waveform |

---

#### RS Audio Save

Export audio to disk with format selection (`wav`, `flac`, `mp3`, `ogg`).

---

#### RS MOSS TTS Loader

*(Optional — requires `transformers`)*

Load MOSS-TTS model variants for text-to-speech generation.

---

#### RS MOSS TTS Batch Save

*(Optional — requires `transformers`)*

Generate TTS audio from a dialogue list with automatic segmentation, per-clip trimming, and batch export. Supports `one_shot`, `all`, and `single` generation modes with Whisper-based word alignment.

---

### Post-Processing & Utilities

#### RS Film Grain

Add realistic film grain with color variation and luminance-aware highlight protection.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `images` | IMAGE | — | Input video frames |
| `intensity` | FLOAT | 0.05 | Grain strength (0–1) |
| `grain_size` | FLOAT | 1.5 | Grain frequency |
| `color_amount` | FLOAT | 0.3 | Color noise ratio |
| `highlight_protection` | FLOAT | 0.5 | Protect bright/dark areas |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | Grained video frames |

---

#### RS Video Trim

Trim video frames and/or audio by time range.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `fps` | FLOAT | 24.0 | Frame rate |
| `in_point` | FLOAT | 0.0 | Start time (seconds) |
| `out_point` | FLOAT | 0.0 | End time (0 = end of clip) |
| `images` | IMAGE | — | *(optional)* Video frames |
| `audio` | AUDIO | — | *(optional)* Audio |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | Trimmed frames |
| `audio` | AUDIO | Trimmed audio |

---

#### RS Free VRAM

Passthrough utility that forces VRAM cleanup between pipeline stages.

| Input | Type | Description |
|---|---|---|
| `any_input` | * (wildcard) | Any data — passed through unchanged |

Unloads all models, runs garbage collection, clears CUDA cache.

---

#### RS Counter

Persistent incrementing counter. State stored in `counter_state.json` across workflow executions.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `start` | INT | 0 | Starting value (used on reset) |
| `step` | INT | 1 | Increment amount |
| `reset` | BOOLEAN | False | Reset to start value |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `value` | INT | Current counter value (increments after output) |

---

## Workflow Examples

### Basic LTXV Video Generation

```
[Load LTXV Model] ──┐
[CLIP Text Encode] ──┼──→ [RS LTXV Generate] ──→ [Save Video]
[CLIP Text Encode] ──┤
[Load VAE] ──────────┘
```

### LoRA Training Pipeline

```
[Load LTXV Model] ──┐
[Load VAE] ──────────┼──→ [RS LTXV Prepare Dataset] ──→ [RS LTXV Train LoRA]
[Load CLIP] ─────────┘                                         │
                                                          (live loss chart)
```

### Audio-Driven Video with TTS

```
[RS Prompt Parser] ──→ [RS Prompt Formatter] ──→ [CLIP Text Encode] ──┐
        │                                                              │
        └──→ [RS MOSS TTS Loader] ──→ [RS MOSS TTS Batch Save] ──┐    │
                                              │                    │    │
                                      [RS Audio Concat] ──────────┼────┼──→ [RS LTXV Generate]
```

### IC-LoRA Structural Control

```
[Load Image] ──→ [RS Canny Preprocessor] ──→ [RS IC-LoRA Guider] ──→ [RS LTXV Generate]
```

### Video Extension + Upscale

```
[RS LTXV Generate] ──→ [RS LTXV Extend] ──→ [RS LTXV Upscale] ──→ [Save Video]
```

---

## Tips & Troubleshooting

### VRAM Management

- **Place RS Free VRAM** between heavy inference nodes and post-processing to reclaim GPU memory.
- Use **`ffn_chunks`** (default: 4) and **`video_attn_scale`** (default: 1.03) to reduce VRAM during generation.
- Enable **`upscale_tiling`** for temporal tiling during upscale on long videos.
- LoRA training uses **layer offloading** to stream blocks one at a time (~0.5 GB instead of ~11 GB).

### Generation Quality

- **`cfg`**: 2.5–4.0 works well for video; higher values can cause artifacts.
- **`audio_cfg`**: 5.0–9.0 is typical for audio.
- **`stg_scale`**: Start at 0. Small values (0.1–0.5) can improve temporal consistency.
- **`rescale`**: 0.7 is a good default.
- **Upscaling**: `upscale_denoise=0.3–0.6` balances sharpness vs. faithfulness.

### LoRA Training

- Use **`fp8-quanto`** quantization (no C++ build tools needed).
- **Subject LoRAs**: enable self-attention + cross-attention. Captions should describe everything *except* the subject.
- **Style LoRAs**: enable self-attention + feed-forward. Captions should describe the visual style in detail.
- **Resume**: set the target total steps before resuming — the LR schedule is rebuilt for the new total.
- The live loss chart shows raw dots, EMA-smoothed line, and a green (decreasing) or red (increasing) trend line.

### Prompt Workflow

- Write scripts with `[s]`, `[a]`, `[d]` tags and feed into RS Prompt Parser.
- RS Prompt Formatter caches to JSON — if the input prompt hasn't changed, Ollama is skipped.
- RS Prompt Formatter Local avoids Ollama entirely by loading Gemma3 12B directly.

---

## License

See [LICENSE](LICENSE) for details.
