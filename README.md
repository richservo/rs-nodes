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
    - [RS Video Save (ProRes)](#rs-video-save-prores)
    - [RS Frame Splitter](#rs-frame-splitter)
    - [RS Frame Collector](#rs-frame-collector)
    - [RS Free VRAM](#rs-free-vram)
    - [RS Counter](#rs-counter)
    - [RS Sigma Scheduler](#rs-sigma-scheduler)
- [Workflow Examples](#workflow-examples)
- [Tips & Troubleshooting](#tips--troubleshooting)

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/richservo/rs-nodes.git
cd rs-nodes
install.bat
```

The install script handles everything: initializes the LTX-2 submodule (for LoRA training) and installs all Python dependencies.

**Linux:**
```bash
./install.sh
```

### External Dependencies

- **[Ollama](https://ollama.com/)** — Required for RS Prompt Formatter and dataset captioning (RS LTXV Prepare Dataset). Install and run locally. The nodes auto-pull models on first use.
- **LTXV Models** — The LTXV generation/extension nodes require LTXV model checkpoints and VAEs loaded through ComfyUI's standard model loading nodes.

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

Scans a folder of videos/images, optionally detects and crops faces, **identifies speakers by voice**, generates captions via Ollama with woven-in dialogue, and preprocesses latents for LoRA training.

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `media_folder` | STRING | Path to folder of videos and/or images |
| `model_path` | CHECKPOINT | LTX-2 checkpoint |
| `text_encoder_path` | STRING | Gemma-3 HF directory (auto-download available) |
| `output_name` | STRING | Name for preprocessed output folder |

**Optional Inputs — Captioning:**

| Input | Type | Default | Description |
|---|---|---|---|
| `resolution_buckets` | STRING | "576x576x49" | WxHxF resolution buckets (semicolon-separated) |
| `lora_trigger` | STRING | "" | Trigger word prepended to all captions |
| `caption_mode` | ENUM | ollama | `ollama`, `skip`, `auto_filename` |
| `caption_style` | ENUM | subject | `subject`, `subject + style`, `style`, `motion`, `general`, `multi_character` |
| `ollama_model` | STRING | "gemma4:26b" | Vision model for captioning |
| `skip_id_pass` | BOOLEAN | False | Skip cast/location ID pass — send all refs directly to captioner |

**Optional Inputs — Face Detection & Character ID:**

| Input | Type | Default | Description |
|---|---|---|---|
| `face_detection` | BOOLEAN | True | Enable InsightFace (antelopev2) face detection + ArcFace embedding |
| `target_face` | IMAGE | — | Reference face for single-character identity matching (uses `lora_trigger` as the character name) |
| `character_refs_folder` | STRING | "" | Multi-character mode: folder of reference face images. Filename stem = character trigger |
| `location_refs_folder` | STRING | "" | Location reference images folder |
| `face_similarity` | FLOAT | 0.40 | Face match threshold |
| `face_padding` | FLOAT | 0.6 | Padding around detected face for face-crop mode |
| `crop_mode` | ENUM | face_crop | `face_crop`, `pan_and_scan`, `full_frame` |

**Optional Inputs — Speech Transcription & Voice Attribution:**

| Input | Type | Default | Description |
|---|---|---|---|
| `transcribe_speech` | BOOLEAN | False | Enable Whisper transcription with Demucs vocal isolation |
| `whisper_model` | ENUM | large-v3 | Whisper model size (`tiny`/`base`/`small`/`medium`/`large-v2`/`large-v3`/`large-v3-turbo`) |
| `voice_refs_folder` | STRING | "" | Folder of voice reference clips (audio OR video). Filename stem must match `character_refs` entries. Enables per-line speaker attribution via speechbrain ECAPA-TDNN |

**Optional Inputs — Other:**

| Input | Type | Default | Description |
|---|---|---|---|
| `target_fps` | FLOAT | 0.0 | Target framerate (0 = source fps). Drops/keeps frames to match without affecting audio speed |
| `max_samples` | INT | 0 | Max total character appearances. Quotas split evenly across characters (e.g. max=200, 4 chars → 50 each) |
| `skip_start_seconds` | FLOAT | 0.0 | Skip first N seconds of every video |
| `skip_end_seconds` | FLOAT | 0.0 | Skip last N seconds of every video |
| `conditioning_folder` | STRING | "" | IC-LoRA conditioning inputs folder |
| `clip` | CLIP | — | Text encoder for in-process condition encoding |
| `vae` | VAE | — | VAE for in-process latent encoding |
| `clip_vision` | CLIP_VISION | — | Optional CLIP Vision model for matching non-human characters (puppets, props) |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `preprocessed_path` | STRING | Path to preprocessed data root |
| `dataset_json_path` | STRING | Path to dataset JSON |

**Key Behaviors:**

*Extraction & Selection:*
- Two-phase processing: clip generation → latent preprocessing
- Face detection via InsightFace (SCRFD + ArcFace, antelopev2 model pack)
- **Multi-character mode**: drop face refs in `character_refs_folder` — auto-identifies cast per clip, balances quotas across characters
- **Location mode**: separate folder for set/location references — Gemma matches location per clip
- **Pan & scan**: face-aware cropping at any aspect ratio while keeping the face in frame
- **Frame-count safety**: clip extraction always overshoots target frame count and trims, so clips never come up short of the bucket minimum
- **Incremental & resumable**: per-clip atomic saves — interrupted runs continue from where they stopped; transcripts and conditions persist as they're generated
- **Quarantine, never delete**: clips that fail (hallucination, no usable speech, no matching characters, etc.) move to `<output_dir>/rejected_clips/<reason>/` instead of being deleted, so false positives can be reviewed and restored

*Speech Transcription:*
- Demucs vocal isolation runs first to strip music/SFX before Whisper transcribes
- Default model is **`large-v3`** for best accuracy on character voices, accents, and unusual vocabulary; configurable down to `tiny` for quick tests
- Silent clips get an empty transcript `""` (recorded once, never re-attempted on subsequent runs)

*Voice Attribution (when `voice_refs_folder` is set):*
- Each character's voice reference is embedded with **speechbrain ECAPA-TDNN** (192-d L2-normalized vector, downloaded automatically — no HuggingFace token needed)
- For each Whisper segment in a clip, ECAPA produces a per-segment embedding that's matched against enrolled voice prints by cosine distance
- **Face-detection hint**: the on-screen character (from face detection) gets a small distance bonus, biasing ambiguous matches toward the visible speaker without overriding strong off-screen voice evidence
- Each segment is tagged with the matched character (or `unknown` for short utterances / unenrolled voices)
- Tagged segments are passed to the captioner as a **DIALOGUE CONTEXT** block, and Gemma weaves the lines naturally into the caption ("X stands by the window and says, '...'") instead of appending dialogue as a separate field

*Caption Encoding (in-process when `clip` is connected):*
- Captions stay clean in `dataset.json` (visual prose only — dialogue is woven in by the captioner)
- At text-encode time, captions are normalized: quote-wrapping single quotes are stripped (so `'pee-wee'` becomes `pee-wee` without affecting `Pee-wee's` or `it's`); known character names are case-normalized using the canonical trigger (so a stray `Cowboy curd is` from a Whisper mistranscription still encodes as `Cowboy Curtis`)
- Original captions in `dataset.json` are never mutated — the normalization only affects the encoded `.pt` tensors

*VRAM Management:*
- All prepper-loaded models (Whisper, Demucs, ECAPA-TDNN, InsightFace) are unloaded between phases, with a final unload sweep before `prepare()` returns so unattended workflows that follow with training start with a clean GPU

**Voice Reference Files:**
- Format: any audio (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`) or video (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`) — audio is auto-extracted from video via ffmpeg
- Naming: filename stem must match the `character_refs_folder` entry (e.g. `cowboy curtis.wav` matches `cowboy curtis.jpg`)
- Length: 10-30 seconds of clean speech is the sweet spot; under 5 seconds risks unstable embeddings
- Variety helps: different sentences/intonations beat one repeated phrase. Concatenating short clips of the same character with ffmpeg works fine — each character can be built from 4-6 short single-speaker clips spliced together
- Demucs runs on enrollment too, so background music/effects in your reference clip get stripped automatically

---

#### RS LTXV Train LoRA

In-process LoRA training for LTX-2, reusing ComfyUI's already-loaded transformer. Includes a **standalone training monitor** that opens in a separate browser tab for full-size, resizable loss charts.

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
| `epochs` | INT | 3 | Number of passes through the full dataset |
| `auto_stop` | BOOLEAN | False | Ignore epoch count — train until divergence detection stops |
| `optimizer` | ENUM | adamw8bit | `adamw8bit`, `adamw`, or `rose` |
| `rose_stabilize` | BOOLEAN | True | Rose only: Smooths noisy gradient ranges. |
| `rose_weight_decay` | FLOAT | 1e-4 | Rose only: Decoupled weight decay strength. |
| `rose_wd_schedule` | BOOLEAN | True | Rose only: Scale weight decay with LR schedule. |
| `scheduler` | ENUM | linear | LR schedule: `linear`, `constant`, `cosine`, `cosine_with_restarts`, `polynomial` |
| `lr_cycle_steps` | INT | 0 | LR schedule cycle length (0 = one cycle per epoch) |
| `lr_cycle_decay` | FLOAT | 1.0 | Multiply LR by this factor each cycle reset (1.0 = no decay) |
| `quantization` | ENUM | fp8-quanto | `fp8-quanto`, `int8-quanto`, `int4-quanto`, `none` |
| `strategy` | ENUM | text_to_video | `text_to_video` or `video_to_video` (IC-LoRA) |
| `clip` | CLIP | — | Text encoder for validation prompt |
| `validation_prompt` | STRING | "" | Prompt for validation video generation |
| `validation_interval` | INT | 250 | Steps between validations |
| `checkpoint_interval` | INT | 500 | Steps between checkpoints |
| `diverge_detect_steps` | INT | 150 | Steps above threshold before entering monitoring |
| `diverge_stop_steps` | INT | 300 | Steps in monitoring without recovery before stopping |
| `diverge_threshold` | FLOAT | 15.0 | % above lowest EMA loss to trigger divergence |
| `resume` | BOOLEAN | False | Resume from latest checkpoint |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `status` | STRING | Training status message |
| `lora_path` | STRING | Path to saved LoRA file |

**Key Behaviors:**
- **In-process**: reuses the loaded 22B transformer — no reload, no double memory
- **Training monitor**: click "Open Training Monitor" on the node to open a full-page loss chart in a new browser tab. Shows raw loss dots, EMA-smoothed line, color-coded trend line, step timing, and divergence status. Loads history from `loss_history.json` on refresh.
- **Rose optimizer**: stateless optimizer — no momentum buffers, lower memory. Rose-specific settings auto-hide when another optimizer is selected.
- **Divergence detection**: monitors EMA loss vs minimum. If loss rises above threshold% for too long, saves a checkpoint and attempts LR reset. If unrecoverable, stops training and rewinds to the pre-divergence checkpoint.
- **LR cycle decay**: progressive learning rate reduction across scheduler cycles (e.g., 0.9 = 10% reduction per cycle)
- **Loss history persistence**: saves `loss_history.json` every 50 steps for chart continuity across resume/restart
- **Layer offloading**: streams transformer blocks CPU↔GPU one at a time (~0.5 GB VRAM instead of ~11 GB)
- **FP8 quantization**: reduces model memory while preserving LoRA weights in float
- **Resume support**: restores optimizer state, RNG, loss history, and rebuilds LR schedule
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
                                                    (click "Open Training Monitor")
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
- **Rose optimizer**: stateless, lower memory than AdamW. LR may need to be ~2x higher than AdamW (model-dependent — experiment).
- **Divergence detection**: uses EMA loss distance from minimum. Threshold of 15% is a good default. Auto-saves checkpoint at detection, attempts LR reset, stops and rewinds if unrecoverable.
- **Resume**: restores optimizer state, loss history, and LR schedule. The training monitor loads history from `loss_history.json` automatically.
- **Training Monitor**: open via the link on the training node. Full-page chart with raw loss dots, EMA-smoothed line, trend line (green=decreasing, red=increasing), step timing, and divergence status. Resizes with the browser window.

### Prompt Workflow

- Write scripts with `[s]`, `[a]`, `[d]` tags and feed into RS Prompt Parser.
- RS Prompt Formatter caches to JSON — if the input prompt hasn't changed, Ollama is skipped.
- RS Prompt Formatter Local avoids Ollama entirely by loading Gemma3 12B directly.

---

## License

See [LICENSE](LICENSE) for details.
