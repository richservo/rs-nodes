# RS Nodes for ComfyUI

A comprehensive custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) focused on **LTXV audio-video generation**, **TTS pipelines**, **prompt engineering**, and **post-processing**. Built for real-world production workflows with emphasis on VRAM efficiency and quality control.

---

## Table of Contents

- [Installation](#installation)
- [Node Reference](#node-reference)
  - [Video Generation](#video-generation)
    - [RS LTXV Generate](#rs-ltxv-generate)
    - [RS LTXV Extend](#rs-ltxv-extend)
  - [Guidance & Control](#guidance--control)
    - [RS IC-LoRA Guider](#rs-ic-lora-guider)
    - [RS Canny Preprocessor](#rs-canny-preprocessor)
  - [Prompt Engineering](#prompt-engineering)
    - [RS Prompt Parser](#rs-prompt-parser)
    - [RS Prompt Formatter](#rs-prompt-formatter)
  - [Audio](#audio)
    - [RS Audio Concat](#rs-audio-concat)
    - [RS Audio Save](#rs-audio-save)
    - [RS MOSS TTS Loader](#rs-moss-tts-loader) *(optional)*
    - [RS MOSS TTS Batch Save](#rs-moss-tts-batch-save) *(optional)*
  - [Post-Processing & Utilities](#post-processing--utilities)
    - [RS Film Grain](#rs-film-grain)
    - [RS Video Trim](#rs-video-trim)
    - [RS Free VRAM](#rs-free-vram)
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
| PyTorch + CUDA | Provided by ComfyUI | GPU computation |
| OpenCV (`cv2`) | Provided by ComfyUI | Canny edge detection |

To enable the optional MOSS-TTS nodes:

```bash
pip install transformers
```

Models are downloaded from HuggingFace on first use, then run fully offline.

### External Dependencies

- **[Ollama](https://ollama.com/)** — Required for RS Prompt Formatter. Install and run locally. The node auto-pulls models on first use.
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
| `frame_rate` | FLOAT | 25.0 | Video frame rate (stamped onto conditioning) |

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
| `audio_stg_scale` | FLOAT | -1.0 | Per-audio STG (-1 = use video value) |
| `stg_blocks` | STRING | "29" | Comma-separated transformer block indices for STG |
| `rescale` | FLOAT | 0.7 | CFG rescaling factor |
| `video_modality_scale` | FLOAT | 0.0 | Video modality isolation (0 = off) |
| `audio_modality_scale` | FLOAT | 3.0 | Audio modality isolation |
| `cfg_end` | FLOAT | -1.0 | End CFG for interpolation (-1 = same as start) |
| `stg_end` | FLOAT | -1.0 | End STG for interpolation |
| `cfg_star_rescale` | BOOLEAN | True | CFG-Zero* rescaling (prevents high-sigma artifacts) |
| `skip_sigma` | FLOAT | 0.0 | Sigma threshold for skipping guidance |

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
| `upscale_lora` | ENUM | none | LoRA to apply during upscale re-diffusion |
| `upscale_lora_strength` | FLOAT | 1.0 | LoRA strength |
| `upscale_steps` | INT | 4 | Re-diffusion steps at upscaled resolution |
| `upscale_cfg` | FLOAT | 1.0 | CFG during re-diffusion |
| `upscale_denoise` | FLOAT | 0.5 | Denoise strength (0 = no re-diffuse, 1 = full) |
| `upscale_fallback` | BOOLEAN | False | Fall back to half-res decode on OOM |
| `upscale_tiling` | BOOLEAN | False | Temporal tiling during upscale (saves VRAM) |
| `upscale_tile_t` | INT | 4 | Temporal tile size (0 = auto) |

**Optional Inputs — Output & Overrides:**

| Input | Type | Default | Description |
|---|---|---|---|
| `decode` | BOOLEAN | True | Decode latents to images |
| `tile_t` | INT | 0 | VAE decode temporal tile size (0 = auto) |
| `guider` | GUIDER | — | Custom guider (e.g., from RS IC-LoRA Guider) |
| `sampler` | SAMPLER | — | Custom sampler (default: euler_ancestral) |
| `sigmas` | SIGMAS | — | Custom sigma schedule |
| `max_shift` | FLOAT | 2.05 | Scheduler max shift |
| `base_shift` | FLOAT | 0.95 | Scheduler base shift |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `latent` | LATENT | Video latent `[B, C, T, H, W]` |
| `audio_latent` | LATENT | Audio latent (if audio_vae used) |
| `images` | IMAGE | Decoded video frames `[B*T, H, W, C]` |
| `audio_output` | AUDIO | Output audio (passthrough or decoded) |

**Key Behaviors:**
- When `audio` + `audio_vae` are connected, `num_frames` is auto-calculated from audio duration
- When `upscale` is enabled, generation happens at half resolution then 2x upscaled
- First image is re-injected at full resolution during upscale re-diffusion
- The node always re-executes (no caching)

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

**Optional Inputs — Extension:**

| Input | Type | Default | Description |
|---|---|---|---|
| `num_new_frames` | INT | 80 | New frames to generate (step: 8) |
| `overlap_frames` | INT | 16 | Frames of overlap for blending (step: 8) |
| `steps` | INT | 20 | Denoising steps |
| `cfg` | FLOAT | 3.0 | Video CFG scale |
| `seed` | INT | 0 | Random seed |
| `frame_rate` | FLOAT | 25.0 | Video FPS |
| `last_image` | IMAGE | — | Keyframe for end of extension |
| `last_strength` | FLOAT | 1.0 | Last frame preservation |
| `overlap_strength` | FLOAT | 1.0 | Overlap region preservation |
| `crf` | INT | 35 | LTXV preprocessing quality |

Also accepts the same **Audio**, **Efficiency**, **Upscale**, **Output**, **Overrides**, and **Scheduler** inputs as RS LTXV Generate.

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `latent` | LATENT | Extended video latent (original + new frames) |
| `images` | IMAGE | Decoded extension frames |
| `audio_output` | AUDIO | Decoded audio (if audio_vae used) |

**Key Behaviors:**
- Extracts overlap region from the tail of the input latent
- Generates extension with overlap as a guide frame
- Linear alpha blending in the overlap region for seamless continuation
- Concatenates original (trimmed) + blended extension

---

### Guidance & Control

#### RS IC-LoRA Guider

Structural control for LTXV generation using IC-LoRA (In-Context LoRA) with preprocessed control images (canny edges, depth maps, pose).

**Required Inputs:**

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | LTXV diffusion model |
| `positive` | CONDITIONING | Positive conditioning |
| `negative` | CONDITIONING | Negative conditioning |
| `vae` | VAE | Video VAE |
| `control_image` | IMAGE | Preprocessed control map (e.g., canny edges) |
| `ic_lora` | ENUM | IC-LoRA safetensors file (from loras directory) |

**Optional Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `lora_strength` | FLOAT | 1.0 | LoRA blend strength |
| `guide_strength` | FLOAT | 1.0 | Frame preservation (1.0 = exact) |
| `guide_frame_idx` | INT | 0 | Which frame to inject guide (0 = first, -1 = last) |
| `crf` | INT | 35 | LTXV preprocessing quality |

Also accepts the same guidance parameters as RS LTXV Generate (`video_cfg`, `audio_cfg`, `stg_scale`, `stg_blocks`, `rescale`, modality scales, scheduler shifts, etc.).

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `guider` | GUIDER | Connect to the `guider` input of RS LTXV Generate or RS LTXV Extend |

**Key Behaviors:**
- Reads `reference_downscale_factor` from LoRA metadata for sparse dilation
- Defers VAE encoding to sample time (when target latent dimensions are known)
- Clones the model and applies IC-LoRA with specified strength

---

#### RS Canny Preprocessor

Canny edge detection with automatic LTXV-safe resolution (128-aligned).

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | Input image |
| `width` | INT | 768 | Target width (step: 128) |
| `height` | INT | 512 | Target height (step: 128) |
| `low_threshold` | INT | 100 | Canny low threshold (0–255) |
| `high_threshold` | INT | 200 | Canny high threshold (0–255) |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Edge map (grayscale replicated to RGB) |
| `width` | INT | Computed 128-aligned width |
| `height` | INT | Computed 128-aligned height |

**Key Behaviors:**
- Preserves aspect ratio and computes the closest 128-aligned resolution from the pixel budget
- Output resolution is LTXV-compatible and can feed directly into RS IC-LoRA Guider

---

### Prompt Engineering

#### RS Prompt Parser

Parse structured dialogue scripts with `[s]`tyle, `[a]`ction, and `[d]`ialogue tags into separate video and audio prompts.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `script` | STRING | "" | Multiline script with tags (see format below) |
| `dialogue_mode` | ENUM | individual | `individual` (one line) or `all` (concatenate all) |
| `dialogue_index` | INT | 1 | Which dialogue line to select (auto-increments) |

**Script Format:**

```
[s] cinematic, natural lighting, handheld camera
[a] a man walks into a room and sits down
[d] Hello, how are you doing today?
[a] the camera pans to reveal another person
[d] I'm doing great, thanks for asking.
```

- `[s]` — **Style**: visual style description (prepended as "Style: ...")
- `[a]` — **Action**: scene description (auto-capitalized, auto-punctuated)
- `[d]` — **Dialogue**: spoken text (quoted in video prompt, plain in audio prompt)

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `video_prompt` | STRING | Combined visual prompt (style + actions + quoted dialogue) |
| `audio_prompt` | STRING | TTS text (selected dialogue or all concatenated) |
| `dialogue_count` | INT | Total number of `[d]` segments |
| `current_index` | INT | Current dialogue index (clamped to valid range) |
| `dialogue_list` | STRING | Numbered list of all dialogue lines |

---

#### RS Prompt Formatter

AI-powered prompt enhancement using a local Ollama model, with optional reference image context and output caching.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `prompt` | STRING | "" | Raw prompt to enhance |
| `system_prompt` | STRING | *(built-in)* | Instructions for the Ollama model |
| `model` | STRING | "gemma3:12b" | Ollama model name |
| `ollama_url` | STRING | "http://localhost:11434" | Ollama server URL |
| `reference_image` | IMAGE | — | *(optional)* Image for visual context |
| `cache_file` | STRING | "formatted_prompt.json" | JSON cache file (stores prompt + output) |
| `output_dir` | STRING | "" | Cache directory (empty = ComfyUI output) |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `formatted_prompt` | STRING | Enhanced prompt text |

**Key Behaviors:**
- Streams responses from Ollama with live token printing to console
- Auto-pulls missing models from the Ollama registry
- Strips `<think>` reasoning blocks from the response
- Caches prompt + output as JSON — automatically skips Ollama when the input prompt hasn't changed
- Falls back gracefully on Ollama connection errors

**Requires:** [Ollama](https://ollama.com/) running locally.

---

### Audio

#### RS Audio Concat

Concatenate up to 20 audio clips with per-clip trimming and configurable pauses.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `clip_count` | INT | 1 | Number of clips to use (1–20) |

For each clip `i` (shown dynamically based on `clip_count`):

| Input | Type | Default | Description |
|---|---|---|---|
| `audio_file_{i}` | ENUM | — | Audio file from ComfyUI input directory (.wav, .mp3, .flac, .ogg, .m4a, .aac) |
| `start_time_{i}` | FLOAT | 0.0 | Seconds to trim from start |
| `end_time_{i}` | FLOAT | 0.0 | Seconds to trim from end |
| `pause_after_{i}` | FLOAT | 0.0 | Silence to insert after clip (0–10s) |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `audio` | AUDIO | Concatenated waveform `{"waveform": Tensor, "sample_rate": int}` |

**Key Behaviors:**
- Auto-resamples all clips to match the first clip's sample rate
- Supports 6+ audio formats
- Caches based on file modification times

---

#### RS Audio Save

Export audio to disk with format selection.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `audio` | AUDIO | — | Audio to save |
| `filename_prefix` | STRING | "clip" | Base filename |
| `index` | INT | 1 | File index (zero-padded in filename) |
| `format` | ENUM | wav | `wav`, `flac`, `mp3`, `ogg` |

**Outputs:** Audio preview widget in ComfyUI UI.

**Key Behaviors:**
- Falls back to WAV if MP3/OGG encoding fails
- Always re-executes (output node)

---

#### RS MOSS TTS Loader

*(Optional — requires `transformers`)*

Load MOSS-TTS model variants for text-to-speech generation.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `model_variant` | ENUM | — | Model selection: "MOSS-TTS (Delay 8B)", "MOSS-TTS (Local 1.7B)", "MOSS-TTSD v1.0", "MOSS-VoiceGenerator", "MOSS-SoundEffect" |
| `local_model_path` | STRING | "" | Local path override (empty = download from HuggingFace) |
| `codec_local_path` | STRING | "" | Local codec path (for TTSD models) |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `MOSS_TTS_PIPE` | PIPE | Model pipeline tuple for RS MOSS TTS Batch Save |

**Key Behaviors:**
- Auto-detects Flash Attention availability
- Models cached in `{models_dir}/moss-tts/`
- Clears VRAM before loading

> **Security Note:** MOSS-TTS models use `trust_remote_code=True` when loading via HuggingFace `transformers`. This is required because the model defines custom architecture code not natively included in the `transformers` library. Without it, the model cannot load. No data is sent externally — the flag only allows the model's bundled Python code to execute locally during loading.

---

#### RS MOSS TTS Batch Save

*(Optional — requires `transformers`)*

Generate TTS audio from a dialogue list with automatic segmentation, per-clip trimming, and batch export.

**Inputs — Core:**

| Input | Type | Default | Description |
|---|---|---|---|
| `run_inference` | BOOLEAN | True | Enable/disable generation (False = use existing files) |
| `filename_prefix` | STRING | "clip" | Output file prefix |
| `format` | ENUM | wav | `wav`, `flac`, `mp3`, `ogg` |
| `mode` | ENUM | one_shot | `one_shot`, `all`, `single` (see below) |

**Inputs — TTS (when `run_inference=True`):**

| Input | Type | Default | Description |
|---|---|---|---|
| `moss_pipe` | MOSS_TTS_PIPE | — | From RS MOSS TTS Loader |
| `dialogue_list` | STRING | — | Numbered dialogue list (from RS Prompt Parser) |
| `reference_audio` | AUDIO | — | Reference voice for few-shot cloning |
| `select_index` | INT | 1 | Line to generate in "single" mode |
| `language` | ENUM | auto | `auto`, `zh`, `en`, `ja`, `ko` |
| `seed` | INT | 0 | RNG seed |
| `temperature` | FLOAT | 1.7 | Sampling temperature |
| `top_p` | FLOAT | 0.8 | Nucleus sampling threshold |
| `top_k` | INT | 25 | Top-K sampling |
| `repetition_penalty` | FLOAT | 1.0 | Repetition penalty |
| `max_new_tokens` | INT | 4096 | Max generation tokens |
| `head_handle` | FLOAT | 0.0 | Silence padding before audio (seconds) |
| `tail_handle` | FLOAT | 0.0 | Silence padding after audio (seconds) |

**Inputs — Per-Clip Trim/Pause** (for each clip `i`, 1–20):

| Input | Type | Default | Description |
|---|---|---|---|
| `pause_before_{i}` | FLOAT | 0.0 | Silence before clip |
| `start_time_{i}` | FLOAT | 0.0 | Trim from start (seconds) |
| `end_time_{i}` | FLOAT | 0.0 | Trim from end (seconds) |
| `pause_after_{i}` | FLOAT | 0.0 | Silence after clip |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `audio` | AUDIO | All clips concatenated |

**Generation Modes:**

| Mode | Behavior |
|---|---|
| `one_shot` | Generate all dialogue as a single audio clip, then segment into individual clips using Whisper word alignment or silence-based dynamic programming |
| `all` | Generate each dialogue line as a separate inference call |
| `single` | Generate only the line at `select_index` |

**Key Behaviors:**
- Saves individual clips as `{prefix}_{001}.{format}` and full concatenation as `{prefix}_full.{format}`
- Whisper-based word alignment for intelligent one-shot segmentation with DP fallback
- When `run_inference=False`, loads existing files from disk (useful for re-trimming)

---

### Post-Processing & Utilities

#### RS Film Grain

Add realistic film grain with color variation and luminance-aware highlight protection.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `images` | IMAGE | — | Input video frames |
| `intensity` | FLOAT | 0.05 | Grain strength (0–1) |
| `grain_size` | FLOAT | 1.5 | Grain frequency (1.0 = pixel-level, 8.0 = large blobs) |
| `color_amount` | FLOAT | 0.3 | Color noise ratio (0 = monochrome, 1 = full color) |
| `highlight_protection` | FLOAT | 0.5 | Protect bright/dark areas (0 = uniform, 1 = midtones only) |
| `seed` | INT | 0 | RNG seed (deterministic per frame) |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | Grained video frames |

**Key Behaviors:**
- Frees all model VRAM before processing (safe to place after inference nodes)
- Processes in batches of 8 frames on GPU (~200 MB peak VRAM)
- Noise generated directly on GPU for speed
- Uses Rec.709 luminance weighting for midtone masking
- Low-res noise with bilinear upscale creates organic grain texture at larger `grain_size` values

---

#### RS Video Trim

Trim video frames and/or audio by time range.

**Inputs:**

| Input | Type | Default | Description |
|---|---|---|---|
| `fps` | FLOAT | 24.0 | Frame rate for time-to-frame conversion |
| `in_point` | FLOAT | 0.0 | Start time in seconds |
| `out_point` | FLOAT | 0.0 | End time in seconds (0 = end of clip) |
| `images` | IMAGE | — | *(optional)* Video frames to trim |
| `audio` | AUDIO | — | *(optional)* Audio to trim |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | Trimmed frames |
| `audio` | AUDIO | Trimmed audio |
| `fps` | FLOAT | Frame rate (passthrough) |
| `frame_count` | INT | Number of frames after trim |

---

#### RS Free VRAM

Passthrough utility node that forces VRAM cleanup between pipeline stages.

**Inputs:**

| Input | Type | Description |
|---|---|---|
| `any_input` | * (wildcard) | *(optional)* Any data type — passed through unchanged |

**Outputs:**

| Output | Type | Description |
|---|---|---|
| `output` | * (wildcard) | Same data as input |

**Key Behaviors:**
- Unloads all models, runs garbage collection, clears CUDA cache
- Prints VRAM freed (GB before/after) to console
- Clones tensor data before cleanup to preserve output
- Always re-executes (no caching)

---

## Workflow Examples

### Basic LTXV Video Generation

```
[Load LTXV Model] ──┐
[CLIP Text Encode] ──┼──→ [RS LTXV Generate] ──→ [Save Video]
[CLIP Text Encode] ──┤
[Load VAE] ──────────┘
```

### Audio-Driven Video with TTS

```
[RS Prompt Parser] ──→ [RS Prompt Formatter] ──→ [CLIP Text Encode] ──┐
        │                                                              │
        └──→ [RS MOSS TTS Loader] ──→ [RS MOSS TTS Batch Save] ──┐    │
                                              │                    │    │
                                      [RS Audio Concat] ──────────┼────┼──→ [RS LTXV Generate]
                                                                  │    │
                                                          (audio) ┘    └── (conditioning)
```

### IC-LoRA Structural Control

```
[Load Image] ──→ [RS Canny Preprocessor] ──→ [RS IC-LoRA Guider] ──→ [RS LTXV Generate]
                                                     │                       ↑
                                              (guider output) ──────→ (guider input)
```

### Video Extension

```
[RS LTXV Generate] ──→ (latent output) ──→ [RS LTXV Extend] ──→ [Save Video]
                                                   ↑
                                           [New Conditioning]
```

### Post-Processing Pipeline

```
[RS LTXV Generate] ──→ [RS Film Grain] ──→ [RS Video Trim] ──→ [Save Video]
```

---

## Tips & Troubleshooting

### VRAM Management

- **Place RS Free VRAM** between heavy inference nodes and post-processing to reclaim GPU memory.
- RS Film Grain and RS LTXV Generate both auto-clean VRAM before their GPU work.
- Use **`ffn_chunks`** (default: 4) and **`video_attn_scale`** (default: 1.03) to reduce VRAM during generation.
- Enable **`upscale_tiling`** for temporal tiling during upscale if running out of memory on long videos.
- Set **`upscale_fallback=True`** to gracefully fall back to half-res decode if upscale OOMs.

### Generation Quality

- **`cfg`**: 2.5–4.0 works well for video; higher values can cause artifacts.
- **`audio_cfg`**: 5.0–9.0 is typical; audio benefits from higher guidance than video.
- **`stg_scale`**: Start at 0 (disabled). Small values (0.1–0.5) can improve temporal consistency.
- **`rescale`**: 0.7 is a good default. Lower values reduce CFG artifacts at the cost of prompt adherence.
- **`cfg_star_rescale`**: Keep enabled to prevent high-sigma initialization artifacts.
- **Upscaling**: `upscale_denoise=0.3–0.6` balances sharpness vs. faithfulness. Steps of 3–6 are usually sufficient.

### Prompt Workflow

- Write scripts with `[s]`, `[a]`, `[d]` tags and feed into RS Prompt Parser.
- RS Prompt Formatter automatically caches to JSON — if the input prompt hasn't changed, Ollama is skipped entirely.
- The `dialogue_index` on RS Prompt Parser auto-increments, making it easy to iterate through dialogue lines in batch workflows.

### Audio

- When `audio` + `audio_vae` are both connected to RS LTXV Generate, the `num_frames` parameter is automatically overridden to match the audio duration.
- RS Audio Concat resamples all clips to the first clip's sample rate.
- RS MOSS TTS Batch Save's `one_shot` mode produces the most natural-sounding results for multi-line dialogue.

---

## License

See [LICENSE](LICENSE) for details.
