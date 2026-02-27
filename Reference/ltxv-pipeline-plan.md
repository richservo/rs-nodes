# RS LTXV Generate — All-in-One Video Pipeline

## Context
KJNodes and ComfyUI-LTXVideo break when core updates internal APIs. The current LTX workflow requires 12-15+ nodes for basic generation. We want 1-2 mega-nodes that internalize the entire pipeline — generation, frame injection, audio, efficiency optimizations, upscaling, VRAM management, and optional decode — while staying plug-and-play with room for advanced expansion.

## Architecture

### External (user connects):
- **model** — from checkpoint loader, with LoRAs pre-applied via Power LoRA Loader
- **positive/negative** — CONDITIONING from CLIP encode
- **vae** — Video VAE

### Internal (our node handles):
- Latent creation, frame injection, scheduling, sampling, guide cropping
- VRAM cleanup between passes (replaces RSFreeVRAM)
- SageAttention / attention mode selection
- FFN chunking for VRAM efficiency
- 2x latent upscale pass (optional)
- VAE decode (optional)

### Override hooks (optional inputs):
- GUIDER, SAMPLER, SIGMAS — override internal defaults when connected

---

## Node 1: `RSLTXVGenerate`
**Display name:** "RS LTXV Generate"
**Category:** `rs-nodes`

### Inputs

**Required:**
| Name | Type | Default | Notes |
|------|------|---------|-------|
| model | MODEL | | Pre-patched with LoRAs if needed |
| positive | CONDITIONING | | |
| negative | CONDITIONING | | |
| vae | VAE | | Video VAE |
| width | INT | 768 | step 32 |
| height | INT | 512 | step 32 |
| num_frames | INT | 97 | step 8, min 9 |
| steps | INT | 20 | |
| cfg | FLOAT | 3.0 | |
| seed | INT | 0 | |
| frame_rate | FLOAT | 25.0 | |

**Optional — Frame Injection:**
| Name | Type | Default | Notes |
|------|------|---------|-------|
| first_image | IMAGE | | Guide at frame 0 |
| middle_image | IMAGE | | Guide at midpoint |
| last_image | IMAGE | | Guide at frame -1 |
| first_strength | FLOAT | 1.0 | Per-guide attention strength |
| middle_strength | FLOAT | 1.0 | |
| last_strength | FLOAT | 1.0 | |
| image_strength | FLOAT | 0.95 | Denoise strength for conditioned frames |
| crf | INT | 35 | H.264 preprocessing quality |

**Optional — Audio:**
| Name | Type | Default | Notes |
|------|------|---------|-------|
| audio | AUDIO | | Audio waveform |
| audio_vae | VAE | | Audio VAE (required if audio provided) |

**Optional — Efficiency:**
| Name | Type | Default | Notes |
|------|------|---------|-------|
| attention_mode | combo | "auto" | [auto, default, sage] — auto detects best for GPU |
| ffn_chunks | INT | 0 | 0=disabled, 2-16 chunks for VRAM savings |

**Optional — Upscale:**
| Name | Type | Default | Notes |
|------|------|---------|-------|
| upscale | BOOLEAN | False | Enable 2x latent upscale pass |
| upscale_model | LATENT_UPSCALE_MODEL | | LTX upscaler (required if upscale=True) |
| upscale_steps | INT | 10 | Steps for upscale refinement pass |
| upscale_cfg | FLOAT | 3.0 | CFG for upscale pass |
| upscale_denoise | FLOAT | 0.5 | Denoise strength for upscale pass |

**Optional — Output:**
| Name | Type | Default | Notes |
|------|------|---------|-------|
| decode | BOOLEAN | False | Also decode to IMAGE |

**Optional — Overrides:**
| Name | Type | Notes |
|------|------|-------|
| guider | GUIDER | Overrides internal CFGGuider (for STG, etc.) |
| sampler | SAMPLER | Overrides internal euler_ancestral |
| sigmas | SIGMAS | Overrides internal LTXVScheduler |

**Optional — Scheduler (ignored if sigmas provided):**
| Name | Type | Default |
|------|------|---------|
| max_shift | FLOAT | 2.05 |
| base_shift | FLOAT | 0.95 |

### Outputs
| Name | Type | Notes |
|------|------|-------|
| latent | LATENT | Always output — denoised video latent |
| images | IMAGE | Only populated if decode=True |

### Internal Flow

```
1. SETUP
   ├── Clone model
   ├── If attention_mode != "default": set transformer_options["optimized_attention_override"]
   ├── If ffn_chunks > 0: apply FFN chunking patches via add_object_patch
   ├── EmptyLTXVLatentVideo(width, height, num_frames)
   └── conditioning_set_values(positive, {"frame_rate": frame_rate})

2. FRAME INJECTION (for each provided image)
   ├── LTXVPreprocess(image, crf)
   └── LTXVAddGuide(positive, negative, vae, latent, image, frame_idx, strength)

3. AUDIO (if provided)
   ├── audio_vae.encode(audio)
   └── Create AV NestedTensor combining video + audio latents

4. SCHEDULING
   ├── ModelSamplingLTXV(model, max_shift, base_shift)
   ├── If no sigmas: LTXVScheduler(steps, ...) → sigmas
   ├── If no sampler: create euler_ancestral
   └── If no guider: create CFGGuider(model, positive, negative, cfg)

5. SAMPLE (first pass)
   └── SamplerCustomAdvanced(noise, guider, sampler, sigmas, latent)

6. POST-SAMPLE
   ├── LTXVCropGuides(latent)
   └── If audio: LTXVSeparateAVLatent → extract video latent

7. UPSCALE (if enabled)
   ├── VRAM cleanup: unload diffusion model, gc.collect(), empty_cache()
   ├── vae.un_normalize(latent) → upscale_model(latent) → vae.normalize(latent)
   ├── Optionally re-sample at upscaled resolution with upscale_denoise
   └── VRAM cleanup: unload upscale model

8. DECODE (if enabled)
   ├── VRAM cleanup if needed
   └── vae.decode(latent) → images

9. RETURN (latent, images)
```

---

## Node 2: `RSLTXVExtend` — Video Extension
**Display name:** "RS LTXV Extend"

Same efficiency features (attention, FFN chunking), extends existing video with overlap blending.

### Inputs
Same pattern as RSLTXVGenerate but replaces width/height/num_frames with:
| Name | Type | Default | Notes |
|------|------|---------|-------|
| latent | LATENT | | Existing video to extend |
| num_new_frames | INT | 80 | step 8 |
| overlap_frames | INT | 16 | step 8 |
| overlap_strength | FLOAT | 1.0 | |

Plus same optional inputs: frame injection (last_image), audio, efficiency, upscale, decode, overrides.

### Internal Flow
1. Extract last N overlap frames from input latent as guide
2. Create empty latent for extension chunk (overlap + new_frames)
3. Inject overlap frames as guide at position 0
4. If last_image: inject as guide at end
5. Apply efficiency patches (attention, FFN chunking)
6. Sample extension chunk
7. Linear blend overlap region
8. Concatenate original[:-overlap] + blended + new
9. If upscale/decode: same as RSLTXVGenerate

---

## Efficiency Implementation Details

### Attention Mode
Uses ComfyUI's built-in attention override system (NOT monkey-patching):
```python
# On cloned model's transformer_options:
if attention_mode == "sage":
    from comfy.ldm.modules.attention import attention_sage
    model.model_options["transformer_options"]["optimized_attention_override"] = attention_sage
elif attention_mode == "auto":
    # Detect: sage if available, else pytorch SDPA
```
This is a stable core API — the `@wrap_attn` decorator in `attention.py` checks this key.

### FFN Chunking
Uses ComfyUI's `add_object_patch` API (stable, used by many nodes):
```python
for idx, block in enumerate(model.diffusion_model.transformer_blocks):
    # Replace ff.forward with chunked version
    model_clone.add_object_patch(
        f"diffusion_model.transformer_blocks.{idx}.ff.forward",
        chunked_forward_bound_method
    )
```
Splits sequence dimension into N chunks, processes sequentially. Trades speed for VRAM.

### Upscale Pass
Uses core's `LatentUpsampler` with VAE normalization:
```python
# 1. Un-normalize latent using VAE's per-channel statistics
latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
# 2. Run upscaler
upsampled = upscale_model(latents)
# 3. Re-normalize
upsampled = vae.first_stage_model.per_channel_statistics.normalize(upsampled)
```
Upscale model loaded via combo selector (`LATENT_UPSCALE_MODEL` type, loaded by core's `LatentUpscaleModelLoader`).

### VRAM Management
Built into the pipeline at stage transitions:
```python
def _free_vram(self):
    mm.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()
    mm.soft_empty_cache()
```
Called between: generation → upscale → decode.

---

## Key Core APIs

### LTX Video Core Nodes (from `comfy_extras/nodes_lt.py`)
```python
from comfy_extras.nodes_lt import (
    LTXVAddGuide,          # Add image guide to conditioning + latent
    LTXVPreprocess,        # H.264 preprocess image for guide injection
    LTXVCropGuides,        # Remove guide frames from final latent
    EmptyLTXVLatentVideo,  # Create empty latent at given resolution
    LTXVScheduler,         # LTX-specific sigma schedule
    ModelSamplingLTXV,     # Apply LTX shift-based model sampling
)
```

### LTX Audio Core Nodes (from `comfy_extras/nodes_lt_audio.py`)
```python
from comfy_extras.nodes_lt_audio import (
    LTXVConcatAVLatent,    # Combine video + audio latents into NestedTensor
    LTXVSeparateAVLatent,  # Split AV latent back into video + audio
)
```

### Sampling Core (from `comfy_extras/nodes_custom_sampler.py`)
```python
from comfy_extras.nodes_custom_sampler import (
    SamplerCustomAdvanced,  # Full custom sampling with guider/sampler/sigmas
)
```

### Attention (from `comfy/ldm/modules/attention.py`)
```python
from comfy.ldm.modules.attention import attention_sage  # SageAttention function
```

### Other Core
```python
import comfy.samplers          # CFGGuider, KSamplerX0Inpainting
import comfy.model_management as mm  # VRAM management
import node_helpers            # conditioning_set_values
```

---

## Files
- **New:** `nodes/ltxv_generate.py` — RSLTXVGenerate
- **New:** `nodes/ltxv_extend.py` — RSLTXVExtend
- **Modified:** `__init__.py` — register both nodes

## Implementation Order
1. `RSLTXVGenerate` — core generation with frame injection + basic sampling
2. Add efficiency features (attention, FFN chunking)
3. Add upscale pass
4. Add audio support
5. `RSLTXVExtend` — reuses patterns from RSLTXVGenerate

## Verification Checklist
1. Basic T2V: model + prompt only → video output
2. I2V: first_image → frame conditioning works
3. First + last guides → both positions correct
4. External LoRAs via Power LoRA Loader → weights applied
5. External STG guider → override works
6. External sampler/sigmas → override works
7. attention_mode=sage → SageAttention active (check console)
8. ffn_chunks=8 → reduced VRAM peak
9. upscale=True → 2x upscaled output
10. decode=True → IMAGE output populated
11. Audio + audio_vae → AV generation works
12. RSLTXVExtend → seamless video continuation
13. Full pipeline with upscale → no OOM (internal VRAM cleanup)
