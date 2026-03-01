# RS IC-LoRA Guider -- Usage Guide

## 1. Overview

The **RS IC-LoRA Guider** node brings structural control to LTX-2 video generation. Using a single Union IC-LoRA file, you can guide video output with depth maps, canny edges, or pose skeletons. The LoRA auto-detects the control type from your input image -- no mode selector needed. Feed it a depth map and you get depth-controlled video; feed it a canny edge map and you get edge-controlled video; feed it a pose skeleton and you get pose-controlled video.

The node outputs a GUIDER that plugs directly into the RS LTXV Generate node's `guider` input, replacing the default guidance pipeline with one that includes your control signal. Everything else -- audio lip sync, first-frame image-to-video, upscaling -- continues to work as usual.

---

## 2. Prerequisites

**Base model**: LTX-2 (the base LTXV model). This is the same model you already use with RS LTXV Generate.

**IC-LoRA file**: Download `ltx-2-19b-ic-lora-union-ref0.5.safetensors` from the [Lightricks LTX-2 IC-LoRA Union Control](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Union-Control) repository on HuggingFace. Place it in your ComfyUI `models/loras/` directory.

**Control image preprocessors**: You need ComfyUI preprocessor nodes to convert your source image into a control signal before feeding it to the guider. Install a preprocessor pack such as [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) or equivalent. You will use nodes like:

- Canny edge detection
- Depth estimation (Lotus Depth, MiDaS, Zoe Depth)
- Pose estimation (DWPose, OpenPose)

---

## 3. Basic Workflow (Video Only)

This is the simplest IC-LoRA workflow: controlled video generation without audio.

### Step-by-step

1. **Load your source image.** Use a Load Image node or any node that provides an IMAGE output.

2. **Run it through a preprocessor.** Connect your source image to a preprocessor node matching the control type you want:
   - For depth control: use a depth estimation node (e.g., Lotus Depth Anything)
   - For edge control: use a Canny edge detection node
   - For pose control: use a DWPose or OpenPose node

   The preprocessor output is the control image -- a depth map, edge map, or pose skeleton.

3. **Connect to RS IC-LoRA Guider.** Wire the preprocessor output to the `control_image` input. Also connect:
   - `model` -- your loaded LTX-2 model
   - `positive` -- your positive CONDITIONING (from a text encode node)
   - `negative` -- your negative CONDITIONING
   - `vae` -- the LTX-2 VAE
   - `ic_lora` -- select `ltx-2-19b-ic-lora-union-ref0.5.safetensors` from the dropdown

4. **Connect guider output to RS LTXV Generate.** Wire the GUIDER output from the IC-LoRA node into the `guider` input on RS LTXV Generate.

5. **Set your prompt on RS LTXV Generate.** Standard text prompting works as normal. The prompt describes the content you want; the control image constrains the structure.

6. **Leave key settings at defaults.** The recommended starting point:
   - `lora_strength` = 1.0
   - `guide_strength` = 1.0
   - `video_cfg` = 3.0

### Node connection summary

```
[Load Image] --> [Preprocessor (Canny/Depth/Pose)] --> control_image
                                                             |
[Load Checkpoint] --> model -----> [RS IC-LoRA Guider] --> GUIDER --> [RS LTXV Generate]
[CLIP Text Encode] --> positive -->        |
[CLIP Text Encode] --> negative -->        |
[VAE Loader] -------> vae ------->         |
                       ic_lora: ltx-2-19b-ic-lora-union-ref0.5.safetensors
```

Note: When using an external guider, the `cfg`, `stg_scale`, `stg_blocks`, `rescale`, `modality_scale`, `cfg_end`, and `stg_end` settings on the RS LTXV Generate node are ignored. Those parameters are controlled on the IC-LoRA Guider node instead. The generate node still handles `width`, `height`, `num_frames`, `steps`, `noise_seed`, `frame_rate`, and all other non-guidance settings.

---

## 4. Workflow with Audio Lip Sync

IC-LoRA control and audio lip sync are fully compatible. Audio injection happens at the generate node level, completely independent of the guider. The IC-LoRA guider handles video control; the generate node handles audio.

### Step-by-step

1. **Set up IC-LoRA control exactly as in Section 3** -- preprocessor, guider node, all the same connections.

2. **Connect audio to RS LTXV Generate.** Wire your audio source to the `audio` input and a loaded audio VAE to the `audio_vae` input on the generate node. This is the same as any audio lip sync workflow.

3. **That is it.** No extra configuration needed. When audio is connected, the generate node automatically:
   - Derives video length from audio duration
   - Packs video and audio latents together
   - Passes the combined latent to the IC-LoRA guider
   - The guider processes them correctly, applying control to video only

4. **Set audio guidance on the IC-LoRA Guider node.** The `audio_cfg` parameter on the guider controls audio classifier-free guidance strength (default 7.0). This replaces the `audio_cfg` setting on the generate node when using an external guider.

### Node connection summary

```
[Preprocessor] --> control_image --> [RS IC-LoRA Guider] --> GUIDER --+
[Model/CLIP/VAE] --> model/pos/neg/vae -->       |                    |
                                                                      v
[Load Audio] ---------> audio -------> [RS LTXV Generate]
[Audio VAE Loader] ---> audio_vae --->         |
```

---

## 5. Workflow with First Image (I2V + IC-LoRA)

You can use a reference first frame (image-to-video) alongside IC-LoRA control. The first image sets the visual identity and appearance of the opening frame; the IC-LoRA control image constrains the structural layout across all frames.

### How it works

First image injection uses "inplace" latent injection -- it directly writes encoded pixels into the first position of the latent tensor. This happens at the generate node before the guider is called, so the guider receives a latent that already contains the first frame. Both mechanisms work together without conflict.

### Step-by-step

1. **Connect your first frame image** to the `first_image` input on RS LTXV Generate. Optionally adjust `first_strength` (default 1.0 means pixel-perfect preservation of the first frame).

2. **Connect the IC-LoRA guider** to the `guider` input on RS LTXV Generate, exactly as in Section 3.

3. **Both work together.** The first image anchors the visual content of frame 1. The IC-LoRA control image guides the structural layout of the generated video. The text prompt describes the desired content and motion.

### Node connection summary

```
[Preprocessor] --> control_image --> [RS IC-LoRA Guider] --> GUIDER --+
[Model/CLIP/VAE] --> ...                                              |
                                                                      v
[Load Image] --> first_image --> [RS LTXV Generate]
```

### Note on middle/last image injection

When using an external guider, middle and last image guide conditioning set on the generate node will not take effect (those modify the generate node's local conditioning, which the external guider does not use). If you need structural control at specific frames, the IC-LoRA control image is the recommended approach. First image injection still works because it modifies the latent tensor directly.

---

## 6. Control Types

The Union IC-LoRA auto-detects the control type from your preprocessed image. Make sure your control image is clearly one type -- do not mix control signals in a single image.

### Depth

**Best for**: Preserving 3D structure, camera geometry, spatial layout, scene composition.

**Preprocessors**: Lotus Depth Anything, MiDaS, Zoe Depth, Depth Anything V2.

**Typical settings**: `guide_strength` = 1.0

**When to use**: You have a scene and want the generated video to match its spatial depth -- foreground/background separation, object distances, room geometry. Good for architectural scenes, landscapes, and maintaining consistent spatial relationships.

### Canny

**Best for**: Preserving edges, silhouettes, fine outlines, text, and hard boundaries.

**Preprocessors**: Canny edge detection.

**Typical settings**: `guide_strength` = 1.0

**When to use**: You want the generated video to follow precise edge boundaries. Good for maintaining sharp outlines of objects, preserving text layout, and style transfer where you want to keep the original composition's line work.

### Pose

**Best for**: Character animation, body movement, gesture control.

**Preprocessors**: DWPose (DWPreprocessor), OpenPose.

**Typical settings**: `guide_strength` = 1.0

**When to use**: You want to control how a character moves -- their body pose, limb positions, and gestures. Good for dance choreography, specific actions, and matching reference performances.

---

## 7. Parameter Reference

### Required Inputs

| Parameter | Type | Description |
|---|---|---|
| `model` | MODEL | The base LTX-2 model. Same model you use with RS LTXV Generate. |
| `positive` | CONDITIONING | Positive text conditioning from a CLIP text encode node. |
| `negative` | CONDITIONING | Negative text conditioning from a CLIP text encode node. |
| `vae` | VAE | The LTX-2 VAE, used to encode the control image into latent space. |
| `control_image` | IMAGE | Preprocessed control image (depth map, canny edges, or pose skeleton). Must be preprocessed before connecting -- the node does not run any preprocessing. |
| `ic_lora` | Dropdown | LoRA file selector. Choose `ltx-2-19b-ic-lora-union-ref0.5.safetensors`. |

### IC-LoRA Settings

| Parameter | Default | Range | Description |
|---|---|---|---|
| `lora_strength` | 1.0 | -10.0 to 10.0 | Strength of the IC-LoRA weights. **Always use 1.0** -- lower values cause artifacts because the LoRA was trained at full strength. |

### Control Settings

| Parameter | Default | Range | Description |
|---|---|---|---|
| `guide_strength` | 1.0 | 0.0 to 1.0 | How strictly the output follows the control image. 1.0 = maximum adherence. Lower values allow more creative deviation from the control signal. |
| `guide_frame_idx` | 0 | 0+ | Which frame index the control image conditions. Usually leave at 0 (first frame). |
| `crf` | 35 | 0 to 100 | Preprocessing quality for the control image encoding. Matches the CRF parameter used in the generate node for guide frame preprocessing. Usually leave at default. |

### Guidance Settings

These replace the equivalent settings on RS LTXV Generate when using the external guider.

| Parameter | Default | Range | Description |
|---|---|---|---|
| `video_cfg` | 3.0 | 0.0 to 100.0 | Video classifier-free guidance scale. Higher = stronger prompt adherence, lower = more natural/creative. 3.0 is a good starting point. |
| `audio_cfg` | 7.0 | 0.0 to 100.0 | Audio classifier-free guidance scale. Only relevant when audio is connected. |
| `stg_scale` | 0.0 | 0.0 to 10.0 | Spatiotemporal guidance scale. Adds temporal coherence. 0.0 = disabled. |
| `stg_blocks` | "29" | String | Which transformer block(s) to apply STG to. Comma-separated integers. Default "29" targets the last block. |
| `rescale` | 0.7 | 0.0 to 1.0 | CFG rescale factor. Reduces CFG artifacts by normalizing the noise prediction magnitude. |
| `modality_scale` | 1.0 | 0.0 to 100.0 | Cross-modal guidance scale. Only relevant for audio-video generation. 1.0 = no cross-modal guidance. |
| `cfg_end` | -1.0 | -1.0 to 100.0 | CFG rollback target. The video CFG linearly interpolates from `video_cfg` to `cfg_end` over the sampling steps. **-1.0 means no rollback** (constant CFG throughout). |
| `stg_end` | -1.0 | -1.0 to 10.0 | STG rollback target. Same as cfg_end but for STG scale. **-1.0 means no rollback.** |

### Scheduler Settings

| Parameter | Default | Range | Description |
|---|---|---|---|
| `max_shift` | 2.05 | 0.0 to 100.0 | Maximum shift for the LTXV model sampling schedule. Controls the noise schedule shape. Should match what you would use on the generate node. |
| `base_shift` | 0.95 | 0.0 to 100.0 | Base shift for the LTXV model sampling schedule. Should match what you would use on the generate node. |

### What you can usually leave at defaults

For most workflows, you only need to set:
- `ic_lora` (select the LoRA file)
- `video_cfg` (if you want to adjust prompt strength)
- `guide_strength` (if you want looser control)

Everything else can stay at defaults.

---

## 8. Tips and Troubleshooting

### LoRA strength must be 1.0

The Union IC-LoRA was trained at full strength. Setting `lora_strength` below 1.0 does not gently reduce the control effect -- it causes visual artifacts and degraded output. If you want less control adherence, lower `guide_strength` instead.

### guide_strength is your main control knob

`guide_strength` determines how strongly the output follows the control image. At 1.0, the structure is tightly constrained. Lowering it (e.g., 0.7 or 0.5) gives the model more freedom to deviate from the control signal while still using it as a loose guide.

### Verify your control image preprocessing

If the output does not follow the control signal as expected, check the preprocessor output first. View the preprocessed image directly to confirm it looks correct:
- Depth maps should show clear near/far gradients
- Canny edges should show clean, well-defined lines (adjust thresholds if too noisy or too sparse)
- Pose skeletons should accurately represent the body positions

### The Union LoRA auto-detects control type

Because the LoRA determines control type from the input image itself, your control image must be unambiguous. A depth map that happens to look like an edge map may confuse the detection. Use standard preprocessor outputs and the auto-detection works reliably.

### Model sampling parameters should match

The `max_shift` and `base_shift` values on the IC-LoRA Guider should match what you would normally use on the generate node. The defaults (2.05 and 0.95) are the standard LTX-2 values and work well in most cases.

### CFG/STG rollback (-1.0 means disabled)

Setting `cfg_end` or `stg_end` to -1.0 disables the rollback schedule, meaning guidance stays constant throughout all sampling steps. If you set a positive value, the guidance linearly interpolates from the starting value to the end value over the course of sampling.

### The generate node's guidance settings are ignored

When you connect an external guider, the generate node's `cfg`, `audio_cfg`, `stg_scale`, `stg_blocks`, `rescale`, `modality_scale`, `cfg_end`, and `stg_end` inputs are bypassed. Set all guidance-related parameters on the IC-LoRA Guider node.

### Settings that still matter on the generate node

Even with an external guider, the generate node still controls:
- `width`, `height`, `num_frames` (generation dimensions)
- `steps` (sampling steps)
- `noise_seed`, `seed_mode` (randomization)
- `frame_rate` (video frame rate)
- `first_image`, `first_strength` (I2V reference frame)
- `audio`, `audio_vae` (audio lip sync)
- `upscale` and all upscale settings
- `attention_mode`, `ffn_chunks` (efficiency)
- `decode` (whether to decode latents to images)

### Combining with first image

When using both `first_image` (on the generate node) and IC-LoRA control, the first image defines the visual appearance of frame 1 while IC-LoRA controls the structural layout. This is useful when you want a specific starting appearance with controlled motion or structure throughout the video.

### VRAM considerations

The IC-LoRA guider adds minimal overhead. The LoRA weights are small, and the control image encoding is a single VAE encode pass. The main VRAM cost remains the same as standard generation.
