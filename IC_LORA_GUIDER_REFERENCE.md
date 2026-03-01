# IC-LoRA Guider Node — Technical Reference & Implementation Plan

## Table of Contents
1. [What Union IC-LoRA Is](#1-what-union-ic-lora-is)
2. [How IC-LoRA Conditioning Works](#2-how-ic-lora-conditioning-works)
3. [Existing Pipeline Architecture](#3-existing-pipeline-architecture)
4. [Audio Lip Sync Compatibility](#4-audio-lip-sync-compatibility)
5. [Design Issues & Solutions](#5-design-issues--solutions)
6. [Node Design — RSICLoRAGuider](#6-node-design--rsicloraguider)
7. [Implementation Plan](#7-implementation-plan)

---

## 1. What Union IC-LoRA Is

**IC-LoRA** (In-Context LoRA / Image Conditioning LoRA) is a LoRA variant that
enables structural/spatial control over video generation — depth, canny edges,
and pose — using the same diffusion transformer with minimal additional weights.

**Union** means a single LoRA file handles all three control types. It
auto-detects the control type from the input image — no mode selector needed.
A depth map produces depth-controlled output, a canny edge map produces
edge-controlled output, a pose skeleton produces pose-controlled output.

### Key characteristics

| Property | Value |
|---|---|
| LoRA strength | **Always 1.0** — lower values cause artifacts |
| Model file | `ltx-2-19b-ic-lora-union-ref0.5.safetensors` (Lightricks) |
| Reference downscale factor | **2** (stored in LoRA safetensors metadata) |
| Control types | Depth, Canny, Pose (auto-detected) |
| Architecture changes | None — standard LoRA applied to transformer |
| CFG formula | **Unchanged** — standard CFG/STG works as-is |

### Reference downscale factor

The Union IC-LoRA was trained with reference images at **half resolution**. At
inference, the control image must be encoded at `target_resolution / 2` to match
training conditions. The factor is stored in the LoRA safetensors metadata:

```python
from safetensors import safe_open
with safe_open(lora_path, framework="pt") as f:
    metadata = f.metadata() or {}
    downscale_factor = int(metadata.get("reference_downscale_factor", 1))
```

### Control image preprocessing

The control image must be preprocessed BEFORE the guider node. Use standard
ComfyUI preprocessor nodes:
- **Depth**: Lotus Depth, MiDaS, Zoe Depth, etc.
- **Canny**: Canny edge detection node
- **Pose**: DWPreprocessor (DWPose), OpenPose, etc.

The guider node receives the already-preprocessed control image.

---

## 2. How IC-LoRA Conditioning Works

**The critical insight: IC-LoRA does NOT modify the guidance formula. It modifies
how the model input is prepared.**

The control signal is injected as a **guide keyframe** — the same mechanism used
for first/middle/last frame injection in the existing generate node.

### Injection flow

```
Control Image (preprocessed depth/canny/pose)
    │
    ▼
Encode at DOWNSCALED resolution (target / downscale_factor)
    │
    ▼
VAE encode → guide latent (smaller spatial dims)
    │
    ▼
LTXVAddGuide.append_keyframe()
    ├─ Appends guide latent to END of main latent tensor (temporal concat)
    ├─ Sets noise_mask ≈ 0 for guide frames (preserve during denoising)
    └─ Stores keyframe_idxs in conditioning metadata
    │
    ▼
Model sees guide frames as context during denoising
(IC-LoRA weights enable the model to interpret control signals)
    │
    ▼
After sampling → strip guide frames from output
```

### What `append_keyframe()` does

1. **Latent tensor**: Concatenates guide latent frames at the end of the video
   latent along the temporal dimension
2. **Noise mask**: Sets `1.0 - strength` for guide positions (strength=1.0 →
   mask=0.0 → fully preserved/clean pixels)
3. **Conditioning**: Adds `keyframe_idxs` metadata so the model knows which
   frames are guides vs. generation targets
4. **Returns**: Updated (positive, negative, latent_samples, noise_mask)

### Guide latent spatial dimensions

For Union IC-LoRA (downscale_factor=2):
- Target video latent: `[B, 128, T, H/32, W/32]`
- Guide latent: `[B, 128, 1, H/64, W/64]` (half spatial)

The `LTXVAddGuide.encode()` method handles resize-to-latent-dims internally.
For IC-LoRA, the control image is encoded at the downscaled pixel resolution,
producing a spatially smaller latent. The official LTX-2 pipeline handles the
spatial mismatch at the token/attention level — the transformer processes
variable-length token sequences with positional encoding.

In ComfyUI's `append_keyframe()`, the guide latent spatial dims must match the
main latent. Two approaches:
- **A**: Encode at full resolution but from a downscaled-then-upscaled source
  image (the VAE sees half-res content at full-res encoding) — matches how the
  LoRA was trained
- **B**: Encode at half resolution, then spatially upsample the latent
  (nearest/bilinear) to match main latent dims

Approach A is simpler and more reliable. Encode flow:
1. Resize control image to `(target_H / downscale_factor, target_W / downscale_factor)`
2. Re-upscale to target pixel resolution (bilinear) — content is half-res but tensor is full-size
3. VAE encode at target resolution
4. Append as guide keyframe (spatial dims match)

This preserves the half-resolution characteristics that the Union IC-LoRA was
trained to interpret while keeping tensor dimensions compatible.

---

## 3. Existing Pipeline Architecture

### Generate node flow (9 stages)

```
┌─ RSLTXVGenerate ─────────────────────────────────────────────┐
│                                                               │
│  1. SETUP          model clone, attention, ffn, latent creation│
│  2. FRAME INJECT   first (inplace), middle/last (guide cond)  │
│  3. AUDIO INJECT   NestedTensor(video, audio) packing         │
│  4. SCHEDULING     model sampling shift, sigmas                │
│  5. GUIDER         MultimodalGuider OR external override ◄────┤
│  6. SAMPLE         guider.sample(noise, latent, sampler, ...)  │
│  7. POST-SAMPLE    separate AV, crop guide keyframes           │
│  8. UPSCALE        optional latent upscale + re-diffusion      │
│  9. DECODE         tiled VAE decode for video + audio          │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                              ▲
                              │
                     guider input (GUIDER override)
                              │
               ┌──────────────┴──────────────┐
               │   RSICLoRAGuider (new node)  │
               │   Outputs: GUIDER            │
               └─────────────────────────────┘
```

### MultimodalGuider guidance formula

```python
noise_pred = (
    pos
    + (cfg - 1) * (pos - neg)                    # CFG term
    + stg_scale * (pos - perturbed)              # STG term
    + (modality_scale - 1) * (pos - modality)    # Cross-modal term
)
```

**This formula does NOT change for IC-LoRA.** The IC-LoRA control is entirely
in the input preparation (guide keyframes), not in the guidance math.

### How the generate node uses the guider override

```python
# Line 425 of ltxv_generate.py
if guider is None:
    guider = MultimodalGuider(m, positive, negative, ...)

# Line 443 — uses guider.model_patcher for channel fix
latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)

# Line 454 — calls guider.sample()
samples = guider.sample(noise, latent_image, sampler, sigmas, ...)
```

When an external guider is passed:
- The generate node's local `m` (model clone) is NOT used by the guider
- The generate node's local `positive`/`negative` conditioning is NOT used by the guider
- The guider has its own model_patcher and conditioning
- The generate node DOES still create the latent tensor, noise, and noise_mask
- The generate node DOES still handle audio NestedTensor packing

---

## 4. Audio Lip Sync Compatibility

### Why it's compatible

Audio injection happens at the **generate node level** (stage 3), AFTER the
guider is created but BEFORE `guider.sample()` is called. The flow:

1. Generate node creates video latent
2. Generate node does frame injection (stage 2) — modifies local latent
3. Generate node packs `NestedTensor(video_latent, audio_latent)` (stage 3)
4. Generate node calls `guider.sample(noise, packed_latent, ...)` (stage 5)

The IC-LoRA guider receives the already-packed NestedTensor. The
MultimodalGuider's `sample()` method peeks at the NestedTensor to get per-
modality shapes, then `predict_noise()` unpacks → processes → repacks at each
step. **No changes needed.**

### Audio guidance isolation (last commit `29050be`)

Audio CFG uses `self.audio_cfg` (constant) and STG uses `self.stg_scale`
(constant) — both are isolated from the video guidance rollback schedule.
This remains unchanged with IC-LoRA.

```python
# Video: uses interpolated cur_cfg and cur_stg (rollback schedule)
v_out = self._calculate(v_pos, v_neg, v_stg, v_mod, cur_cfg, cur_stg)

# Audio: uses constant self.audio_cfg and self.stg_scale (no rollback)
a_out = self._calculate(a_pos, a_neg, a_stg, a_mod, self.audio_cfg, self.stg_scale)
```

---

## 5. Design Issues & Solutions

### Issue 1: Model sampling shift

The generate node computes a **shift** for `ModelSamplingFlux` based on token
count (latent dimensions). It applies this to its local model `m`. When an
external guider is used, `m` isn't the guider's model — the shift is lost.

**Solution**: The IC-LoRA guider takes `max_shift` and `base_shift` as inputs
and applies model sampling in its own `sample()` method, where the latent
dimensions are known.

```python
def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
    # Apply model sampling shift (deferred until we know latent dims)
    self._apply_model_sampling(latent_image)
    return super().sample(noise, latent_image, sampler, sigmas, **kwargs)
```

### Issue 2: Guide keyframe management

The IC-LoRA guide latent needs to be:
- **Appended** to the latent tensor before sampling
- **Reflected** in the guider's conditioning (keyframe_idxs)
- **Stripped** from the output after sampling

The generate node's post-sample cleanup (stage 7) crops keyframes using
`get_keyframe_idxs(positive)` — but its local `positive` won't have the IC-LoRA
keyframes (those are in the guider's conditioning).

**Solution**: The IC-LoRA guider handles guide frame lifecycle internally:
1. During construction: modify conditioning with `append_keyframe()`, store the
   modified latent appendage and keyframe count
2. In `sample()`: append guide latent to input, call parent, strip from output
3. The generate node's post-sample cleanup sees 0 keyframes (its local
   `positive` is unmodified) — no conflict

### Issue 3: Frame injection interaction

When an external guider is used, the generate node's middle/last image guide
conditioning (via `append_keyframe`) modifies local `positive`/`negative` which
the guider doesn't see. First image inplace injection still works (it modifies
the latent tensor directly, which IS passed to the guider).

**Solution for now**: For IC-LoRA workflows, the control image replaces the
role of middle/last frame guides. First image (I2V reference) still works since
it's inplace. If combined middle/last + IC-LoRA is needed later, the guider
node can accept additional guide images as optional inputs.

### Issue 4: NestedTensor guide frame appending

When audio is active, the latent is a NestedTensor. `append_keyframe()` needs
a plain tensor. The guide frames are video-only (no audio guide frames).

**Solution**: In `sample()`, if the latent is a NestedTensor:
1. Unbind → `[video_latent, audio_latent]`
2. Append IC-LoRA guide frames to video_latent only
3. Repack as NestedTensor (video is now longer by N guide frames, audio unchanged)
4. After sampling, unbind again, strip guide frames from video, repack

This keeps audio latent untouched — lip sync is fully preserved.

---

## 6. Node Design — RSICLoRAGuider

### Node overview

```
┌─ RS IC-LoRA Guider ──────────────────────────────────┐
│                                                       │
│  Required inputs:                                     │
│    model        (MODEL)    — base model               │
│    positive     (CONDITIONING)                        │
│    negative     (CONDITIONING)                        │
│    vae          (VAE)      — for encoding control img │
│    control_image (IMAGE)   — preprocessed depth/canny/pose │
│    ic_lora      (combo)    — LoRA file selector       │
│                                                       │
│  Optional inputs (with defaults):                     │
│    lora_strength   FLOAT  1.0                         │
│    guide_strength  FLOAT  1.0   (guide adherence)     │
│    video_cfg       FLOAT  3.0                         │
│    audio_cfg       FLOAT  7.0                         │
│    stg_scale       FLOAT  0.0                         │
│    stg_blocks      STRING "29"                        │
│    rescale         FLOAT  0.7                         │
│    modality_scale  FLOAT  1.0                         │
│    cfg_end         FLOAT  -1.0  (rollback target)     │
│    stg_end         FLOAT  -1.0  (rollback target)     │
│    max_shift       FLOAT  2.05                        │
│    base_shift      FLOAT  0.95                        │
│    guide_frame_idx INT    0     (which frame to condition) │
│    crf             INT    35    (preprocessing quality)│
│                                                       │
│  Output:                                              │
│    GUIDER  → plugs into RSLTXVGenerate guider input   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Workflow

```
[Preprocessor Node]          [LoRA Selector]
  (Depth/Canny/Pose)              │
        │                         │
        ▼                         ▼
┌─ RS IC-LoRA Guider ────────────────────┐
│  1. Load & apply IC-LoRA to model      │
│  2. Read reference_downscale_factor    │
│  3. Encode control image (downscaled)  │
│  4. Add as guide to conditioning       │
│  5. Create ICLoRAGuider (extends       │
│     MultimodalGuider)                  │
└─────────────┬──────────────────────────┘
              │ GUIDER
              ▼
┌─ RS LTXV Generate ─────────────────────┐
│  (unchanged — uses guider override)    │
│  Audio injection still happens here    │
│  First image I2V still works           │
└────────────────────────────────────────┘
```

---

## 7. Implementation Plan

### File structure

```
rs-nodes/
  nodes/
    ic_lora_guider.py    ← NEW: RSICLoRAGuider node
  utils/
    multimodal_guider.py ← MODIFY: extract ICLoRAGuider subclass
  web/
    ic_lora_guider.js    ← NEW: UI section headers
  __init__.py            ← MODIFY: register new node
```

### Step 1: ICLoRAGuider class (`utils/multimodal_guider.py`)

Create `ICLoRAGuider(MultimodalGuider)` that adds:

```python
class ICLoRAGuider(MultimodalGuider):
    def __init__(self, model, positive, negative,
                 guide_latent,           # Encoded control image latent
                 num_guide_frames,       # How many frames appended
                 max_shift, base_shift,  # For model sampling
                 **kwargs):              # All MultimodalGuider params
        super().__init__(model, positive, negative, **kwargs)
        self._guide_latent = guide_latent
        self._num_guide_frames = num_guide_frames
        self._max_shift = max_shift
        self._base_shift = base_shift

    def _apply_model_sampling(self, latent_image):
        """Apply ModelSamplingFlux shift based on latent dims."""
        # Compute token count from video latent (exclude audio if nested)
        if latent_image.is_nested:
            tokens = math.prod(latent_image.unbind()[0].shape[2:])
        else:
            tokens = math.prod(latent_image.shape[2:])
        # Standard LTXV shift formula
        mm_shift = (self._max_shift - self._base_shift) / (4096 - 1024)
        b = self._base_shift - mm_shift * 1024
        shift = tokens * mm_shift + b
        # Apply to model
        sampling_cls = type("ModelSamplingAdv",
                           (ModelSamplingFlux, CONST), {})
        sampling_obj = sampling_cls(self.model_patcher.model.model_config)
        sampling_obj.set_parameters(shift=shift)
        self.model_patcher.add_object_patch("model_sampling", sampling_obj)

    def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
        # 1. Apply model sampling shift
        self._apply_model_sampling(latent_image)

        # 2. Append IC-LoRA guide frames to the latent
        denoise_mask = kwargs.get("denoise_mask", None)
        latent_image, denoise_mask = self._append_guide(
            latent_image, denoise_mask
        )
        if denoise_mask is not None:
            kwargs["denoise_mask"] = denoise_mask

        # 3. Fix noise to match expanded latent
        noise = comfy.sample.prepare_noise(latent_image, kwargs.get("seed", 0))

        # 4. Run parent sampling
        result = super().sample(noise, latent_image, sampler, sigmas, **kwargs)

        # 5. Strip guide frames from output
        return self._strip_guide(result)

    def _append_guide(self, latent_image, denoise_mask):
        """Append guide latent frames. Handles NestedTensor for AV."""
        if latent_image.is_nested:
            parts = latent_image.unbind()
            video = parts[0]
            audio = parts[1]
            video = torch.cat([video, self._guide_latent], dim=2)
            latent_image = NestedTensor((video, audio))
            if denoise_mask is not None:
                dm_parts = denoise_mask.unbind()
                guide_mask = torch.zeros(
                    *self._guide_latent.shape[:2], self._guide_latent.shape[2], 1, 1
                )
                video_dm = torch.cat([dm_parts[0], guide_mask], dim=2)
                denoise_mask = NestedTensor((video_dm, dm_parts[1]))
        else:
            latent_image = torch.cat([latent_image, self._guide_latent], dim=2)
            if denoise_mask is not None:
                guide_mask = torch.zeros(...)
                denoise_mask = torch.cat([denoise_mask, guide_mask], dim=2)
        return latent_image, denoise_mask

    def _strip_guide(self, result):
        """Remove guide frames from sampled output."""
        n = self._num_guide_frames
        if n == 0:
            return result
        if result.is_nested:
            parts = result.unbind()
            video = parts[0][:, :, :-n]
            return NestedTensor((video, parts[1]))
        return result[:, :, :-n]
```

### Step 2: RSICLoRAGuider node (`nodes/ic_lora_guider.py`)

```python
class RSICLoRAGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "control_image": ("IMAGE",),
                "ic_lora": (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                "lora_strength": ("FLOAT", {"default": 1.0, ...}),
                "guide_strength": ("FLOAT", {"default": 1.0, ...}),
                "video_cfg": ("FLOAT", {"default": 3.0, ...}),
                "audio_cfg": ("FLOAT", {"default": 7.0, ...}),
                # ... all MultimodalGuider params ...
                "max_shift": ("FLOAT", {"default": 2.05, ...}),
                "base_shift": ("FLOAT", {"default": 0.95, ...}),
                "guide_frame_idx": ("INT", {"default": 0, ...}),
                "crf": ("INT", {"default": 35, ...}),
            },
        }
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "rs-nodes"

    def create_guider(self, model, positive, negative, vae,
                      control_image, ic_lora, **kwargs):
        # 1. Load IC-LoRA and extract downscale factor
        lora_path = folder_paths.get_full_path_or_raise("loras", ic_lora)
        downscale_factor = self._read_downscale_factor(lora_path)
        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # 2. Apply LoRA to model clone
        m = model.clone()
        m, _ = comfy.sd.load_lora_for_models(
            m, None, lora_data, kwargs.get("lora_strength", 1.0), 0
        )

        # 3. Encode control image at downscaled resolution
        guide_latent = self._encode_control(
            vae, control_image, downscale_factor, kwargs.get("crf", 35)
        )

        # 4. Modify conditioning with guide keyframe info
        positive, negative, num_guide = self._add_guide_conditioning(
            positive, negative, vae, guide_latent,
            kwargs.get("guide_frame_idx", 0),
            kwargs.get("guide_strength", 1.0),
        )

        # 5. Create ICLoRAGuider
        guider = ICLoRAGuider(
            m, positive, negative,
            guide_latent=guide_latent,
            num_guide_frames=num_guide,
            max_shift=kwargs.get("max_shift", 2.05),
            base_shift=kwargs.get("base_shift", 0.95),
            video_cfg=kwargs.get("video_cfg", 3.0),
            audio_cfg=kwargs.get("audio_cfg", 7.0),
            stg_scale=kwargs.get("stg_scale", 0.0),
            # ... etc
        )
        return (guider,)

    def _read_downscale_factor(self, lora_path):
        from safetensors import safe_open
        try:
            with safe_open(lora_path, framework="pt") as f:
                metadata = f.metadata() or {}
                return int(metadata.get("reference_downscale_factor", 1))
        except Exception:
            return 1

    def _encode_control(self, vae, image, downscale_factor, crf):
        """Encode control image at downscaled resolution.

        1. Resize to target/downscale_factor
        2. Re-upscale to target res (bilinear) — preserves half-res character
        3. VAE encode — latent dims match main latent
        """
        from comfy_extras.nodes_lt import preprocess as ltxv_preprocess
        h, w = image.shape[1], image.shape[2]
        if downscale_factor > 1:
            small_h, small_w = h // downscale_factor, w // downscale_factor
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1), small_w, small_h, "bilinear", "center"
            )
            image = comfy.utils.common_upscale(
                image, w, h, "bilinear", "center"
            ).movedim(1, -1)

        processed = ltxv_preprocess(image[0], crf)
        return vae.encode(processed.unsqueeze(0)[:, :, :, :3])
```

### Step 3: Register node (`__init__.py`)

```python
from .nodes.ic_lora_guider import RSICLoRAGuider

NODE_CLASS_MAPPINGS["RSICLoRAGuider"] = RSICLoRAGuider
NODE_DISPLAY_NAME_MAPPINGS["RSICLoRAGuider"] = "RS IC-LoRA Guider"
```

### Step 4: UI section headers (`web/ic_lora_guider.js`)

Add section headers for the node's input groups:
- IC-LoRA (lora selector, strength)
- Control (control_image, guide_strength, guide_frame_idx, crf)
- Guidance (video_cfg, audio_cfg, stg_scale, stg_blocks, rescale, modality_scale, cfg_end, stg_end)
- Scheduler (max_shift, base_shift)

### Step 5: Testing checklist

- [ ] IC-LoRA guider with depth control image → verify structural adherence
- [ ] IC-LoRA guider with canny control image → verify edge adherence
- [ ] IC-LoRA guider with pose control image → verify pose adherence
- [ ] IC-LoRA guider + audio input → verify lip sync still works
- [ ] IC-LoRA guider + first_image → verify I2V reference still works
- [ ] IC-LoRA guider + first_image + audio → verify all three work together
- [ ] Guide frame stripping → verify output has correct frame count
- [ ] Model sampling shift → verify denoising quality matches direct generation
- [ ] LoRA strength = 1.0 → verify no artifacts
- [ ] Various guide_strength values (0.5, 0.75, 1.0) → verify control intensity

---

## Sources

- [In-Context LoRA for Diffusion Transformers (arXiv:2410.23775)](https://arxiv.org/abs/2410.23775)
- [LTX-2-19b-IC-LoRA-Union-Control (HuggingFace)](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Union-Control)
- [LTX-2 IC-LoRA Documentation](https://docs.ltx.video/open-source-model/usage-guides/ic-lo-ra)
- [Lightricks/LTX-2 GitHub (ic_lora.py)](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py)
- [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)
- [ali-vilab/In-Context-LoRA](https://github.com/ali-vilab/In-Context-LoRA)
