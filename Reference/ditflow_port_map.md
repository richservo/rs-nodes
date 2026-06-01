# DiTFlow → LTX Port Map (Phase 0 Recon)

Reconnaissance output from reading the DiTFlow repo at
`C:\Users\Richard\research\ditflow\`. Maps every meaningful concept
in DiTFlow's CogVideoX implementation to its LTX 2.3 equivalent.

## DiTFlow repo size

1,173 lines total. Very tractable codebase.

| File | LOC | Purpose |
|---|---|---|
| `motion_guidance.py` | 809 | Entry point, Guidance class, sampling loop with AMF |
| `guidance_utils/custom_modules.py` | 120 | `InjectionProcessor` (attention wrapper) + `ModuleWithGuidance` |
| `guidance_utils/custom_transformer.py` | 154 | `ControlledTransformer` (CogVideoX subclass for trainable RoPE/pos_embed) |
| `guidance_utils/motion_flow_utils.py` | 58 | `compute_motion_flow` — the AMF computation itself |
| `guidance_utils/custom_embeddings.py` | 33 | RoPE prep helper |

## The 58-line AMF core

`motion_flow_utils.py:compute_motion_flow(q, k, h, w, temp, nframes, argmax)`:

1. Take Q and K from one attention block
2. Compute attention map A = softmax((q @ kᵀ) / √d_k)
3. Reshape attention to per-frame-pair blocks: `[f1, f2, hw1, hw2]`
4. For each frame pair (i, j), compute displacement field:
   - Either argmax (hard correspondence) or soft expected displacement
   - dx, dy per spatial patch
5. Sum across heads, average

This is the ENTIRE AMF computation. Trivial to port — pure tensor
math, no model-specific assumptions except the `226:` text token
offset (CogVideoX has 226 text tokens prepended to the visual
sequence; we slice them off).

## Port surface area: CONCEPT → LTX EQUIVALENT

### Concept 1: Get Q and K from one attention block

**DiTFlow:** Patches the CogVideoX attention processor
(`InjectionProcessor` in `custom_modules.py`). Replaces the default
processor on each block via `attn1.set_processor(processor)`. Inside,
when forward runs, copies query/key into `self.query`, `self.key`
**before** sdpa is called.

**LTX:** `comfy/ldm/lightricks/model.py:361` — `class CrossAttention`.
It computes Q/K/V via `self.to_q/k/v(x)` then calls
`comfy.ldm.modules.attention.optimized_attention(q, k, v, ...)` at
line 414. Either:

- **Forward pre-hook on `optimized_attention`** — wrap the function call
- **Module-level forward hook on `CrossAttention`** — capture inputs to the
  internal call by saving Q/K from `to_q`/`to_k` projections via a
  pre-hook on the parent module
- **Subclass `CrossAttention`** and override `forward` to save Q/K
  before delegating to `optimized_attention` — cleanest

Recommended: subclass-and-monkey-replace approach mirroring DiTFlow's
processor pattern. Replace the `CrossAttention` instance on each
`BasicAVTransformerBlock` we want to probe.

### Concept 2: "guidance blocks" — which transformer blocks to capture from

**DiTFlow:** Hardcoded in config — `guidance_blocks_5b: [20]` (1
block out of CogVideoX-5B's ~30 blocks). Block 15 for 2B.

**LTX:** Unknown until Phase 1 probe. LTX 2.3 22B has **48 blocks**
in `transformer_blocks`. Probe all of them, pick the best 1-3.
Empirical step.

### Concept 3: Save & reuse intermediate block outputs (for SMM/MOFT — not strictly needed for the `flow` path)

**DiTFlow:** `ModuleWithGuidance` wraps a block, calls module forward,
captures `out[-1]` as `saved_features`.

**LTX:** Same pattern. Wrap a `BasicAVTransformerBlock`'s forward via
`nn.Module` subclass. Trivial.

**Decision:** Skip SMM/MOFT paths for v1. Focus on `flow` loss only
(the canonical DiTFlow contribution). Saves complexity.

### Concept 4: Pass-through optimization (skip later blocks during guidance forward)

**DiTFlow:** `change_mode(train=True)` swaps `block.forward = dummy_pass`
for all blocks AFTER the highest guidance block. Massive speedup —
no point running blocks downstream of where we extract Q/K.

**LTX:** Same pattern, same speedup applies. Direct port.

### Concept 5: 3D Positional Embeddings (RoPE)

**DiTFlow:** `prepare_rotary_positional_embeddings` in
`custom_embeddings.py` — wraps `diffusers.models.embeddings.get_3d_rotary_pos_embed`
with CogVideoX-specific spatial/temporal scales. The result is passed
as `rope=(q_rope, k_rope)` to each block's attention.

**LTX:** LTX uses 3D RoPE too, computed inside
`comfy/ldm/lightricks/symmetric_patchifier.py`. The RoPE construction
is LTX-specific (different grid math). For RoPE optimization (Phase
4), we'd need to:
1. Extract LTX's RoPE construction into a wrapped/forkable version
2. Make it differentiable + trainable
3. Inject the trained values back into the model's RoPE source

Phase 4 work. Phase 1-3 doesn't need this.

### Concept 6: ControlledTransformer subclass — adds trainable pos_embedding + RoPE

**DiTFlow:** Subclasses `CogVideoXTransformer3DModel`, adds
`trainable_pos_embedding` and `trainable_rope` attributes, overrides
forward to inject them.

**LTX equivalent:** Would subclass LTX's `LTXVideoTransformer3DModel`
(at `comfy/ldm/lightricks/av_model.py` — the runtime AV variant). For
Phase 1-2 (no RoPE optimization), we don't need this subclass — we
just need forward hooks on individual blocks. For Phase 4 we'd need
something similar.

### Concept 7: Guidance loop — modify denoising step

**DiTFlow:** In the denoise loop, for each guidance timestep:
1. Run guidance loop (5 sub-steps per main step) optimizing `z_t`
2. Each sub-step: forward through DiT → compute AMF from current
   block Q/K → MSE loss vs cached source AMF → autograd grad → step
3. Use `change_mode(train=True)` to skip downstream blocks
4. Switch to `change_mode(train=False)` for the actual denoising
   step

**LTX:** ComfyUI's sampling is in `comfy.samplers.sample`. We'd
override at the guider level — extend `CFGGuider` (or compose with
it). Same pattern as existing `MultimodalGuider` and `ICLoRAGuider`
in `utils/multimodal_guider.py`. The guider's `sample()` method gets
to drive its own sampling loop.

### Concept 8: KV injection at block 0

**DiTFlow:** `injection_blocks: [0]` — at block 0, inject the source
video's K/V into the generated video's K/V (the `inject_kv` path in
`InjectionProcessor`). This is the "structural anchor at the start
of denoising" the spec mentioned.

**LTX:** Same pattern, direct port. Wrap `BasicAVTransformerBlock 0`'s
attention with a KV-injection hook.

## Defaults from DiTFlow's config to inherit (and tune)

```yaml
lr: [0.002, 0.001]               # high → low across guidance steps
optimization_steps: 5             # 5 sub-steps per denoising step
guidance_timestep_range: [50, 40] # guide steps 0-10 of 50 total
motion_temp: 2                    # softmax temperature in AMF
threshloss: True                  # ignore zero-flow patches
argmax_motion_flow: True          # hard correspondence in reference
prop_motion: 0.04                 # MOFT % (not used in flow path)
```

**Numbers that need recalibration for LTX:**

- `lr` — DiTFlow's tuning was for CogVideoX DDIM. LTX uses rectified
  flow; gradient magnitudes differ. Sweep starting at DiTFlow's values
  and adjust by orders of magnitude until guidance is stable but
  effective.
- `guidance_timestep_range` — DiTFlow uses 50-step schedules.
  LTX 2.3 uses ~12-25 step shifted/scaled rectified flow. Equivalent
  range probably "first ~30-50%" of steps, but tune empirically.
- `motion_temp` — likely fine to start at 2, may need higher (3-5)
  if AMF signal is noisier on LTX.

## Open questions (carried over from plan)

1. **Does optimized_attention let us hook Q/K cleanly?** Yes — we
   subclass `CrossAttention` and intercept Q/K before the
   `optimized_attention(q, k, v, ...)` call at line 414. Confirmed.

2. **Multimodal attention (AV) — separate video/audio token paths?**
   Looking at `av_model.py`: video and audio go through separate
   attention chains (`audio_to_video_attn` at line 153, etc.).
   The "main" self-attention on each block is video-only, then
   cross-attentions add audio interactions. For AMF extraction we
   hook the video self-attention, ignore audio paths.

3. **Patch grid resolution at typical user resolutions?** Patch
   size is part of LTX config — need to read from
   `transformer.config.patch_size` at runtime. For typical 768×512
   output: ~3-4 patches × 24×16 = roughly 3500 patches per latent
   frame. Compatible with DiTFlow's per-frame attention math (their
   30×45 = 1350 patches).

4. **RoPE structure for Phase 4?** LTX RoPE is in
   `symmetric_patchifier.py`. Defer to Phase 4 deep-dive.

## What does NOT port

- **diffusers pipeline (`CogVideoXPipeline`)** — LTX uses ComfyUI's
  native sampler infrastructure. We don't use diffusers. The
  `Guidance` class scaffold becomes a `MotionLockGuider` class in
  `utils/amf_guider.py`.
- **CogVideoX VAE specifics** — LTX has its own causal video VAE.
  We use it as-is via the existing ComfyUI VAE pipeline (`vae.encode`,
  `vae.decode`).
- **Text encoder calls** — already handled by upstream ComfyUI nodes
  (CLIP encode → CONDITIONING).
- **`enable_model_cpu_offload`, `enable_slicing`, `enable_tiling`** —
  ComfyUI handles VRAM management differently. Skip.

## Verification status

DiTFlow's quick-start expects running their `motion_guidance.py` on
CogVideoX. **We are NOT running this verification step.** The
reference codebase exists, has a thin enough surface that the
reading-only recon is sufficient to plan the port. If Phase 1
hits a structural issue, we'll come back here.

## Phase 1 work item readiness

Phase 1 (AMF probe) has clear marching orders now:

1. Write `tools/ltx_amf_probe.py`:
   - Subclass `CrossAttention` to capture Q/K on instances
   - Replace `CrossAttention` instance on each
     `BasicAVTransformerBlock` via attribute swap
   - Forward source latent through DiT at noise levels
     t ∈ {0.4, 0.5, 0.6, 0.7, 0.8}
   - For each (block_idx, timestep), pull Q/K and call
     `compute_motion_flow` (direct port from
     `motion_flow_utils.py`)
   - Compute RAFT optical flow on source as ground truth (use
     `torchvision.models.optical_flow.raft_large` or similar)
   - Correlate AMF vector field with RAFT flow field per
     (block, timestep), output a results table

2. Phase 1 hard gate: ANY block at ANY timestep with correlation
   > 0.6 vs RAFT → proceed. Best in 0.3-0.5 → try multi-block
   combination. < 0.3 everywhere → STOP, project dead.

## Estimated Phase 1 effort

Code: ~2-3 hrs (mostly setting up the LTX model load and the
hook plumbing; the AMF math is a 58-line copy).

GPU time: ~1-2 hours for the probe (5 timesteps × 48 blocks × 5-10
clips × ~30 sec per forward pass).

Output: a markdown table + recommendation. Ready to start Phase 2
as soon as we have the table.
