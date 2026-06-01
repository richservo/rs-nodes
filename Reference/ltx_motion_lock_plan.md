# LTX Motion-Locked V2V — Implementation Plan

Companion to `ltx_motion_lock_spec.md` (the architectural rationale). This
document is the build plan: concrete file paths, integration points,
phase gates, and risk register. Optimized for AI-assisted execution
(hours per phase, not weeks) with user-in-the-loop review between
phases.

## Source spec

Architecture and motivation: `~/Downloads/ltx_motion_lock_spec.md`
(import to `Reference/` when implementation starts).

Method papers / repos:
- **DiTFlow** — CVPR 2025. `github.com/ditflow/ditflow`. Reference impl
  for CogVideoX. AMF extraction + guidance + RoPE optimization. **Port
  from this repo, do not implement from paper.**
- LTX-Video — `arxiv.org/abs/2501.00103`
- OnlyFlow — `arxiv.org/abs/2411.10501` (related, optical-flow approach)

## Goal

Lock LTX 2.3 generative output to source-video motion at the latent
level, while allowing full appearance transformation driven by a
reference frame. Solve the motion drift that 8:1 temporal VAE
compression imposes on existing V2V pipelines (the failure we
empirically verified — bulk-encode is the only stable IC-LoRA path, but
it costs frame-accurate motion).

## Hardware target

96 GB RTX 6000 on RunPod. VRAM is not a constraint. Plan does NOT
optimize for VRAM at the prototype stage. Local 16 GB GPU support is
out-of-scope for v1; revisit after motion lock works end-to-end.

---

## Architectural ground truth (LTX 2.3 AV runtime)

Runtime model code lives in ComfyUI core, not the LTX-2 submodule.
DiT and attention plumbing:

| Component | Path | Notes |
|---|---|---|
| AV model | `comfy/ldm/lightricks/av_model.py` | LTX 2.3 22B AV variant |
| Block class | `BasicAVTransformerBlock` (line 83) | per-block: self-attn, cross-attn, FFN |
| Block stack | `self.transformer_blocks` (ModuleList) | 48 blocks for the 22B |
| Forward loop | `_process_transformer_blocks` (line 850) → `for i, block in enumerate(self.transformer_blocks)` (line 877) | hook target |
| Patchifier / RoPE | `comfy/ldm/lightricks/symmetric_patchifier.py` | 3D RoPE generation |
| VAE | `comfy/ldm/lightricks/vae/` | causal 8:1 temporal compression |

### Hook strategy (no core modifications)

All attention extraction is done via `register_forward_hook` on
individual `BasicAVTransformerBlock` instances from rs-nodes. We do
NOT modify `comfy/ldm/*`. Hooks are attached at sampler-entry and
removed at sampler-exit so leaving the workflow returns the model to
unmodified state.

For attention-map extraction specifically: pytorch's standard
`scaled_dot_product_attention` doesn't expose the attention matrix.
We need a per-block forward hook that wraps the self-attention
sub-module's forward to compute (or estimate) the attention map for
the cross-frame portion. Options:

1. **Direct attention recompute via flash-attn-style helper** — call
   the attention math without `sdpa` for the blocks we're probing
   only, capturing the softmax output. Slower but exact.
2. **Q/K logging + offline matmul** — capture Q and K at the hook
   point, compute attention scores in a side-buffer. Cheaper than
   recompute; sufficient for AMF.

Option 2 is what we should try first.

---

## Phased plan

Each phase has an explicit **go / no-go gate**. We do not proceed
until the prior phase produces a clear result. Most phase work is AI
coding; user time is concentrated in render evaluation between phases.

### Phase 0 — Bootstrap (1-2 hrs me)

1. Clone DiTFlow into `~/research/ditflow/` (NOT inside rs-nodes).
2. Read `ditflow/attention_hook.py`, `ditflow/amf.py`,
   `ditflow/guidance.py` (or whatever the equivalent files are
   named). Document the porting surface area.
3. Run DiTFlow's reference example on CogVideoX (using their setup
   instructions) to confirm it works on the architecture it was
   designed for.
4. Write a one-page port map: "DiTFlow concept X → LTX equivalent
   Y". Save to `Reference/ditflow_port_map.md`.

**Gate:** DiTFlow runs end-to-end on CogVideoX in our environment.
If not, debug their repo before proceeding.

### Phase 1 — AMF probe (2-4 hrs me + GPU runs)

Standalone script, NOT yet integrated as a node.

**Deliverable:** `tools/ltx_amf_probe.py` (new directory in rs-nodes)
that:

1. Loads LTX 2.3 22B via ComfyUI's model loader API.
2. Encodes a source clip via the LTX video VAE.
3. Adds noise at t ∈ {0.4, 0.5, 0.6, 0.7, 0.8} (probe each).
4. Forward passes through the DiT with per-block hooks active,
   capturing Q and K from `BasicAVTransformerBlock`'s self-attention
   submodule.
5. Computes cross-frame attention maps from the captured Q/K. The
   cross-frame portion specifically: tokens corresponding to latent
   frame `t` attending to tokens at latent frame `t-1` and `t+1`.
6. Derives AMF = patch-wise correspondence vectors between
   consecutive latent frames from the attention maps.
7. Computes RAFT optical flow on the source clip (downsample to
   match latent patch resolution).
8. Correlates AMF (per block, per timestep) with RAFT flow.

**Output:** a CSV / markdown table with columns
`(block_idx, timestep, correlation)` for a probe set of 5-10 clips
with varied motion profiles.

**Gate:** does ANY block at ANY timestep show correlation > 0.6 with
RAFT? If yes, proceed. If best is 0.3-0.5, multi-block combination
might rescue it (extend probe to weighted combinations). If best is
< 0.3 across all blocks, the approach is dead for LTX. STOP and
write up findings.

**Probe set:** 5-10 clips that span:
- Camera pan (rigid translation)
- Body motion (non-rigid, predictable)
- Hand gesture (non-rigid, fast)
- Mostly-static talking head (low motion baseline)
- High-motion action (stress test)

User responsibility: provide the probe clips. Suggest 2-3 second clips
at the user's normal generation resolution.

### Phase 2 — Vanilla AMF guidance (3-6 hrs me + tuning)

Now integrate guidance into the sampling loop.

**Deliverable:** A new guider class `AMFGuider` in
`utils/amf_guider.py` (NOT modifying `multimodal_guider.py` — separate
path while we prove this works).

Plumbing:

1. Source AMF is computed once at sampler entry (same hook
   machinery from Phase 1).
2. At each denoising step, run a forward pass with hooks active on
   the current latent `z_t`, compute current AMF.
3. Compute `L_AMF = ||AMF_source - AMF_current||_2`.
4. Compute `grad_z = autograd.grad(L_AMF, z_t)`.
5. Apply guidance: `z_t = z_t - lr(t) * grad_z` BEFORE the standard
   sampler step.
6. Guidance window: configurable. DiTFlow used 20-30% of steps;
   start there, sweep later.
7. LR schedule: start with `lr = 0.001 * (1 - t/t_max_guided)`,
   then tune. The spec's `0.002 → 0.001` is from CogVideoX
   epsilon-prediction; LTX is rectified flow so the gradient scale
   will be different — empirical sweep required.

**Identity test (gate):**
Run with `source == reference` (no appearance change).
If guided output ≈ source within tolerance, the guidance loop is
functional. If not, the gradient is going the wrong direction or
the AMF extraction is broken.

**V2V test (gate):**
Run with the user's actual problem case (the gesture-occlusion clip
that motivated this whole exercise). Compare:
- Existing IC-LoRA Union bulk encode (current baseline)
- Pure motion-locked V2V (AMF guidance + reference image, no IC-LoRA)

User judges which produces motion closest to source AND comparable
appearance quality.

### Phase 3 — Reference appearance conditioning (1-2 hrs me)

Pure V2V path (no IC-LoRA, no control video preprocessing).
Appearance comes entirely from the reference image via LTX's I2V
cross-attention conditioning pathway.

1. Encode reference frame via LTX image VAE (already exists in
   ComfyUI's LTX implementation).
2. Wire reference latent into LTX's I2V cross-attention conditioning
   pathway. LTX already has this — extend its weight / repeat across
   timesteps rather than building a new path.
3. Skip the trainable projection adapter for v1. Add only if
   appearance leaks structure (visible failure mode).

### Phase 4 — RoPE optimization (4-8 hrs me)

DiTFlow's biggest motion-fidelity wins came from this stage despite
the spec marketing it as "optional." Worth doing for production
quality.

1. Before the main sampling pass, run a short optimization loop
   on the 3D RoPE embeddings used by LTX's patchifier.
2. Minimize `L_AMF(generated_with_optimized_RoPE, source)` via
   gradient descent on the RoPE parameters.
3. Inject optimized RoPE into the actual sampling pass.

This requires backprop through a full DiT forward pass during the
optimization loop. On 96 GB this is fine. On 16 GB it's not — gating
this stage to "high-VRAM mode only" is acceptable for v1.

### Phase 5 — ComfyUI integration (2-3 hrs me)

Wrap the working pipeline as a node:

`RSLTXVMotionLock` in `nodes/ltxv_motion_lock.py`. Standalone — does
NOT chain with IC-LoRA. Inputs:

- `model`, `positive`, `negative`, `vae` (standard)
- `source_video` (IMAGE batch — raw pixel frames of source, no
  preprocessing required)
- `reference_image` (IMAGE — single appearance reference)
- `amf_block_idx` (INT — chosen primary block from Phase 1)
- `amf_block_idx_2` (INT, optional — secondary block for weighted combo)
- `amf_strength` (FLOAT — guidance weight)
- `amf_guide_window` (FLOAT — fraction of steps guided, 0-1)
- `amf_lr_start`, `amf_lr_end` (FLOAT — LR schedule endpoints)
- `enable_rope_opt` (BOOLEAN — Phase 4 toggle)

Output: GUIDER (plugs into `RSLTXVGenerate.guider`).

Reuses existing rs-nodes patterns from `ltxv_iclora_guider.py` for
the wrapper shape.

---

## IC-LoRA: not in the loop

**Pure motion-locked V2V — no IC-LoRA, no control video preprocessing.**

Union/Motion-Track/any IC-LoRA conditions on VAE-encoded control
video, which is exactly the path that produces the inter-latent drift
we're trying to eliminate. Stacking AMF on top of Union just compounds
signals that fight each other.

The user's goal is explicit: leverage LTX's generative physics on
RAW footage as source with frame-accurate motion. No canny/depth/pose,
no Union, no Motion-Track. AMF guidance + reference appearance, full
stop.

Historical context (kept for reference):
1. **Replace** — Pure V2V. AMF guidance + reference appearance, no
   IC-LoRA in the loop. **THIS IS THE PATH.** Matches the user's
   actual goal: raw footage in, motion-locked physics-aware regen out.

2. **Augment** — IC-LoRA Union runs as today; AMF guidance adds on
   top. IC-LoRA controls appearance/structure via its trained
   pathway; AMF controls motion trajectory. Conceptually
   orthogonal. Risk: they fight each other.

Decision: replace, full stop. The augment path was considered and
ruled out — the user wants to BEAT Union, not stack on top of it.

---

## Risk register (likelihood × impact)

| Risk | Likelihood | Impact | Mitigation | Detected at |
|---|---|---|---|---|
| No LTX block carries clean motion signal | Medium | Project-killer | Multi-block weighted combo; attention-entropy auto-selection | Phase 1 |
| Rectified flow + classifier guidance instability | Medium | Output quality degrades | Gentle LR schedule, gradient clipping, guidance window tuning | Phase 2 |
| Over-constraining flattens physics ("copy-like" output) | High initially | Quality regression | Soft constraint weight, restrict to fewer blocks | Phase 2 |
| Appearance leaks structure via reference | High without adapter | Output looks like reference instead of following motion | Add projection adapter (trainable layer) — extra complexity | Phase 3 |
| LTX 2.3 AV multimodal attention confuses AMF | Medium | AMF signal noisy | Hook only video-attention paths, skip audio | Phase 1 |
| Hyperparameter transfer from CogVideoX fails | Near-certain | Need recalibration | Plan empirical sweep, don't trust paper values | Phase 2 |
| Fused VAE-decoder final step disrupts guidance | Medium | Decode-time artifacts | Disable guidance before final step (spec calls this out) | Phase 2 |
| RoPE optimization unstable on LTX | Medium | Phase 4 fails | Phase 4 is gated optional; skip if Phase 1-3 already good enough | Phase 4 |

---

## Open questions (resolve before / during Phase 1)

1. Does ComfyUI's LTX inference use `sdpa` or a custom attention
   kernel (e.g. SageAttention)? Affects whether Q/K capture via
   forward hook gets us what we need or if we need a deeper hook
   into the attention math.
2. How does the AV model's multimodal attention work — are video
   tokens and audio tokens interleaved in the same self-attention,
   or are they separate streams? If interleaved, AMF extraction
   needs to mask audio tokens.
3. What's the patch grid resolution at typical user resolutions
   (768×512, 1024×576)? RAFT flow needs to be downsampled to match
   the latent patch grid for AMF↔flow correlation.
4. Does LTX 2.3 22B's RoPE accept per-block customization, or is
   RoPE shared across all blocks? Affects Phase 4 design.

---

## File layout (when implementation starts)

```
rs-nodes/
├── tools/
│   └── ltx_amf_probe.py            # Phase 1 — standalone probe
├── utils/
│   ├── multimodal_guider.py        # untouched, existing IC-LoRA path
│   ├── amf_guider.py               # Phase 2 — new guider with AMF guidance
│   ├── amf_hooks.py                # attention hook utilities, shared
│   └── amf_loss.py                 # L_AMF computation
├── nodes/
│   └── ltxv_motion_lock.py         # Phase 5 — ComfyUI node wrapper
└── Reference/
    ├── ltx_motion_lock_spec.md     # source spec
    ├── ltx_motion_lock_plan.md     # this file
    ├── ditflow_port_map.md         # Phase 0 deliverable
    └── ltx_amf_probe_results.md    # Phase 1 deliverable
```

NO modifications to `comfy/ldm/*`. NO modifications to the LTX-2
submodule. All integration via PyTorch forward hooks attached at
sampler entry, removed at sampler exit.

---

## Time budget (AI execution, not human)

| Phase | Coding time | GPU/render time | User review |
|---|---|---|---|
| 0. Bootstrap DiTFlow | 1-2 hrs | minutes | minimal |
| 1. AMF probe | 2-4 hrs | hours (probe set × timesteps × blocks) | review results table, decide gate |
| 2. Vanilla guidance | 3-6 hrs | hours of tuning iterations | review identity test + V2V comparison |
| 3. Reference appearance | 1-2 hrs | hours | review appearance fidelity |
| 4. RoPE optimization (optional) | 4-8 hrs | hours | review motion fidelity delta |
| 5. ComfyUI integration | 2-3 hrs | minimal | usability review |
| **Total** | **13-25 hrs** | days of wall-clock GPU | several hours total |

Calendar time depends on render iteration cadence. With responsive
user testing between phases, end-to-end in 3-5 days is realistic. With
gaps, longer.

---

## Success criteria

V1 success: on the user's gesture-occlusion test case, motion in the
generated output lands within ±1 latent frame of the source motion at
every measured keyframe AND appearance is visibly transformed by the
reference. Specifically:

1. The gesture moment (originally drifting by visible frames) lands
   at the correct latent slot.
2. Output character/lighting/style follows the reference, not the
   source.
3. No visible "slideshow" or "slow-motion" artifacts (our
   previously-encountered failure modes).
4. Quality at least matches current IC-LoRA bulk-encode baseline.

If 1-3 hold but 4 fails, this is still a useful feature with a
quality trade-off knob — ship it as opt-in.
