# Prompt Relay — Build Plan

Implementation plan for porting the **Prompt Relay** method (arXiv 2604.10030, Chen / Huang / Liu 2026) into the rs-nodes LTXV pipeline.

## Method summary (paper)

Prompt Relay is an **inference-time, training-free, plug-and-play** modification to the cross-attention logits of a text-conditioned video diffusion model that routes each text-prompt segment to a specific span of latent video frames. Originally evaluated on Wan2.2; the math is architecture-agnostic and applies to any DiT that does cross-attention from spatiotemporal video tokens to text tokens — which is exactly what LTX 2.3 does (`attn2` in `BasicTransformerBlock`).

**Modified attention:**

```
Attn(Q, K, V) = softmax(QK^T / sqrt(d) − C(Q, K)) V
C(i, j)       = 1[j ∈ K_s] · ReLU(|f(i) − m_s| − w)² / (2 σ²)
```

- `f(i)` — latent frame index of query token `i`
- `m_s = (t_s_start + t_s_end) / 2` — temporal midpoint of segment `s` (in latent frames)
- `K_s` — K-token range belonging to segment `s` (after concatenating segment encodings along the token axis)
- `w` — local window inside which no penalty is applied. Paper default: `w = L − 2` where `L = t_end − t_start`.
- `σ` — boundary decay. With `ε = 0.1` and `w = L − 2`: `σ = (L − w) / sqrt(2 · ln(1/ε))`.
- Global-prompt tokens (a separate prompt that conditions the whole video) get `C = 0` — they attend freely.

Applied **at every cross-attention block, every denoising step**. Self-attention untouched. No model surgery.

## LTX-specific facts (verified against `comfy/ldm/lightricks/`)

- `BasicTransformerBlock.forward` (`model.py:470`) calls `self.attn2(x, context=context, mask=attention_mask, ...)` — classical text cross-attention. Already accepts a float additive `attention_mask`.
- `_prepare_attention_mask` (`model.py:852`) reshapes a bool mask `[B, K]` → float `[B, 1, 1, K]` via `(mask − 1)`, then it gets broadcast into `optimized_attention_masked`. **If we pre-supply a float-typed `[B, 1, Q, K]` mask, the existing check `if attention_mask is not None and not torch.is_floating_point(attention_mask)` skips the reshape and our mask flows straight through.**
- `SymmetricPatchifier.patchify` (`symmetric_patchifier.py:97`) flattens `b c f h w → b (f h w) c` with patch size `(1, 1, 1)` — so for a query token at index `i` in the flattened sequence: **`f(i) = i // (H_lat · W_lat)`**.
- AV model (`av_model.py`) adds `self.audio_attn2 = CrossAttention(...)` for audio tokens cross-attending the same text context. `AudioPatchifier.patchify` flattens `b c t f → b t (c f)` — so for audio token `i`: **`f(i) = i`** directly.
- Latent shape: `[B, 128, T_lat, H_lat, W_lat]` where `T_lat = (num_frames − 1) // 8 + 1`. First latent frame maps to pixel frame 0; latent frame `t ≥ 1` covers pixel frames `1 + (t−1)·8` … `1 + t·8`.
- Audio latent timestamp formula (`AudioPatchifier._get_audio_latent_time_in_sec`): `(audio_mel_frame · hop_length / sample_rate)` where `audio_mel_frame = audio_latent_frame · audio_latent_downsample_factor` (with causal fix for first frame).

## Data contract

### LLM / canonical input JSON

```json
{
  "global": "cinematic 35mm, woman in red dress, warm tungsten light",
  "segments": [
    {"t_start": 0.0, "t_end": 1.5, "prompt": "she walks into the room"},
    {"t_start": 1.5, "t_end": 3.0, "prompt": "she turns to camera and smiles"}
  ]
}
```

- Times in **seconds**.
- `global` optional (empty string → no global prompt).
- Segments may overlap or have gaps (paper allows it).
- Empty / missing `segments` → degrade to standard CLIP encoding (no metadata stamped).

### `prompt_relay` metadata (stamped on CONDITIONING)

```python
{
  "prompt_relay": {
    "global_len":   int,    # # text tokens in the global prompt slice (0 if none)
    "segments": [           # ordered, parallel to JSON segments, in K-token order
      {
        "start_token": int, # half-open [start, end) in K
        "end_token":   int,
        "t_start_sec": float,
        "t_end_sec":   float,
      },
      ...
    ],
    "epsilon":       float, # default 0.1
    "window_mode":   "L-2" | "L-1" | "custom",
    "window_custom": int,   # used iff window_mode == "custom"
    "frame_rate":    float, # gen frame_rate snapshot (informational; primary read is cond.frame_rate)
    "num_frames":    int,
  }
}
```

Only **seconds** are persisted. Per-stream latent-frame conversion happens at mask-build time so video and audio paths each compute against their own temporal scale.

## File layout

| Status | Path | Purpose |
|---|---|---|
| **NEW** | `utils/prompt_relay.py` | Pure mask-builder function (no ComfyUI imports). Unit-testable. |
| **NEW** | `nodes/prompt_relay_encode.py` | `RSPromptRelayEncode` — drop-in CLIPTextEncode alternative with optional Ollama formatter. |
| **NEW** | `system_prompts/relay_default.txt` | Default LLM system prompt for relay JSON output. |
| **MOD** | `nodes/ltxv_generate.py` | Auto-detect `prompt_relay` on cond → install attention-mask wrapper. |
| **MOD** | `__init__.py` (root) | Register `RSPromptRelayEncode` in `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. |

## Component specs

### `utils/prompt_relay.py`

Single public entry, fully synchronous, torch-only:

```python
def build_relay_mask(
    stream: Literal["video", "audio"],
    *,
    B: int,
    K_total: int,
    pr: dict,                          # the prompt_relay metadata dict
    base_mask: torch.Tensor | None,    # existing padding mask, possibly bool [B, K] or float [B, 1, ?, K]
    dtype: torch.dtype,
    device: torch.device,
    # video stream
    T_lat: int | None = None, H_lat: int | None = None, W_lat: int | None = None,
    frame_rate: float | None = None,
    # audio stream
    audio_T_lat: int | None = None,
    audio_latent_downsample_factor: int = 4,
    hop_length: int = 160,
    sample_rate: int = 16000,
    audio_is_causal: bool = True,
) -> torch.Tensor:                     # returns [B, 1, Q, K_total] float additive mask
```

Algorithm:

1. Compute query-token frame index vector `f` of length `Q`:
   - video: `f = arange(T_lat).repeat_interleave(H_lat * W_lat)` → `[Q]`, `Q = T_lat·H_lat·W_lat`
   - audio: `f = arange(audio_T_lat)` → `[Q]`, `Q = audio_T_lat`
2. For each segment, convert `t_start_sec / t_end_sec` to **stream-specific latent frames**:
   - video: `t_lat = 0 if t_sec == 0 else clamp(1 + floor((t_sec*fr - 1) / 8), 0, T_lat)` (matches `keyframe_idxs` math used elsewhere in the codebase)
   - audio: invert `_get_audio_latent_time_in_sec`: `audio_lat = round(t_sec * sample_rate / hop_length / audio_latent_downsample_factor)` clamped to `audio_T_lat`. Honor `audio_is_causal` for `t_sec == 0` → `audio_lat = 0`.
3. Per segment compute `L_s`, `m_s`, `w_s` (per `window_mode`), `σ_s` (with `eps_floor = 1e-4` to avoid div-by-zero on degenerate segments).
4. Penalty: `c_s = ReLU(|f - m_s| - w_s)² / (2 σ_s²)` shape `[Q]`. Build the additive mask:
   - Initialize `mask = zeros([B, 1, Q, K_total], dtype=dtype, device=device)`
   - For each segment `s`: `mask[:, :, :, K_s_start:K_s_end] = -c_s.unsqueeze(-1)`
   - Global tokens (`K[0:global_len]`): leave at 0
5. Combine with `base_mask`:
   - If bool/int → convert padded positions to a large negative float (e.g. `-1e4`) at all Q rows, then add.
   - If float `[B, 1, ?, K]` → broadcast and add.
6. Cast to `dtype`, return.

Edge cases:
- `global_len == K_total` (no segments) → return base mask broadcast to `[B, 1, Q, K]` (or just return None upstream).
- `T_lat == 1` (single-frame video) → `f` is constant; penalty trivially zero unless segment midpoint differs (which would mask everything). Return base.
- Segment with `t_end <= t_start` → skip.
- Stream='audio' but `audio_T_lat is None` → raise.

Include a `__main__` block with synthetic shape/value assertions:
- 2-segment, video stream, B=2, T_lat=12, H_lat=4, W_lat=4 → mask shape `[2,1,192,K]`, segment-0 penalty zero in segment-0 frames, large in segment-1 frames.
- Audio stream variant.

### `nodes/prompt_relay_encode.py` — `RSPromptRelayEncode`

INPUT_TYPES:

```python
required:
  clip:               CLIP
  prompt:             STRING (multiline, default: "")
optional:
  num_frames:         INT     (default 97, min 9, max 8192, step 8)
  frame_rate:         FLOAT   (default 25.0, min 0.1, max 1000.0, step 0.01)
  # Ollama (LLM formatter)
  use_ollama:         BOOLEAN (default False)
  ollama_url:         STRING  (default "http://localhost:11434")
  ollama_model:       STRING  (default "qwen2.5:32b")
  ollama_system:      STRING  (multiline, default "" → load relay_default.txt)
  ollama_seed:        INT     (default 0, min 0, max 0xffffffffffffffff)
  ollama_temperature: FLOAT   (default 0.3, min 0.0, max 2.0, step 0.01)
  # Relay tunables
  epsilon:            FLOAT   (default 0.1, min 0.001, max 0.999, step 0.001)
  window_mode:        (["L-2", "L-1", "custom"],)
  window_custom:      INT     (default 0, min 0, max 1024)
  # Debug
  debug_print_json:   BOOLEAN (default False)
```

RETURN_TYPES: `("CONDITIONING",)`
RETURN_NAMES: `("conditioning",)`
FUNCTION: `"encode"`
CATEGORY: `"rs-nodes"`

`IS_CHANGED`: hash of `(prompt, use_ollama, ollama_url, ollama_model, ollama_system, ollama_seed, ollama_temperature)` only. Changing relay tunables doesn't re-hit the LLM.

`encode(...)` flow:

1. **Determine input mode:**
   - If `prompt.strip().startswith("{")`: parse as JSON directly.
   - Elif `use_ollama`: call Ollama with `format: "json"`, system prompt loaded from `ollama_system` or `system_prompts/relay_default.txt`. Parse JSON.
   - Else: treat `prompt` as single global, no segments.
   - On JSON parse failure: `logger.warning`, fall through to single-global mode.
2. **Validate JSON:**
   - Coerce `global` to str, `segments` to list of dicts.
   - Drop segments with `t_end <= t_start`.
   - If no segments after validation → single-global mode (no metadata).
3. **Encode through CLIP:**
   - Encode global (if non-empty) via `clip.encode_from_tokens_scheduled`.
   - Encode each segment prompt the same way.
   - Each call returns a list of `[(cond_tensor, opts_dict)]` — assume 1 entry per call (no scheduling).
4. **Concatenate K tokens:**
   - Cat cond tensors along token axis (dim=1). Record `(start_token, end_token)` per segment + `global_len`.
   - Concatenate per-prompt padding masks the same way (if present in opts).
   - Pooled output: prefer global's `pooled_output`; else first segment's; else None.
5. **Build `prompt_relay` dict** (see §Data contract).
6. **Stamp via `node_helpers.conditioning_set_values`:**
   ```python
   cond = node_helpers.conditioning_set_values(
       cond, {"prompt_relay": pr_dict, "attention_mask": concat_mask, "pooled_output": pooled}
   )
   ```
7. If `debug_print_json` → log the JSON + segment→K-range table.

**Ollama call detail:**
- POST `{ollama_url}/api/generate` with `{"model": ..., "prompt": user_prompt, "system": sys_prompt, "format": "json", "options": {"seed": ..., "temperature": ...}, "stream": false}`.
- Reuse the request/cache pattern from existing `nodes/prompt_formatter.py` for consistency.
- Cache LLM responses keyed on the IS_CHANGED hash so re-running the workflow with same inputs is free.

**`system_prompts/relay_default.txt`** — instructs the LLM to:
- Output STRICTLY a JSON object matching the schema, no prose
- Use times in seconds matching the requested duration (passed as a hint in the user prompt prefix or system prompt template)
- Provide a `global` line for persistent visual context
- Provide 2–6 `segments`, each with non-overlapping or lightly-overlapping time spans
- Include 1–2 few-shot examples

### `nodes/ltxv_generate.py` (modifications)

**Insert** in `_generate_impl`, just after the attention-mode setup block (~line 290–295):

```python
# --- Prompt Relay (auto-enabled when conditioning carries metadata) ---
pr_meta = self._extract_prompt_relay(positive)
if pr_meta is not None:
    logger.info(f"Prompt Relay active: {len(pr_meta['segments'])} segment(s), "
                f"global_len={pr_meta['global_len']}, ε={pr_meta['epsilon']}")
    self._install_prompt_relay(m, pr_meta)
```

**Add helpers** to `RSLTXVGenerate`:

```python
def _extract_prompt_relay(self, conditioning):
    for _, opts in conditioning:
        pr = opts.get("prompt_relay")
        if pr is not None:
            return pr
    return None

def _install_prompt_relay(self, m, pr_meta):
    """Wrap diffusion_model.forward to inject the Prompt Relay penalty into
    attention_mask BEFORE _prepare_attention_mask sees it. We pass a float-typed
    [B, 1, Q, K] mask, which the existing is_floating_point() check passes through
    unchanged."""

    from ..utils.prompt_relay import build_relay_mask

    # Capture the unwrapped forward
    orig_forward = m.model.diffusion_model.forward

    def wrapped(self_, x, timestep, context, attention_mask=None,
                frame_rate=25, transformer_options={}, keyframe_idxs=None,
                denoise_mask=None, **kwargs):
        B = x.shape[0]
        K_total = context.shape[1] if context.ndim >= 2 else 0

        if K_total > 0:
            video_mask = build_relay_mask(
                "video",
                B=B, K_total=K_total, pr=pr_meta,
                base_mask=attention_mask,
                dtype=x.dtype, device=x.device,
                T_lat=x.shape[2], H_lat=x.shape[3], W_lat=x.shape[4],
                frame_rate=float(frame_rate),
            )
            attention_mask = video_mask  # already float [B, 1, Q, K]

        return orig_forward(
            x, timestep, context, attention_mask=attention_mask,
            frame_rate=frame_rate, transformer_options=transformer_options,
            keyframe_idxs=keyframe_idxs, denoise_mask=denoise_mask, **kwargs,
        )

    # add_object_patch with __get__ binding so `self_` is the model
    bound = wrapped.__get__(m.model.diffusion_model, type(m.model.diffusion_model))
    m.add_object_patch("diffusion_model.forward", bound)
```

**Audio (AV model) variant:**
Detect `isinstance(m.model.diffusion_model, LTXVAVTransformer)` (or check for `audio_attn2` attribute on a sample block) and wrap its forward with both `build_relay_mask("video", ...)` and `build_relay_mask("audio", ...)`. The audio mask is threaded into `transformer_options["prompt_relay_audio_mask"]`; the AV block forward is wrapped to read from there at the `audio_attn2` call site. Defer to **Step D** — get video working first.

## Build order (incremental, each step revertable)

| Step | Deliverable | Acceptance test | Commit? |
|---|---|---|---|
| **A** | `utils/prompt_relay.py` + `__main__` test block | Manual run prints expected shapes; penalty is 0 inside segment, large outside | yes |
| **B** | `RSPromptRelayEncode` (raw-JSON path only, no Ollama) + registration | Workflow encodes 2-segment JSON → CONDITIONING carries `prompt_relay` dict; visible in a debug print | yes |
| **C** | `_install_prompt_relay` hook in `RSLTXVGenerate` (video only) | 2-segment 97-frame T2V renders; visual events shift ordering vs single-prompt baseline of same total prompt | yes |
| **D** | Audio (`audio_attn2`) extension for AV model | AV model 2-segment generation; both vis and audio align to segment timing | yes |
| **E** | Ollama path in `RSPromptRelayEncode` + `system_prompts/relay_default.txt` | Plain-prompt input → LLM → JSON → encode; verify JSON in logs | yes |
| **F** | `IS_CHANGED` smart caching | Tweak `epsilon` between runs → Ollama not re-called (logs confirm cache hit) | yes |

## Composability checks (already verified during research)

- **SAGE attention path** — both `attention_pytorch` and `attention_sage` accept float additive `mask`. ✓
- **`optimized_attention_override`** (transformer_options key already used for SAGE) — orthogonal; receives our combined `mask`. ✓
- **Self-attention guide masks** (`self_attention_mask`, `_build_guide_self_attention_mask`) — different parameter, different code path. ✓
- **Audio cross-attention** (`audio_attn2`) — separate attention call; out of scope until Step D, then handled with a parallel `prompt_relay_audio_mask` thread.
- **Negative conditioning** — wrapper inspects per-call. Negatives without `prompt_relay` get standard treatment.

## Risks / verify during build

- **Float-mask convention.** `_prepare_attention_mask` uses `(mask − 1)` for bool input → -1 additive (soft, not -inf). Confirm our combined mask values play nicely with `attention_pytorch` / `attention_sage`. If empirically the penalty needs more headroom, consider a `penalty_scale` knob (default 1.0).
- **Cond rotation / CFG batch.** ComfyUI runs positive+negative as B=2. Our wrapper builds the mask off `x.shape[0]` so it broadcasts correctly. K must match between positive and negative — `_prepare_context` already handles padding.
- **`frame_rate` source.** Cond carries `frame_rate` (set at `ltxv_generate.py:432`); read it from there in preference to `pr_meta['frame_rate']` so temporal upscale halving is respected.
- **`global_len == 0`.** No global prompt → first K token belongs to segment 0. Make sure mask init handles the K[0:0] empty global slice cleanly.

## Out of scope (v1)

- Hand-authored timing UI (timeline editor) inside the node.
- Per-block / per-step penalty schedule. Paper applies everywhere.
- Sharing relay metadata across multiple cond branches (style + relay merged).
- Wan 2.2 or other architectures — LTX-only.
- IS_CHANGED-based partial encode invalidation when only some segments change. Re-encode all on any prompt-text change.

## References

- Paper: arXiv 2604.10030 — *Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation* (Chen, Huang, Liu, 2026).
- Original repo (Wan2.2): https://github.com/GordonChen19/Prompt-Relay
- Project page: https://gordonchen19.github.io/Prompt-Relay/
- Kijai's LTX 2.3 port (reference only — we follow the paper, not this): https://github.com/kijai/ComfyUI-PromptRelay
