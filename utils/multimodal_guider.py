"""MultimodalGuider for LTXV AV (audio-video) models.

Provides per-modality CFG, Spatiotemporal Guidance (STG), cross-modal
guidance, and CFG rescaling — all in one guider that extends ComfyUI's
CFGGuider.
"""

from dataclasses import dataclass, field

import torch
import comfy.model_patcher
import comfy.samplers
import comfy.utils


# ---------------------------------------------------------------------------
# STG (Spatiotemporal Guidance) helpers
# ---------------------------------------------------------------------------

@dataclass
class STGFlag:
    """Mutable flag shared by all block wrappers, toggled per forward pass."""
    do_skip: bool = False
    skip_layers: list = field(default_factory=list)


class STGAttentionOverride:
    """Attention override that returns v directly (identity) for self-attention.

    In a BasicAVTransformerBlock, attention calls happen in order:
      0: attn1       (video self-attention)    → SKIP
      1: attn2       (video cross-attention)
      2: audio_attn1 (audio self-attention)    → SKIP
      3: audio_attn2 (audio cross-attention)
      4: audio_to_video_attn (cross-modal)
      5: video_to_audio_attn (cross-modal)
    """
    SELF_ATTN_INDICES = {0, 2}

    def __init__(self, existing_override=None):
        self.call_count = 0
        self.existing_override = existing_override

    def __call__(self, func, q, k, v, *args, **kwargs):
        idx = self.call_count
        self.call_count += 1
        if idx in self.SELF_ATTN_INDICES:
            return v
        if self.existing_override is not None:
            return self.existing_override(func, q, k, v, *args, **kwargs)
        return func(q, k, v, *args, **kwargs)


class STGBlockWrapper:
    """Wraps a transformer block to optionally apply STG attention skipping."""

    def __init__(self, idx, flag):
        self.idx = idx
        self.flag = flag

    def __call__(self, args, extra_args):
        if self.flag.do_skip and self.idx in self.flag.skip_layers:
            args = dict(args)
            to = dict(args.get("transformer_options", {}))
            existing = to.get("optimized_attention_override", None)
            to["optimized_attention_override"] = STGAttentionOverride(existing)
            args["transformer_options"] = to
        return extra_args["original_block"](args)


# ---------------------------------------------------------------------------
# MultimodalGuider
# ---------------------------------------------------------------------------

class MultimodalGuider(comfy.samplers.CFGGuider):
    """Guidance for LTXV audio-video models with per-modality CFG,
    Spatiotemporal Guidance (STG), cross-modal guidance, and CFG rescale."""

    def __init__(
        self,
        model,
        positive,
        negative,
        video_cfg=3.0,
        audio_cfg=7.0,
        stg_scale=1.0,
        stg_blocks=None,
        rescale=0.7,
        modality_scale=3.0,
    ):
        if stg_blocks is None:
            stg_blocks = [29]

        super().__init__(model)

        self.video_cfg = video_cfg
        self.audio_cfg = audio_cfg
        self.stg_scale = stg_scale
        self.stg_blocks = stg_blocks
        self.rescale = rescale
        self.modality_scale = modality_scale
        self._latent_shapes = None

        self.inner_set_conds({"positive": positive, "negative": negative})

        # Install STG block wrappers on specified blocks
        self.stg_flag = STGFlag(skip_layers=list(stg_blocks))
        if self.stg_scale > 0:
            for idx in self.stg_blocks:
                wrapper = STGBlockWrapper(idx, self.stg_flag)
                self.model_patcher.set_model_patch_replace(
                    wrapper, "dit", "double_block", idx
                )

    # ------------------------------------------------------------------
    # sample() — peek at NestedTensor shapes before parent packs them
    # ------------------------------------------------------------------

    def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
        if latent_image.is_nested:
            self._latent_shapes = [t.shape for t in latent_image.unbind()]
        else:
            self._latent_shapes = [latent_image.shape]
        return super().sample(noise, latent_image, sampler, sigmas, **kwargs)

    # ------------------------------------------------------------------
    # predict_noise() — custom multi-pass guidance
    # ------------------------------------------------------------------

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        model = self.inner_model

        need_stg = self.stg_scale > 0
        need_mod = self.modality_scale != 1.0

        # Fast path: no STG or modality — single batched forward pass
        if not need_stg and not need_mod:
            return self._predict_noise_fast(x, timestep, model_options, seed)

        # Full multimodal guidance (2-3 passes)
        need_cfg = self.video_cfg != 1.0 or self.audio_cfg != 1.0

        # --- Pass 1: positive + negative batched together ---
        if need_cfg and negative is not None:
            results = comfy.samplers.calc_cond_batch(
                model, [positive, negative], x, timestep, model_options
            )
            pos_out, neg_out = results[0], results[1]
        else:
            pos_out = comfy.samplers.calc_cond_batch(
                model, [positive], x, timestep, model_options
            )[0]
            neg_out = pos_out

        # --- Pass 2: STG (perturbed) — requires different block behavior ---
        if need_stg:
            self.stg_flag.do_skip = True
            try:
                stg_out = comfy.samplers.calc_cond_batch(
                    model, [positive], x, timestep, model_options
                )[0]
            finally:
                self.stg_flag.do_skip = False
        else:
            stg_out = pos_out

        # --- Pass 3: modality-isolated (no cross-modal attention) ---
        if need_mod:
            mo = comfy.model_patcher.create_model_options_clone(model_options)
            to = mo.setdefault("transformer_options", {})
            to["a2v_cross_attn"] = False
            to["v2a_cross_attn"] = False
            mod_out = comfy.samplers.calc_cond_batch(
                model, [positive], x, timestep, mo
            )[0]
        else:
            mod_out = pos_out

        # --- Apply per-modality guidance ---
        if self._latent_shapes is not None and len(self._latent_shapes) > 1:
            return self._apply_multimodal_guidance(
                pos_out, neg_out, stg_out, mod_out
            )
        else:
            return self._apply_single_guidance(pos_out, neg_out, stg_out)

    def _predict_noise_fast(self, x, timestep, model_options, seed):
        """Single batched pass — per-modality CFG only, no STG/modality."""
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        model = self.inner_model

        need_cfg = self.video_cfg != 1.0 or self.audio_cfg != 1.0
        if need_cfg and negative is not None:
            results = comfy.samplers.calc_cond_batch(
                model, [positive, negative], x, timestep, model_options
            )
            pos_out, neg_out = results[0], results[1]
        else:
            pos_out = comfy.samplers.calc_cond_batch(
                model, [positive], x, timestep, model_options
            )[0]
            neg_out = pos_out

        # Per-modality CFG (stg/mod terms vanish since stg_out=mod_out=pos_out)
        if self._latent_shapes is not None and len(self._latent_shapes) > 1:
            return self._apply_multimodal_guidance(
                pos_out, neg_out, pos_out, pos_out
            )
        else:
            return self._apply_single_guidance(pos_out, neg_out, pos_out)

    # ------------------------------------------------------------------
    # Guidance math
    # ------------------------------------------------------------------

    def _apply_multimodal_guidance(self, pos_out, neg_out, stg_out, mod_out):
        """Unpack per-modality, apply guidance formula, repack."""
        v_pos, a_pos = comfy.utils.unpack_latents(pos_out, self._latent_shapes)
        v_neg, a_neg = comfy.utils.unpack_latents(neg_out, self._latent_shapes)
        v_stg, a_stg = comfy.utils.unpack_latents(stg_out, self._latent_shapes)
        v_mod, a_mod = comfy.utils.unpack_latents(mod_out, self._latent_shapes)

        # Video: pos + (vcfg-1)*(pos-neg) + stg*(pos-ptb) + (mod-1)*(pos-mod_isolated)
        v_out = v_pos.clone()
        if self.video_cfg != 1.0:
            v_out += (self.video_cfg - 1.0) * (v_pos - v_neg)
        if self.stg_scale > 0:
            v_out += self.stg_scale * (v_pos - v_stg)
        if self.modality_scale != 1.0:
            v_out += (self.modality_scale - 1.0) * (v_pos - v_mod)
        v_out = self._apply_rescale(v_out, v_pos)

        # Audio: pos + (acfg-1)*(pos-neg) + stg*(pos-ptb) + (mod-1)*(pos-mod_isolated)
        a_out = a_pos.clone()
        if self.audio_cfg != 1.0:
            a_out += (self.audio_cfg - 1.0) * (a_pos - a_neg)
        if self.stg_scale > 0:
            a_out += self.stg_scale * (a_pos - a_stg)
        if self.modality_scale != 1.0:
            a_out += (self.modality_scale - 1.0) * (a_pos - a_mod)
        a_out = self._apply_rescale(a_out, a_pos)

        packed, _ = comfy.utils.pack_latents([v_out, a_out])
        return packed

    def _apply_single_guidance(self, pos_out, neg_out, stg_out):
        """Single-modality (video-only) guidance fallback."""
        out = pos_out.clone()
        if self.video_cfg != 1.0:
            out += (self.video_cfg - 1.0) * (pos_out - neg_out)
        if self.stg_scale > 0:
            out += self.stg_scale * (pos_out - stg_out)
        out = self._apply_rescale(out, pos_out)
        return out

    def _apply_rescale(self, guided, positive):
        """CFG rescale: blend guided toward matching positive's std."""
        if self.rescale <= 0 or self.rescale >= 1.0:
            return guided
        std_pos = positive.std()
        std_guided = guided.std()
        if std_guided > 0:
            rescaled = guided * (std_pos / std_guided)
            return self.rescale * rescaled + (1.0 - self.rescale) * guided
        return guided
