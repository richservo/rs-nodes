"""MultimodalGuider for LTXV AV (audio-video) models.

Provides per-modality CFG, Spatiotemporal Guidance (STG), cross-modal
guidance, and CFG rescaling — all in one guider that extends ComfyUI's
CFGGuider.

STG implementation matches the official ComfyUI-LTXVideo approach using
PatchAttention context managers for global attention monkey-patching.
"""

import contextlib
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import comfy.ldm.modules.attention
import comfy.model_patcher
import comfy.samplers
import comfy.utils


# ---------------------------------------------------------------------------
# STG (Spatiotemporal Guidance) helpers — matches official ComfyUI-LTXVideo
# ---------------------------------------------------------------------------

@dataclass
class STGFlag:
    """Mutable flag shared by all block wrappers, toggled per forward pass."""
    do_skip: bool = False
    skip_layers: List[int] = None


class PatchAttention(contextlib.AbstractContextManager):
    """Context manager that temporarily replaces global attention functions
    to return v directly (identity) for specified attention indices.

    Matches the official ComfyUI-LTXVideo stg.py PatchAttention.
    """

    def __init__(self, attn_idx: Optional[Union[int, List[int]]] = None):
        self.current_idx = -1
        if isinstance(attn_idx, int):
            self.attn_idx = [attn_idx]
        elif attn_idx is None:
            self.attn_idx = [0]
        else:
            self.attn_idx = list(attn_idx)

    def __enter__(self):
        self.original_attention = comfy.ldm.modules.attention.optimized_attention
        self.original_attention_masked = (
            comfy.ldm.modules.attention.optimized_attention_masked
        )
        comfy.ldm.modules.attention.optimized_attention = self.stg_attention
        comfy.ldm.modules.attention.optimized_attention_masked = (
            self.stg_attention_masked
        )

    def __exit__(self, exc_type, exc_value, traceback):
        comfy.ldm.modules.attention.optimized_attention = self.original_attention
        comfy.ldm.modules.attention.optimized_attention_masked = (
            self.original_attention_masked
        )
        self.original_attention = None
        self.original_attention_masked = None

    def stg_attention(self, q, k, v, heads, *args, **kwargs):
        self.current_idx += 1
        if self.current_idx in self.attn_idx:
            return v
        return self.original_attention(q, k, v, heads, *args, **kwargs)

    def stg_attention_masked(self, q, k, v, heads, *args, **kwargs):
        self.current_idx += 1
        if self.current_idx in self.attn_idx:
            return v
        return self.original_attention_masked(q, k, v, heads, *args, **kwargs)


class STGBlockWrapper:
    """Wraps a transformer block to optionally apply STG attention skipping
    via PatchAttention context manager. Matches official implementation."""

    def __init__(self, block, stg_flag: STGFlag, idx: int):
        self.flag = stg_flag
        self.idx = idx
        self.block = block

    def __call__(self, args, extra_args):
        context_manager = contextlib.nullcontext()
        stg_indexes = args["transformer_options"].get("stg_indexes", [0])
        if self.flag.do_skip and self.idx in self.flag.skip_layers:
            context_manager = PatchAttention(stg_indexes)
        with context_manager:
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

        # Install STG block wrappers on ALL blocks (matching official)
        self.stg_flag = STGFlag(skip_layers=list(stg_blocks))
        self._patch_model(self.model_patcher, self.stg_flag)

    @classmethod
    def _patch_model(cls, model, stg_flag):
        """Wrap all transformer blocks with STGBlockWrapper."""
        transformer_blocks = cls._get_transformer_blocks(model)
        for i, block in enumerate(transformer_blocks):
            model.set_model_patch_replace(
                STGBlockWrapper(block, stg_flag, i), "dit", "double_block", i
            )

    @staticmethod
    def _get_transformer_blocks(model):
        """Get transformer blocks, handling both model architectures."""
        diffusion_model = model.get_model_object("diffusion_model")
        key = "diffusion_model.transformer_blocks"
        if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
            key = "diffusion_model.transformer.transformer_blocks"
        return model.get_model_object(key)

    @staticmethod
    def _calc_stg_indexes(run_vx, run_ax):
        """Compute which self-attention indices to skip for STG.

        Attention call order depends on active modalities:
          Video only:  [0: v_self, 1: v_cross]
          Audio only:  [0: a_self, 1: a_cross]
          Both (AV):   [0: v_self, 1: v_cross, 2: a_self, 3: a_cross, 4: a2v, 5: v2a]
        """
        stg_indexes = set()
        num_self_attns = int(run_vx) + int(run_ax)
        video_attn_idx = 0
        audio_attn_idx = 0 if num_self_attns == 1 else 2
        if run_vx:
            stg_indexes.add(video_attn_idx)
        if run_ax:
            stg_indexes.add(audio_attn_idx)
        return list(stg_indexes)

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
    # predict_noise() — multi-pass guidance matching official flow
    # ------------------------------------------------------------------

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        model = self.inner_model

        is_av = self._latent_shapes is not None and len(self._latent_shapes) > 1
        need_stg = self.stg_scale > 0
        need_mod = self.modality_scale != 1.0 and is_av
        need_cfg = self.video_cfg != 1.0 or self.audio_cfg != 1.0

        # Fast path: no STG or modality — single batched forward pass
        if not need_stg and not need_mod:
            return self._predict_noise_fast(x, timestep, model_options, seed)

        run_vx = True
        run_ax = is_av
        to = model_options.setdefault("transformer_options", {})

        # --- Pass 1: positive ---
        try:
            to["run_vx"] = run_vx
            to["run_ax"] = run_ax
            pos_out = comfy.samplers.calc_cond_batch(
                model, [positive], x, timestep, model_options
            )[0]
        finally:
            to.pop("run_vx", None)
            to.pop("run_ax", None)

        # Initialize all per-modality predictions as pos (so any unused
        # guidance term gives pos - pos = 0 in the formula)
        if is_av:
            v_pos, a_pos = comfy.utils.unpack_latents(pos_out, self._latent_shapes)
            v_neg, a_neg = v_pos, a_pos
            v_stg, a_stg = v_pos, a_pos
            v_mod, a_mod = v_pos, a_pos
        else:
            v_pos = pos_out
            v_neg = v_stg = v_mod = v_pos

        # --- Pass 2: negative ---
        if need_cfg and negative is not None:
            try:
                to["run_vx"] = run_vx
                to["run_ax"] = run_ax
                neg_out = comfy.samplers.calc_cond_batch(
                    model, [negative], x, timestep, model_options
                )[0]
                if is_av:
                    v_neg, a_neg = comfy.utils.unpack_latents(
                        neg_out, self._latent_shapes
                    )
                else:
                    v_neg = neg_out
            finally:
                to.pop("run_vx", None)
                to.pop("run_ax", None)

        # --- Pass 3: STG (perturbed) ---
        if need_stg:
            stg_indexes = self._calc_stg_indexes(run_vx, run_ax)
            try:
                to["run_vx"] = run_vx
                to["run_ax"] = run_ax
                to["ptb_index"] = 0
                to["stg_indexes"] = stg_indexes
                self.stg_flag.do_skip = True
                stg_out = comfy.samplers.calc_cond_batch(
                    model, [positive], x, timestep, model_options
                )[0]
                if is_av:
                    v_stg, a_stg = comfy.utils.unpack_latents(
                        stg_out, self._latent_shapes
                    )
                else:
                    v_stg = stg_out
            finally:
                self.stg_flag.do_skip = False
                to.pop("ptb_index", None)
                to.pop("stg_indexes", None)
                to.pop("run_vx", None)
                to.pop("run_ax", None)

        # --- Pass 4: modality-isolated (no cross-modal attention) ---
        if need_mod:
            try:
                to["run_vx"] = run_vx
                to["run_ax"] = run_ax
                to["a2v_cross_attn"] = False
                to["v2a_cross_attn"] = False
                mod_out = comfy.samplers.calc_cond_batch(
                    model, [positive], x, timestep, model_options
                )[0]
                v_mod, a_mod = comfy.utils.unpack_latents(
                    mod_out, self._latent_shapes
                )
            finally:
                to.pop("a2v_cross_attn", None)
                to.pop("v2a_cross_attn", None)
                to.pop("run_vx", None)
                to.pop("run_ax", None)

        # --- Apply per-modality guidance ---
        v_out = self._calculate(v_pos, v_neg, v_stg, v_mod, self.video_cfg)

        if is_av:
            a_out = self._calculate(a_pos, a_neg, a_stg, a_mod, self.audio_cfg)
            packed, _ = comfy.utils.pack_latents([v_out, a_out])
            return packed
        return v_out

    def _predict_noise_fast(self, x, timestep, model_options, seed):
        """Single batched pass — per-modality CFG only, no STG/modality."""
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        model = self.inner_model

        is_av = self._latent_shapes is not None and len(self._latent_shapes) > 1
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

        if is_av:
            v_pos, a_pos = comfy.utils.unpack_latents(
                pos_out, self._latent_shapes
            )
            if neg_out is pos_out:
                v_neg, a_neg = v_pos, a_pos
            else:
                v_neg, a_neg = comfy.utils.unpack_latents(
                    neg_out, self._latent_shapes
                )
            v_out = self._calculate(v_pos, v_neg, v_pos, v_pos, self.video_cfg)
            a_out = self._calculate(a_pos, a_neg, a_pos, a_pos, self.audio_cfg)
            packed, _ = comfy.utils.pack_latents([v_out, a_out])
            return packed
        else:
            return self._calculate(
                pos_out, neg_out, pos_out, pos_out, self.video_cfg
            )

    # ------------------------------------------------------------------
    # Guidance math — matches official GuiderParameters.calculate
    # ------------------------------------------------------------------

    def _calculate(self, pos, neg, perturbed, modality, cfg):
        """Apply the full guidance formula:
        pos + (cfg-1)*(pos-neg) + stg*(pos-perturbed) + (mod-1)*(pos-modality)
        """
        noise_pred = (
            pos
            + (cfg - 1) * (pos - neg)
            + self.stg_scale * (pos - perturbed)
            + (self.modality_scale - 1) * (pos - modality)
        )
        if self.rescale != 0:
            factor = pos.std() / noise_pred.std()
            factor = self.rescale * factor + (1 - self.rescale)
            noise_pred = noise_pred * factor
        return noise_pred
