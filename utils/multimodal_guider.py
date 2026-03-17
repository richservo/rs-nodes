"""MultimodalGuider for LTXV AV (audio-video) models.

Provides per-modality CFG, Spatiotemporal Guidance (STG), cross-modal
guidance, and CFG rescaling — all in one guider that extends ComfyUI's
CFGGuider.

STG implementation matches the official ComfyUI-LTXVideo approach using
PatchAttention context managers for global attention monkey-patching.

ICLoRAGuider extends MultimodalGuider to add IC-LoRA guide frame injection,
model sampling shift, and guide frame lifecycle management.
"""

import contextlib
import logging
import math
import types
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import comfy.ldm.common_dit
import comfy.ldm.modules.attention
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
import comfy.utils

logger = logging.getLogger(__name__)


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
# VRAM-efficient block forward — based on kjnodes LTX2ForwardPatch
# ---------------------------------------------------------------------------

def _ltx2_forward(
    self,
    x: Tuple[torch.Tensor, torch.Tensor],
    v_context=None, a_context=None, attention_mask=None,
    v_timestep=None, a_timestep=None,
    v_pe=None, a_pe=None, v_cross_pe=None, a_cross_pe=None,
    v_cross_scale_shift_timestep=None, a_cross_scale_shift_timestep=None,
    v_cross_gate_timestep=None, a_cross_gate_timestep=None,
    transformer_options=None, **extra_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VRAM-efficient LTXAV block forward with per-modality attention scaling.

    Replaces the default block forward with explicit intermediate tensor
    deletion to reduce peak VRAM usage for long video generation.
    Based on kjnodes ltx2_forward.
    """
    run_vx = transformer_options.get("run_vx", True)
    run_ax = transformer_options.get("run_ax", True)
    video_scale = getattr(self, "video_scale", 1.0)
    audio_scale = getattr(self, "audio_scale", 1.0)
    audio_to_video_scale = getattr(self, "audio_to_video_scale", 1.0)
    video_to_audio_scale = getattr(self, "video_to_audio_scale", 1.0)

    vx, ax = x
    run_ax = run_ax and ax.numel() > 0 and audio_scale != 0.0
    run_a2v = (run_vx and transformer_options.get("a2v_cross_attn", True)
               and ax.numel() > 0 and audio_to_video_scale != 0.0)
    run_v2a = (run_ax and transformer_options.get("v2a_cross_attn", True)
               and video_to_audio_scale != 0.0)

    # Video self-attention
    if run_vx:
        vshift_msa, vscale_msa = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], v_timestep, slice(0, 2))
        norm_vx = comfy.ldm.common_dit.rms_norm(vx) * (1 + vscale_msa) + vshift_msa
        del vshift_msa, vscale_msa
        attn1_out = self.attn1(norm_vx, pe=v_pe, transformer_options=transformer_options)
        del norm_vx
        vgate_msa = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], v_timestep, slice(2, 3))[0]
        vx += attn1_out * vgate_msa * video_scale
        del vgate_msa, attn1_out
        vx.add_(self.attn2(
            comfy.ldm.common_dit.rms_norm(vx),
            context=v_context.contiguous() if v_context is not None else None,
            mask=attention_mask, transformer_options=transformer_options,
        ), alpha=video_scale)

    # Audio self-attention
    if run_ax:
        ashift_msa, ascale_msa = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(0, 2))
        norm_ax = comfy.ldm.common_dit.rms_norm(ax) * (1 + ascale_msa) + ashift_msa
        del ashift_msa, ascale_msa
        attn1_out = self.audio_attn1(norm_ax, pe=a_pe, transformer_options=transformer_options)
        del norm_ax
        agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(2, 3))[0]
        ax += attn1_out * agate_msa * audio_scale
        del agate_msa, attn1_out
        ax.add_(self.audio_attn2(
            comfy.ldm.common_dit.rms_norm(ax),
            context=a_context.contiguous() if a_context is not None else None,
            mask=attention_mask, transformer_options=transformer_options,
        ), alpha=audio_scale)

    # Audio-Video cross attention
    if run_a2v or run_v2a:
        vx_norm3 = comfy.ldm.common_dit.rms_norm(vx)
        ax_norm3 = comfy.ldm.common_dit.rms_norm(ax)

        if run_a2v:
            scale_v, shift_v = self.get_ada_values(
                self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0],
                v_cross_scale_shift_timestep, slice(0, 2))
            vx_scaled = vx_norm3 * (1 + scale_v) + shift_v
            del scale_v, shift_v
            scale_a, shift_a = self.get_ada_values(
                self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0],
                a_cross_scale_shift_timestep, slice(0, 2))
            ax_scaled = ax_norm3 * (1 + scale_a) + shift_a
            del scale_a, shift_a
            a2v_out = self.audio_to_video_attn(
                vx_scaled, context=ax_scaled, pe=v_cross_pe,
                k_pe=a_cross_pe, transformer_options=transformer_options)
            del vx_scaled, ax_scaled
            gate_a2v = self.get_ada_values(
                self.scale_shift_table_a2v_ca_video[4:, :], vx.shape[0],
                v_cross_gate_timestep, slice(0, 1))[0]
            vx += a2v_out * gate_a2v * audio_to_video_scale
            del gate_a2v, a2v_out

        if run_v2a:
            scale_v, shift_v = self.get_ada_values(
                self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0],
                v_cross_scale_shift_timestep, slice(2, 4))
            vx_scaled = vx_norm3 * (1 + scale_v) + shift_v
            del scale_v, shift_v, vx_norm3
            scale_a, shift_a = self.get_ada_values(
                self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0],
                a_cross_scale_shift_timestep, slice(2, 4))
            ax_scaled = ax_norm3 * (1 + scale_a) + shift_a
            del scale_a, shift_a, ax_norm3
            v2a_out = self.video_to_audio_attn(
                ax_scaled, context=vx_scaled, pe=a_cross_pe,
                k_pe=v_cross_pe, transformer_options=transformer_options)
            del ax_scaled, vx_scaled
            gate_v2a = self.get_ada_values(
                self.scale_shift_table_a2v_ca_audio[4:, :], ax.shape[0],
                a_cross_gate_timestep, slice(0, 1))[0]
            ax += v2a_out * gate_v2a * video_to_audio_scale
            del gate_v2a, v2a_out

    # Video feedforward
    if run_vx:
        vshift_mlp, vscale_mlp = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], v_timestep, slice(3, 5))
        vx_scaled = comfy.ldm.common_dit.rms_norm(vx) * (1 + vscale_mlp) + vshift_mlp
        del vshift_mlp, vscale_mlp
        ff_out = self.ff(vx_scaled)
        del vx_scaled
        vgate_mlp = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], v_timestep, slice(5, 6))[0]
        vx += ff_out * vgate_mlp * video_scale
        del vgate_mlp, ff_out

    # Audio feedforward
    if run_ax:
        ashift_mlp, ascale_mlp = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(3, 5))
        ax_scaled = comfy.ldm.common_dit.rms_norm(ax) * (1 + ascale_mlp) + ashift_mlp
        del ashift_mlp, ascale_mlp
        ff_out = self.audio_ff(ax_scaled)
        del ax_scaled
        agate_mlp = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(5, 6))[0]
        ax += ff_out * agate_mlp * audio_scale
        del agate_mlp, ff_out

    return vx, ax


class _AttentionTunerPatch:
    """Descriptor that wraps a block's forward with _ltx2_forward + scale factors."""
    def __init__(self, video_scale=1.0, audio_scale=1.0,
                 a2v_scale=1.0, v2a_scale=1.0):
        self.video_scale = video_scale
        self.audio_scale = audio_scale
        self.a2v_scale = a2v_scale
        self.v2a_scale = v2a_scale

    def __get__(self, obj, objtype=None):
        vs = self.video_scale
        as_ = self.audio_scale
        a2vs = self.a2v_scale
        v2as = self.v2a_scale
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.video_scale = vs
            self_module.audio_scale = as_
            self_module.audio_to_video_scale = a2vs
            self_module.video_to_audio_scale = v2as
            return _ltx2_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)


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
        video_cfg_end=None,
        stg_scale_end=None,
        # Per-modality overrides (None = use shared value above)
        audio_stg_scale=None,
        audio_rescale=None,
        audio_modality_scale=None,
        video_modality_scale=None,
        # CFG-Zero* (project neg onto pos before guidance formula)
        cfg_star_rescale=True,
        # Skip-steps: return zeros when sigma > threshold (CFG-Zero init)
        skip_sigma_threshold=0.0,
        # VRAM-efficient forward + attention scaling
        video_attn_scale=1.0,
    ):
        if stg_blocks is None:
            stg_blocks = [29]

        # Clone model to isolate guider modifications (matching official)
        model = model.clone()

        self._current_step = 0
        self._total_steps = 1
        self.last_denoised_v = None
        self.last_denoised_a = None

        # Register ON_PRE_RUN callback to reset step counter (matching official)
        model.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_PRE_RUN,
            "mm_guider_on_pre_run",
            self._reset_state,
        )

        super().__init__(model)

        self.video_cfg = video_cfg
        self.audio_cfg = audio_cfg
        self.stg_scale = stg_scale
        self.stg_blocks = stg_blocks
        self.rescale = rescale
        self.modality_scale = modality_scale
        self.video_cfg_end = video_cfg_end if video_cfg_end is not None else video_cfg
        self.stg_scale_end = stg_scale_end if stg_scale_end is not None else stg_scale

        # Per-modality parameters (fall back to shared values)
        self.audio_stg_scale = audio_stg_scale if audio_stg_scale is not None else stg_scale
        self.audio_rescale = audio_rescale if audio_rescale is not None else rescale
        self.audio_modality_scale = audio_modality_scale if audio_modality_scale is not None else modality_scale
        self.video_modality_scale = video_modality_scale if video_modality_scale is not None else modality_scale

        self.cfg_star_rescale = cfg_star_rescale
        self.skip_sigma_threshold = skip_sigma_threshold

        self._latent_shapes = None

        self.inner_set_conds({"positive": positive, "negative": negative})

        # Install STG block wrappers on ALL blocks (matching official)
        self.stg_flag = STGFlag(skip_layers=list(stg_blocks))
        self._patch_model(self.model_patcher, self.stg_flag)

        # Apply VRAM-efficient forward with attention scaling (LTXAV only)
        self._apply_attention_tuner(self.model_patcher,
                                    video_scale=video_attn_scale)

    def _reset_state(self, model_patcher=None):
        """Reset per-run state. Called via ON_PRE_RUN callback."""
        self._current_step = 0
        self.last_denoised_v = None
        self.last_denoised_a = None

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

    @classmethod
    def _apply_attention_tuner(cls, model, video_scale=1.0, audio_scale=1.0):
        """Patch all LTXAV blocks with VRAM-efficient forward + attention scales.

        Replaces each block's forward with _ltx2_forward, which aggressively
        deletes intermediate tensors to reduce peak VRAM for long generations.
        Only applies to LTXAV models (skips plain LTXV).
        """
        diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
            return  # Plain LTXV — different block structure
        blocks = cls._get_transformer_blocks(model)
        for idx in range(len(blocks)):
            block = blocks[idx]
            patched = _AttentionTunerPatch(
                video_scale=video_scale, audio_scale=audio_scale,
            ).__get__(block, block.__class__)
            model.add_object_patch(
                f"diffusion_model.transformer_blocks.{idx}.forward", patched,
            )
        logger.info(f"Attention tuner: {len(blocks)} blocks "
                    f"(video_scale={video_scale})")

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
        self._current_step = 0
        self._total_steps = max(sigmas.shape[-1] - 1, 1)
        return super().sample(noise, latent_image, sampler, sigmas, **kwargs)

    # ------------------------------------------------------------------
    # predict_noise() — multi-pass guidance matching official flow
    # ------------------------------------------------------------------

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        # CFG-Zero init: skip guidance on near-pure noise (high sigma steps)
        if self.skip_sigma_threshold > 0 and timestep > self.skip_sigma_threshold:
            self._current_step += 1
            return torch.zeros_like(x)

        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        model = self.inner_model

        # Per-step interpolation (plain floats — no tensor graph retention)
        current_step = self._current_step
        self._current_step = current_step + 1
        t = current_step / max(self._total_steps - 1, 1)
        cur_cfg = float(self.video_cfg + t * (self.video_cfg_end - self.video_cfg))
        cur_stg = float(self.stg_scale + t * (self.stg_scale_end - self.stg_scale))

        is_av = self._latent_shapes is not None and len(self._latent_shapes) > 1
        need_stg = (self.stg_scale > 0 or self.stg_scale_end > 0
                    or (is_av and self.audio_stg_scale > 0))
        need_mod = is_av and (self.video_modality_scale != 1.0
                              or self.audio_modality_scale != 1.0)
        need_cfg = (self.video_cfg != 1.0 or self.video_cfg_end != 1.0
                    or self.audio_cfg != 1.0)

        # Fast path: no STG or modality — single batched forward pass
        if not need_stg and not need_mod:
            return self._predict_noise_fast(x, timestep, model_options, seed,
                                            cur_cfg=cur_cfg)

        run_vx = True
        run_ax = is_av
        # Explicit cross-attention flags (matching official — always set
        # even though the model defaults to True when absent)
        run_a2v = True
        run_v2a = True
        to = model_options.setdefault("transformer_options", {})

        # Track packed outputs for post-cfg hooks
        noise_pred_neg = 0
        noise_pred_perturbed = 0

        # --- Pass 1: positive (with explicit cross-attn flags) ---
        try:
            to["run_vx"] = run_vx
            to["run_ax"] = run_ax
            to["a2v_cross_attn"] = run_a2v
            to["v2a_cross_attn"] = run_v2a
            noise_pred_pos = comfy.samplers.calc_cond_batch(
                model, [positive], x, timestep, model_options
            )[0]
        finally:
            del to["run_vx"], to["run_ax"]
            del to["a2v_cross_attn"], to["v2a_cross_attn"]

        # Unpack per-modality predictions; initialize guidance terms to pos
        # so that any unused term gives (pos - pos) = 0 in the formula.
        # (The official code uses 0, but that only works because it always
        # runs in AV mode where need_mod is gated by do_modality().)
        if is_av:
            v_pos, a_pos = comfy.utils.unpack_latents(noise_pred_pos, self._latent_shapes)
            v_neg, a_neg = v_pos, a_pos
            v_stg, a_stg = v_pos, a_pos
            v_mod, a_mod = v_pos, a_pos
        else:
            v_pos = noise_pred_pos
            v_neg = v_stg = v_mod = v_pos

        # --- Pass 2: negative ---
        if need_cfg and negative is not None:
            try:
                to["run_vx"] = run_vx
                to["run_ax"] = run_ax
                to["a2v_cross_attn"] = run_a2v
                to["v2a_cross_attn"] = run_v2a
                noise_pred_neg = comfy.samplers.calc_cond_batch(
                    model, [negative], x, timestep, model_options
                )[0]

                # CFG-Zero* rescale: project negative onto positive direction
                # to prevent garbage negative predictions at high sigma.
                # Reference: https://arxiv.org/abs/2503.18886
                if self.cfg_star_rescale:
                    batch_size = noise_pred_pos.shape[0]
                    pos_flat = noise_pred_pos.view(batch_size, -1)
                    neg_flat = noise_pred_neg.view(batch_size, -1)
                    dot = torch.sum(pos_flat * neg_flat, dim=1, keepdim=True)
                    sq_norm = torch.sum(neg_flat ** 2, dim=1, keepdim=True) + 1e-8
                    alpha = dot / sq_norm
                    noise_pred_neg = alpha * noise_pred_neg

                if is_av:
                    v_neg, a_neg = comfy.utils.unpack_latents(
                        noise_pred_neg, self._latent_shapes
                    )
                else:
                    v_neg = noise_pred_neg
            finally:
                del to["run_vx"], to["run_ax"]
                del to["a2v_cross_attn"], to["v2a_cross_attn"]

        # --- Pass 3: STG (perturbed) ---
        if need_stg:
            stg_indexes = self._calc_stg_indexes(run_vx, run_ax)
            try:
                to["run_vx"] = run_vx
                to["run_ax"] = run_ax
                to["a2v_cross_attn"] = run_a2v
                to["v2a_cross_attn"] = run_v2a
                to["ptb_index"] = 0
                to["stg_indexes"] = stg_indexes
                self.stg_flag.do_skip = True
                noise_pred_perturbed = comfy.samplers.calc_cond_batch(
                    model, [positive], x, timestep, model_options
                )[0]
                if is_av:
                    v_stg, a_stg = comfy.utils.unpack_latents(
                        noise_pred_perturbed, self._latent_shapes
                    )
                else:
                    v_stg = noise_pred_perturbed
            finally:
                self.stg_flag.do_skip = False
                del to["ptb_index"], to["stg_indexes"]
                del to["run_vx"], to["run_ax"]
                del to["a2v_cross_attn"], to["v2a_cross_attn"]

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
                del to["a2v_cross_attn"], to["v2a_cross_attn"]
                del to["run_vx"], to["run_ax"]

        # --- Apply per-modality guidance ---
        v_out = self._calculate(v_pos, v_neg, v_stg, v_mod, cur_cfg, cur_stg,
                                modality_scale=self.video_modality_scale)

        if is_av:
            a_out = self._calculate(a_pos, a_neg, a_stg, a_mod,
                                    self.audio_cfg, self.audio_stg_scale,
                                    rescale=self.audio_rescale,
                                    modality_scale=self.audio_modality_scale)
            result, _ = comfy.utils.pack_latents([v_out, a_out])
        else:
            result = v_out

        # Replicate sampler_post_cfg_function hooks (matching official
        # MultimodalGuider — normally called inside cfg_function, but we
        # bypass it for multi-pass guidance)
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": result,
                "cond": positive,
                "uncond": negative,
                "model": model,
                "uncond_denoised": noise_pred_neg if need_cfg else 0,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                "perturbed_cond": positive,
                "perturbed_cond_denoised": noise_pred_perturbed if need_stg else 0,
            }
            result = fn(args)

        # Track last denoised per modality (matching official — needed for
        # skip_step support and ensures hooks see consistent state)
        if is_av:
            self.last_denoised_v, self.last_denoised_a = comfy.utils.unpack_latents(
                result, self._latent_shapes
            )
        else:
            self.last_denoised_v = result

        return result

    def _predict_noise_fast(self, x, timestep, model_options, seed,
                            cur_cfg=None):
        """Single batched pass — per-modality CFG only, no STG/modality."""
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        model = self.inner_model

        if cur_cfg is None:
            cur_cfg = self.video_cfg

        is_av = self._latent_shapes is not None and len(self._latent_shapes) > 1
        need_cfg = cur_cfg != 1.0 or self.audio_cfg != 1.0

        if need_cfg and negative is not None:
            results = comfy.samplers.calc_cond_batch(
                model, [positive, negative], x, timestep, model_options
            )
            pos_out, neg_out = results[0], results[1]

            # CFG-Zero* rescale (same as multi-pass path)
            if self.cfg_star_rescale:
                batch_size = pos_out.shape[0]
                pos_flat = pos_out.view(batch_size, -1)
                neg_flat = neg_out.view(batch_size, -1)
                dot = torch.sum(pos_flat * neg_flat, dim=1, keepdim=True)
                sq_norm = torch.sum(neg_flat ** 2, dim=1, keepdim=True) + 1e-8
                alpha = dot / sq_norm
                neg_out = alpha * neg_out
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
            v_out = self._calculate(v_pos, v_neg, v_pos, v_pos, cur_cfg, 0)
            a_out = self._calculate(a_pos, a_neg, a_pos, a_pos,
                                    self.audio_cfg, 0)
            result, _ = comfy.utils.pack_latents([v_out, a_out])
        else:
            result = self._calculate(
                pos_out, neg_out, pos_out, pos_out, cur_cfg, 0
            )

        # Replicate sampler_post_cfg_function hooks
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": result,
                "cond": positive,
                "uncond": negative,
                "model": model,
                "uncond_denoised": neg_out if need_cfg else 0,
                "cond_denoised": pos_out,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
            }
            result = fn(args)

        # Track last denoised per modality
        if is_av:
            self.last_denoised_v, self.last_denoised_a = comfy.utils.unpack_latents(
                result, self._latent_shapes
            )
        else:
            self.last_denoised_v = result

        return result

    # ------------------------------------------------------------------
    # Guidance math — matches official GuiderParameters.calculate
    # ------------------------------------------------------------------

    def _calculate(self, pos, neg, perturbed, modality, cfg, stg_scale,
                    rescale=None, modality_scale=None):
        """Apply the full guidance formula (matching official GuiderParameters.calculate):
        pos + (cfg-1)*(pos-neg) + stg*(pos-perturbed) + (mod-1)*(pos-modality)

        When neg/perturbed/modality is 0 (scalar), the corresponding term
        reduces to just pos*(factor) which the formula handles naturally.
        """
        if rescale is None:
            rescale = self.rescale
        if modality_scale is None:
            modality_scale = self.modality_scale
        noise_pred = (
            pos
            + (cfg - 1) * (pos - neg)
            + stg_scale * (pos - perturbed)
            + (modality_scale - 1) * (pos - modality)
        )
        if rescale != 0:
            factor = pos.std() / noise_pred.std()
            factor = rescale * factor + (1 - rescale)
            noise_pred = noise_pred * factor
        return noise_pred


# ---------------------------------------------------------------------------
# ICLoRAGuider — MultimodalGuider with IC-LoRA guide frame injection
# ---------------------------------------------------------------------------

class ICLoRAGuider(MultimodalGuider):
    """Extends MultimodalGuider to handle IC-LoRA guide frame lifecycle.

    Encoding of the control image is deferred to sample() where the actual
    video latent dimensions are known. This ensures the guide latent spatial
    dims always match the video latent.

    Manages:
    - Deferred VAE encoding of control pixels (at correct latent dims)
    - Deferred ModelSamplingFlux shift (applied when latent dims known)
    - Guide keyframe conditioning injection (keyframe_idxs metadata)
    - Guide latent appending before sampling (video-only, NestedTensor safe)
    - Denoise mask update for guide frames (mask ~ 0 to preserve)
    - Noise regeneration to match expanded latent
    - Guide frame stripping from sampled output
    """

    def __init__(
        self,
        model,
        positive,
        negative,
        control_pixels,
        vae,
        downscale_factor,
        guide_strength,
        guide_frame_idx,
        max_shift,
        base_shift,
        **kwargs,
    ):
        super().__init__(model, positive, negative, **kwargs)
        self._control_pixels = control_pixels
        self._vae = vae
        self._downscale_factor = downscale_factor
        self._guide_strength = guide_strength
        self._guide_frame_idx = guide_frame_idx
        self._max_shift = max_shift
        self._base_shift = base_shift
        # Set after encoding in sample()
        self._num_guide_frames = 0
        # Save original conds so we can reset before each encode pass
        # (prevents keyframe_idxs from accumulating across re-runs)
        self._orig_positive = positive
        self._orig_negative = negative

    # ------------------------------------------------------------------
    # Model sampling shift — deferred until latent dims are known
    # ------------------------------------------------------------------

    def _apply_model_sampling(self, latent_image):
        """Apply ModelSamplingFlux shift based on actual latent dimensions.

        Uses the same shift formula as ltxv_generate.py (lines 395-402).
        """
        # Compute token count from video latent (exclude audio if nested)
        if latent_image.is_nested:
            tokens = math.prod(latent_image.unbind()[0].shape[2:])
        else:
            tokens = math.prod(latent_image.shape[2:])

        x1, x2 = 1024, 4096
        mm_shift = (self._max_shift - self._base_shift) / (x2 - x1)
        b = self._base_shift - mm_shift * x1
        shift = tokens * mm_shift + b

        # Build combined ModelSamplingFlux + CONST class
        class ModelSamplingAdvanced(
            comfy.model_sampling.ModelSamplingFlux,
            comfy.model_sampling.CONST,
        ):
            pass

        sampling_obj = ModelSamplingAdvanced(self.model_patcher.model.model_config)
        sampling_obj.set_parameters(shift=shift)
        self.model_patcher.add_object_patch("model_sampling", sampling_obj)
        logger.info(f"Applied model sampling shift={shift:.4f} (tokens={tokens})")

    # ------------------------------------------------------------------
    # Deferred encoding — called once at the start of sample()
    # ------------------------------------------------------------------

    def _encode_and_inject_guide(self, latent_image, denoise_mask):
        """Encode control pixels and inject guide into latent + conditioning.

        Called at the start of sample() when the actual video latent
        dimensions are known. Replicates the official LTXAddVideoICLoRAGuide
        execute() flow exactly — uses append_keyframe with the REAL video
        latent (not a dummy) so the returned latent and noise_mask are the
        final ones used for sampling.
        """
        from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask

        # Reset conditioning to original state (prevents keyframe_idxs
        # from accumulating across re-runs when guider is reused)
        self.inner_set_conds({
            "positive": self._orig_positive,
            "negative": self._orig_negative,
        })

        # --- Extract video latent (handle NestedTensor for AV) ---
        if latent_image.is_nested:
            parts = latent_image.unbind()
            video = parts[0]
            audio = parts[1] if len(parts) > 1 else None
        else:
            video = latent_image
            audio = None

        _, _, latent_t, latent_h, latent_w = video.shape
        scale_factors = self._vae.downscale_index_formula
        time_sf = scale_factors[0]
        dsf = self._downscale_factor

        # --- Build video latent dict + noise_mask (matches official flow) ---
        video_dict = {"samples": video}
        if denoise_mask is not None:
            if denoise_mask.is_nested:
                video_dict["noise_mask"] = denoise_mask.unbind()[0]
            else:
                video_dict["noise_mask"] = denoise_mask
        noise_mask = get_noise_mask(video_dict)

        # --- Causal fix (matches official iclora.py:166-172) ---
        images = self._control_pixels
        num_frames_to_keep = ((images.shape[0] - 1) // time_sf) * time_sf + 1
        images = images[:num_frames_to_keep]
        causal_fix = self._guide_frame_idx == 0 or num_frames_to_keep == 1
        if not causal_fix:
            images = torch.cat([images[:1], images], dim=0)

        # --- Encode at reduced resolution (matches official iclora.py:174-185) ---
        # The official IC-LoRA encode computes:
        #   target = int(latent_dim * scale_factor / dsf)
        # We pass latent_dim/dsf to base LTXVAddGuide.encode which multiplies
        # by scale_factor internally, producing the same pixel resolution.
        enc_w = latent_w // dsf if dsf > 1 else latent_w
        enc_h = latent_h // dsf if dsf > 1 else latent_h
        _, guide_latent = LTXVAddGuide.encode(
            self._vae, enc_w, enc_h, images, scale_factors
        )

        # Strip prepended causal frame
        if not causal_fix:
            guide_latent = guide_latent[:, :, 1:, :, :]
            images = images[1:]

        logger.info(f"Encoded guide latent: {list(guide_latent.shape)}")

        # --- Sparse dilation (matches official LTXVDilateLatent exactly) ---
        guide_mask = None
        if dsf > 1:
            if latent_w % dsf != 0 or latent_h % dsf != 0:
                raise ValueError(
                    f"Latent spatial size {latent_w}x{latent_h} must be "
                    f"divisible by latent_downscale_factor {dsf}"
                )
            dil_h = guide_latent.shape[3] * dsf
            dil_w = guide_latent.shape[4] * dsf
            dilated = torch.zeros(
                guide_latent.shape[:3] + (dil_h, dil_w),
                device=guide_latent.device, dtype=guide_latent.dtype,
            )
            dilated[..., ::dsf, ::dsf] = guide_latent
            guide_mask = torch.full(
                (dilated.shape[0], 1, dilated.shape[2], dil_h, dil_w),
                -1.0, device=guide_latent.device, dtype=guide_latent.dtype,
            )
            guide_mask[..., ::dsf, ::dsf] = 1.0
            guide_latent = dilated
            logger.info(f"Dilated to: {list(guide_latent.shape)}")

        self._num_guide_frames = guide_latent.shape[2]

        # --- Keyframe injection (matches official iclora.py:219-240) ---
        positive = self._get_positive()
        negative = self._get_negative()

        frame_idx_actual, _ = LTXVAddGuide.get_latent_index(
            positive, latent_t,
            len(images), self._guide_frame_idx, scale_factors
        )

        # Use REAL video latent with append_keyframe (not a dummy).
        # This sets keyframe_idxs in conditioning AND concatenates the
        # guide latent to the video latent with proper noise_mask — the
        # exact same thing the official IC-LoRA node does.
        positive, negative, video_out, noise_mask_out = LTXVAddGuide.append_keyframe(
            positive, negative, frame_idx_actual,
            video, noise_mask,
            guide_latent, self._guide_strength, scale_factors,
            guide_mask=guide_mask,
            latent_downscale_factor=dsf,
            causal_fix=causal_fix,
        )
        logger.info(f"Guide keyframe at frame_idx={frame_idx_actual}, "
                    f"strength={self._guide_strength}, dsf={dsf}, "
                    f"causal_fix={causal_fix}")

        self.inner_set_conds({"positive": positive, "negative": negative})
        # --- Reassemble NestedTensor if AV ---
        if audio is not None:
            latent_out = comfy.nested_tensor.NestedTensor((video_out, audio))
            # Preserve audio denoise mask
            if denoise_mask is not None and denoise_mask.is_nested:
                audio_dm = denoise_mask.unbind()[1] if len(denoise_mask.unbind()) > 1 else None
            else:
                audio_dm = None
            if audio_dm is not None:
                denoise_out = comfy.nested_tensor.NestedTensor((noise_mask_out, audio_dm))
            else:
                denoise_out = comfy.nested_tensor.NestedTensor((noise_mask_out,))
        else:
            latent_out = video_out
            denoise_out = noise_mask_out

        return latent_out, denoise_out

    def _get_positive(self):
        """Reconstruct positive conditioning in original tuple format."""
        return self._reconstruct_conds("positive")

    def _get_negative(self):
        """Reconstruct negative conditioning in original tuple format."""
        return self._reconstruct_conds("negative")

    def _reconstruct_conds(self, key):
        """Convert internal conds dict format back to ComfyUI tuple format.

        inner_set_conds stores: [{cross_attn: T, model_conds: {}, uuid: ..., ...}]
        Original format is: [[T, {model_conds: {}, ...}], ...]
        """
        conds = self.original_conds.get(key, [])
        result = []
        for c in conds:
            cross_attn = c.get("cross_attn", None)
            metadata = {k: v for k, v in c.items()
                        if k not in ("cross_attn", "uuid")}
            result.append([cross_attn, metadata])
        return result

    # ------------------------------------------------------------------
    # Guide frame stripping
    # ------------------------------------------------------------------

    def _strip_guide(self, result):
        """Remove appended guide frames from the sampled output."""
        n = self._num_guide_frames
        if n == 0:
            return result

        if result.is_nested:
            parts = result.unbind()
            video = parts[0][:, :, :-n]
            if len(parts) > 1:
                return comfy.nested_tensor.NestedTensor((video, parts[1]))
            return comfy.nested_tensor.NestedTensor((video,))

        return result[:, :, :-n]

    # ------------------------------------------------------------------
    # sample() override — full guide frame lifecycle
    # ------------------------------------------------------------------

    def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
        # 1. Apply model sampling shift (deferred until latent dims known)
        self._apply_model_sampling(latent_image)

        # 2. Encode guide + inject into latent and conditioning
        denoise_mask = kwargs.get("denoise_mask", None)
        latent_image, denoise_mask = self._encode_and_inject_guide(
            latent_image, denoise_mask
        )
        kwargs["denoise_mask"] = denoise_mask

        # 3. Regenerate noise to match expanded latent dimensions
        noise = comfy.sample.prepare_noise(latent_image, kwargs.get("seed", 0))

        logger.info(f"Guide frames appended: {self._num_guide_frames} "
                    f"frame(s), latent temporal dim now "
                    f"{latent_image.unbind()[0].shape[2] if latent_image.is_nested else latent_image.shape[2]}")

        # 4. Run parent sampling (MultimodalGuider -> CFGGuider)
        result = super().sample(noise, latent_image, sampler, sigmas, **kwargs)

        # 5. Strip guide frames from the output
        pre_strip_dim = result.unbind()[0].shape[2] if result.is_nested else result.shape[2]
        result = self._strip_guide(result)
        post_strip_dim = result.unbind()[0].shape[2] if result.is_nested else result.shape[2]
        logger.info(f"Guide strip: {pre_strip_dim} → {post_strip_dim} "
                    f"(removed {self._num_guide_frames} guide frame(s))")

        return result
