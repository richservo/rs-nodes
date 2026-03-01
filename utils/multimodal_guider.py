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
import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import comfy.ldm.modules.attention
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
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
        video_cfg_end=None,
        stg_scale_end=None,
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
        self.video_cfg_end = video_cfg_end if video_cfg_end is not None else video_cfg
        self.stg_scale_end = stg_scale_end if stg_scale_end is not None else stg_scale
        self._latent_shapes = None
        self._current_step = 0
        self._total_steps = 1

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
        self._current_step = 0
        self._total_steps = max(sigmas.shape[-1] - 1, 1)
        return super().sample(noise, latent_image, sampler, sigmas, **kwargs)

    # ------------------------------------------------------------------
    # predict_noise() — multi-pass guidance matching official flow
    # ------------------------------------------------------------------

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        model = self.inner_model

        # Per-step interpolation (plain floats — no tensor graph retention)
        t = self._current_step / max(self._total_steps - 1, 1)
        self._current_step += 1
        cur_cfg = float(self.video_cfg + t * (self.video_cfg_end - self.video_cfg))
        cur_stg = float(self.stg_scale + t * (self.stg_scale_end - self.stg_scale))

        is_av = self._latent_shapes is not None and len(self._latent_shapes) > 1
        need_stg = self.stg_scale > 0 or self.stg_scale_end > 0
        need_mod = self.modality_scale != 1.0 and is_av
        need_cfg = (self.video_cfg != 1.0 or self.video_cfg_end != 1.0
                    or self.audio_cfg != 1.0)

        # Fast path: no STG or modality — single batched forward pass
        if not need_stg and not need_mod:
            return self._predict_noise_fast(x, timestep, model_options, seed,
                                            cur_cfg=cur_cfg)

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
        v_out = self._calculate(v_pos, v_neg, v_stg, v_mod, cur_cfg, cur_stg)

        if is_av:
            a_out = self._calculate(a_pos, a_neg, a_stg, a_mod,
                                    self.audio_cfg, self.stg_scale)
            packed, _ = comfy.utils.pack_latents([v_out, a_out])
            return packed
        return v_out

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
            packed, _ = comfy.utils.pack_latents([v_out, a_out])
            return packed
        else:
            return self._calculate(
                pos_out, neg_out, pos_out, pos_out, cur_cfg, 0
            )

    # ------------------------------------------------------------------
    # Guidance math — matches official GuiderParameters.calculate
    # ------------------------------------------------------------------

    def _calculate(self, pos, neg, perturbed, modality, cfg, stg_scale):
        """Apply the full guidance formula:
        pos + (cfg-1)*(pos-neg) + stg*(pos-perturbed) + (mod-1)*(pos-modality)
        """
        noise_pred = (
            pos
            + (cfg - 1) * (pos - neg)
            + stg_scale * (pos - perturbed)
            + (self.modality_scale - 1) * (pos - modality)
        )
        if self.rescale != 0:
            factor = pos.std() / noise_pred.std()
            factor = self.rescale * factor + (1 - self.rescale)
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
        guide_strength,
        guide_frame_idx,
        max_shift,
        base_shift,
        **kwargs,
    ):
        super().__init__(model, positive, negative, **kwargs)
        self._control_pixels = control_pixels
        self._vae = vae
        self._guide_strength = guide_strength
        self._guide_frame_idx = guide_frame_idx
        self._max_shift = max_shift
        self._base_shift = base_shift
        # Set after encoding in sample()
        self._guide_latent = None
        self._num_guide_frames = 0

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
        print(f"[RSICLoRAGuider] Applied model sampling shift={shift:.4f} (tokens={tokens})")

    # ------------------------------------------------------------------
    # Deferred encoding — called once at the start of sample()
    # ------------------------------------------------------------------

    def _encode_guide(self, latent_image):
        """Encode control pixels and inject guide keyframe into conditioning.

        Called at the start of sample() when the actual video latent
        dimensions are known. Uses LTXVAddGuide.encode() which resizes
        the control image to match the target latent spatial dims.
        """
        from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask
        import node_helpers

        # Get video latent dims (exclude audio if nested)
        if latent_image.is_nested:
            video_latent = latent_image.unbind()[0]
        else:
            video_latent = latent_image
        _, _, latent_t, latent_h, latent_w = video_latent.shape

        scale_factors = self._vae.downscale_index_formula

        # Encode control pixels at the video latent's spatial dims
        # LTXVAddGuide.encode() resizes pixels to match latent_w * width_sf x latent_h * height_sf
        _, guide_latent = LTXVAddGuide.encode(
            self._vae, latent_w, latent_h, self._control_pixels, scale_factors
        )
        self._guide_latent = guide_latent
        self._num_guide_frames = guide_latent.shape[2]
        print(f"[RSICLoRAGuider] Encoded guide latent: {list(guide_latent.shape)} "
              f"({self._num_guide_frames} frame(s))")

        # Compute the latent index for the guide frame
        frame_idx_actual, _ = LTXVAddGuide.get_latent_index(
            self._get_positive(), latent_t,
            len(self._control_pixels), self._guide_frame_idx, scale_factors
        )

        # Build a dummy latent + noise_mask for append_keyframe
        # (we only need the conditioning metadata — the actual latent
        # concatenation is done separately in _append_guide)
        dummy_latent = torch.zeros(
            (1, 128, latent_t, latent_h, latent_w),
            device=video_latent.device,
        )
        noise_mask = get_noise_mask({"samples": dummy_latent})

        # Get current conditioning from the guider
        positive = self._get_positive()
        negative = self._get_negative()

        # Append keyframe to conditioning (sets keyframe_idxs metadata)
        positive, negative, _, _ = LTXVAddGuide.append_keyframe(
            positive, negative, frame_idx_actual,
            dummy_latent, noise_mask,
            guide_latent, self._guide_strength, scale_factors,
        )
        print(f"[RSICLoRAGuider] Guide keyframe added at frame_idx={frame_idx_actual}, "
              f"strength={self._guide_strength}")

        # Update the guider's conditioning
        self.inner_set_conds({"positive": positive, "negative": negative})

        # Free references we no longer need
        self._control_pixels = None
        self._vae = None

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
    # Guide frame lifecycle
    # ------------------------------------------------------------------

    def _append_guide(self, latent_image, denoise_mask):
        """Append IC-LoRA guide latent frames to the video latent.

        For NestedTensor (AV mode), only the video portion is extended.
        The denoise mask is updated with ~ 0 for guide positions so they
        are preserved during denoising.
        """
        guide = self._guide_latent.to(latent_image.device if not latent_image.is_nested
                                      else latent_image.unbind()[0].device)

        if latent_image.is_nested:
            parts = latent_image.unbind()
            video = parts[0]
            audio = parts[1] if len(parts) > 1 else None

            # Pad guide channels if video has more (e.g. AV channel padding)
            if video.shape[1] > guide.shape[1]:
                pad_len = video.shape[1] - guide.shape[1]
                guide = torch.nn.functional.pad(
                    guide, (0, 0, 0, 0, 0, 0, 0, pad_len), value=0
                )

            video = torch.cat([video, guide], dim=2)

            # Update denoise mask
            if denoise_mask is not None and denoise_mask.is_nested:
                dm_parts = denoise_mask.unbind()
                video_dm = dm_parts[0]
                audio_dm = dm_parts[1] if len(dm_parts) > 1 else None
                # Guide mask ~ 0 -> preserve guide frames
                guide_mask = torch.full(
                    (video_dm.shape[0], 1, self._num_guide_frames,
                     video_dm.shape[3], video_dm.shape[4]),
                    1.0 - self._guide_strength,
                    dtype=video_dm.dtype, device=video_dm.device,
                )
                video_dm = torch.cat([video_dm, guide_mask], dim=2)
                if audio_dm is not None:
                    denoise_mask = comfy.nested_tensor.NestedTensor((video_dm, audio_dm))
                else:
                    denoise_mask = comfy.nested_tensor.NestedTensor((video_dm,))
            elif denoise_mask is not None:
                guide_mask = torch.full(
                    (denoise_mask.shape[0], 1, self._num_guide_frames, 1, 1),
                    1.0 - self._guide_strength,
                    dtype=denoise_mask.dtype, device=denoise_mask.device,
                )
                denoise_mask = torch.cat([denoise_mask, guide_mask], dim=2)

            if audio is not None:
                latent_image = comfy.nested_tensor.NestedTensor((video, audio))
            else:
                latent_image = comfy.nested_tensor.NestedTensor((video,))
        else:
            # Plain tensor (video only)
            if latent_image.shape[1] > guide.shape[1]:
                pad_len = latent_image.shape[1] - guide.shape[1]
                guide = torch.nn.functional.pad(
                    guide, (0, 0, 0, 0, 0, 0, 0, pad_len), value=0
                )

            latent_image = torch.cat([latent_image, guide], dim=2)

            if denoise_mask is not None:
                guide_mask = torch.full(
                    (denoise_mask.shape[0], 1, self._num_guide_frames,
                     denoise_mask.shape[3], denoise_mask.shape[4]),
                    1.0 - self._guide_strength,
                    dtype=denoise_mask.dtype, device=denoise_mask.device,
                )
                denoise_mask = torch.cat([denoise_mask, guide_mask], dim=2)

        return latent_image, denoise_mask

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

        # 2. Encode control image and inject guide conditioning
        #    (deferred to here because we now know the video latent dims)
        self._encode_guide(latent_image)

        # 3. Append IC-LoRA guide frames to the latent
        denoise_mask = kwargs.get("denoise_mask", None)
        latent_image, denoise_mask = self._append_guide(latent_image, denoise_mask)
        if denoise_mask is not None:
            kwargs["denoise_mask"] = denoise_mask

        # 4. Regenerate noise to match expanded latent dimensions
        noise = comfy.sample.prepare_noise(latent_image, kwargs.get("seed", 0))

        print(f"[RSICLoRAGuider] Guide frames appended: {self._num_guide_frames} "
              f"frame(s), latent temporal dim now "
              f"{latent_image.unbind()[0].shape[2] if latent_image.is_nested else latent_image.shape[2]}")

        # 5. Run parent sampling (MultimodalGuider -> CFGGuider)
        result = super().sample(noise, latent_image, sampler, sigmas, **kwargs)

        # 6. Strip guide frames from the output
        result = self._strip_guide(result)
        print(f"[RSICLoRAGuider] Guide frames stripped, output temporal dim: "
              f"{result.unbind()[0].shape[2] if result.is_nested else result.shape[2]}")

        return result
