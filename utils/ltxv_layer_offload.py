"""Layer offloading for LTX-2 LoRA training on low-VRAM GPUs.

Streams transformer blocks between CPU and GPU one at a time during
forward and backward passes.  Only 1 block lives on GPU at any moment,
dropping model VRAM from ~11 GB (fp8) to ~0.5 GB.

Works with ComfyUI's LTXAVModel block interface (BasicAVTransformerBlock).

Usage (inside InProcessTrainer):
    setup_layer_offloading(transformer, device)
    # then call model forward normally — the patched
    # _process_transformer_blocks handles streaming.
"""

import logging
import types

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private CUDA stream for offload copies
# ---------------------------------------------------------------------------
# All H2D / D2H copies for layer offload run on a dedicated stream so they can
# overlap compute on the default stream (this is what gave us 0.17-era speed).
# Cross-stream data dependencies are enforced via cuda.Event waits.
#
# Why a private stream specifically:
#  - Enables H2D of block N+1 / D2H of block N to pipeline behind compute(N).
#  - Isolates our copies from whatever ComfyUI 0.20.1's dynamic VRAM scheduler
#    is doing on the default stream, which previously caused C-level aborts
#    when pinned-memory async copies on the default stream raced the scheduler.
#  - We synchronize this stream at the boundary of the forward block loop so
#    when control returns to ComfyUI no copies are in flight.

_offload_streams: dict = {}


def _get_offload_stream(device: torch.device) -> torch.cuda.Stream:
    key = (device.type, device.index if device.index is not None else 0)
    s = _offload_streams.get(key)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _offload_streams[key] = s
    return s


# ---------------------------------------------------------------------------
# Custom autograd Function: gradient checkpoint + layer offload combined
# ---------------------------------------------------------------------------

class _OffloadCheckpointFn(torch.autograd.Function):
    """Runs a single transformer block with combined offloading + checkpointing.

    Forward:  load block → clone inputs → run (no_grad) → evict → return detached
    Backward: load block → recompute forward (with grad) → backward → evict
    """

    @staticmethod
    def forward(ctx, vx, ax, block, block_kwargs, device, has_audio, block_idx):
        # Clone inputs BEFORE the block mutates them in-place (addcmul_ etc.)
        # IMPORTANT: always clone both — never return an input tensor from
        # a custom Function (PyTorch forbids it and silently breaks the graph).
        # Save on CPU — all 48 blocks' saved tensors coexist during forward,
        # and keeping them on GPU would accumulate ~5 GB of hidden states.
        compute_stream = torch.cuda.current_stream(device)
        offload_stream = _get_offload_stream(device)

        # Offload stream must wait for compute stream before reading vx/ax —
        # those tensors were last written by the previous block's compute.
        compute_done = torch.cuda.Event()
        compute_done.record(compute_stream)
        offload_stream.wait_event(compute_done)

        with torch.cuda.stream(offload_stream):
            # Clone on GPU first (so the in-place mutations inside the block
            # below can't race with the in-flight D2H), then async D2H. The
            # CPU saved copies are consumed in backward, far in the future.
            saved_vx_cpu = vx.detach().clone().to("cpu", non_blocking=True)
            saved_ax_cpu = ax.detach().clone().to("cpu", non_blocking=True)
            # Async H2D of block params on the same stream.
            block.to(device, non_blocking=True)
            block_loaded = torch.cuda.Event()
            block_loaded.record(offload_stream)

        ctx.save_for_backward(saved_vx_cpu, saved_ax_cpu)
        ctx.block = block
        ctx.block_kwargs = block_kwargs
        ctx.device = device
        ctx.has_audio = has_audio
        ctx.block_idx = block_idx

        # Compute stream must wait until block params are fully on GPU.
        compute_stream.wait_event(block_loaded)

        with torch.no_grad():
            out_vx, out_ax = block((vx, ax), **block_kwargs)

        # Offload stream must wait for compute to finish using block params
        # before evicting them. Then async D2H, fire-and-forget — the next
        # iteration's H2D queues behind it on the offload stream and is
        # gated on its own event.
        fwd_done = torch.cuda.Event()
        fwd_done.record(compute_stream)
        with torch.cuda.stream(offload_stream):
            offload_stream.wait_event(fwd_done)
            block.to("cpu", non_blocking=True)

        # Always return NEW tensors (detached from block's no_grad graph).
        # The Function mechanism re-attaches grad_fn to these.
        return out_vx.detach().clone(), out_ax.detach().clone()

    @staticmethod
    def backward(ctx, grad_vx, grad_ax):
        saved_vx, saved_ax = ctx.saved_tensors
        block = ctx.block
        device = ctx.device
        has_audio = ctx.has_audio

        # Periodically clear VRAM cache during backward to prevent
        # fragmentation-induced OOM (which runs in C++ autograd and
        # can't be caught by Python try/except).  Every 12 blocks
        # balances overhead (~4 clears per backward) vs safety.
        block_idx = getattr(ctx, 'block_idx', -1)
        if block_idx >= 0 and block_idx % 12 == 0:
            torch.cuda.empty_cache()

        compute_stream = torch.cuda.current_stream(device)
        offload_stream = _get_offload_stream(device)

        # Async H2D of block params + saved hidden states on offload stream.
        with torch.cuda.stream(offload_stream):
            block.to(device, non_blocking=True)
            saved_vx = saved_vx.to(device, non_blocking=True)
            saved_ax = saved_ax.to(device, non_blocking=True)
            h2d_done = torch.cuda.Event()
            h2d_done.record(offload_stream)

        # Compute stream waits until everything is on GPU before recompute.
        compute_stream.wait_event(h2d_done)

        # Create leaf tensors for gradient tracking.
        vx_leaf = saved_vx.detach().requires_grad_(True)
        if has_audio:
            ax_leaf = saved_ax.detach().requires_grad_(True)
        else:
            ax_leaf = None

        # ComfyUI's transformer blocks use addcmul_ (in-place) extensively.
        # saved_tensors_hooks clones on save to avoid version-counter errors
        # from in-place mutations invalidating saved tensors.
        #
        # CRITICAL: clone() and block() must run INSIDE enable_grad() —
        # backward() runs under no_grad by default, so clone() outside
        # enable_grad() produces a tensor with no grad_fn, severing the
        # gradient chain from block output back to vx_leaf.
        # Override attention to PyTorch SDPA during backward recompute.
        # SageAttention's CUDA kernel isn't autograd-compatible, so attn2
        # (cross-attention) LoRA layers get zero gradients when sage is active.
        # Using PyTorch attention here ensures all 12/12 LoRA layers train.
        from comfy.ldm.modules.attention import attention_pytorch
        # The override is called as override(func, *args, **kwargs) by the
        # @wrap_attn decorator, so we must accept and discard the first arg.
        def _pytorch_override(_func, *args, **kwargs):
            return attention_pytorch(*args, **kwargs)
        bwd_kwargs = dict(ctx.block_kwargs)
        bwd_tf_opts = dict(bwd_kwargs.get("transformer_options", {}))
        bwd_tf_opts["optimized_attention_override"] = _pytorch_override
        bwd_kwargs["transformer_options"] = bwd_tf_opts

        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: t.clone(),   # pack: fresh storage+version (no detach!)
            lambda t: t,            # unpack: return as-is
        ):
            # cache_enabled=False: the bf16 weight cache only helps when the
            # same param is cast multiple times within one autocast scope.
            # We cast each param once per backward, so the cache provides
            # no benefit — and clearing it at __exit__ was a known crash
            # site on ComfyUI 0.20.1.
            with torch.enable_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, cache_enabled=False
            ):
                # Clone INSIDE enable_grad so CloneBackward links to vx_leaf
                vx = vx_leaf.clone()
                ax = ax_leaf.clone() if has_audio else saved_ax
                out_vx, out_ax = block((vx, ax), **bwd_kwargs)
            # Drain compute stream before tensors created inside autocast
            # leave scope, so their destructors don't race in-flight kernels.
            compute_stream.synchronize()

        outputs = [out_vx]
        grads = [grad_vx]
        if has_audio:
            outputs.append(out_ax)
            grads.append(grad_ax)

        # Use torch.autograd.grad with EXPLICIT inputs so we get gradients
        # back directly — no reliance on .grad attribute accumulation.
        inputs_list = [vx_leaf]
        if has_audio:
            inputs_list.append(ax_leaf)
        param_list = [p for p in block.parameters() if p.requires_grad]
        all_inputs = inputs_list + param_list

        input_grads = torch.autograd.grad(
            outputs, all_inputs, grads,
            allow_unused=True, retain_graph=False,
        )

        vx_grad_out = input_grads[0]
        ax_grad_out = input_grads[1] if has_audio else None
        param_grads = input_grads[len(inputs_list):]

        # Async D2H of grads + block on offload stream. Offload waits for
        # compute to finish autograd.grad before reading the produced grads.
        # Block eviction queues after the grads on the same stream — fire-and-
        # forget, the next backward's H2D will queue behind it on this stream.
        grads_computed = torch.cuda.Event()
        grads_computed.record(compute_stream)

        grad_map: dict = {}
        with torch.cuda.stream(offload_stream):
            offload_stream.wait_event(grads_computed)
            for p, g in zip(param_list, param_grads):
                if g is not None:
                    grad_map[p] = g.to("cpu", non_blocking=True)
            grads_landed = torch.cuda.Event()
            grads_landed.record(offload_stream)
            block.to("cpu", non_blocking=True)

        # Host-side wait until all grad D2Hs have actually landed in CPU
        # memory — required before we can read them with p.grad.add_() below.
        # This does NOT wait for the block eviction queued behind it.
        grads_landed.synchronize()

        # Accumulate into .grad on CPU (AdamW reads from .grad)
        for p, g in grad_map.items():
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)

        # Returns for: vx, ax, block, block_kwargs, device, has_audio, block_idx
        return (
            vx_grad_out,
            ax_grad_out,
            None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Patched block-processing loop (ComfyUI LTXAVModel interface)
# ---------------------------------------------------------------------------

def _offloaded_process_blocks(
    model, x, context, attention_mask, timestep, pe,
    transformer_options={}, self_attention_mask=None, **kwargs
):
    """Drop-in replacement for LTXAVModel._process_transformer_blocks.

    Same signature as ComfyUI's version — unpacks the packed args,
    then streams each block through GPU one at a time.
    """
    device = model._offload_device

    # Unpack exactly as ComfyUI's _process_transformer_blocks does
    vx = x[0]
    ax = x[1]
    v_context = context[0]
    a_context = context[1]
    v_timestep = timestep[0]
    a_timestep = timestep[1]
    (v_pe, av_cross_video_freq_cis) = pe[0]
    (a_pe, av_cross_audio_freq_cis) = pe[1]

    (
        av_ca_audio_scale_shift_timestep,
        av_ca_video_scale_shift_timestep,
        av_ca_a2v_gate_noise_timestep,
        av_ca_v2a_gate_noise_timestep,
    ) = timestep[2]

    v_prompt_timestep = timestep[3]
    a_prompt_timestep = timestep[4]

    has_audio = ax.numel() > 0

    # Build kwargs dict matching BasicAVTransformerBlock.forward()
    block_kwargs = dict(
        v_context=v_context,
        a_context=a_context,
        attention_mask=attention_mask,
        v_timestep=v_timestep,
        a_timestep=a_timestep,
        v_pe=v_pe,
        a_pe=a_pe,
        v_cross_pe=av_cross_video_freq_cis,
        a_cross_pe=av_cross_audio_freq_cis,
        v_cross_scale_shift_timestep=av_ca_video_scale_shift_timestep,
        a_cross_scale_shift_timestep=av_ca_audio_scale_shift_timestep,
        v_cross_gate_timestep=av_ca_a2v_gate_noise_timestep,
        a_cross_gate_timestep=av_ca_v2a_gate_noise_timestep,
        transformer_options=transformer_options,
        self_attention_mask=self_attention_mask,
        v_prompt_timestep=v_prompt_timestep,
        a_prompt_timestep=a_prompt_timestep,
    )

    for i, block in enumerate(model.transformer_blocks):
        # Pass the real ax (even if empty) — the block checks ax.numel() > 0
        # to decide whether to run audio processing.
        cur_ax = ax

        new_vx, new_ax = _OffloadCheckpointFn.apply(
            vx, cur_ax, block, block_kwargs, device, has_audio, i,
        )

        vx = new_vx
        if has_audio:
            ax = new_ax

    # Boundary sync: drain the offload stream before returning to the rest
    # of ComfyUI's forward path. Inside the loop, copies pipeline behind
    # compute on a private stream — but we never want in-flight copies to
    # leak past this function, where ComfyUI 0.20.1's dynamic VRAM
    # scheduler may inspect / move model params. (This is a single host
    # wait per forward, not per block — perf-neutral.)
    _get_offload_stream(device).synchronize()

    return [vx, ax]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_layer_offloading(peft_model, device: torch.device) -> None:
    """Enable layer offloading on an LTX-2 transformer (PEFT-wrapped or plain).

    - Moves non-block parameters (patchify, adaln, proj_out, etc.) to GPU.
    - Keeps transformer blocks on CPU.
    - Monkey-patches ``_process_transformer_blocks`` so that each forward/
      backward pass streams one block at a time.

    Args:
        peft_model: The transformer (possibly wrapped by PEFT ``get_peft_model``).
        device: Target GPU device.
    """
    # Resolve through PEFT wrapper to the raw model
    base = peft_model
    if hasattr(base, "get_base_model"):
        base = base.get_base_model()

    if not hasattr(base, "transformer_blocks"):
        raise ValueError(
            f"Model {type(base).__name__} has no 'transformer_blocks' attribute — "
            "layer offloading is only supported for LTX-2 models."
        )

    blocks = base.transformer_blocks
    num_blocks = len(blocks)

    # Move non-block child modules to GPU using .to() so that quanto's
    # special tensor types (QBytesWeight, etc.) are moved correctly.
    # Individual p.data = p.data.to() does NOT work for quantized weights.
    for name, child in base.named_children():
        if name != "transformer_blocks":
            child.to(device)

    # Ensure all blocks are on CPU
    for block in blocks:
        block.to("cpu")

    # Store device reference for the patched function
    base._offload_device = device

    logger.info(
        f"Layer offloading enabled: {num_blocks} blocks on CPU, "
        f"non-block params on GPU. "
        f"VRAM after setup: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
    )

    # Monkey-patch the block loop
    base._process_transformer_blocks = types.MethodType(
        _offloaded_process_blocks, base,
    )


def teardown_layer_offloading(peft_model) -> None:
    """Undo layer offloading — move everything back to CPU."""
    try:
        base = peft_model
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()
        peft_model.to("cpu")
    except Exception as e:
        logger.warning(f"teardown_layer_offloading: {e}")
