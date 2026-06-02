"""Phase 1 AMF probe — measure motion signal strength per LTX transformer block.

Hooks each `BasicAVTransformerBlock`'s video self-attention (attn1) on the
loaded LTX 2.3 22B AV model, captures Q/K during a forward pass on noised
source latents, computes the Attention Motion Flow (AMF) for each
(block, timestep) pair, and correlates it with RAFT optical flow as
ground truth.

Gate (from `Reference/ltx_motion_lock_plan.md`):
- ANY (block, timestep) correlation > 0.6 → proceed to Phase 2 with that block
- best in 0.3-0.5 → try multi-block weighted combinations
- all < 0.3 → project dead, STOP

This is a standalone tools script — NOT a ComfyUI node. Designed to run
from within the ComfyUI environment (so `import comfy.*` works). On the
96 GB RunPod, run from ComfyUI's root directory:

    cd /path/to/ComfyUI
    python custom_nodes/rs-nodes/tools/ltx_amf_probe.py \\
        --model-path models/checkpoints/ltx-2.3-22b.safetensors \\
        --clips path/to/clip1.mp4 path/to/clip2.mp4 \\
        --timesteps 0.4 0.5 0.6 0.7 0.8 \\
        --width 768 --height 512 --num-frames 73 \\
        --output Reference/ltx_amf_probe_results

Outputs `<output>.csv` (full table) and `<output>.md` (ranked summary).
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange


# --- Locate ComfyUI root and set up imports ---------------------------------

def _find_comfyui_root() -> Path:
    """Walk up from this file to find a directory containing 'comfy/'."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "comfy"
        if candidate.is_dir() and (candidate / "sd.py").exists():
            return parent
    # Fallback: try CWD
    cwd = Path.cwd()
    if (cwd / "comfy" / "sd.py").exists():
        return cwd
    raise RuntimeError(
        "Could not find ComfyUI root (a directory containing comfy/sd.py). "
        "Run this script from ComfyUI's root or pass --comfyui-root."
    )


def _setup_imports(comfyui_root: Optional[str] = None):
    """Insert ComfyUI's root onto sys.path so `import comfy` works."""
    root = Path(comfyui_root) if comfyui_root else _find_comfyui_root()
    sp = str(root)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return root


# --- AMF core (ported from DiTFlow's motion_flow_utils.py) ------------------

def compute_motion_flow(
    q: torch.Tensor,
    k: torch.Tensor,
    h: int,
    w: int,
    nframes: int,
    temp: float = 2.0,
    argmax: bool = False,
) -> torch.Tensor:
    """Compute Attention Motion Flow (AMF) from Q, K.

    Args:
        q, k: shape [heads, total_seq, head_dim] for one block, one batch.
        h, w: patch grid resolution per latent frame (after patchify).
        nframes: number of latent frames in the sequence.
        temp: softmax temperature.
        argmax: if True, use hard argmax correspondence; if False, soft
            expected displacement.

    Returns:
        flows: shape [nframes*nframes, h*w, 2] — per (i, j) frame pair,
            per patch, (dx, dy) displacement in patch units.
    """
    # Attention matrix per head (using normalized scale)
    A_all = torch.matmul(q, k.transpose(-1, -2))
    A_all = A_all / torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype, device=q.device))
    # Average across heads (DiTFlow uses .mean(0, keepdim=True) over the head dim)
    A_all = A_all.mean(0, keepdim=True)  # [1, total_seq, total_seq]

    total_predicted_flows = 0.0
    for head in range(A_all.shape[0]):
        A_head = A_all[head]  # [total_seq, total_seq]

        # Reshape attention to per-frame-pair blocks: [f1*hw1, f2, hw2]
        A_head = rearrange(A_head, "s (f hw) -> s f hw", f=nframes)
        A_head = F.softmax(A_head * temp, dim=-1)
        A_head = rearrange(
            A_head,
            "(f1 s1) f2 s2 -> f1 f2 s1 s2",
            f1=nframes, f2=nframes, s1=h * w, s2=h * w,
        )

        predicted_flows = []
        for fi in range(nframes):
            for fj in range(nframes):
                A = A_head[fi, fj]  # [hw, hw]
                if argmax:
                    matches = A.argmax(dim=-1)
                    x1 = torch.arange(A.shape[0], device=A.device) % w
                    y1 = torch.arange(A.shape[0], device=A.device) // w
                    x2 = matches % w
                    y2 = matches // w
                    disp = torch.stack([x2 - x1, y2 - y1], dim=-1).float()
                else:
                    y_g, x_g = torch.meshgrid(
                        torch.arange(h, device=A.device),
                        torch.arange(w, device=A.device),
                        indexing="ij",
                    )
                    rel_x = x_g.flatten().unsqueeze(0) - x_g.flatten().unsqueeze(1)
                    rel_y = y_g.flatten().unsqueeze(0) - y_g.flatten().unsqueeze(1)
                    dx = (rel_x.float() * A).sum(dim=1)
                    dy = (rel_y.float() * A).sum(dim=1)
                    disp = torch.stack([dx, dy], dim=-1)
                predicted_flows.append(disp)

        predicted_flows = torch.stack(predicted_flows, dim=0)  # [f*f, hw, 2]
        total_predicted_flows = total_predicted_flows + predicted_flows

    return total_predicted_flows / A_all.shape[0]


# --- Q/K capture hook -------------------------------------------------------

class QKCapture:
    """Captures Q, K from a CrossAttention block's forward pass.

    Replaces the instance's forward with a re-implementation that captures
    q, k after q_norm/k_norm and RoPE, just before the optimized_attention
    call. Also implements per-head output gating (`to_gate_logits`) so the
    block's behavior is identical to the original — important because LTX
    2.3 22B uses gating on every block. Gating only affects the attention
    output (post Q/K), so AMF extraction is unaffected.
    """

    def __init__(self, attn_module):
        self.attn = attn_module
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self._original_forward = attn_module.forward
        # All blocks supported now; flag retained for compatibility.
        self.supports_capture = True

    def install(self):
        # Lazy imports inside the closure so install happens before sys.path is final
        import comfy.ldm.modules.attention as comfy_attn
        from comfy.ldm.lightricks.model import apply_rotary_emb

        attn = self.attn
        cap = self

        def probed_forward(x, context=None, mask=None, pe=None, k_pe=None,
                           transformer_options={}):
            q = attn.to_q(x)
            actual_context = x if context is None else context
            k = attn.to_k(actual_context)
            v = attn.to_v(actual_context)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            if pe is not None:
                q = apply_rotary_emb(q, pe)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe)
            cap.q = q.detach().clone()
            cap.k = k.detach().clone()
            if mask is None:
                out = comfy_attn.optimized_attention(
                    q, k, v, attn.heads,
                    attn_precision=attn.attn_precision,
                    transformer_options=transformer_options,
                )
            else:
                out = comfy_attn.optimized_attention_masked(
                    q, k, v, attn.heads, mask,
                    attn_precision=attn.attn_precision,
                    transformer_options=transformer_options,
                )
            # Per-head output gating (LTX 2.3 22B uses this on every block)
            if attn.to_gate_logits is not None:
                gate_logits = attn.to_gate_logits(x)  # (B, T, H)
                b, t, _ = out.shape
                out = out.view(b, t, attn.heads, attn.dim_head)
                gates = 2.0 * torch.sigmoid(gate_logits)
                out = out * gates.unsqueeze(-1)
                out = out.view(b, t, attn.heads * attn.dim_head)
            return attn.to_out(out)

        attn.forward = probed_forward

    def uninstall(self):
        self.attn.forward = self._original_forward
        self.q = None
        self.k = None


# --- RAFT optical flow (ground truth) ---------------------------------------

def compute_raft_flow_per_latent_pair(
    pixel_frames: torch.Tensor,
    nframes: int,
    time_sf: int,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute RAFT optical flow between latent-frame centers, downsampled
    to the patch grid resolution.

    Args:
        pixel_frames: [T, C, H, W] in [-1, 1] (CHW float)
        nframes: number of latent frames
        time_sf: temporal scale factor (8 for LTX)
        target_h, target_w: patch grid resolution
        device: CUDA device

    Returns:
        gt_flows: [nframes*nframes, target_h*target_w, 2] in patch units
    """
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    weights = Raft_Large_Weights.DEFAULT
    raft = raft_large(weights=weights, progress=False).to(device).eval()
    transforms = weights.transforms()

    T = pixel_frames.shape[0]
    # Latent k>0 maps to pixel chunk centered at ((k-1)*time_sf + time_sf//2).
    # Latent k=0 maps to pixel 0 directly.
    def latent_center_pixel(k: int) -> int:
        if k == 0:
            return 0
        return min((k - 1) * time_sf + time_sf // 2, T - 1)

    flows = []
    raft_h, raft_w = pixel_frames.shape[-2], pixel_frames.shape[-1]
    # RAFT expects multiples of 8
    raft_h_aligned = (raft_h // 8) * 8
    raft_w_aligned = (raft_w // 8) * 8
    px_align = pixel_frames[:, :, :raft_h_aligned, :raft_w_aligned]

    for fi in range(nframes):
        for fj in range(nframes):
            if fi == fj:
                flows.append(torch.zeros(target_h * target_w, 2, device=device))
                continue
            i_px = latent_center_pixel(fi)
            j_px = latent_center_pixel(fj)
            img1 = px_align[i_px:i_px + 1].to(device)
            img2 = px_align[j_px:j_px + 1].to(device)
            img1_t, img2_t = transforms(img1, img2)
            with torch.no_grad():
                flow_list = raft(img1_t, img2_t)
                flow = flow_list[-1]  # [1, 2, H, W] in pixels
            # Downsample to patch grid
            flow_down = F.interpolate(flow, size=(target_h, target_w),
                                      mode="bilinear", align_corners=False)
            # Convert flow from pixel units to patch units
            patch_size_x = raft_w_aligned / target_w
            patch_size_y = raft_h_aligned / target_h
            flow_down[:, 0] = flow_down[:, 0] / patch_size_x
            flow_down[:, 1] = flow_down[:, 1] / patch_size_y
            flow_vec = flow_down.squeeze(0).permute(1, 2, 0).reshape(-1, 2)
            flows.append(flow_vec)

    del raft
    torch.cuda.empty_cache()
    return torch.stack(flows, dim=0)


# --- Correlation metric -----------------------------------------------------

def correlate_amf_with_gt(amf: torch.Tensor, gt: torch.Tensor) -> float:
    """Pearson correlation between AMF and GT flow, computed across all
    (frame_pair, patch, xy) entries, weighted toward non-zero GT patches.

    Both tensors are shape [nframes*nframes, h*w, 2].
    """
    # Flatten everything to a single vector pair
    amf_flat = amf.reshape(-1).float()
    gt_flat = gt.reshape(-1).float()
    # Mask: only patches where GT has appreciable motion
    gt_mag = gt.norm(dim=-1).reshape(-1)
    # Repeat mask for x and y components
    mask_pair = (gt_mag > 0.1).repeat_interleave(2)
    if mask_pair.sum() < 2:
        return 0.0
    a = amf_flat[mask_pair]
    g = gt_flat[mask_pair]
    a = a - a.mean()
    g = g - g.mean()
    num = (a * g).sum()
    den = torch.sqrt((a * a).sum() * (g * g).sum() + 1e-8)
    return float((num / den).item())


# --- Model loading + DiT forward orchestration ------------------------------

@dataclass
class ProbeContext:
    """Bundle of model + VAE + tokens needed for one probe pass."""
    model_patcher: object   # ComfyUI ModelPatcher
    diffusion_model: torch.nn.Module  # the DiT itself (model.model.diffusion_model)
    vae: object             # ComfyUI VAE
    text_cond: torch.Tensor
    device: torch.device
    time_sf: int = 8


def load_ltx(model_path: str, device: torch.device,
             gemma_path: Optional[str] = None,
             t5_path: Optional[str] = None) -> ProbeContext:
    """Load LTX 2.3 checkpoint via ComfyUI's loader.

    Returns a ProbeContext with everything needed for the probe.
    Text encoders are loaded separately if provided (LTX 2.3 ships them
    as separate files, not bundled in the DiT checkpoint).
    """
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm

    log.info(f"Loading LTX checkpoint: {model_path}")
    out = comfy.sd.load_checkpoint_guess_config(
        model_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=None,
    )
    model_patcher = out[0]
    clip = out[1]
    vae = out[2]

    # If checkpoint didn't include the text encoder (typical for LTX 2.3)
    # and the user provided gemma/t5 paths, load them via comfy.sd.load_clip.
    if clip is None and gemma_path and t5_path:
        log.info(f"Loading text encoders separately: gemma={gemma_path}, t5={t5_path}")
        clip = comfy.sd.load_clip(
            ckpt_paths=[gemma_path, t5_path],
            embedding_directory=None,
            clip_type=comfy.sd.CLIPType.LTXV,
        )
    log.info(f"Loaded: model={type(model_patcher).__name__}, vae={type(vae).__name__}")

    # Move model onto device
    mm.load_models_gpu([model_patcher])
    diffusion_model = model_patcher.model.diffusion_model

    # Build text conditioning. LTX 2.3 checkpoints don't bundle the text
    # encoder (Gemma3+T5 are separate files), so clip is usually None when
    # loading just the DiT checkpoint. For the probe we don't need real text:
    # we're measuring self-attention motion signal, not generating output.
    # Cross-attention to a zero context contributes ~zero to the residual
    # stream, leaving self-attention to see visual content cleanly.
    if clip is not None:
        log.info("Using real text encoder from checkpoint")
        tokens = clip.tokenize("")
        cond, _pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    else:
        # LTX 2.3 AV expects context of dim (cross_attention_dim + audio_cross_attention_dim).
        # The first half is video/text context (cross_attention_dim=4096 for 22B);
        # the second half is audio context (audio_cross_attention_dim=2048).
        # preprocess_text_embeds passes through unchanged if context.shape[-1]
        # matches this sum (line 567 in av_model.py).
        if hasattr(diffusion_model, "audio_cross_attention_dim"):
            ctx_dim = diffusion_model.cross_attention_dim + diffusion_model.audio_cross_attention_dim
            log.info(f"AV model detected; ctx_dim = {diffusion_model.cross_attention_dim} "
                     f"(video) + {diffusion_model.audio_cross_attention_dim} (audio) = {ctx_dim}")
        else:
            first_block = diffusion_model.transformer_blocks[0]
            ctx_dim = first_block.attn2.to_k.in_features
        # Get model dtype from a weight
        any_param = next(diffusion_model.parameters())
        dtype = any_param.dtype
        log.warning("=" * 70)
        log.warning("NO TEXT ENCODER LOADED — using zero context tensor.")
        log.warning("This is OUT-OF-DISTRIBUTION for the model and will likely")
        log.warning("produce near-noise AMF correlations. To fix: pass real")
        log.warning("text encoders via --gemma-path and --t5-path.")
        log.warning(f"ctx tensor: shape=[1, 256, {ctx_dim}] dtype={dtype}")
        log.warning("=" * 70)
        cond = torch.zeros(1, 256, ctx_dim, device=device, dtype=dtype)

    return ProbeContext(
        model_patcher=model_patcher,
        diffusion_model=diffusion_model,
        vae=vae,
        text_cond=cond,
        device=device,
    )


def load_video_to_pixel_tensor(path: str, num_frames: int, w: int, h: int,
                               device: torch.device) -> torch.Tensor:
    """Load mp4 to [T, C, H, W] in [-1, 1] float32."""
    from torchvision.io import read_video
    video, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
    video = video[:num_frames].float() / 255.0  # [T, C, H, W] in [0,1]
    if video.shape[-1] != w or video.shape[-2] != h:
        video = F.interpolate(video, size=(h, w), mode="bilinear", align_corners=False)
    video = (video - 0.5) * 2.0  # → [-1, 1]
    return video.to(device)


def vae_encode_video(vae, pixel_frames: torch.Tensor) -> torch.Tensor:
    """Encode pixel frames via LTX VAE → latents.

    pixel_frames: [T, C, H, W] in [-1, 1]
    returns: [1, C, T_latent, H_latent, W_latent]
    """
    # ComfyUI's VAE.encode expects [B, H, W, C] image batches and handles
    # video natively for video-VAEs. The LTX video VAE returns latents.
    # We call encode_video if available, else fall back.
    if hasattr(vae, "encode_video"):
        latent = vae.encode_video(pixel_frames.unsqueeze(0).permute(0, 2, 1, 3, 4))
    else:
        # Generic fallback: vae.encode handles BHWC images
        bhwc = pixel_frames.permute(0, 2, 3, 1)  # [T, H, W, C]
        latent = vae.encode(bhwc)
    # Normalize shape to [1, C, T, H, W]
    if latent.dim() == 5 and latent.shape[1] != latent.shape[2]:
        # Already [B, C, T, H, W] — good
        pass
    return latent


def add_noise_rectified_flow(latent: torch.Tensor, sigma: float,
                             seed: int = 0) -> torch.Tensor:
    """Rectified-flow noising: x_t = (1 - sigma) * x_0 + sigma * noise."""
    g = torch.Generator(device=latent.device).manual_seed(seed)
    noise = torch.randn(latent.shape, generator=g, device=latent.device, dtype=latent.dtype)
    return (1.0 - sigma) * latent + sigma * noise


def run_dit_forward_with_capture(
    ctx: ProbeContext,
    noised_latent: torch.Tensor,
    sigma: float,
    captures: list[QKCapture],
):
    """Run one DiT forward pass on the noised latent.

    Captures install themselves before this call; this function just
    needs to invoke the DiT in a way LTX expects.

    The exact forward signature is LTX-AV-model-specific. If this errors,
    inspect comfy/ldm/lightricks/av_model.py's main forward() and adjust.
    """
    # Build inputs the way LTX AV expects.
    # The AV model's forward separates audio from video via:
    #   vx = x[0]; ax = x[1] if len(x) > 1 else zeros(...)
    # So x MUST be a list/tuple [video_latent] (or [video, audio]).
    timestep = torch.tensor([sigma], device=ctx.device, dtype=noised_latent.dtype)
    context = ctx.text_cond.to(ctx.device, dtype=noised_latent.dtype)

    try:
        with torch.no_grad():
            ctx.diffusion_model(
                x=[noised_latent],
                timestep=timestep,
                context=context,
            )
    except TypeError as e:
        log.error(f"DiT forward signature mismatch: {e}")
        log.error("Inspect comfy/ldm/lightricks/av_model.py's forward signature "
                  "and update run_dit_forward_with_capture() in this script.")
        raise


# --- Main probe loop --------------------------------------------------------

log = logging.getLogger("ltx_amf_probe")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Path to LTX 2.3 checkpoint")
    p.add_argument("--clips", required=True, nargs="+",
                   help="Source clip paths. Accepts individual files OR a directory "
                        "containing video files (mp4/mov/mkv/webm).")
    p.add_argument("--timesteps", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.2, 0.3],
                   help="Sigma values to probe at (rectified flow normalized 0-1). "
                        "DiTFlow extracts AMF on CLEAN latents (sigma=0). Higher noise "
                        "degrades motion signal. Defaults span clean to lightly-noised.")
    p.add_argument("--gemma-path", default=None,
                   help="Path to Gemma3 text encoder (.safetensors/.gguf). If provided "
                        "together with --t5-path, real text conditioning is used. Without "
                        "this, zero context is used, which is OUT-OF-DISTRIBUTION for the "
                        "model and produces near-noise AMF correlations.")
    p.add_argument("--t5-path", default=None,
                   help="Path to T5-XXL text encoder (.safetensors).")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=73,
                   help="Pixel frames per clip (must be 8n+1)")
    p.add_argument("--blocks", type=int, nargs="+", default=None,
                   help="Block indices to probe (default: all 48)")
    p.add_argument("--output", default="ltx_amf_probe_results",
                   help="Output basename (writes <name>.csv and <name>.md)")
    p.add_argument("--comfyui-root", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = _setup_imports(args.comfyui_root)
    log.info(f"ComfyUI root: {root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Expand --clips: a directory becomes all video files inside it
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    expanded_clips = []
    for clip_arg in args.clips:
        clip_path = Path(clip_arg)
        if clip_path.is_dir():
            found = sorted([p for p in clip_path.iterdir()
                            if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
            if not found:
                log.warning(f"No video files in directory: {clip_path}")
            expanded_clips.extend(str(p) for p in found)
        elif clip_path.is_file():
            expanded_clips.append(str(clip_path))
        else:
            log.warning(f"Skipping (not a file or directory): {clip_arg}")
    if not expanded_clips:
        log.error("No clips found after expansion. Aborting.")
        return
    args.clips = expanded_clips
    log.info(f"Probing {len(args.clips)} clip(s): {[Path(c).name for c in args.clips]}")

    ctx = load_ltx(args.model_path, device,
                   gemma_path=args.gemma_path, t5_path=args.t5_path)

    # Determine which blocks to probe
    all_blocks = list(ctx.diffusion_model.transformer_blocks)
    block_indices = args.blocks if args.blocks else list(range(len(all_blocks)))
    log.info(f"Probing {len(block_indices)} blocks of {len(all_blocks)} total")

    # Patch grid resolution.
    # The VAE has temporal scale_factor=8 and spatial scale_factor=32, giving
    # us latent dims [T_lat, H_lat, W_lat]. The DiT's patchifier then further
    # sub-patches the latent grid with patch_size=(1, p, p). The actual DiT
    # sequence length is T_lat * (H_lat//p) * (W_lat//p). Introspect p from
    # the model so we don't have to guess.
    patchifier_p = ctx.diffusion_model.patchifier.patch_size[1]
    vae_spatial = 32  # LTX video VAE spatial scale factor
    vae_temporal = ctx.time_sf  # 8
    latent_h = args.height // vae_spatial
    latent_w = args.width // vae_spatial
    h_patches = latent_h // patchifier_p
    w_patches = latent_w // patchifier_p
    n_latent_frames = (args.num_frames - 1) // vae_temporal + 1
    log.info(f"VAE latent: {n_latent_frames}x{latent_h}x{latent_w}, "
             f"patchifier p={patchifier_p} -> patch grid "
             f"{n_latent_frames} frames x {h_patches}x{w_patches}")

    # Results accumulator: list of dicts
    rows = []

    for clip_idx, clip_path in enumerate(args.clips):
        clip_name = Path(clip_path).stem
        log.info(f"=== Clip {clip_idx+1}/{len(args.clips)}: {clip_name} ===")

        # Load pixels
        pixel_frames = load_video_to_pixel_tensor(
            clip_path, args.num_frames, args.width, args.height, device
        )
        log.info(f"Loaded pixel frames: {tuple(pixel_frames.shape)}")

        # VAE encode
        with torch.no_grad():
            source_latent = vae_encode_video(ctx.vae, pixel_frames)
        # ComfyUI's VAE may return latents on CPU and in float32 even when the
        # DiT runs in bfloat16. Force both device and dtype to match the DiT.
        model_dtype = next(ctx.diffusion_model.parameters()).dtype
        source_latent = source_latent.to(device=device, dtype=model_dtype)
        log.info(f"Source latent: {tuple(source_latent.shape)} on {source_latent.device} dtype={source_latent.dtype}")

        # RAFT ground truth
        log.info("Computing RAFT optical flow (ground truth)...")
        gt_flow = compute_raft_flow_per_latent_pair(
            pixel_frames, n_latent_frames, ctx.time_sf,
            h_patches, w_patches, device,
        )
        log.info(f"Ground-truth flow shape: {tuple(gt_flow.shape)}")

        # For each timestep, install hooks, forward, compute AMF per block
        for sigma in args.timesteps:
            log.info(f"--- sigma = {sigma} ---")
            noised = add_noise_rectified_flow(source_latent, sigma, seed=args.seed)

            # Install Q/K captures on selected blocks
            captures = []
            for bi in block_indices:
                cap = QKCapture(all_blocks[bi].attn1)
                cap.install()
                captures.append((bi, cap))

            try:
                # Forward
                run_dit_forward_with_capture(ctx, noised, sigma, [c for _, c in captures])

                # For each block, compute AMF and correlate
                for bi, cap in captures:
                    if not cap.supports_capture:
                        log.info(f"  block {bi}: skipped (gated attention)")
                        continue
                    if cap.q is None or cap.k is None:
                        log.info(f"  block {bi}: no Q/K captured (block may not have run)")
                        continue

                    # Q/K shapes: [B, total_seq, inner_dim]. Reshape to per-head.
                    # In LTX CrossAttention, q,k haven't been head-split (that
                    # happens inside optimized_attention). Reshape here:
                    B, S, D = cap.q.shape
                    H = all_blocks[bi].attn1.heads
                    Dh = D // H
                    q_h = cap.q.reshape(B, S, H, Dh).permute(0, 2, 1, 3)[0]  # [H, S, Dh]
                    k_h = cap.k.reshape(B, S, H, Dh).permute(0, 2, 1, 3)[0]  # [H, S, Dh]

                    # Validate seq length matches n_frames*h*w
                    expected_S = n_latent_frames * h_patches * w_patches
                    if S != expected_S:
                        log.warning(
                            f"  block {bi}: seq len {S} != expected {expected_S} "
                            f"(n_frames={n_latent_frames}, hw={h_patches*w_patches}). "
                            f"Update --width/--height/--num-frames OR LTX uses a different patch size."
                        )
                        continue

                    amf = compute_motion_flow(
                        q_h, k_h, h=h_patches, w=w_patches,
                        nframes=n_latent_frames, temp=2.0,
                    )
                    corr = correlate_amf_with_gt(amf, gt_flow)
                    rows.append({
                        "block_idx": bi,
                        "sigma": sigma,
                        "clip": clip_name,
                        "correlation": corr,
                    })
                    log.info(f"  block {bi:>2d} sigma={sigma:.2f}: corr = {corr:+.4f}")
            finally:
                # Uninstall all hooks for this timestep before next iteration
                for _, cap in captures:
                    cap.uninstall()
                captures.clear()
                gc.collect()
                torch.cuda.empty_cache()

        # Free this clip's latents/flows before next
        del source_latent, gt_flow, pixel_frames
        gc.collect()
        torch.cuda.empty_cache()

    # --- Write outputs ---
    out_csv = Path(args.output).with_suffix(".csv")
    out_md = Path(args.output).with_suffix(".md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block_idx", "sigma", "clip", "correlation"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {len(rows)} rows to {out_csv}")

    # Summary: average correlation per (block_idx, sigma) across clips, then per block
    from collections import defaultdict
    per_block = defaultdict(list)
    per_block_sigma = defaultdict(list)
    for r in rows:
        per_block[r["block_idx"]].append(r["correlation"])
        per_block_sigma[(r["block_idx"], r["sigma"])].append(r["correlation"])

    block_avg = sorted(
        [(bi, sum(vs) / len(vs)) for bi, vs in per_block.items()],
        key=lambda x: -abs(x[1]),
    )

    best_corr = block_avg[0][1] if block_avg else 0.0
    gate = ("PROCEED to Phase 2" if abs(best_corr) > 0.6
            else "MULTI-BLOCK COMBO" if abs(best_corr) > 0.3
            else "STOP — project dead")

    lines = []
    lines.append(f"# LTX AMF Probe Results")
    lines.append("")
    lines.append(f"- Clips: {[Path(c).stem for c in args.clips]}")
    lines.append(f"- Timesteps (sigma): {args.timesteps}")
    lines.append(f"- Resolution: {args.width}x{args.height}, {args.num_frames} frames")
    lines.append(f"- Blocks probed: {len(block_indices)} of {len(all_blocks)}")
    lines.append(f"- Total measurements: {len(rows)}")
    lines.append("")
    lines.append(f"## Gate decision")
    lines.append("")
    if not block_avg:
        lines.append("**No measurements recorded.** Every block was skipped — check the "
                     "log above for the reason (gated attention, seq-len mismatch, etc).")
        lines.append("")
        lines.append("**Verdict: INDETERMINATE — fix the probe and re-run.**")
    else:
        lines.append(f"**Best avg |correlation|: {abs(best_corr):.4f} (block {block_avg[0][0]})**")
        lines.append("")
        lines.append(f"**Verdict: {gate}**")
        lines.append("")
        lines.append("## Top 10 blocks by avg |correlation|")
        lines.append("")
        lines.append("| Rank | Block | Avg corr | Best sigma |")
        lines.append("|------|-------|----------|------------|")
        for rank, (bi, avg) in enumerate(block_avg[:10], start=1):
            # Find best sigma for this block
            per_sig = {sig: c for (b, sig), corrs in per_block_sigma.items()
                       if b == bi for c in [sum(corrs) / len(corrs)]}
            best_sig = max(per_sig.items(), key=lambda x: abs(x[1])) if per_sig else (None, 0.0)
            lines.append(f"| {rank} | {bi} | {avg:+.4f} | sigma={best_sig[0]}: {best_sig[1]:+.4f} |")

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Wrote summary to {out_md}")
    log.info(f"Gate verdict: {gate}")
    if block_avg:
        log.info(f"Best block: {block_avg[0][0]} (avg corr {block_avg[0][1]:+.4f})")


if __name__ == "__main__":
    main()
