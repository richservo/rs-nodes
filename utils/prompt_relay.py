"""Prompt Relay (arXiv 2604.10030) — pure mask builder.

Implements the cross-attention penalty that routes each text-prompt segment to
its assigned span of latent video/audio frames:

    Attn(Q, K, V) = softmax(QK^T / sqrt(d) - C(Q, K)) V
    C(i, j)       = 1[j in K_s] * ReLU(|f(i) - m_s| - w)^2 / (2 sigma^2)

Returns a float additive mask shaped [B, 1, Q, K_total] suitable for handing
straight to ComfyUI's optimized attention kernels (which add it to logits).

No ComfyUI imports — torch only. See Reference/prompt-relay-plan.md for the
broader integration design.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

EPS_FLOOR = 1e-4


def _seconds_to_video_latent(t_sec: float, frame_rate: float, T_lat: int) -> float:
    """Convert seconds to a (possibly fractional) LTX video latent-frame index.

    LTX layout: latent t=0 -> pixel frame 0; latent t>=1 covers pixel frames
    [1+(t-1)*8 .. 1+t*8). Inverse: pixel f -> latent t = 0 if f<=0 else
    1 + (f-1)/8.
    """
    if t_sec <= 0 or frame_rate <= 0:
        return 0.0
    pixel_frame = t_sec * frame_rate
    if pixel_frame <= 1:
        # Below the first 8-stride bin -> first latent frame.
        return min(1.0, float(T_lat))
    return min(1.0 + (pixel_frame - 1.0) / 8.0, float(T_lat))


def _seconds_to_audio_latent(
    t_sec: float,
    *,
    sample_rate: int,
    hop_length: int,
    audio_latent_downsample_factor: int,
    audio_T_lat: int,
    causal: bool,
) -> float:
    """Inverse of AudioPatchifier._get_audio_latent_time_in_sec.

    Forward (from comfy/ldm/lightricks/symmetric_patchifier.py):
        mel = lat * dsf
        if causal: mel = max(0, mel + 1 - dsf)
        sec = mel * hop / sr

    Inverse (causal): for sec > 0, mel = sec * sr / hop and
        lat = max(0, 1 + (mel - 1) / dsf)
    Inverse (non-causal): lat = mel / dsf.
    """
    if t_sec <= 0:
        return 0.0
    mel = t_sec * sample_rate / hop_length
    if causal:
        lat = 1.0 + (mel - 1.0) / max(audio_latent_downsample_factor, 1)
    else:
        lat = mel / max(audio_latent_downsample_factor, 1)
    return min(max(lat, 0.0), float(audio_T_lat))


def _segment_window_sigma(L: float, mode: str, custom: int, epsilon: float) -> tuple[float, float]:
    """Return (w, sigma) for a segment of latent length L.

    Paper: w = L - 2 by default, sigma = (L - w) / sqrt(2 * ln(1/epsilon)).
    """
    L = max(float(L), 0.0)
    if mode == "custom":
        w = float(max(0, min(custom, int(L))))
    elif mode == "L-1":
        w = max(0.0, L - 1.0)
    else:  # "L-2" default
        w = max(0.0, L - 2.0)
    denom = math.sqrt(2.0 * math.log(1.0 / max(epsilon, 1e-6)))
    sigma = max((L - w) / max(denom, 1e-6), EPS_FLOOR)
    return w, sigma


def _convert_base_mask(
    base_mask: torch.Tensor | None,
    B: int,
    K_total: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Normalize an incoming attention_mask to a float additive tensor that broadcasts
    against [B, 1, Q, K_total]. Returns None if base_mask is None.

    Bool/int input is converted via (mask - 1) * finfo.max, matching the convention in
    LTX's _prepare_attention_mask. Float input is passed through (with reshape to add
    the head/Q dims as needed).
    """
    if base_mask is None:
        return None
    if base_mask.dtype == torch.bool or not torch.is_floating_point(base_mask):
        big = torch.finfo(dtype).max if dtype.is_floating_point else 1e4
        m = (base_mask.to(dtype) - 1.0) * big   # 0 for valid, -big for padding
    else:
        m = base_mask.to(dtype)
    # Normalize shape -> [B, 1, ?, K]. We allow Q-dim to be 1 (broadcast) or full.
    if m.ndim == 2:           # [B, K]
        m = m.unsqueeze(1).unsqueeze(2)        # [B, 1, 1, K]
    elif m.ndim == 3:         # [B, Q, K]
        m = m.unsqueeze(1)                      # [B, 1, Q, K]
    elif m.ndim == 4:         # [B, H, Q, K] or [B, 1, Q, K]
        pass
    else:
        raise ValueError(f"Unsupported base_mask ndim={m.ndim}, shape={tuple(m.shape)}")
    return m.to(device=device, dtype=dtype)


def build_relay_mask(
    stream: Literal["video", "audio"],
    *,
    B: int,
    K_total: int,
    pr: dict,
    base_mask: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
    # video stream
    T_lat: int | None = None,
    H_lat: int | None = None,
    W_lat: int | None = None,
    frame_rate: float | None = None,
    # audio stream
    audio_T_lat: int | None = None,
    audio_latent_downsample_factor: int = 4,
    hop_length: int = 160,
    sample_rate: int = 16000,
    audio_is_causal: bool = True,
) -> torch.Tensor:
    """Build the [B, 1, Q, K_total] additive Prompt Relay mask for one cross-attention path.

    See the module docstring for the math. Q is determined by `stream`:
        video: Q = T_lat * H_lat * W_lat,  f(i) = i // (H_lat * W_lat)
        audio: Q = audio_T_lat,            f(i) = i

    `pr` is the prompt_relay metadata dict stamped on CONDITIONING by RSPromptRelayEncode.
    """
    if stream == "video":
        if T_lat is None or H_lat is None or W_lat is None or frame_rate is None:
            raise ValueError("video stream requires T_lat, H_lat, W_lat, frame_rate")
        Q = int(T_lat) * int(H_lat) * int(W_lat)
        f = torch.arange(T_lat, device=device, dtype=torch.float32).repeat_interleave(H_lat * W_lat)
    elif stream == "audio":
        if audio_T_lat is None:
            raise ValueError("audio stream requires audio_T_lat")
        Q = int(audio_T_lat)
        f = torch.arange(audio_T_lat, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown stream: {stream!r}")

    # Build the additive mask [B, 1, Q, K_total]. Start from zeros; layer in the
    # per-segment penalty, then add the (broadcast) base padding mask.
    mask = torch.zeros((1, 1, Q, K_total), device=device, dtype=torch.float32)

    epsilon = float(pr.get("epsilon", 0.1))
    window_mode = str(pr.get("window_mode", "L-2"))
    window_custom = int(pr.get("window_custom", 0))
    segments = pr.get("segments", [])

    for seg in segments:
        k0 = int(seg["start_token"])
        k1 = int(seg["end_token"])
        if k1 <= k0:
            continue
        if seg.get("t_end_sec", 0.0) <= seg.get("t_start_sec", 0.0):
            continue
        if stream == "video":
            t_start = _seconds_to_video_latent(float(seg["t_start_sec"]), float(frame_rate), int(T_lat))
            t_end   = _seconds_to_video_latent(float(seg["t_end_sec"]),   float(frame_rate), int(T_lat))
        else:
            t_start = _seconds_to_audio_latent(
                float(seg["t_start_sec"]),
                sample_rate=sample_rate, hop_length=hop_length,
                audio_latent_downsample_factor=audio_latent_downsample_factor,
                audio_T_lat=int(audio_T_lat), causal=audio_is_causal,
            )
            t_end = _seconds_to_audio_latent(
                float(seg["t_end_sec"]),
                sample_rate=sample_rate, hop_length=hop_length,
                audio_latent_downsample_factor=audio_latent_downsample_factor,
                audio_T_lat=int(audio_T_lat), causal=audio_is_causal,
            )

        L_s = max(t_end - t_start, 0.0)
        if L_s <= 0:
            continue
        m_s = 0.5 * (t_start + t_end)
        w_s, sigma_s = _segment_window_sigma(L_s, window_mode, window_custom, epsilon)

        # Penalty c_s[i] = ReLU(|f(i) - m_s| - w_s)^2 / (2 sigma_s^2)
        delta = (f - m_s).abs() - w_s
        delta = delta.clamp(min=0.0)
        c_s = (delta * delta) / (2.0 * sigma_s * sigma_s)              # [Q]

        # Negate (additive penalty subtracts from logits) and scatter into K range.
        mask[:, :, :, k0:k1] = -c_s.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # broadcast over [B,H,K]

    # Cast and broadcast to batch.
    mask = mask.to(dtype=dtype).expand(B, -1, -1, -1).contiguous()

    # Combine with base padding mask if provided.
    base = _convert_base_mask(base_mask, B, K_total, dtype, device)
    if base is not None:
        # base may be [B,1,1,K] (broadcast over Q) or [B,1,Q,K]; addition broadcasts.
        # If base's batch dim is 1 (e.g. shared mask), broadcasting handles it.
        mask = mask + base

    return mask


# ---------------------------------------------------------------------------
# Sanity tests — run with: python -m utils.prompt_relay
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Two segments cleanly tiling a 12-latent-frame, 97-pixel-frame, 25fps clip.
    # global occupies K[0:4], seg-0 K[4:12], seg-1 K[12:20].
    pr = {
        "global_len": 4,
        "segments": [
            {"start_token": 4,  "end_token": 12, "t_start_sec": 0.0,  "t_end_sec": 1.92},
            {"start_token": 12, "end_token": 20, "t_start_sec": 1.92, "t_end_sec": 3.84},
        ],
        "epsilon": 0.1,
        "window_mode": "L-2",
        "window_custom": 0,
        "frame_rate": 25.0,
        "num_frames": 97,
    }

    # --- Video stream ---
    T_lat, H_lat, W_lat = 12, 4, 4
    Q_video = T_lat * H_lat * W_lat
    K_total = 20

    base = torch.ones(2, K_total, dtype=torch.bool)  # all valid, no padding
    mask = build_relay_mask(
        "video",
        B=2, K_total=K_total, pr=pr, base_mask=base,
        dtype=torch.float32, device=torch.device("cpu"),
        T_lat=T_lat, H_lat=H_lat, W_lat=W_lat, frame_rate=25.0,
    )
    assert mask.shape == (2, 1, Q_video, K_total), mask.shape

    # Global tokens (K[0:4]) should be ~0 everywhere (only base mask, all valid -> 0).
    assert mask[:, :, :, :4].abs().max().item() < 1e-6, "global tokens should be unpenalised"

    # Query at frame 0: should be NEAR seg-0 midpoint (frame ~3) -> low penalty in K[4:12],
    # FAR from seg-1 midpoint (frame ~9) -> strong penalty in K[12:20].
    q_f0 = 0  # token at frame 0 (any spatial position works since H=W are same frame)
    pen_seg0_at_f0 = -mask[0, 0, q_f0, 8].item()   # any token inside seg-0 K range
    pen_seg1_at_f0 = -mask[0, 0, q_f0, 16].item()  # any token inside seg-1 K range
    assert pen_seg0_at_f0 < pen_seg1_at_f0, (pen_seg0_at_f0, pen_seg1_at_f0)
    assert pen_seg1_at_f0 > 1.0, f"expected large seg-1 penalty at frame 0, got {pen_seg1_at_f0}"

    # Query at frame 9 (middle of seg-1): inverse — should have low penalty in K[12:20].
    q_f9 = 9 * H_lat * W_lat
    pen_seg0_at_f9 = -mask[0, 0, q_f9, 8].item()
    pen_seg1_at_f9 = -mask[0, 0, q_f9, 16].item()
    assert pen_seg1_at_f9 < pen_seg0_at_f9, (pen_seg0_at_f9, pen_seg1_at_f9)

    print(f"[video] shape={tuple(mask.shape)}  "
          f"f0(seg0,seg1)=({pen_seg0_at_f0:.3f},{pen_seg1_at_f0:.3f})  "
          f"f9(seg0,seg1)=({pen_seg0_at_f9:.3f},{pen_seg1_at_f9:.3f})")

    # Padding mask test: mark K[18:20] as padding -> those positions get -PAD_NEG added.
    base_pad = torch.ones(1, K_total, dtype=torch.bool)
    base_pad[0, 18:] = False
    mask_pad = build_relay_mask(
        "video",
        B=1, K_total=K_total, pr=pr, base_mask=base_pad,
        dtype=torch.float32, device=torch.device("cpu"),
        T_lat=T_lat, H_lat=H_lat, W_lat=W_lat, frame_rate=25.0,
    )
    pad_val = mask_pad[0, 0, 0, 19].item()
    valid_global = mask_pad[0, 0, 0, 0].item()
    assert valid_global == 0.0
    # Padding contributes -finfo.max (effectively -inf for softmax) on top of any penalty.
    assert pad_val < -1e30, f"padding should be very large negative (-finfo.max), got {pad_val}"
    print(f"[video pad] valid_global={valid_global} pad_val={pad_val:.3e}  (expected very large negative)")

    # --- Audio stream ---
    audio_T_lat = 24  # 24 latent audio frames -> with dsf=4 hop=160 sr=16000:
                      # last sec ~= (1 + 23*4) * 160 / 16000 = 0.93s ... not enough.
    # Adjust: pretend audio runs ~4s. With dsf=4, hop=160, sr=16000 -> sec_per_lat = 4*160/16000 = 0.04s
    # so 4s needs 100 latent frames. Use 100 for the test.
    audio_T_lat = 100
    mask_a = build_relay_mask(
        "audio",
        B=1, K_total=K_total, pr=pr, base_mask=None,
        dtype=torch.float32, device=torch.device("cpu"),
        audio_T_lat=audio_T_lat,
    )
    assert mask_a.shape == (1, 1, audio_T_lat, K_total), mask_a.shape
    # Audio token at index 0 (= t_sec = 0) should have low seg-0 penalty, high seg-1.
    pen_a0_seg0 = -mask_a[0, 0, 0, 8].item()
    pen_a0_seg1 = -mask_a[0, 0, 0, 16].item()
    assert pen_a0_seg0 < pen_a0_seg1, (pen_a0_seg0, pen_a0_seg1)
    print(f"[audio] shape={tuple(mask_a.shape)}  "
          f"a0(seg0,seg1)=({pen_a0_seg0:.3f},{pen_a0_seg1:.3f})")

    # --- Empty segments -> zero relay penalty (just base mask) ---
    pr_empty = {**pr, "segments": []}
    mask_e = build_relay_mask(
        "video",
        B=1, K_total=K_total, pr=pr_empty, base_mask=None,
        dtype=torch.float32, device=torch.device("cpu"),
        T_lat=T_lat, H_lat=H_lat, W_lat=W_lat, frame_rate=25.0,
    )
    assert mask_e.abs().max().item() == 0.0
    print("[empty] zero mask OK")

    print("all tests passed")
