"""Sigma schedule builder for LTX-Video 2.3.

Build custom sigma schedules with spike injection, power curves, bias,
contrast, and anti-plateau jitter — based on research showing that breaking
plateaus in the sigma schedule with deliberate turbulence produces better
motion and physics than a smooth decay curve.

Outputs a SIGMAS tensor that plugs into RSLTXVGenerate's optional sigmas input.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)


class RSSigmaScheduler:
    """Build a custom sigma schedule with spike injection and curve shaping."""

    # Official Lightricks distilled IC-LoRA sigma schedule
    DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "distilled": ("BOOLEAN", {"default": False,
                                          "tooltip": "Use Lightricks distilled sigma schedule (for IC-LoRA). Ignores steps/shift/resolution."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01,
                                        "tooltip": "Upper shift bound (match your generate node)"}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01,
                                         "tooltip": "Lower shift bound (match your generate node)"}),
                "width": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32,
                                  "tooltip": "Video width for shift computation"}),
                "height": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32,
                                   "tooltip": "Video height for shift computation"}),
                "frames": ("INT", {"default": 97, "min": 1, "max": 1024, "step": 1,
                                   "tooltip": "Video frames for shift computation"}),
                "power": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01,
                                    "tooltip": "Curve exponent. >1 = steeper (compress mid-range), <1 = flatter"}),
                "bias": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                                   "tooltip": "Shift curve left/right. + = more coarse structure time, - = more fine detail time"}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
                                       "tooltip": "Expand/compress dynamic range around mean sigma"}),
            },
            "optional": {
                "spikes": ("STRING", {"default": "", "multiline": False,
                                      "tooltip": "Comma-separated frame numbers for spike injection. E.g. '15,48,90'. Empty = no spikes."}),
                "spike_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Spike depth as fraction of sigma at that point. 0.5 = halve the sigma"}),
                "spike_width": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                                        "tooltip": "Steps affected by each spike (1 = sharp dip, more = broader valley)"}),
                "jitter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.001,
                                     "tooltip": "Anti-plateau: random perturbation amplitude. Prevents drift from smooth sigma regions. 0.01-0.03 = subtle."}),
                "jitter_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                                        "tooltip": "Seed for jitter randomness (0 = random)"}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Blend: 1.0 = full custom schedule, 0.0 = stock schedule"}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "build_sigmas"
    CATEGORY = "rs-nodes"

    def build_sigmas(self, distilled, steps, max_shift, base_shift, width, height, frames,
                     power=1.0, bias=0.0, contrast=1.0,
                     spikes="", spike_strength=0.3, spike_width=1,
                     jitter=0.0, jitter_seed=0, mix=1.0):

        if distilled:
            # Lightricks distilled schedule — fixed 8 steps, ignores shift/resolution
            sig = torch.tensor(self.DISTILLED_SIGMAS, dtype=torch.float64)
            steps = len(sig) - 1
            logger.info(f"Distilled mode: base schedule {sig.tolist()}")
        else:
            # Compute shift from resolution (same formula as RSLTXVGenerate)
            tokens = (width // 32) * (height // 32) * max(1, (frames - 1) // 8)
            x1, x2 = 1024, 4096
            tokens_clamped = min(tokens, x2 * 2)
            mm_shift = (max_shift - base_shift) / (x2 - x1)
            b = base_shift - mm_shift * x1
            shift = tokens_clamped * mm_shift + b

            # Build base sigma schedule (same as generate node)
            sig = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
            sig = torch.where(
                sig != 0,
                math.exp(shift) / (math.exp(shift) + (1.0 / sig - 1.0) ** 1),
                torch.zeros_like(sig),
            )
            # Stretch so terminal sigma = 0.1
            non_zero_mask = sig != 0
            non_zero_sigmas = sig[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - 0.1)
            sig[non_zero_mask] = 1.0 - (one_minus_z / scale_factor)

        base_sigmas = sig.clone()
        modified = sig.clone()

        # --- Apply power curve ---
        if power != 1.0 and steps > 1:
            interior = modified[1:-1]
            s_max = modified[0]
            s_min = modified[-2] if modified[-2] > 0 else 1e-6
            norm = ((s_max - interior) / (s_max - s_min)).clamp(0, 1)
            norm = norm ** power
            modified[1:-1] = s_max - norm * (s_max - s_min)

        # --- Apply bias ---
        if bias != 0.0 and steps > 1:
            interior = modified[1:-1]
            weight = interior * (1.0 - interior)
            modified[1:-1] = (interior + bias * weight).clamp(0, 1)

        # --- Apply contrast ---
        if contrast != 1.0 and steps > 1:
            interior = modified[1:-1]
            mean = interior.mean()
            modified[1:-1] = (mean + contrast * (interior - mean)).clamp(0, 1)

        # --- Anti-plateau jitter ---
        if jitter > 0 and steps > 2:
            gen = torch.Generator()
            if jitter_seed > 0:
                gen.manual_seed(jitter_seed)
            else:
                gen.seed()
            noise = torch.randn(steps - 1, generator=gen, dtype=torch.float64) * jitter
            # Scale noise by local delta so jitter is proportional to the schedule's pace
            deltas = (modified[:-2] - modified[2:]) / 2  # average delta around each interior point
            modified[1:-1] = modified[1:-1] + noise * deltas
            modified[1:-1] = modified[1:-1].clamp(0, 1)

        # --- Inject spikes ---
        spike_positions = self._parse_spikes(spikes, frames)
        if spike_positions and spike_strength > 0 and steps > 2:
            half_w = spike_width // 2
            for pos in spike_positions:
                idx = round(pos * (steps - 1)) + 1
                for i in range(max(1, idx - half_w), min(steps, idx + half_w + 1)):
                    # Gaussian falloff for wider spikes
                    if spike_width > 1:
                        dist = abs(i - idx)
                        falloff = math.exp(-0.5 * (dist / max(1, half_w * 0.5)) ** 2)
                    else:
                        falloff = 1.0
                    modified[i] = modified[i] * (1.0 - spike_strength * falloff)

        # --- Enforce monotonically decreasing ---
        for i in range(1, len(modified)):
            if modified[i] >= modified[i - 1]:
                modified[i] = modified[i - 1] - 1e-6
        modified[-1] = 0.0
        modified = modified.clamp(min=0.0)

        # --- Mix with base ---
        if mix < 1.0:
            modified = mix * modified + (1.0 - mix) * base_sigmas

        sigmas = modified.float()

        # Log the schedule
        vals = [f"{s:.4f}" for s in sigmas.tolist()]
        logger.info(f"Sigma schedule ({len(sigmas)}): [{', '.join(vals)}]")
        if spike_positions:
            logger.info(f"Spikes at positions: {spike_positions} (strength={spike_strength}, width={spike_width})")
        if jitter > 0:
            logger.info(f"Jitter: {jitter} (seed={jitter_seed})")

        return (sigmas,)

    @staticmethod
    def _parse_spikes(spikes_str: str, total_frames: int) -> list[float]:
        """Parse comma-separated frame numbers, convert to normalized 0-1 schedule positions."""
        if not spikes_str or not spikes_str.strip() or total_frames <= 1:
            return []
        positions = []
        for part in spikes_str.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                frame = int(float(part))
                if 0 < frame < total_frames:
                    positions.append(frame / total_frames)
            except ValueError:
                continue
        return sorted(positions)
