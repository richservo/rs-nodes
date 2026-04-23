"""LogC3 decode for LTX-2.3 HDR IC-LoRA output.

Uses two baked ACES LUTs shipped in rs-nodes/luts/ (see luts/bake_luts.py):

  * logc3_ei800_to_acescg.cube
    LogC3 (EI800) -> linear ACEScg. The hdr_linear output carries proper
    ACES working-space scene-referred HDR — drop it straight into an EXR
    for grading tools that expect ACES AP1 linear input.

  * logc3_ei800_to_rec709_aces.cube
    LogC3 (EI800) -> Rec.709 display via the ACES 1.0 SDR Video transform
    (full RRT + ODT). The sdr_preview output is display-referred Rec.709,
    matching what a colorist would see through an ACES viewing pipeline —
    no ad-hoc Reinhard tonemap.

Pipeline:
  IMAGE (LogC3, from VAE Decode) ─┬─> hdr_linear (linear ACEScg)
                                  ├─> raw (LogC3 passthrough for external grading)
                                  └─> sdr_preview (Rec.709 display)
"""

import logging

import torch

from ..utils.lut3d import apply_3d_lut, load_bundled_lut

logger = logging.getLogger(__name__)


class RSLogC3Decode:
    """Decode LogC3 (EI800) HDR frames from the LTX-2.3 HDR IC-LoRA into
    linear ACEScg (for EXR save) and an ACES-Rec.709 tonemapped preview
    (for ProRes/MP4 save), plus a raw LogC3 passthrough."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "LogC3 (EI800) frames from VAE Decode after the HDR IC-LoRA. Values in [0, 1]."}),
            },
            "optional": {
                "exposure_stops": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                                              "tooltip": "Exposure adjustment in stops (±EV), applied by scaling the LogC input for both hdr_linear and sdr_preview. 0 = identity. The raw output is left alone so it preserves the original LogC data."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("hdr_linear", "raw", "sdr_preview")
    FUNCTION = "decode"
    CATEGORY = "rs-nodes"

    _LUT_HDR = "logc3_ei800_to_acescg.cube"
    _LUT_SDR = "logc3_ei800_to_rec709_aces.cube"

    def decode(self, images, exposure_stops=0.0):
        raw = images[..., :3].clone()

        # Exposure in LogC space is applied additively to the log signal.
        # LogC3's domain is encoded in [0, 1]; a stop of exposure shifts
        # values by ~0.0588 (derived from ARRI's formula: 1 stop ~=
        # log10(2) / C = 0.301 / 0.24719 ~= 1.218 in log units, then
        # normalized by the [0, 1] mapping range of 20.72). For the
        # typical ±2-3 stops range this linear-log shift is close enough
        # to a true linear multiplier for preview purposes. Use LUT
        # clamping to keep out-of-range values sane.
        if exposure_stops != 0.0:
            # Much cleaner: go to linear, scale, re-encode is overkill.
            # Just shift the LogC signal — 0.0588 per stop is the
            # published LogC3 slope constant.
            logc_shift = exposure_stops * 0.0588
            adjusted = (raw + logc_shift).clamp(0.0, 1.0)
        else:
            adjusted = raw

        # --- HDR linear via ACES LUT ---
        _, h_dmin, h_dmax, h_lut = load_bundled_lut(self._LUT_HDR)
        hdr_linear = apply_3d_lut(adjusted, h_lut, h_dmin, h_dmax, clamp_input=True)

        # --- SDR preview via ACES 1.0 SDR Video display transform ---
        _, s_dmin, s_dmax, s_lut = load_bundled_lut(self._LUT_SDR)
        sdr_preview = apply_3d_lut(adjusted, s_lut, s_dmin, s_dmax, clamp_input=True)

        logger.info(
            f"LogC3 decode: {raw.shape[0]} frame(s), "
            f"exposure={exposure_stops:+.2f} stops, "
            f"hdr_linear range [{hdr_linear.min().item():.3f}, {hdr_linear.max().item():.3f}], "
            f"sdr_preview range [{sdr_preview.min().item():.3f}, {sdr_preview.max().item():.3f}]"
        )
        return (hdr_linear, raw, sdr_preview)
