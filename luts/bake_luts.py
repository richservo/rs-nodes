"""One-shot bake script for the ACES LUTs shipped with rs-nodes.

Produces two .cube files in this directory:

  * logc3_ei800_to_rec709_aces.cube
    LogC3 (EI800) -> Rec.709 display via ACES 1.0 SDR Video.
    Uses DisplayViewTransform so the bake includes the full
    ACES RRT + Rec.709 ODT pipeline (tone-mapping + gamma).
    Apply to the LogC3 VAE output for a proper ACES display
    transform (replaces ad-hoc Reinhard + sRGB gamma).

  * logc3_ei800_to_acescg.cube
    LogC3 (EI800) -> linear ACEScg (ACES AP1 scene-linear).
    Straight colorspace-to-colorspace bake via OCIO Baker with
    a 1D input shaper (size 1024) so highlight precision stays
    usable despite the modest 33^3 output cube. Values can
    exceed 1.0 for super-whites — that's HDR territory.

Only needs re-running if the ACES config version or colorspace
naming changes. Install-time dependency: `pip install opencolorio`.
"""

import os

import numpy as np
import PyOpenColorIO as OCIO

CONFIG_URI = "ocio://studio-config-v2.2.0_aces-v1.3_ocio-v2.4"
SRC = "ARRI LogC3 (EI800)"
CUBE_SIZE = 33
SHAPER_SIZE = 1024


def _write_3d_cube(path: str, size: int, samples: np.ndarray, title: str) -> None:
    """samples shape [B, G, R, 3] with B varying slowest."""
    with open(path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title}\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0 0 0\n")
        f.write("DOMAIN_MAX 1 1 1\n")
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    v = samples[b, g, r]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def _bake_via_processor(proc: "OCIO.Processor", size: int) -> np.ndarray:
    """Sample a size^3 grid through an OCIO processor. Returns [B, G, R, 3]."""
    cpu = proc.getDefaultCPUProcessor()
    out = np.empty((size, size, size, 3), dtype=np.float32)
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rgb = cpu.applyRGB([r / (size - 1), g / (size - 1), b / (size - 1)])
                out[b, g, r] = rgb
    return out


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = OCIO.Config.CreateFromFile(CONFIG_URI)

    # --- LUT 1: LogC3 EI800 -> Rec.709 display via ACES 1.0 SDR Video ---
    # DisplayViewTransform applies the full RRT + ODT. Bake by sampling a
    # 33^3 grid through the processor manually — OCIO.Baker doesn't accept
    # a display/view target directly.
    dvt = OCIO.DisplayViewTransform()
    dvt.setSrc(SRC)
    dvt.setDisplay("Rec.1886 Rec.709 - Display")
    dvt.setView("ACES 1.0 - SDR Video")
    proc = cfg.getProcessor(dvt)
    samples = _bake_via_processor(proc, CUBE_SIZE)
    path_sdr = os.path.join(out_dir, "logc3_ei800_to_rec709_aces.cube")
    _write_3d_cube(
        path_sdr, CUBE_SIZE, samples,
        "LogC3 (EI800) -> Rec.709 display via ACES 1.0 SDR Video",
    )
    print(f"[1/2] {path_sdr} ({CUBE_SIZE}^3, via DisplayViewTransform)")

    # --- LUT 2: LogC3 EI800 -> linear ACEScg (scene-referred HDR) ---
    # Colorspace-to-colorspace — no display transform, values may exceed 1.
    # OCIO.Baker handles this with a 1D input shaper for highlight precision.
    b2 = OCIO.Baker()
    b2.setConfig(cfg)
    b2.setFormat("resolve_cube")
    b2.setInputSpace(SRC)
    b2.setTargetSpace("ACEScg")
    b2.setCubeSize(CUBE_SIZE)
    b2.setShaperSize(SHAPER_SIZE)
    path_hdr = os.path.join(out_dir, "logc3_ei800_to_acescg.cube")
    b2.bake(path_hdr)
    print(f"[2/2] {path_hdr} ({CUBE_SIZE}^3, shaper={SHAPER_SIZE})")


if __name__ == "__main__":
    main()
