"""3D LUT parse + GPU apply helpers.

Used internally by RSLogC3Decode to apply the baked ACES LUTs under
rs-nodes/luts/. Kept as a plain utility module (not a ComfyUI node)
so the decode node stays self-contained from the user's perspective.
"""

import os
import re

import torch
import torch.nn.functional as F


def parse_cube(path: str):
    """Parse a .cube 3D LUT. Returns (size, domain_min, domain_max, table).

    table shape: [N, N, N, 3] with B varying slowest and R fastest — matches
    the .cube file's row order.
    """
    size = None
    dom_min = [0.0, 0.0, 0.0]
    dom_max = [1.0, 1.0, 1.0]
    values: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            u = line.upper()
            if u.startswith("TITLE"):
                continue
            if u.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
                continue
            if u.startswith("LUT_1D_SIZE"):
                raise ValueError(f"{path}: expected a 3D LUT, got 1D")
            if u.startswith("DOMAIN_MIN"):
                dom_min = [float(x) for x in line.split()[1:4]]
                continue
            if u.startswith("DOMAIN_MAX"):
                dom_max = [float(x) for x in line.split()[1:4]]
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 3:
                try:
                    values.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    pass
    if size is None:
        raise ValueError(f"{path}: no LUT_3D_SIZE")
    expected = size * size * size
    if len(values) != expected:
        raise ValueError(f"{path}: expected {expected} samples, got {len(values)}")
    table = torch.tensor(values, dtype=torch.float32).view(size, size, size, 3)
    return size, dom_min, dom_max, table


def apply_3d_lut(
    images: torch.Tensor,
    lut: torch.Tensor,
    domain_min: list[float],
    domain_max: list[float],
    clamp_input: bool = True,
) -> torch.Tensor:
    """Apply a 3D LUT to [B, H, W, 3] IMAGE tensors via trilinear interp.

    Implemented with torch.nn.functional.grid_sample (5D / volumetric) so
    it runs on CUDA with no per-pixel Python. The LUT is treated as a
    density volume with B (blue) as the depth axis, G as height, R as
    width — matching the .cube file ordering.
    """
    device = images.device
    dtype = images.dtype
    lut = lut.to(device=device, dtype=dtype)
    dmin = torch.tensor(domain_min, device=device, dtype=dtype)
    dmax = torch.tensor(domain_max, device=device, dtype=dtype)

    uvw = (images - dmin) / (dmax - dmin).clamp(min=1e-9)
    if clamp_input:
        uvw = uvw.clamp(0.0, 1.0)

    N = lut.shape[0]
    lut_vol = lut.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # [1, 3, N_B, N_G, N_R]

    B, H, W, _ = images.shape
    grid = uvw * 2.0 - 1.0  # [B, H, W, 3] as (R, G, B) in [-1, 1]
    grid5 = grid.view(B, 1, H, W, 3)
    lut_vol_b = lut_vol.expand(B, -1, -1, -1, -1)
    out = F.grid_sample(
        lut_vol_b, grid5,
        mode="bilinear", padding_mode="border", align_corners=True,
    )  # [B, 3, 1, H, W]
    return out.squeeze(2).permute(0, 2, 3, 1).contiguous()


_LUT_CACHE: dict[str, tuple] = {}


def load_bundled_lut(filename: str) -> tuple:
    """Load a .cube from rs-nodes/luts/ with process-level caching."""
    if filename in _LUT_CACHE:
        return _LUT_CACHE[filename]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rs-nodes/
    path = os.path.join(here, "luts", filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bundled LUT not found: {path}")
    result = parse_cube(path)
    _LUT_CACHE[filename] = result
    return result
