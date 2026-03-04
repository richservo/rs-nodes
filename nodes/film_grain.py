import torch
import torch.nn.functional as F


class RSFilmGrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grain_size": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.1}),
                "color_amount": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "highlight_protection": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = "RS Nodes"

    def apply_grain(self, images, intensity, grain_size, color_amount, highlight_protection, seed):
        if intensity == 0:
            return (images,)

        import comfy.model_management as mm
        device = mm.get_torch_device()
        images = images.to(device)

        B, H, W, C = images.shape

        # Rec.709 luminance
        lum_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device)
        luminance = (images * lum_weights).sum(dim=-1)  # (B, H, W)

        # Midtone mask: parabola peaks at 0.5, zero at 0 and 1
        midtone_mask = 4.0 * luminance * (1.0 - luminance)
        # Blend between uniform (1.0) and midtone-only based on highlight_protection
        mask = 1.0 - highlight_protection + highlight_protection * midtone_mask  # (B, H, W)
        mask = mask.unsqueeze(-1)  # (B, H, W, 1)

        # Generate grain per frame
        noise_h = max(1, round(H / grain_size))
        noise_w = max(1, round(W / grain_size))
        need_upscale = noise_h != H or noise_w != W

        mono_noise = torch.empty(B, H, W, 1, device=device)
        color_noise = torch.empty(B, H, W, C, device=device)

        for i in range(B):
            gen = torch.Generator(device="cpu").manual_seed(seed + i)

            if need_upscale:
                small_mono = torch.randn(1, 1, noise_h, noise_w, generator=gen)
                mono_noise[i] = F.interpolate(small_mono, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).squeeze(0).to(device)

                small_color = torch.randn(1, C, noise_h, noise_w, generator=gen)
                color_noise[i] = F.interpolate(small_color, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).squeeze(0).to(device)
            else:
                mono_noise[i] = torch.randn(H, W, 1, generator=gen).to(device)
                color_noise[i] = torch.randn(H, W, C, generator=gen).to(device)

        # Blend mono/color grain
        noise = mono_noise * (1.0 - color_amount) + color_noise * color_amount

        result = torch.clamp(images + noise * intensity * mask, 0.0, 1.0)
        # Return on CPU (ComfyUI IMAGE convention)
        return (result.cpu(),)
