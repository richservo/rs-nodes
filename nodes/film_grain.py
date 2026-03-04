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

        import gc
        import comfy.model_management as mm

        # Free VRAM from prior inference so GPU is available
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()

        device = mm.get_torch_device()
        B, H, W, C = images.shape

        noise_h = max(1, round(H / grain_size))
        noise_w = max(1, round(W / grain_size))
        need_upscale = noise_h != H or noise_w != W

        lum_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device)

        # Process in batches — 8 frames ≈ 200 MB image data + noise overhead
        batch_size = 8
        result = torch.empty_like(images)  # output stays on CPU

        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            chunk = images[start:end].to(device)
            n = end - start

            # Luminance and mask
            luminance = (chunk * lum_weights).sum(dim=-1)
            midtone_mask = 4.0 * luminance * (1.0 - luminance)
            mask = (1.0 - highlight_protection + highlight_protection * midtone_mask).unsqueeze(-1)
            del luminance, midtone_mask

            # Generate and apply noise per frame — all on GPU
            for i in range(n):
                gen = torch.Generator(device=device).manual_seed(seed + start + i)

                if need_upscale:
                    small_mono = torch.randn(1, 1, noise_h, noise_w, device=device, generator=gen)
                    mono = F.interpolate(small_mono, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).squeeze(0)
                    small_color = torch.randn(1, C, noise_h, noise_w, device=device, generator=gen)
                    color = F.interpolate(small_color, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).squeeze(0)
                    del small_mono, small_color
                else:
                    mono = torch.randn(H, W, 1, device=device, generator=gen)
                    color = torch.randn(H, W, C, device=device, generator=gen)

                noise = mono.lerp(color, color_amount)
                chunk[i].add_(noise.mul_(intensity).mul_(mask[i]))
                del mono, color, noise

            result[start:end] = chunk.clamp_(0.0, 1.0).cpu()
            del chunk, mask

        return (result,)
