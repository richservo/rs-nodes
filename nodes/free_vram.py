import gc

import torch
import comfy.model_management as mm


class RSFreeVRAM:
    """Passthrough node that frees VRAM between pipeline stages."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "any_input": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "rs-nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, any_input=None):
        # Clone tensor data before cleanup so it's independent of model state
        output = self._clone_tensors(any_input)

        free_before = torch.cuda.mem_get_info()[0] / (1024 ** 3)

        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()

        free_after = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        print(f"[RS Free VRAM] {free_before:.2f} GB free → {free_after:.2f} GB free (recovered {free_after - free_before:.2f} GB)")

        return (output,)

    @staticmethod
    def _clone_tensors(data):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.clone()
        if isinstance(data, dict):
            return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        return data
