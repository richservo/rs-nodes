"""ProRes video save node for feature film production.

Saves IMAGE sequences as ProRes .mov files using PyAV (ffmpeg).
Supports ProRes 422 profiles and optional audio muxing.
"""

import logging
import os
from fractions import Fraction

import torch

logger = logging.getLogger(__name__)


class RSVideoSave:
    """Save video frames as ProRes .mov for professional post-production."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "output"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "profile": (["proxy", "lt", "standard", "hq", "4444", "4444xq"],
                            {"default": "hq",
                             "tooltip": "ProRes profile. hq = good balance of quality/size for editing. 4444 = near-lossless with alpha support."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional audio to mux into the .mov file."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1,
                                  "tooltip": "File index suffix. 0 = auto-increment."}),
                "output_dir": ("STRING", {"default": "", "tooltip": "Subdirectory under ComfyUI output. Empty = output root."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_video"
    CATEGORY = "rs-nodes"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # ProRes profile → ffmpeg profile_id
    _PROFILE_MAP = {
        "proxy": 0,
        "lt": 1,
        "standard": 2,
        "hq": 3,
        "4444": 4,
        "4444xq": 5,
    }

    def save_video(self, images, filename_prefix, fps, profile,
                   audio=None, index=0, output_dir=""):
        import av
        import folder_paths

        # Resolve output path
        base = folder_paths.get_output_directory()
        if output_dir.strip():
            base = os.path.join(base, output_dir.strip())
        os.makedirs(base, exist_ok=True)

        # Build filename — split prefix into subdirectory + basename
        if os.sep in filename_prefix or "/" in filename_prefix:
            sub_dir, name_base = os.path.split(filename_prefix)
            base = os.path.join(base, sub_dir)
            os.makedirs(base, exist_ok=True)
        else:
            name_base = filename_prefix

        if index > 0:
            filename = f"{name_base}_{index:05d}.mov"
        else:
            # Auto-increment: find highest existing counter
            counter = 0
            for f in os.listdir(base):
                if f.startswith(name_base) and f.endswith(".mov"):
                    try:
                        num = int(f[len(name_base)+1:-4])
                        counter = max(counter, num)
                    except ValueError:
                        pass
            filename = f"{name_base}_{counter + 1:05d}.mov"

        out_path = os.path.join(base, filename)

        # Determine pixel format based on profile
        # 4444/4444xq support alpha (yuva444p10le), others use yuv422p10le
        if profile in ("4444", "4444xq"):
            pix_fmt = "yuva444p10le" if images.shape[-1] == 4 else "yuv444p10le"
        else:
            pix_fmt = "yuv422p10le"

        profile_id = self._PROFILE_MAP[profile]

        # Open output container
        container = av.open(out_path, mode="w")

        # Video stream
        stream = container.add_stream("prores_ks", rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[2]
        stream.height = images.shape[1]
        stream.pix_fmt = pix_fmt
        stream.options = {"profile": str(profile_id)}

        # Audio stream — must be added before encoding starts
        audio_stream = None
        if audio is not None:
            try:
                waveform = audio["waveform"][0]  # [C, S]
                sample_rate = audio["sample_rate"]
                layout = "stereo" if waveform.shape[0] >= 2 else "mono"
                audio_stream = container.add_stream("pcm_s16le", rate=sample_rate)
                audio_stream.layout = layout
            except Exception as e:
                logger.warning(f"Failed to create audio stream: {e}")
                audio_stream = None

        # Encode video frames
        num_frames = images.shape[0]
        for i in range(num_frames):
            frame_data = torch.clamp(images[i][..., :3] * 255, min=0, max=255)
            frame_np = frame_data.to(device="cpu", dtype=torch.uint8).numpy()
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        # Flush video
        for packet in stream.encode():
            container.mux(packet)

        # Encode audio
        if audio_stream is not None:
            try:
                # Convert to int16 [C, S] numpy
                audio_np = (waveform.cpu().float().clamp(-1, 1) * 32767).to(torch.int16).numpy()

                # PyAV from_ndarray expects [channels, samples] for planar s16p
                # or [samples, channels] for interleaved — use planar for simplicity
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)  # mono: [1, S]

                audio_frame = av.AudioFrame.from_ndarray(audio_np, format="s16p", layout=audio_stream.layout.name)
                audio_frame.sample_rate = sample_rate
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)
                for packet in audio_stream.encode():
                    container.mux(packet)
            except Exception as e:
                logger.warning(f"Failed to encode audio: {e}")

        container.close()

        file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        logger.info(f"Saved ProRes {profile} ({num_frames} frames, {fps}fps): {out_path} ({file_size_mb:.1f} MB)")

        return (out_path,)
