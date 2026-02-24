import os
import torch
import torchaudio
import folder_paths


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
MAX_CLIPS = 20


def get_audio_files():
    input_dir = folder_paths.get_input_directory()
    files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )
    return ["none"] + files


class RSAudioConcat:
    @classmethod
    def INPUT_TYPES(cls):
        files = get_audio_files()

        inputs = {
            "required": {
                "clip_count": ("INT", {"default": 1, "min": 1, "max": MAX_CLIPS, "step": 1}),
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000, "step": 1}),
            },
        }
        for i in range(1, MAX_CLIPS + 1):
            inputs["required"][f"audio_file_{i}"] = (files,)
            inputs["required"][f"pause_after_{i}"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1})

        return inputs

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "concat"
    CATEGORY = "rs-nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def concat(self, clip_count: int, sample_rate: int, **kwargs):
        input_dir = folder_paths.get_input_directory()

        waveforms = []
        pauses = []
        for i in range(1, clip_count + 1):
            filename = kwargs.get(f"audio_file_{i}", "none")
            if filename == "none":
                continue

            filepath = os.path.join(input_dir, filename)
            if not os.path.isfile(filepath):
                raise ValueError(f"File not found: {filepath}")

            wav, sr = torchaudio.load(filepath)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            waveforms.append(wav)

            pause = kwargs.get(f"pause_after_{i}", 0.5)
            pauses.append(pause)

        if not waveforms:
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": sample_rate},)

        parts = []
        for i, wav in enumerate(waveforms):
            parts.append(wav)
            if i < len(waveforms) - 1:
                pause_samples = int(pauses[i] * sample_rate)
                if pause_samples > 0:
                    parts.append(torch.zeros(1, pause_samples))

        combined = torch.cat(parts, dim=1)
        combined = combined.unsqueeze(0)  # [1, 1, total_samples]

        return ({"waveform": combined, "sample_rate": sample_rate},)
