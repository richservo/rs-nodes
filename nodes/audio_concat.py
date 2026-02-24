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
            },
            "optional": {},
        }
        for i in range(1, MAX_CLIPS + 1):
            inputs["optional"][f"audio_file_{i}"] = (files,)
            inputs["optional"][f"start_time_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.01})
            inputs["optional"][f"end_time_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.01})
            inputs["optional"][f"pause_after_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1})

        return inputs

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "concat"
    CATEGORY = "rs-nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def concat(self, clip_count: int, **kwargs):
        input_dir = folder_paths.get_input_directory()

        # First pass: load all clips to determine sample rate from first clip.
        raw_clips = []
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
            raw_clips.append((wav, sr, i))

        if not raw_clips:
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100},)

        # Use the first clip's sample rate as the target.
        sample_rate = raw_clips[0][1]

        waveforms = []
        pauses = []
        for wav, sr, i in raw_clips:
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)

            # Trim: start_time = seconds to cut from the beginning,
            #        end_time = seconds to cut from the end.
            start_trim = kwargs.get(f"start_time_{i}", 0.0)
            end_trim = kwargs.get(f"end_time_{i}", 0.0)
            start_sample = int(start_trim * sample_rate)
            end_sample = wav.shape[1] - int(end_trim * sample_rate)
            wav = wav[:, start_sample:max(start_sample, end_sample)]

            if wav.shape[1] == 0:
                continue
            waveforms.append(wav)

            pause = kwargs.get(f"pause_after_{i}", 0.0)
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
