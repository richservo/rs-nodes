import os
import torch
import torchaudio
import folder_paths


_NATIVE_FORMATS = {"wav", "flac"}


class RSAudioSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "clip"}),
                "index": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
                "format": (["wav", "flac", "mp3", "ogg"],),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_audio"
    CATEGORY = "rs-nodes"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save_audio(self, audio, filename_prefix, index, format):
        waveform = audio["waveform"]  # [B, C, S]
        sample_rate = audio["sample_rate"]
        wav = waveform[0]  # [C, S]

        filename = f"{filename_prefix}_{index:03d}.{format}"
        output_path = os.path.join(folder_paths.get_input_directory(), filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        saved_filename = filename

        if format in _NATIVE_FORMATS:
            torchaudio.save(output_path, wav, sample_rate, format=format)
        else:
            try:
                torchaudio.save(output_path, wav, sample_rate, format=format)
            except Exception:
                saved_filename = f"{filename_prefix}_{index:03d}.wav"
                fallback_path = os.path.join(
                    folder_paths.get_input_directory(), saved_filename
                )
                print(
                    f"[RSAudioSave] Warning: could not encode as {format}. "
                    f"Falling back to wav -> {saved_filename}"
                )
                torchaudio.save(fallback_path, wav, sample_rate, format="wav")

        print(f"[RSAudioSave] Saved: {saved_filename}")

        subfolder = ""
        view_filename = saved_filename
        if "/" in saved_filename or "\\" in saved_filename:
            parts = saved_filename.replace("\\", "/")
            subfolder = parts.rsplit("/", 1)[0]
            view_filename = parts.rsplit("/", 1)[1]

        return {
            "ui": {
                "audio": [
                    {
                        "filename": view_filename,
                        "subfolder": subfolder,
                        "type": "input",
                    }
                ]
            },
            "result": (),
        }
