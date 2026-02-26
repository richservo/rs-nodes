import difflib
import gc
import hashlib
import os
import re
import torch
import torchaudio
import folder_paths
import comfy.model_management as mm


_NATIVE_FORMATS = {"wav", "flac"}
_MAX_CLIPS = 20

_LOCAL_MODEL_IDS = {"OpenMOSS-Team/MOSS-TTS-Local-Transformer"}


def _parse_dialogue_list(dialogue_list):
    """Parse numbered dialogue list from the parser into plain lines."""
    lines = []
    for line in dialogue_list.strip().splitlines():
        cleaned = re.sub(r"^\d+\.\s*", "", line.strip())
        if cleaned:
            lines.append(cleaned)
    return lines


def _apply_handles(tensor_1d, sample_rate, head_seconds, tail_seconds):
    parts = []
    if head_seconds > 0:
        parts.append(torch.zeros(int(head_seconds * sample_rate), dtype=tensor_1d.dtype))
    parts.append(tensor_1d)
    if tail_seconds > 0:
        parts.append(torch.zeros(int(tail_seconds * sample_rate), dtype=tensor_1d.dtype))
    return torch.cat(parts) if len(parts) > 1 else tensor_1d


def _run_generation(model, input_ids, attention_mask, model_id, processor,
                    temperature, top_p, top_k, repetition_penalty, max_new_tokens):
    if model_id in _LOCAL_MODEL_IDS:
        from transformers import GenerationConfig

        class _LocalGenerationConfig(GenerationConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.layers = kwargs.get("layers", [{} for _ in range(32)])
                self.do_samples = kwargs.get("do_samples", None)
                self.n_vq_for_inference = kwargs.get("n_vq_for_inference", 32)

        text_layer = {"repetition_penalty": 1.0, "temperature": 1.5, "top_p": 1.0, "top_k": top_k}
        audio_layer = {"repetition_penalty": repetition_penalty, "temperature": temperature, "top_p": top_p, "top_k": top_k}

        gen_config = _LocalGenerationConfig(
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=151653,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_cache=True,
            do_sample=False,
            n_vq_for_inference=model.channels - 1,
            do_samples=[True] * model.channels,
            layers=[text_layer] + [audio_layer] * (model.channels - 1),
        )

        return model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_config)
    else:
        return model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            audio_temperature=temperature, audio_top_p=top_p,
            audio_top_k=top_k, audio_repetition_penalty=repetition_penalty,
        )


def _discover_clip_count(filename_prefix, fmt):
    """Count how many clip files exist on disk for the given prefix/format."""
    input_dir = folder_paths.get_input_directory()
    count = 0
    for i in range(1, _MAX_CLIPS + 1):
        path = os.path.join(input_dir, f"{filename_prefix}_{i:03d}.{fmt}")
        if os.path.isfile(path):
            count = i
        else:
            break
    return count


class RSMossTTSSave:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "run_inference": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "clip"}),
                "format": (["wav", "flac", "mp3", "ogg"],),
                "mode": (["one_shot", "all", "single"],),
            },
            "optional": {
                "moss_pipe": ("MOSS_TTS_PIPE", {"lazy": True}),
                "dialogue_list": ("STRING", {"forceInput": True, "lazy": True}),
                "reference_audio": ("AUDIO", {"lazy": True}),
                "select_index": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
                "language": (["auto", "zh", "en", "ja", "ko"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 1.7, "min": 0.0, "max": 5.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
                "head_handle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "tail_handle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
        }

        for i in range(1, _MAX_CLIPS + 1):
            if i == 1:
                inputs["optional"][f"pause_before_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1})
            inputs["optional"][f"start_time_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.01})
            inputs["optional"][f"end_time_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.01})
            inputs["optional"][f"pause_after_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1})

        return inputs

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_and_save"
    CATEGORY = "rs-nodes"
    OUTPUT_NODE = True

    def check_lazy_status(self, run_inference, **kwargs):
        needed = []
        mode = kwargs.get("mode", "all")
        if run_inference:
            if kwargs.get("moss_pipe") is None:
                needed.append("moss_pipe")
            if kwargs.get("dialogue_list") is None:
                needed.append("dialogue_list")
            if kwargs.get("reference_audio") is None:
                needed.append("reference_audio")
        else:
            # Need dialogue_list for clip labels and one_shot segmentation
            if kwargs.get("dialogue_list") is None:
                needed.append("dialogue_list")
        return needed

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("run_inference", True):
            return float("nan")
        # When not inferencing, hash file mtimes so we re-run when clips change
        prefix = kwargs.get("filename_prefix", "clip")
        fmt = kwargs.get("format", "wav")
        input_dir = folder_paths.get_input_directory()
        h = hashlib.md5()
        for i in range(1, _MAX_CLIPS + 1):
            filepath = os.path.join(input_dir, f"{prefix}_{i:03d}.{fmt}")
            try:
                mtime = os.path.getmtime(filepath)
                h.update(f"{i}:{mtime}".encode())
            except OSError:
                break
        # Also hash per-clip trim/pause values
        for i in range(1, _MAX_CLIPS + 1):
            for field in ("start_time", "end_time", "pause_after", "pause_before"):
                val = kwargs.get(f"{field}_{i}", 0.0)
                h.update(f"{field}_{i}:{val}".encode())
        return h.hexdigest()

    def _generate_one(
        self, model, processor, sample_rate, device, model_id,
        text, seed, temperature, top_p, top_k, repetition_penalty,
        max_new_tokens, head_handle, tail_handle, language, reference,
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        lang = None if language == "auto" else language

        user_msg = processor.build_user_message(
            text=text, reference=reference, tokens=None, language=lang,
        )

        batch = processor([[user_msg]], mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = _run_generation(
                model, input_ids, attention_mask, model_id, processor,
                temperature, top_p, top_k, repetition_penalty, max_new_tokens,
            )

        messages = processor.decode(outputs)
        del outputs, input_ids, attention_mask

        if messages[0] is None:
            return None
        wav = messages[0].audio_codes_list[0]
        wav = _apply_handles(wav.cpu(), sample_rate, head_handle, tail_handle)
        return wav

    # ------------------------------------------------------------------
    #  Whisper-based word alignment
    # ------------------------------------------------------------------

    def _whisper_get_words(self, wav_1d, sample_rate):
        """Transcribe audio with Whisper, return [(word, start_sec, end_sec)] or None."""
        try:
            import whisper
        except ImportError:
            print("[RSMossTTSSave] openai-whisper not installed — skipping Whisper alignment")
            return None

        import numpy as np

        # Resample to 16 kHz mono (Whisper requirement)
        if sample_rate != 16000:
            wav_16k = torchaudio.functional.resample(
                wav_1d.unsqueeze(0), sample_rate, 16000,
            ).squeeze(0)
        else:
            wav_16k = wav_1d
        audio_np = wav_16k.cpu().numpy().astype(np.float32)

        # Use ComfyUI's whisper model cache dir if it exists
        cache_dir = os.path.join(folder_paths.models_dir, "stt", "whisper")
        if not os.path.isdir(cache_dir):
            cache_dir = None

        try:
            print("[RSMossTTSSave] Loading Whisper base model...")
            model = whisper.load_model("base", download_root=cache_dir)
            result = model.transcribe(audio_np, word_timestamps=True)
        except Exception as e:
            print(f"[RSMossTTSSave] Whisper transcription failed: {e}")
            return None
        finally:
            try:
                del model
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                text = w["word"].strip()
                if text:
                    words.append((text, w["start"], w["end"]))

        if not words:
            print("[RSMossTTSSave] Whisper returned no words")
            return None

        transcript = " ".join(w[0] for w in words)
        print(f"[RSMossTTSSave] Whisper ({len(words)} words): {transcript[:120]}...")
        return words

    def _align_words_to_lines(self, words, lines, sample_rate):
        """Match Whisper words to script lines via sequence alignment.

        Returns list of (len(lines)-1) boundary sample positions, or None.
        """

        def norm(text):
            return re.sub(r"[^a-z0-9]", "", text.lower())

        # Flat script word list with line-index tracking
        script_words = []
        word_line_idx = []
        for li, line in enumerate(lines):
            for w in line.split():
                n = norm(w)
                if n:
                    script_words.append(n)
                    word_line_idx.append(li)
        if not script_words:
            return None

        w_norms = [norm(w[0]) for w in words]

        # Sequence-align Whisper output against the known script
        sm = difflib.SequenceMatcher(None, w_norms, script_words, autojunk=False)

        # Assign each Whisper word a line index
        whisper_line = [-1] * len(words)
        matched_count = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    whisper_line[i1 + k] = word_line_idx[j1 + k]
                matched_count += i2 - i1
            elif tag == "replace":
                w_len, s_len = i2 - i1, j2 - j1
                for k in range(w_len):
                    s_idx = j1 + min(int(k * s_len / w_len), s_len - 1)
                    whisper_line[i1 + k] = word_line_idx[s_idx]

        # Interpolate unassigned words from neighbours
        for i in range(len(whisper_line)):
            if whisper_line[i] != -1:
                continue
            left = right = -1
            for j in range(i - 1, -1, -1):
                if whisper_line[j] != -1:
                    left = whisper_line[j]
                    break
            for j in range(i + 1, len(whisper_line)):
                if whisper_line[j] != -1:
                    right = whisper_line[j]
                    break
            whisper_line[i] = left if left != -1 else (right if right != -1 else 0)

        total = max(len(w_norms), len(script_words))
        ratio = matched_count / total if total else 0
        print(f"[RSMossTTSSave] Alignment: {matched_count}/{total} words matched ({ratio:.0%})")
        if ratio < 0.4:
            print("[RSMossTTSSave] Alignment too poor, skipping Whisper boundaries")
            return None

        # Place boundaries between adjacent lines
        needed = len(lines) - 1
        boundaries = []
        for li in range(needed):
            last_wi = first_next_wi = -1
            for wi in range(len(whisper_line)):
                if whisper_line[wi] == li:
                    last_wi = wi
            for wi in range(len(whisper_line)):
                if whisper_line[wi] == li + 1:
                    first_next_wi = wi
                    break

            if last_wi >= 0 and first_next_wi >= 0:
                boundary_sec = (words[last_wi][2] + words[first_next_wi][1]) / 2.0
            elif last_wi >= 0:
                boundary_sec = words[last_wi][2]
            elif first_next_wi >= 0:
                boundary_sec = words[first_next_wi][1]
            else:
                print(f"[RSMossTTSSave] No words for line boundary {li+1}, alignment failed")
                return None

            boundaries.append(int(boundary_sec * sample_rate))
            print(f"[RSMossTTSSave] Boundary {li+1}: {boundary_sec:.3f}s (sample {boundaries[-1]})")

        return boundaries

    # ------------------------------------------------------------------
    #  Silence-based fallback (DP)
    # ------------------------------------------------------------------

    def _silence_dp_boundaries(self, wav_1d, lines, sample_rate, processor=None):
        """Fallback: pick cut points at silence regions via token-weighted DP."""
        num_segments = len(lines)
        total_samples = wav_1d.shape[0]
        needed = num_segments - 1

        # Proportional weights
        if processor is not None:
            try:
                weights = [len(processor.tokenizer.encode(line)) for line in lines]
            except Exception:
                weights = [len(line) for line in lines]
        else:
            weights = [len(line) for line in lines]

        total_weight = sum(weights) or num_segments
        if total_weight == 0:
            weights = [1] * num_segments
            total_weight = num_segments

        cum_targets = []
        cumulative = 0
        for w in weights[:-1]:
            cumulative += w
            cum_targets.append(cumulative / total_weight)

        # Silence detection
        window_size = int(0.020 * sample_rate)
        hop_size = int(0.010 * sample_rate)
        min_silence_frames = int(0.100 / 0.010)
        num_frames = (wav_1d.shape[0] - window_size) // hop_size + 1

        silence_centers = []
        if num_frames > 0:
            rms = torch.zeros(num_frames)
            for f in range(num_frames):
                start = f * hop_size
                frame = wav_1d[start:start + window_size].float()
                rms[f] = torch.sqrt(torch.mean(frame ** 2))

            threshold = rms.mean() * 0.1
            is_silent = rms < threshold

            start_frame = None
            for f in range(num_frames):
                if is_silent[f]:
                    if start_frame is None:
                        start_frame = f
                else:
                    if start_frame is not None:
                        length = f - start_frame
                        if length >= min_silence_frames:
                            silence_centers.append((start_frame + length // 2) * hop_size)
                        start_frame = None
            if start_frame is not None:
                length = num_frames - start_frame
                if length >= min_silence_frames:
                    silence_centers.append((start_frame + length // 2) * hop_size)

        K = len(silence_centers)
        print(f"[RSMossTTSSave] Silence fallback: {K} regions, {needed} cuts needed")

        if K < needed:
            return [int(total_samples * t) for t in cum_targets]

        INF = float("inf")
        dp = [[INF] * K for _ in range(needed)]
        parent = [[-1] * K for _ in range(needed)]

        for k in range(K):
            dp[0][k] = (silence_centers[k] / total_samples - cum_targets[0]) ** 2

        for j in range(1, needed):
            best_prev, best_prev_k = INF, -1
            for k in range(K):
                if k > 0 and dp[j - 1][k - 1] < best_prev:
                    best_prev = dp[j - 1][k - 1]
                    best_prev_k = k - 1
                if best_prev < INF:
                    cost = best_prev + (silence_centers[k] / total_samples - cum_targets[j]) ** 2
                    if cost < dp[j][k]:
                        dp[j][k] = cost
                        parent[j][k] = best_prev_k

        best_k = min(range(K), key=lambda k: dp[needed - 1][k])
        chosen = [0] * needed
        chosen[needed - 1] = best_k
        for j in range(needed - 2, -1, -1):
            chosen[j] = parent[j + 1][chosen[j + 1]]

        return [silence_centers[ci] for ci in chosen]

    # ------------------------------------------------------------------
    #  Main segmentation entry point
    # ------------------------------------------------------------------

    def _segment_oneshot(self, wav_1d, lines, sample_rate, processor=None):
        """Split one-shot waveform into per-line segments.

        Tries Whisper word-level alignment first for precise cuts,
        falls back to silence-based DP if Whisper is unavailable or fails.
        """
        if len(lines) <= 1:
            return [wav_1d]

        # --- Try Whisper ---
        boundaries = None
        whisper_words = self._whisper_get_words(wav_1d, sample_rate)
        if whisper_words is not None:
            boundaries = self._align_words_to_lines(whisper_words, lines, sample_rate)
            if boundaries is not None:
                print("[RSMossTTSSave] Using Whisper word-aligned boundaries")

        # --- Fallback: silence DP ---
        if boundaries is None:
            print("[RSMossTTSSave] Falling back to silence-based segmentation")
            boundaries = self._silence_dp_boundaries(wav_1d, lines, sample_rate, processor)

        segments = []
        prev = 0
        for b in boundaries:
            segments.append(wav_1d[prev:b])
            prev = b
        segments.append(wav_1d[prev:])

        return segments

    def _save_one(self, wav_1d, sample_rate, filename_prefix, index, fmt):
        wav_2d = wav_1d.unsqueeze(0)  # [1, S]

        filename = f"{filename_prefix}_{index:03d}.{fmt}"
        output_path = os.path.join(folder_paths.get_input_directory(), filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        saved_filename = filename
        if fmt in _NATIVE_FORMATS:
            torchaudio.save(output_path, wav_2d, sample_rate, format=fmt)
        else:
            try:
                torchaudio.save(output_path, wav_2d, sample_rate, format=fmt)
            except Exception:
                saved_filename = f"{filename_prefix}_{index:03d}.wav"
                fallback_path = os.path.join(
                    folder_paths.get_input_directory(), saved_filename
                )
                print(f"[RSMossTTSSave] Fallback to wav -> {saved_filename}")
                torchaudio.save(fallback_path, wav_2d, sample_rate, format="wav")

        print(f"[RSMossTTSSave] Saved: {saved_filename}")

        subfolder = ""
        view_filename = saved_filename
        if "/" in saved_filename or "\\" in saved_filename:
            parts = saved_filename.replace("\\", "/")
            subfolder = parts.rsplit("/", 1)[0]
            view_filename = parts.rsplit("/", 1)[1]

        return {"filename": view_filename, "subfolder": subfolder, "type": "input"}

    def _save_oneshot(self, wav_1d, sample_rate, filename_prefix, fmt):
        wav_2d = wav_1d.unsqueeze(0)  # [1, S]

        filename = f"{filename_prefix}_oneshot.{fmt}"
        output_path = os.path.join(folder_paths.get_input_directory(), filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        saved_filename = filename
        if fmt in _NATIVE_FORMATS:
            torchaudio.save(output_path, wav_2d, sample_rate, format=fmt)
        else:
            try:
                torchaudio.save(output_path, wav_2d, sample_rate, format=fmt)
            except Exception:
                saved_filename = f"{filename_prefix}_oneshot.wav"
                fallback_path = os.path.join(
                    folder_paths.get_input_directory(), saved_filename
                )
                print(f"[RSMossTTSSave] Fallback to wav -> {saved_filename}")
                torchaudio.save(fallback_path, wav_2d, sample_rate, format="wav")

        print(f"[RSMossTTSSave] Saved one-shot: {saved_filename}")

        subfolder = ""
        view_filename = saved_filename
        if "/" in saved_filename or "\\" in saved_filename:
            parts = saved_filename.replace("\\", "/")
            subfolder = parts.rsplit("/", 1)[0]
            view_filename = parts.rsplit("/", 1)[1]

        return {"filename": view_filename, "subfolder": subfolder, "type": "input"}

    @staticmethod
    def _split_filename(filename):
        """Split a filename with possible subfolder into view-compatible dict."""
        subfolder = ""
        view_filename = filename
        if "/" in filename or "\\" in filename:
            parts = filename.replace("\\", "/")
            subfolder = parts.rsplit("/", 1)[0]
            view_filename = parts.rsplit("/", 1)[1]
        return {"filename": view_filename, "subfolder": subfolder, "type": "input"}

    def _save_full(self, audio_dict, filename_prefix, fmt):
        """Save the concatenated AUDIO dict as {prefix}_full.{fmt}."""
        wav = audio_dict["waveform"][0]  # [C, S]
        sr = audio_dict["sample_rate"]

        filename = f"{filename_prefix}_full.{fmt}"
        output_path = os.path.join(folder_paths.get_input_directory(), filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        saved_filename = filename
        if fmt in _NATIVE_FORMATS:
            torchaudio.save(output_path, wav, sr, format=fmt)
        else:
            try:
                torchaudio.save(output_path, wav, sr, format=fmt)
            except Exception:
                saved_filename = f"{filename_prefix}_full.wav"
                fallback_path = os.path.join(
                    folder_paths.get_input_directory(), saved_filename
                )
                print(f"[RSMossTTSSave] Fallback to wav -> {saved_filename}")
                torchaudio.save(fallback_path, wav, sr, format="wav")

        print(f"[RSMossTTSSave] Saved full clip: {saved_filename}")
        return self._split_filename(saved_filename)

    def _concat_clips(self, clip_count, filename_prefix, fmt, **kwargs):
        """Load clips from disk, apply per-clip trim/pause, concatenate."""
        input_dir = folder_paths.get_input_directory()
        raw_clips = []

        for i in range(1, clip_count + 1):
            filepath = os.path.join(input_dir, f"{filename_prefix}_{i:03d}.{fmt}")
            if not os.path.isfile(filepath):
                print(f"[RSMossTTSSave] Warning: clip file not found, skipping: {filepath}")
                continue
            wav, sr = torchaudio.load(filepath)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            raw_clips.append((wav, sr, i))

        if not raw_clips:
            return {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}

        sample_rate = raw_clips[0][1]
        waveforms = []
        pauses = []

        for wav, sr, i in raw_clips:
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)

            start_trim = kwargs.get(f"start_time_{i}", 0.0)
            end_trim = kwargs.get(f"end_time_{i}", 0.0)
            start_sample = int(start_trim * sample_rate)
            end_sample = wav.shape[1] - int(end_trim * sample_rate)
            wav = wav[:, start_sample:max(start_sample, end_sample)]

            if wav.shape[1] == 0:
                continue
            waveforms.append(wav)
            pauses.append(kwargs.get(f"pause_after_{i}", 0.0))

        if not waveforms:
            return {"waveform": torch.zeros(1, 1, 1), "sample_rate": sample_rate}

        parts = []
        # Pause before first clip
        pause_before = kwargs.get("pause_before_1", 0.0)
        if pause_before > 0:
            parts.append(torch.zeros(1, int(pause_before * sample_rate)))

        for j, wav in enumerate(waveforms):
            parts.append(wav)
            if j < len(waveforms) - 1:
                pause_samples = int(pauses[j] * sample_rate)
                if pause_samples > 0:
                    parts.append(torch.zeros(1, pause_samples))

        combined = torch.cat(parts, dim=1).unsqueeze(0)  # [1, 1, total_samples]
        return {"waveform": combined, "sample_rate": sample_rate}

    def generate_and_save(self, run_inference, filename_prefix, format, mode, **kwargs):
        moss_pipe = kwargs.get("moss_pipe")
        dialogue_list = kwargs.get("dialogue_list")
        reference_audio = kwargs.get("reference_audio")
        select_index = kwargs.get("select_index", 1)
        language = kwargs.get("language", "auto")
        seed = kwargs.get("seed", 0)
        temperature = kwargs.get("temperature", 1.7)
        top_p = kwargs.get("top_p", 0.8)
        top_k = kwargs.get("top_k", 25)
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        max_new_tokens = kwargs.get("max_new_tokens", 4096)
        head_handle = kwargs.get("head_handle", 0.0)
        tail_handle = kwargs.get("tail_handle", 0.0)

        concat_count = 0
        ui_audio = []
        clip_labels = []

        # --- Generation phase ---
        if run_inference:
            if moss_pipe is None:
                raise ValueError("moss_pipe must be connected when run_inference is enabled")
            if not dialogue_list:
                raise ValueError("dialogue_list must be connected when run_inference is enabled")

            model, processor, sample_rate, device, model_id = moss_pipe

            # Ensure model is on GPU (may have been offloaded after previous run)
            model.to(device)
            if hasattr(processor, "audio_tokenizer"):
                processor.audio_tokenizer.to(device)
            gc.collect()  # Free stale CPU tensor memory from previous offload

            # Encode reference audio once
            reference = None
            if reference_audio is not None:
                wav = reference_audio["waveform"][0].mean(dim=0)
                orig_sr = reference_audio["sample_rate"]
                if orig_sr != sample_rate:
                    wav = torchaudio.functional.resample(wav.unsqueeze(0), orig_sr, sample_rate).squeeze(0)
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                reference = processor.encode_audios_from_wav(
                    wav_list=[wav], sampling_rate=sample_rate,
                )

            lines = _parse_dialogue_list(dialogue_list)
            total = len(lines)
            clip_labels = lines

            if total == 0:
                empty_audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": sample_rate}
                return {"ui": {"audio": [], "clip_count": [0], "clip_labels": []}, "result": (empty_audio,)}

            if mode == "one_shot":
                # Concat all dialogue lines into a single text, generate once
                combined_text = " ".join(lines)
                print(f"[RSMossTTSSave] One-shot generating all {total} lines: {combined_text[:80]}...")

                wav_1d = self._generate_one(
                    model, processor, sample_rate, device, model_id,
                    combined_text, seed, temperature, top_p, top_k, repetition_penalty,
                    max_new_tokens, head_handle, tail_handle, language, reference,
                )

                if wav_1d is None:
                    raise ValueError("One-shot generation failed — model returned no audio")

                # Save the raw one-shot file as a reference copy
                self._save_oneshot(wav_1d, sample_rate, filename_prefix, format)

                # Segment into individual clips
                segments = self._segment_oneshot(wav_1d, lines, sample_rate, processor)
                for clip_num, seg in enumerate(segments, 1):
                    self._save_one(seg, sample_rate, filename_prefix, clip_num, format)
            else:
                if mode == "single":
                    idx = min(select_index, total) - 1
                    indices = [idx]
                else:
                    indices = list(range(total))

                for i in indices:
                    text = lines[i]
                    clip_num = i + 1
                    print(f"[RSMossTTSSave] Generating clip {clip_num}/{total}: {text[:60]}...")

                    wav_1d = self._generate_one(
                        model, processor, sample_rate, device, model_id,
                        text, seed, temperature, top_p, top_k, repetition_penalty,
                        max_new_tokens, head_handle, tail_handle, language, reference,
                    )

                    if wav_1d is None:
                        print(f"[RSMossTTSSave] Warning: generation failed for clip {clip_num}, skipping")
                        continue

                    self._save_one(wav_1d, sample_rate, filename_prefix, clip_num, format)

            concat_count = total

            # Build ui_audio from ALL clips on disk (not just generated ones)
            input_dir = folder_paths.get_input_directory()
            for i in range(1, concat_count + 1):
                filename = f"{filename_prefix}_{i:03d}.{format}"
                filepath = os.path.join(input_dir, filename)
                if os.path.isfile(filepath):
                    ui_audio.append(self._split_filename(filename))

            # Offload model to CPU to free VRAM for other workflows
            model.cpu()
            if hasattr(processor, "audio_tokenizer"):
                processor.audio_tokenizer.cpu()
            del reference
            gc.collect()  # Collect stale GPU tensor references before clearing CUDA cache
            torch.cuda.empty_cache()
            mm.soft_empty_cache()
        else:
            # No inference — load from disk based on mode
            input_dir = folder_paths.get_input_directory()

            if mode == "one_shot":
                # Discover numbered clips from previous segmentation
                concat_count = _discover_clip_count(filename_prefix, format)
                if concat_count == 0:
                    # No numbered clips — try to segment the oneshot file
                    oneshot_filename = f"{filename_prefix}_oneshot.{format}"
                    oneshot_path = os.path.join(input_dir, oneshot_filename)
                    if not os.path.isfile(oneshot_path):
                        raise ValueError(
                            f"No clip files or one-shot file found for prefix '{filename_prefix}'. "
                            f"Enable run_inference with mode=one_shot to generate them."
                        )
                    if not dialogue_list:
                        raise ValueError(
                            "dialogue_list must be connected to segment the one-shot file. "
                            "Connect the dialogue list or run inference first."
                        )
                    lines = _parse_dialogue_list(dialogue_list)
                    num_lines = len(lines)
                    if num_lines == 0:
                        raise ValueError("dialogue_list is empty — cannot segment one-shot file.")
                    wav_os, sr_os = torchaudio.load(oneshot_path)
                    wav_1d_os = wav_os.mean(dim=0) if wav_os.shape[0] > 1 else wav_os.squeeze(0)
                    segments = self._segment_oneshot(wav_1d_os, lines, sr_os)
                    for clip_num, seg in enumerate(segments, 1):
                        self._save_one(seg, sr_os, filename_prefix, clip_num, format)
                    concat_count = num_lines
                    clip_labels = lines

                # Parse labels from dialogue_list if available and not already set
                if not clip_labels and dialogue_list:
                    clip_labels = _parse_dialogue_list(dialogue_list)

                # Build ui_audio from numbered clips
                for i in range(1, concat_count + 1):
                    filename = f"{filename_prefix}_{i:03d}.{format}"
                    filepath = os.path.join(input_dir, filename)
                    if os.path.isfile(filepath):
                        ui_audio.append(self._split_filename(filename))
            else:
                # all or single — discover numbered clips on disk
                concat_count = _discover_clip_count(filename_prefix, format)
                if concat_count == 0:
                    raise ValueError(
                        f"No clip files found for prefix '{filename_prefix}' "
                        f"(expected {filename_prefix}_001.{format} in input directory). "
                        f"Enable run_inference to generate them."
                    )

                if dialogue_list:
                    clip_labels = _parse_dialogue_list(dialogue_list)

                # Build ui_audio entries for preview (split subfolder like _save_one)
                for i in range(1, concat_count + 1):
                    filename = f"{filename_prefix}_{i:03d}.{format}"
                    filepath = os.path.join(input_dir, filename)
                    if os.path.isfile(filepath):
                        subfolder = ""
                        view_filename = filename
                        if "/" in filename or "\\" in filename:
                            parts = filename.replace("\\", "/")
                            subfolder = parts.rsplit("/", 1)[0]
                            view_filename = parts.rsplit("/", 1)[1]
                        ui_audio.append({"filename": view_filename, "subfolder": subfolder, "type": "input"})

        # --- Output phase (unified for all modes) ---
        audio_dict = self._concat_clips(concat_count, filename_prefix, format, **kwargs)
        full_audio_info = self._save_full(audio_dict, filename_prefix, format)

        return {
            "ui": {"audio": ui_audio, "full_audio": [full_audio_info], "clip_count": [concat_count], "clip_labels": clip_labels},
            "result": (audio_dict,),
        }
