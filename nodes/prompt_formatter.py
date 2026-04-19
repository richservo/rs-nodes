import base64
import io
import json
import logging
import os
import re
import sys
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image


class OllamaHTTPError(Exception):
    def __init__(self, code, reason, body):
        self.code = code
        self.reason = reason
        self.body = body
        super().__init__(f"{code} {reason}")


DEFAULT_SYSTEM_PROMPT = """You are a prompt formatter. Your job is to restructure the user's text into a specific tag format and enhance the prompt using the image as reference. Do not change anything about the spirit of the prompt just add visual details to make the output better. Make sure to include a detailed description of the person in the scene and the environment and add any other missing visual details without adding any new elements. Do NOT change the order, make sure actions and dialogue follow the exact order of the original prompt. Anything in quotation marks are dialogue, anything not in quotation marks are action. The standard flow will be action followed by dialogue followed by action. There will never be a case where multiple actions should ever be bundled into a single action or dialogue should be one after another. It is very important to indicate when a person is speaking.

Format rules:
- [s] = style/setting (appears once at the start, describes the visual style)
- [a] = action (describes what is happening visually — movement, expressions, scenery) you must include everything, enhance, never remove. If action indicates speaking it has to be included, he says, saying, he said, etc. This is very important. DO NOT REMOVE ANY DIALOGUE TAGS!!!!!
- [d] = dialogue (raw spoken text only, no quotation marks, no "he said")

Instructions:
1. Extract the visual style/setting and place it after [s]
2. Break the remaining text into action and dialogue segments in chronological order
3. Each [a] should describe visual action, expression, or scene detail
4. Each [d] should contain ONLY the spoken words — no quotes, no attribution
5. Preserve the original order of events exactly
6. Do NOT invent new details, actions, or dialogue that aren't in the original
7. Output ONLY the formatted text with tags — no explanations, no preamble
8. Try to break up each phrase with actions between them.


Example output:
[s] cinematic-realistic
[a] A man in his late 30s, with short, neatly trimmed dark hair, wearing a beige blazer over an orange t-shirt, sits in a recording studio. He is wearing black headphones and facing a microphone. The background shows a wall with framed artwork depicting fantasy and sci-fi scenes. The man initially looks to his left, displaying a confused and concerned expression. He then turns his head to look over his right shoulder, glancing towards someone off-screen to his left. A subtle hum of studio equipment fills the air. Then, the man speaks in a slightly worried tone, directed towards the unseen person
[d] Tim, are you feeling okay?
[a] He squints, his brow furrowing, and continues in a concerned voice
[d] You don't look so good, man.
[a] The background ambience remains constant, a gentle white noise underlying the man's speech.

[s] = style
[a] = action, and dialogue attribution
[d] = dialgue (should be raw text, no quotation marks)"""


class RSPromptFormatter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {"default": DEFAULT_SYSTEM_PROMPT, "multiline": True}),
                "model": ("STRING", {"default": "gemma3:12b"}),
                "ollama_url": ("STRING", {"default": "http://localhost:11434"}),
            },
            "optional": {
                "first_image": ("IMAGE", {"tooltip": "First frame / opening image for the scene."}),
                "middle_image": ("IMAGE", {"tooltip": "Middle frame / key moment of the scene."}),
                "last_image": ("IMAGE", {"tooltip": "Last frame / ending image for the scene."}),
                "cache_file": ("STRING", {"default": "formatted_prompt.json", "tooltip": "JSON cache file. Re-runs Ollama only when the input prompt changes."}),
                "output_dir": ("STRING", {"default": "", "tooltip": "Directory for cache file. Empty = ComfyUI output folder."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_prompt",)
    FUNCTION = "format_prompt"
    CATEGORY = "rs-nodes"

    def _pull_model(self, base_url, model_name):
        """Pull a model from Ollama's registry via the /api/pull endpoint."""
        url = f"{base_url}/api/pull"
        data = json.dumps({"name": model_name, "stream": True}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                last_status = ""
                for line in resp:
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if "error" in msg:
                        raise RuntimeError(f"Ollama pull failed: {msg['error']}")
                    status = msg.get("status", "")
                    if status != last_status:
                        logger.info(f"Pull: {status}")
                        last_status = status
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to pull model '{model_name}': {e}")

    def _stream_chat(self, base_url, payload):
        """Send a streaming chat request and print tokens as they arrive."""
        url = f"{base_url}/api/chat"
        payload["stream"] = True
        # Don't override think — caller controls whether it's in the payload
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise OllamaHTTPError(e.code, e.reason, body) from e

        thinking_started = False
        content_started = False
        caption_parts = []
        try:
            for line in resp:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                msg = chunk.get("message", {})
                think_chunk = msg.get("thinking", "")
                content_chunk = msg.get("content", "")
                if think_chunk:
                    if not thinking_started:
                        sys.stderr.write("[thinking] ")
                        thinking_started = True
                    sys.stderr.write(think_chunk)
                    sys.stderr.flush()
                if content_chunk:
                    if thinking_started and not content_started:
                        sys.stderr.write("\nGenerating: ")
                    elif not content_started:
                        sys.stderr.write("Generating: ")
                    content_started = True
                    sys.stderr.write(content_chunk)
                    sys.stderr.flush()
                    caption_parts.append(content_chunk)
                if chunk.get("done"):
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                    break
        finally:
            resp.close()

        cleaned = "".join(caption_parts).strip()
        # Safety: strip any <think> tags that leaked into content
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def _resolve_cache_path(self, output_dir, cache_file):
        import folder_paths
        base = folder_paths.get_output_directory()
        if output_dir.strip():
            d = os.path.join(base, output_dir.strip())
        else:
            d = base
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, cache_file.strip())

    def _encode_image(self, image_tensor, label: str = ""):
        """Convert an IMAGE tensor to a labeled, high-res base64 JPEG for Ollama.
        Burns the label onto the image so the model can visually identify each frame."""
        frame = image_tensor[0]  # first frame [H, W, C]
        img_array = (frame.cpu().numpy() * 255).astype(np.uint8)

        # Burn label onto image (same approach as data prepper)
        if label:
            import cv2
            bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.7, bgr.shape[1] / 800)  # scale with image width
            thickness = max(2, int(font_scale * 2.5))
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(bgr, (0, 0), (tw + 8, th + 10), (0, 0, 0), -1)
            cv2.putText(bgr, label, (4, th + 6), font, font_scale, (255, 255, 255), thickness)
            img_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def format_prompt(self, prompt: str, system_prompt: str, model: str, ollama_url: str,
                      first_image=None, middle_image=None, last_image=None,
                      cache_file="formatted_prompt.json", output_dir=""):
        cache_path = self._resolve_cache_path(output_dir, cache_file)

        # Build labeled image list from connected inputs
        images = []
        if first_image is not None:
            images.append(("First frame", first_image))
        if middle_image is not None:
            images.append(("Middle frame", middle_image))
        if last_image is not None:
            images.append(("Last frame", last_image))

        # Build a key describing which image slots are connected
        image_key = ",".join(label for label, _ in images) if images else ""

        # Check JSON cache — skip Ollama if prompt and image configuration haven't changed
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if (cache.get("prompt") == prompt
                        and cache.get("system_prompt") == system_prompt
                        and cache.get("image_key", "") == image_key):
                    logger.info("Prompt unchanged — using cached output")
                    return (cache["output"],)
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt or old format, re-run

        base = ollama_url.rstrip("/")

        # Encode images with burned-in labels
        encoded_images = []
        if images:
            for label, img_tensor in images:
                encoded_images.append(self._encode_image(img_tensor, label=label))
            frame_listing = ", ".join(label for label, _ in images)
            label_text = (
                f"Attached are {len(images)} labeled reference frames from "
                f"the video ({frame_listing}). Each frame has its label "
                f"burned into the top-left corner.\n\n"
            )
            content = f"{label_text}{prompt}"
        else:
            content = prompt

        user_message = {"role": "user", "content": content}
        if encoded_images:
            user_message["images"] = encoded_images

        # Check if model supports thinking via /api/show capabilities
        supports_think = False
        try:
            show_url = f"{base}/api/show"
            show_data = json.dumps({"name": model}).encode("utf-8")
            show_req = urllib.request.Request(show_url, data=show_data,
                                             headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(show_req, timeout=10) as show_resp:
                show_info = json.loads(show_resp.read().decode("utf-8"))
                caps = show_info.get("capabilities", [])
                supports_think = "thinking" in caps
                logger.info(f"Model '{model}' capabilities: {caps}")
        except Exception:
            pass  # If we can't check, default to no thinking

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "keep_alive": 0,
        }
        if supports_think:
            payload["think"] = True

        try:
            formatted = self._stream_chat(base, payload)
        except OllamaHTTPError as e:
            detail = ""
            try:
                err_json = json.loads(e.body)
                detail = err_json.get("error", "")
            except Exception:
                detail = e.body[:200] if e.body else ""

            if "not found" in detail.lower():
                logger.info(f"Model '{model}' not found — pulling from Ollama...")
                self._pull_model(base, model)
                logger.info(f"Model '{model}' ready.")
                formatted = self._stream_chat(base, payload)
            elif "think" in detail.lower():
                # Model doesn't support thinking — retry without it
                logger.info(f"Model '{model}' doesn't support thinking — retrying without")
                payload.pop("think", None)
                formatted = self._stream_chat(base, payload)
            else:
                raise RuntimeError(f"Ollama error ({e.code}): {detail or e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot connect to Ollama at {base}: {e}")

        if not formatted:
            raise RuntimeError("Ollama returned an empty response")

        formatted = formatted.strip()

        # Save prompt + output to JSON cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "system_prompt": system_prompt, "image_key": image_key, "output": formatted}, f, indent=2)
        logger.info(f"Saved output to {cache_path}")

        return (formatted,)
