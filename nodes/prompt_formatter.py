import base64
import io
import json
import re
import sys
import urllib.request
import urllib.error

import numpy as np
from PIL import Image


class OllamaHTTPError(Exception):
    def __init__(self, code, reason, body):
        self.code = code
        self.reason = reason
        self.body = body
        super().__init__(f"{code} {reason}")


DEFAULT_SYSTEM_PROMPT = """You are a prompt formatter. Your job is to restructure the user's text into a specific tag format and enhance the prompt using the image as reference. Do not change anything about the spirit of the prompt just add visual details to make the output better. Make sure to include a detailed description of the person in the scene and the environment and add any other missing visual details without adding any new elements. Do NOT change the order, make sure actions and dialogue follow the exact order of the original prompt. Anything in quotation marks are dialogue, anything not in quotation marks are action. The standard flow will be action followed by dialogue followed by action. There will never be a case where multiple actions should ever be bundled into a single action or dialogue should be one after another.

Format rules:
- [s] = style/setting (appears once at the start, describes the visual style)
- [a] = action (describes what is happening visually — movement, expressions, scenery)
- [d] = dialogue (raw spoken text only, no quotation marks, no "he said")

Instructions:
1. Extract the visual style/setting and place it after [s]
2. Break the remaining text into action and dialogue segments in chronological order
3. Each [a] should describe visual action, expression, or scene detail
4. Each [d] should contain ONLY the spoken words — no quotes, no attribution
5. Preserve the original order of events exactly
6. Do NOT invent new details, actions, or dialogue that aren't in the original
7. Output ONLY the formatted text with tags — no explanations, no preamble


Example output:
[s] cinematic-realistic
[a] A man in his late 30s, with short, neatly trimmed dark hair, wearing a beige blazer over an orange t-shirt, sits in a recording studio. He is wearing black headphones and facing a microphone. The background shows a wall with framed artwork depicting fantasy and sci-fi scenes. The man initially looks to his left, displaying a confused and concerned expression. He then turns his head to look over his right shoulder, glancing towards someone off-screen to his left. A subtle hum of studio equipment fills the air. Then, the man speaks in a slightly worried tone, directed towards the unseen person
[d] Tim, are you feeling okay?
[a] He squints, his brow furrowing, and continues in a concerned voice
[d] You don't look so good, man.
[a] The background ambience remains constant, a gentle white noise underlying the man's speech.

[s] = style
[a] = action
[d] = dialogue (should be raw text, no quotation marks)"""


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
                "reference_image": ("IMAGE",),
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
                        print(f"[RS Prompt Formatter] Pull: {status}")
                        last_status = status
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to pull model '{model_name}': {e}")

    def _stream_chat(self, base_url, payload):
        """Send a streaming chat request and print tokens as they arrive."""
        url = f"{base_url}/api/chat"
        payload["stream"] = True
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

        print("[RS Prompt Formatter] Generating:", flush=True)
        full_text = []
        in_think = False
        try:
            for line in resp:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_text.append(token)
                if chunk.get("done"):
                    break
        finally:
            resp.close()
            print(flush=True)

        raw = "".join(full_text)
        # Strip <think>...</think> blocks (including partial/unclosed)
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
        return cleaned

    def format_prompt(self, prompt: str, system_prompt: str, model: str, ollama_url: str, reference_image=None):
        base = ollama_url.rstrip("/")

        images = None
        if reference_image is not None:
            frame = reference_image[0]  # first frame [H, W, C]
            img_array = (frame.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images = [base64.b64encode(buf.getvalue()).decode("utf-8")]

        user_message = {"role": "user", "content": prompt}
        if images:
            user_message["images"] = images

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "keep_alive": 0,
        }

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
                print(f"[RS Prompt Formatter] Model '{model}' not found — pulling from Ollama...")
                self._pull_model(base, model)
                print(f"[RS Prompt Formatter] Model '{model}' ready.")
                formatted = self._stream_chat(base, payload)
            else:
                raise RuntimeError(f"Ollama error ({e.code}): {detail or e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot connect to Ollama at {base}: {e}")

        if not formatted:
            raise RuntimeError("Ollama returned an empty response")

        return (formatted.strip(),)
