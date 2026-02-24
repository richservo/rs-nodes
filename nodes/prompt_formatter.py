import base64
import io
import json
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


DEFAULT_SYSTEM_PROMPT = """You are a prompt formatter. Your ONLY job is to restructure the user's text into a specific tag format. Do NOT add, remove, or change any content. Only reorganize it.

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

Example input:
A cinematic realistic shot. A man sits at a desk looking worried. He says "Are you okay?" while leaning forward. The room is dimly lit.

Example output:
[s] cinematic-realistic
[a] A man sits at a desk looking worried. The room is dimly lit. He leans forward.
[d] Are you okay?"""


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

    def _http_post(self, url, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Read the response body for Ollama's error message
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise OllamaHTTPError(e.code, e.reason, body) from e

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
            "stream": False,
        }

        try:
            result = self._http_post(f"{base}/api/chat", payload)
        except OllamaHTTPError as e:
            detail = ""
            try:
                err_json = json.loads(e.body)
                detail = err_json.get("error", "")
            except Exception:
                detail = e.body[:200] if e.body else ""

            if "not found" in detail.lower():
                # Auto-pull the model, then retry
                print(f"[RS Prompt Formatter] Model '{model}' not found — pulling from Ollama...")
                self._pull_model(base, model)
                print(f"[RS Prompt Formatter] Model '{model}' ready.")
                result = self._http_post(f"{base}/api/chat", payload)
            else:
                raise RuntimeError(f"Ollama error ({e.code}): {detail or e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot connect to Ollama at {base}: {e}")

        formatted = result.get("message", {}).get("content", "")
        if not formatted:
            raise RuntimeError("Ollama returned an empty response")

        return (formatted.strip(),)
