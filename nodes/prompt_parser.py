import re


class RSPromptParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {"default": "", "multiline": True}),
                "dialogue_mode": (["individual", "all"],),
                "dialogue_index": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "control_after_generate": "increment"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("video_prompt", "audio_prompt", "dialogue_count", "dialogue_list")
    FUNCTION = "parse"
    CATEGORY = "rs-nodes"

    def parse(self, script: str, dialogue_mode: str, dialogue_index: int):
        tokens = re.findall(r'\[([sad])\]\s*(.*?)(?=\[[sad]\]|$)', script, re.DOTALL)

        video_parts = []
        audio_prompts = []

        for tag, content in tokens:
            content = content.strip()
            if not content:
                continue

            if tag == "s":
                video_parts.append(f"Style: {content}.")

            elif tag == "a":
                text = content[0].upper() + content[1:] if content else content
                if not text.endswith((".", "!", "?")):
                    text += "."
                video_parts.append(text)

            elif tag == "d":
                video_parts.append(f'"{content}"')
                audio_prompts.append(content)

        video_prompt = " ".join(video_parts)
        dialogue_count = len(audio_prompts)

        if dialogue_mode == "all":
            audio_prompt = " ".join(audio_prompts)
        else:
            idx = dialogue_index - 1
            audio_prompt = audio_prompts[idx] if 0 <= idx < dialogue_count else ""

        dialogue_list = "\n".join(f"{i+1}. {line}" for i, line in enumerate(audio_prompts))

        return (video_prompt, audio_prompt, dialogue_count, dialogue_list)
