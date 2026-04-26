"""RSPromptRelayEncode — multi-segment CLIP encoder for Prompt Relay.

Drop-in alternative to CLIPTextEncode. Accepts a JSON-formatted multi-segment
prompt (or plain text → single-segment) and emits a CONDITIONING that carries
`prompt_relay` metadata. RSLTXVGenerate auto-detects this metadata and installs
the cross-attention penalty patch from arXiv 2604.10030.

Step B scope: raw-JSON path only. Ollama formatter wiring deferred to Step E.
See Reference/prompt-relay-plan.md.
"""

from __future__ import annotations

import json
import logging

import torch
import node_helpers

logger = logging.getLogger(__name__)


class RSPromptRelayEncode:
    """CLIP encode with optional Prompt Relay multi-segment routing.

    Input modes:
      • JSON string starting with '{' — parsed as {"global": str, "segments": [...]}
      • Plain text — encoded as a single global prompt (no segments, behaves like
        CLIPTextEncode).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip":   ("CLIP",),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                       "tooltip": (
                                           "JSON object with 'global' and 'segments' "
                                           "(each {'t_start','t_end','prompt'}, seconds), "
                                           "OR plain text for single-prompt encoding."
                                       )}),
            },
            "optional": {
                "num_frames":   ("INT",   {"default": 97,   "min": 9,   "max": 8192,  "step": 8}),
                "frame_rate":   ("FLOAT", {"default": 25.0, "min": 0.1, "max": 1000.0, "step": 0.01}),
                "epsilon":      ("FLOAT", {"default": 0.1,  "min": 0.001, "max": 0.999, "step": 0.001}),
                "window_mode":  (["L-2", "L-1", "custom"],),
                "window_custom":("INT",   {"default": 0, "min": 0, "max": 1024,
                                            "tooltip": "Used iff window_mode == 'custom'. Latent frames."}),
                "debug_print":  ("BOOLEAN", {"default": False,
                                              "tooltip": "Log parsed JSON and per-segment K-token ranges."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "rs-nodes"

    # ------------------------------------------------------------------
    def encode(
        self,
        clip,
        prompt,
        num_frames=97,
        frame_rate=25.0,
        epsilon=0.1,
        window_mode="L-2",
        window_custom=0,
        debug_print=False,
    ):
        relay = self._parse_input(prompt)

        # Single-segment path: ordinary CLIPTextEncode behaviour.
        if relay is None or not relay["segments"]:
            text = relay["global"] if relay else prompt
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
            if debug_print:
                logger.info(f"RSPromptRelayEncode: single-prompt mode, len={len(text)}")
            return (cond,)

        # Multi-segment path: encode global + each segment, then cat along K.
        global_text = relay["global"].strip()
        segments    = relay["segments"]

        blocks = []  # [(label, cond_tensor, opts_dict)]
        if global_text:
            cond_list = clip.encode_from_tokens_scheduled(clip.tokenize(global_text))
            t, opts = cond_list[0]
            blocks.append(("global", t, opts))
        for i, seg in enumerate(segments):
            cond_list = clip.encode_from_tokens_scheduled(clip.tokenize(seg["prompt"]))
            t, opts = cond_list[0]
            blocks.append((f"seg{i}", t, opts))

        # Concatenate token (K) axis. Cond shape: [B, N_tokens, D].
        cond_concat = torch.cat([b[1] for b in blocks], dim=1)

        # Build per-block (start, end) offsets.
        offsets, cursor = [], 0
        for _, t, _ in blocks:
            n = t.shape[1]
            offsets.append((cursor, cursor + n))
            cursor += n

        if global_text:
            global_len  = offsets[0][1]
            seg_offsets = offsets[1:]
        else:
            global_len  = 0
            seg_offsets = offsets

        pr_segments = [
            {
                "start_token": int(k0),
                "end_token":   int(k1),
                "t_start_sec": float(seg["t_start"]),
                "t_end_sec":   float(seg["t_end"]),
            }
            for (k0, k1), seg in zip(seg_offsets, segments)
        ]

        pr_meta = {
            "global_len":    int(global_len),
            "segments":      pr_segments,
            "epsilon":       float(epsilon),
            "window_mode":   str(window_mode),
            "window_custom": int(window_custom),
            "frame_rate":    float(frame_rate),
            "num_frames":    int(num_frames),
        }

        # Use the first block's opts as the carrier (preserves keys like
        # `unprocessed_ltxav_embeds`, `pooled_output`, hooks, etc.). The token
        # tensor is replaced with the concatenation; `prompt_relay` is stamped
        # via conditioning_set_values so it merges cleanly.
        base_opts = dict(blocks[0][2])
        cond = [[cond_concat, base_opts]]
        cond = node_helpers.conditioning_set_values(cond, {"prompt_relay": pr_meta})

        if debug_print:
            logger.info(
                f"RSPromptRelayEncode: cond shape={tuple(cond_concat.shape)}, "
                f"global_len={global_len}, segments={pr_segments}"
            )
            logger.info(f"  parsed JSON:\n{json.dumps(relay, indent=2)}")

        return (cond,)

    # ------------------------------------------------------------------
    def _parse_input(self, text: str) -> dict | None:
        """Parse the prompt input.

        Returns:
          {"global": str, "segments": [{"prompt", "t_start", "t_end"}, ...]} on success
          {"global": text, "segments": []} for plain (non-JSON) text
          None only if `text` is empty/whitespace
        """
        s = text.strip()
        if not s:
            return None

        if s.startswith("{"):
            try:
                data = json.loads(s)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"RSPromptRelayEncode: JSON parse failed ({e}); treating as plain text")
                return {"global": text, "segments": []}

            if not isinstance(data, dict):
                logger.warning("RSPromptRelayEncode: JSON root is not an object; treating as plain text")
                return {"global": text, "segments": []}

            global_text = str(data.get("global", "") or "")
            raw_segments = data.get("segments", []) or []
            if not isinstance(raw_segments, list):
                raw_segments = []

            clean = []
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue
                p = seg.get("prompt")
                if not isinstance(p, str) or not p.strip():
                    continue
                try:
                    t_start = float(seg.get("t_start", 0.0))
                    t_end   = float(seg.get("t_end", 0.0))
                except (TypeError, ValueError):
                    continue
                if t_end <= t_start:
                    continue
                clean.append({"prompt": p.strip(), "t_start": t_start, "t_end": t_end})

            return {"global": global_text, "segments": clean}

        # Plain text → single global, no segments.
        return {"global": text, "segments": []}
