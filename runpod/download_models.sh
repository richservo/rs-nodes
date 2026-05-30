#!/usr/bin/env bash
# rs-nodes pod-side: download model weights needed by specific workflows
# from HuggingFace directly to the network volume. Faster than scp from a
# home network, and the volume persists so this only runs once.
#
# Default manifest covers the Koi_NoPeople IC-LoRA workflow (~48 GB total).
# Add new entries to MODELS below as you bring more workflows online.
#
# Idempotent: skips files that already exist with the expected size,
# resumes partial downloads via `wget -c`.
#
# Usage:
#   bash /workspace/download_models.sh
#   MODELS_ROOT=/workspace/ComfyUI/models bash /workspace/download_models.sh

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/workspace/ComfyUI/models}"

# DNS guard.
if ! getent hosts huggingface.co >/dev/null 2>&1; then
    echo "[download_models] DNS broken; injecting public resolvers"
    { echo "nameserver 8.8.8.8"; echo "nameserver 1.1.1.1"; } > /etc/resolv.conf || true
fi

# Optional HuggingFace auth — set HF_TOKEN env var for gated repos / higher
# rate limits. Get a "Read" token at https://huggingface.co/settings/tokens .
WGET_AUTH_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
    echo "[download_models] HF_TOKEN set; using authenticated requests"
    WGET_AUTH_HEADER=(--header="Authorization: Bearer ${HF_TOKEN}")
fi

# -----------------------------------------------------------------------------
# Manifest: <subdir>|<filename>|<url>
# Subdir is appended under $MODELS_ROOT (e.g. checkpoints, loras, ...).
# fp8 quantizations live at Lightricks/LTX-2.3-fp8 (separate repo from
# the bf16 originals at Lightricks/LTX-2.3).
# -----------------------------------------------------------------------------
MODELS=(
    "checkpoints|ltx-2.3-22b-dev-fp8.safetensors|https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors"
    "text_encoders|gemma_3_12B_it_fp4_mixed.safetensors|https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
    "latent_upscale_models|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    "loras|ltx-2.3-22b-distilled-lora-384-1.1.safetensors|https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    "loras|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors|https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    "loras|ltx-2-19b-ic-lora-detailer.safetensors|https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors"
)

mkdir -p "$MODELS_ROOT"
echo "[download_models] Target: $MODELS_ROOT"
echo ""

for entry in "${MODELS[@]}"; do
    IFS='|' read -r subdir name url <<< "$entry"
    target_dir="$MODELS_ROOT/$subdir"
    target_file="$target_dir/$name"
    mkdir -p "$target_dir"

    if [ -f "$target_file" ]; then
        # If the local file is non-empty assume it's complete. wget -c
        # would catch an incomplete one but a quick HEAD lets us skip
        # already-good files without making any network calls beyond
        # what we'd do anyway.
        size=$(stat -c%s "$target_file")
        if [ "$size" -gt 0 ]; then
            printf '[ok]    %-55s (%s bytes)\n' "$subdir/$name" "$size"
            continue
        fi
    fi

    printf '[fetch] %-55s\n' "$subdir/$name"
    # -c resumes partial downloads. --tries=3 covers transient HF hiccups.
    # --no-verbose keeps the log clean while still showing errors.
    wget -c --tries=3 --no-verbose --show-progress \
        "${WGET_AUTH_HEADER[@]}" \
        -O "$target_file" "$url" || {
            echo "  ERROR: download failed for $name"
            rm -f "$target_file"  # don't leave a 0-byte placeholder
            continue
        }
    final_size=$(stat -c%s "$target_file")
    echo "  done: $final_size bytes"
done

echo ""
echo "[download_models] All entries processed."
echo "Files under $MODELS_ROOT:"
du -sh "$MODELS_ROOT"/* 2>/dev/null || true
