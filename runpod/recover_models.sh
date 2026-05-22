#!/usr/bin/env bash
# Manual model-weight recovery for rs-nodes LTX-2 pods.
#
# When to use this:
#   - A pod's network volume is missing model weights that should be there
#     (e.g. interrupted bootstrap destroyed files via the now-fixed
#     rm-on-failure bug)
#   - You want to grab specific LTX-2 models without running the full
#     bootstrap (no torch reinstall, no ollama pulls, no other side effects)
#
# Safety guarantees:
#   - Anything currently on disk with non-zero size is SKIPPED (never
#     overwritten or deleted)
#   - Downloads land in a staging dir first; only mv'd to final on success
#   - Failed downloads leave your existing $file COMPLETELY untouched
#
# Usage:
#   bash /workspace/recover_models.sh
#
#   The script will prompt you for your HF token if HF_TOKEN isn't
#   already set in the environment. Get a token (read scope is enough)
#   at https://huggingface.co/settings/tokens.
#
#   To skip the prompt:
#     export HF_TOKEN=hf_XXXXXXXXXXXXX
#     bash /workspace/recover_models.sh
#
#   (Optional) Edit the MODELS=(...) manifest below to add/remove
#   specific files. Format: subdir|filename|repo_id|repo_path
#
# Output:
#   - "=== Current status ===" — what exists vs missing on the volume
#   - "=== Downloading missing ===" — fetches just the missing files
#   - "=== Final state ===" — ls of the model directories
#
# Tips:
#   - Re-run any time; idempotent. Already-present files are skipped.
#   - To force re-download of a specific file, delete it first then re-run.
#   - HF_HUB_ENABLE_HF_TRANSFER=1 is set automatically for parallel downloads
#     (~50-500 MB/s typical with auth + good peering).

set -eu

MODELS_ROOT="${MODELS_ROOT:-/workspace/ComfyUI/models}"

# ----------------------------------------------------------------------------
# HF_TOKEN prompt — read from env if already set, otherwise ask the user.
# Hidden input (no echo) so the token doesn't end up in scrollback or logs.
# ----------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Enter your HuggingFace token (input hidden)."
    echo "Get one at https://huggingface.co/settings/tokens (read scope is enough)."
    printf "HF_TOKEN: "
    # -s = silent (don't echo), -r = raw (don't interpret backslashes)
    read -rs HF_TOKEN
    echo
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: empty token entered. Aborting."
        exit 1
    fi
fi

export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1

# ----------------------------------------------------------------------------
# hf CLI check
# ----------------------------------------------------------------------------
if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not found on PATH."
    echo
    echo "Install with:"
    echo "  /workspace/.venv/bin/pip install -U 'huggingface_hub[hf_transfer]<1.0,>=0.34'"
    echo
    echo "Or activate the venv first:"
    echo "  source /workspace/.venv/bin/activate"
    exit 1
fi

# ----------------------------------------------------------------------------
# Model manifest. Edit this list to add or remove files.
# Format:
#   subdir | filename | hf_repo_id | hf_repo_path
#
# - subdir: subdirectory under MODELS_ROOT (e.g. checkpoints, loras)
# - filename: name of the file as it lands locally
# - hf_repo_id: HuggingFace repo id (org/name)
# - hf_repo_path: path within the HF repo (often same as filename but
#   sometimes nested like split_files/text_encoders/x.safetensors)
# ----------------------------------------------------------------------------
MODELS=(
    # === LTX-2.3 base checkpoint (bf16 dev) ===
    "checkpoints|ltx-2.3-22b-dev.safetensors|Lightricks/LTX-2.3|ltx-2.3-22b-dev.safetensors"
    # fp8 checkpoint omitted — uncomment if you specifically want it:
    # "checkpoints|ltx-2.3-22b-dev-fp8.safetensors|Lightricks/LTX-2.3-fp8|ltx-2.3-22b-dev-fp8.safetensors"

    # === Text encoder ===
    "text_encoders|gemma_3_12B_it_fp4_mixed.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"

    # === Spatial upscaler ===
    "latent_upscale_models|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

    # === Distilled LoRA (for fast 8-step generation) ===
    "loras|ltx-2.3-22b-distilled-lora-384-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

    # === IC-LoRAs ===
    "loras|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    "loras|ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control|ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"

    # === Add more here as needed. Examples (uncomment + verify repo paths) ===
    # "loras|ltx-2.3-22b-ic-lora-hdr-0.9.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-HDR|ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"
    # "loras|ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-Lipdub|ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"
    # "loras|ltx-2.3-22b-ic-lora-refocus.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-Refocus|ltx-2.3-22b-ic-lora-refocus.safetensors"
    # "loras|ltx-2.3-22b-ic-lora-uncompress.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-Uncompress|ltx-2.3-22b-ic-lora-uncompress.safetensors"
    # "loras|ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-HDR|ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"
    # "loras|ltx-2.3-ID-LoRA-CelebVHQ-3K.safetensors|Lightricks/LTX-2.3-ID-LoRA|ltx-2.3-ID-LoRA-CelebVHQ-3K.safetensors"
)

# ----------------------------------------------------------------------------
# Phase 1: status check
# ----------------------------------------------------------------------------
echo "=== Current status ==="
missing_count=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r subdir name repo_id repo_path <<< "$entry"
    file="$MODELS_ROOT/$subdir/$name"
    if [ -s "$file" ]; then
        size=$(du -h "$file" 2>/dev/null | cut -f1)
        printf "  OK       %8s  %s/%s\n" "$size" "$subdir" "$name"
    else
        printf "  MISSING            %s/%s\n" "$subdir" "$name"
        missing_count=$((missing_count + 1))
    fi
done
echo

# ----------------------------------------------------------------------------
# Phase 2: download missing models (skipped if nothing missing)
# ----------------------------------------------------------------------------
ok_count=0
fail_count=0
if [ "$missing_count" -eq 0 ]; then
    echo "All models already present — skipping model download phase."
    echo
else
    echo "=== Downloading $missing_count missing file(s) ==="
    echo "(safe: existing files are never touched)"
    echo
fi
for entry in "${MODELS[@]}"; do
    IFS='|' read -r subdir name repo_id repo_path <<< "$entry"
    dir="$MODELS_ROOT/$subdir"
    file="$dir/$name"
    [ -s "$file" ] && continue

    mkdir -p "$dir"
    staging=$(mktemp -d -p /workspace)
    echo "→ $subdir/$name"
    echo "  repo: $repo_id"
    if hf download "$repo_id" "$repo_path" --local-dir "$staging"; then
        if [ -f "$staging/$repo_path" ]; then
            mv -f "$staging/$repo_path" "$file"
            rm -rf "$staging"
            size=$(du -h "$file" 2>/dev/null | cut -f1)
            echo "  ✓ done ($size)"
            ok_count=$((ok_count + 1))
        else
            echo "  ✗ FAILED — hf returned 0 but file not found at $staging/$repo_path"
            rm -rf "$staging"
            fail_count=$((fail_count + 1))
        fi
    else
        echo "  ✗ FAILED — existing file (if any) NOT touched"
        rm -rf "$staging"
        fail_count=$((fail_count + 1))
    fi
    echo
done

# ----------------------------------------------------------------------------
# Phase 3: custom-node packs. Same pattern — manifest, check, clone if
# missing, leave existing clones alone (just `git pull` to update).
# ----------------------------------------------------------------------------
CUSTOM_NODES_ROOT="${CUSTOM_NODES_ROOT:-/workspace/ComfyUI/custom_nodes}"
PIP="${COMFY_VENV_PIP:-/workspace/.venv/bin/pip}"

# Format: dirname | git url
NODE_PACKS=(
    "ComfyUI-LTXVideo|https://github.com/Lightricks/ComfyUI-LTXVideo.git"
    "ComfyUI-VideoHelperSuite|https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
    "comfyui_controlnet_aux|https://github.com/Fannovel16/comfyui_controlnet_aux.git"
    "ComfyUI_essentials|https://github.com/cubiq/ComfyUI_essentials.git"
    "RES4LYF|https://github.com/ClownsharkBatwing/RES4LYF.git"
    "ComfyUI-SeedVR2_VideoUpscaler|https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
    "ComfyUI-Video-Depth-Anything|https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git"
    # rs-nodes intentionally NOT included — managed via your RS_NODES env var workflow.
)

echo "=== Custom-node packs status ==="
node_missing=0
for entry in "${NODE_PACKS[@]}"; do
    IFS='|' read -r dirname url <<< "$entry"
    if [ -d "$CUSTOM_NODES_ROOT/$dirname/.git" ]; then
        printf "  OK       %s\n" "$dirname"
    else
        printf "  MISSING  %s\n" "$dirname"
        node_missing=$((node_missing + 1))
    fi
done
echo

if [ "$node_missing" -gt 0 ]; then
    echo "=== Cloning $node_missing missing node pack(s) ==="
    mkdir -p "$CUSTOM_NODES_ROOT"
    node_ok=0
    node_fail=0
    for entry in "${NODE_PACKS[@]}"; do
        IFS='|' read -r dirname url <<< "$entry"
        target="$CUSTOM_NODES_ROOT/$dirname"
        [ -d "$target/.git" ] && continue
        echo "→ $dirname"
        echo "  url: $url"
        if git clone "$url" "$target"; then
            if [ -f "$target/requirements.txt" ] && [ -x "$PIP" ]; then
                echo "  installing $dirname requirements..."
                "$PIP" install --no-cache-dir -r "$target/requirements.txt" 2>&1 | sed 's/^/    /' || \
                    echo "  WARN: pip install for $dirname returned non-zero"
            fi
            if [ -f "$target/install.py" ]; then
                ( cd "$target" && /workspace/.venv/bin/python install.py ) 2>&1 | sed 's/^/    /' || true
            fi
            echo "  ✓ done"
            node_ok=$((node_ok + 1))
        else
            echo "  ✗ FAILED to clone $url"
            node_fail=$((node_fail + 1))
        fi
        echo
    done
    echo "Custom nodes: $node_ok cloned, $node_fail failed."
else
    echo "All custom-node packs already present."
fi

# ----------------------------------------------------------------------------
# Phase 4: final report
# ----------------------------------------------------------------------------
echo
echo "=== Summary ==="
echo "  Models downloaded:  $ok_count"
echo "  Models failed:      $fail_count"
echo
echo "=== Models on disk ==="
for d in checkpoints latent_upscale_models loras text_encoders; do
    if [ -d "$MODELS_ROOT/$d" ]; then
        echo "--- $MODELS_ROOT/$d/ ---"
        ls -lh "$MODELS_ROOT/$d/" 2>/dev/null | tail -n +2
        echo
    fi
done

echo "=== Custom nodes on disk ==="
ls -1d "$CUSTOM_NODES_ROOT"/*/ 2>/dev/null | sed 's/^/  /'

if [ "$fail_count" -gt 0 ]; then
    echo
    echo "Some downloads failed. Re-run the script — it will only retry the missing files."
    exit 1
fi
