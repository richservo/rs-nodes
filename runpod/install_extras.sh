#!/usr/bin/env bash
# rs-nodes pod-side: install extra custom-node packs needed by specific
# workflows. Run this ONCE on a fresh /workspace volume after startup.sh
# has created /workspace/ComfyUI. The installed packs live under
# /workspace/ComfyUI/custom_nodes/ which persists with the volume, so
# subsequent pods don't need to reinstall them.
#
# Idempotent: re-running clones missing packs and `git pull`s existing
# ones. Safe to run after every dispatch if you want updates.
#
# Usage:
#   bash /workspace/install_extras.sh             # install default set
#   bash /workspace/install_extras.sh +sam3       # add packs to set
#   bash /workspace/install_extras.sh -ltxvideo   # remove packs from set
#
# To add a new pack: append a "<key>|<git url>" line to PACKS below.

set -euo pipefail

CUSTOM_NODES_DIR="${CUSTOM_NODES_DIR:-/workspace/ComfyUI/custom_nodes}"

if [ ! -d "$CUSTOM_NODES_DIR" ]; then
    echo "ERROR: $CUSTOM_NODES_DIR does not exist."
    echo "Run startup.sh first so ComfyUI is cloned, then re-run this."
    exit 1
fi

# DNS guard mirrors startup.sh — some RunPod regions ship an empty
# /etc/resolv.conf so even github.com fails to resolve. Fix it once.
if ! getent hosts github.com >/dev/null 2>&1; then
    echo "[install_extras] DNS broken; injecting public resolvers"
    { echo "nameserver 8.8.8.8"; echo "nameserver 1.1.1.1"; } > /etc/resolv.conf || true
fi

# -----------------------------------------------------------------------------
# Pack registry: <local_dirname>|<git_url>
# Default set covers the Koi_NoPeople workflow.
# -----------------------------------------------------------------------------
declare -A PACKS=(
    [vhs]="ComfyUI-VideoHelperSuite|https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
    [controlnet_aux]="comfyui_controlnet_aux|https://github.com/Fannovel16/comfyui_controlnet_aux.git"
    [essentials]="ComfyUI_essentials|https://github.com/cubiq/ComfyUI_essentials.git"
    [ltxvideo]="ComfyUI-LTXVideo|https://github.com/Lightricks/ComfyUI-LTXVideo.git"
    [sam3]="ComfyUI-SAM3|https://github.com/PozzettiAndrea/ComfyUI-SAM3.git"
    [seedvr2]="ComfyUI-SeedVR2_VideoUpscaler|https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
    [res4lyf]="RES4LYF|https://github.com/ClownsharkBatwing/RES4LYF.git"
    [video_depth_anything]="ComfyUI-Video-Depth-Anything|https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git"
)

# Default selection (all keys above — Linux-friendly stack including
# SeedVR2 as a substitute for Windows-only RTX upscaling, plus RES4LYF
# for the res_2s sampler family used by LTX-2 distilled workflows).
SELECTED=(vhs controlnet_aux essentials ltxvideo sam3 seedvr2 res4lyf video_depth_anything)

# Apply +pack / -pack overrides from args.
for arg in "$@"; do
    case "$arg" in
        +*)
            key="${arg#+}"
            if [ -z "${PACKS[$key]:-}" ]; then
                echo "ERROR: unknown pack '$key'. Known: ${!PACKS[*]}"
                exit 1
            fi
            SELECTED+=("$key")
            ;;
        -*)
            key="${arg#-}"
            SELECTED=("${SELECTED[@]/$key/}")
            ;;
        *)
            echo "ERROR: arg '$arg' must start with + or -"
            exit 1
            ;;
    esac
done

# Dedup and drop empties.
declare -A seen=()
FINAL=()
for k in "${SELECTED[@]}"; do
    [ -z "$k" ] && continue
    [ -n "${seen[$k]:-}" ] && continue
    seen[$k]=1
    FINAL+=("$k")
done

cd "$CUSTOM_NODES_DIR"
echo "[install_extras] Installing into $CUSTOM_NODES_DIR"
echo "[install_extras] Packs: ${FINAL[*]}"
echo ""

# -----------------------------------------------------------------------------
# Clone or pull each selected pack.
# -----------------------------------------------------------------------------
for key in "${FINAL[@]}"; do
    spec="${PACKS[$key]}"
    name="${spec%%|*}"
    url="${spec##*|}"
    target="$CUSTOM_NODES_DIR/$name"

    if [ -d "$target/.git" ]; then
        echo "[$key] $name exists; pulling..."
        git -C "$target" pull --ff-only || \
            echo "  WARN: pull failed; keeping existing revision"
    else
        echo "[$key] cloning $url -> $name"
        git clone "$url" "$target" || {
            echo "  ERROR: clone failed; skipping $key"
            continue
        }
    fi

    # Install Python deps if the pack ships a requirements.txt.
    if [ -f "$target/requirements.txt" ]; then
        echo "  installing deps..."
        pip install --no-cache-dir -r "$target/requirements.txt" || \
            echo "  WARN: pip install failed for $key (some nodes may not load)"
    fi

    # Some packs ship an install.py instead of (or alongside) requirements.txt.
    if [ -f "$target/install.py" ]; then
        echo "  running install.py..."
        (cd "$target" && python install.py) || \
            echo "  WARN: install.py failed for $key"
    fi

    echo ""
done

echo "[install_extras] Done. Restart ComfyUI on the pod to load the new nodes."
