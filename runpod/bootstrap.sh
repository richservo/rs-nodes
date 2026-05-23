#!/usr/bin/env bash
# rs-nodes pod-side: ONE-SHOT setup for a fresh RunPod network volume.
# Run this once on a brand-new pod with the volume mounted at /workspace
# and it will end with ComfyUI serving on port 8188.
#
# Idempotent: safe to re-run after a crash, after a partial download,
# or on a re-deployed pod. Each phase skips work that's already done.
#
# Configure via env vars before running (all optional):
#   HF_TOKEN          — HuggingFace token for downloads (read scope is enough)
#   OLLAMA_MODEL      — Ollama model(s) to pull (space-separated, default: gemma4:31b gemma4:26b)
#   RS_INSTALL_OLLAMA — set to "0" to skip Ollama (default: "1")
#   RS_LAUNCH_COMFY   — set to "0" to skip the final ComfyUI launch (default: "1")
#   RS_NODE_PACKS     — space-separated key list for install_extras pack set
#                       (default: "vhs controlnet_aux essentials ltxvideo sam3")
#
# Examples:
#   bash /workspace/bootstrap.sh
#   HF_TOKEN=hf_xxx OLLAMA_MODEL="gemma4:31b gemma4:26b" bash /workspace/bootstrap.sh
#   RS_LAUNCH_COMFY=0 bash /workspace/bootstrap.sh   # provision then exit

set -euo pipefail

# -----------------------------------------------------------------------------
# Config (override via env vars before invocation)
# -----------------------------------------------------------------------------
WORKSPACE=/workspace
COMFY_DIR="$WORKSPACE/ComfyUI"
RS_NODES_DIR="$COMFY_DIR/custom_nodes/rs-nodes"
MODELS_ROOT="$COMFY_DIR/models"
VENV="$WORKSPACE/.venv"
PORT="${COMFY_PORT:-8188}"

OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:31b gemma4:26b}"
OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
# RunPod containers can ship NVIDIA_VISIBLE_DEVICES=void which hides all
# GPUs from ollama serve and forces silent CPU fallback. Force visibility
# to 'all' (override via NVIDIA_VISIBLE_DEVICES_OVERRIDE if needed).
NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES_OVERRIDE:-all}"
RS_INSTALL_OLLAMA="${RS_INSTALL_OLLAMA:-1}"   # default ON — RSPromptFormatter needs it
RS_LAUNCH_COMFY="${RS_LAUNCH_COMFY:-1}"
RS_NODE_PACKS="${RS_NODE_PACKS:-vhs controlnet_aux essentials ltxvideo seedvr2 res4lyf}"

export OLLAMA_MODELS OLLAMA_HOST NVIDIA_VISIBLE_DEVICES

# Ubuntu 24.04 (PEP 668) marks the system Python as externally-managed,
# blocking pip installs unless we opt in. The container is single-
# purpose so we accept this globally.
export PIP_BREAK_SYSTEM_PACKAGES=1

# Pin JIT + HF caches to the network volume so they survive container
# restarts. See startup.sh for details — same idea, exported here so
# first-boot bootstrap.sh (which doesn't go through startup.sh) also
# uses /workspace cache paths.
mkdir -p /workspace/.cache/triton \
         /workspace/.cache/torch_inductor \
         /workspace/.cache/huggingface
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/torch_inductor}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
banner() {
    echo ""
    echo "========================================================================"
    echo "  $*"
    echo "========================================================================"
}
log() { printf '[bootstrap] %s\n' "$*"; }

# -----------------------------------------------------------------------------
# Phase 0 — DNS guard
# Some RunPod regions ship containers with an empty /etc/resolv.conf.
# Lives on the container disk so this has to run every cold boot.
# -----------------------------------------------------------------------------
banner "Phase 0/7  DNS check"
if ! getent hosts github.com >/dev/null 2>&1; then
    log "DNS resolution broken; injecting public resolvers"
    { echo "nameserver 8.8.8.8"; echo "nameserver 1.1.1.1"; } > /etc/resolv.conf || \
        log "WARN: could not write /etc/resolv.conf"
    if ! getent hosts github.com >/dev/null 2>&1; then
        log "ERROR: DNS still broken after injection. Bail out."
        exit 1
    fi
fi
log "OK"

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# -----------------------------------------------------------------------------
# Phase 0.5 — sshd survival
#   When this script is the Container Start Command, RunPod's stock
#   init never runs and sshd has no host keys. Persist them on the
#   volume so they survive every container reset and SSH clients
#   don't see "host key changed" warnings. Also persist authorized_keys
#   the same way.
# -----------------------------------------------------------------------------
banner "Phase 0.5/7  sshd host keys + authorized_keys"

mkdir -p /workspace/.ssh /workspace/.ssh/host_keys /etc/ssh /root/.ssh
chmod 700 /workspace/.ssh /root/.ssh

if compgen -G "/workspace/.ssh/host_keys/ssh_host_*" > /dev/null; then
    log "Restoring sshd host keys from /workspace/.ssh/host_keys"
    cp -f /workspace/.ssh/host_keys/ssh_host_* /etc/ssh/
elif compgen -G "/etc/ssh/ssh_host_*" > /dev/null; then
    log "Persisting existing sshd host keys to /workspace/.ssh/host_keys"
    cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
else
    log "Generating fresh sshd host keys (ssh-keygen -A)"
    ssh-keygen -A
    cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
fi
chmod 600 /etc/ssh/ssh_host_*_key 2>/dev/null || true
chmod 644 /etc/ssh/ssh_host_*_key.pub 2>/dev/null || true

if [ -s /workspace/.ssh/authorized_keys ]; then
    log "Restoring authorized_keys from /workspace/.ssh"
    cp /workspace/.ssh/authorized_keys /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
elif [ -s /root/.ssh/authorized_keys ]; then
    log "Persisting authorized_keys -> /workspace/.ssh"
    cp /root/.ssh/authorized_keys /workspace/.ssh/authorized_keys
    chmod 600 /workspace/.ssh/authorized_keys
fi

if command -v service >/dev/null 2>&1; then
    service ssh restart 2>&1 | sed 's/^/[sshd] /' || true
elif [ -x /usr/sbin/sshd ]; then
    pkill -f /usr/sbin/sshd 2>/dev/null || true
    /usr/sbin/sshd
fi

# -----------------------------------------------------------------------------
# Phase 1 — ComfyUI + rs-nodes checkout
# -----------------------------------------------------------------------------
banner "Phase 1/7  ComfyUI + rs-nodes"
if [ ! -d "$COMFY_DIR/.git" ]; then
    log "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
    log "ComfyUI present; pulling..."
    git -C "$COMFY_DIR" pull --ff-only || log "WARN: ComfyUI pull failed"
fi

if [ ! -d "$RS_NODES_DIR/.git" ]; then
    log "Cloning rs-nodes..."
    git clone https://github.com/richservo/rs-nodes.git "$RS_NODES_DIR"
else
    log "rs-nodes present; pulling..."
    git -C "$RS_NODES_DIR" pull --ff-only || log "WARN: rs-nodes pull failed"
fi
git -C "$RS_NODES_DIR" submodule update --init --recursive || \
    log "WARN: submodule update failed"

# Mirror pod-side helper scripts (startup.sh, install_*, monitor.sh, etc.)
# to /workspace/ so the pod template's Container Start Command finds
# them on subsequent boots without re-running bootstrap.sh.
if [ -d "$RS_NODES_DIR/runpod" ]; then
    log "Mirroring rs-nodes/runpod scripts to /workspace/"
    cp -f "$RS_NODES_DIR/runpod/"*.sh /workspace/ 2>/dev/null || true
    chmod +x /workspace/*.sh 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# Phase 2 — Base Python deps
# -----------------------------------------------------------------------------
banner "Phase 2/7  Python deps (ComfyUI + rs-nodes + ROSE)"
# Persistent venv on the network volume so installs survive container
# resets. --system-site-packages inherits CUDA / system libs from the
# base image while letting us override pkg versions in the venv.
#
# Check BOTH python and activate — `python3 -m venv` isn't atomic, so
# a container restart mid-creation leaves bin/python without bin/
# activate, which then boot-loops the pod (Phase 2 skips creation,
# fails on source activate, container exits, restart, repeat). If
# either file is missing or empty, wipe and rebuild.
if [ ! -s "$VENV/bin/python" ] || [ ! -s "$VENV/bin/activate" ]; then
    if [ -e "$VENV" ]; then
        log "Existing venv at $VENV is incomplete (bin/python or bin/activate missing); rebuilding"
        rm -rf "$VENV"
    else
        log "Creating venv at $VENV (one-time)"
    fi
    python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
log "venv active: $(which python) ($(python --version 2>&1))"

# Defensive cleanup: cu130 torch upgrade and other --upgrade pip operations
# can leave numpy's .dist-info in a corrupt state where importlib.metadata
# can't read the version (`found=None`), which then breaks transformers
# import at ComfyUI startup. Wipe stale/broken numpy metadata so pip
# resolves cleanly into a fresh install.
PYSITE="$VENV/lib/python3.12/site-packages"
if compgen -G "$PYSITE/numpy-*.dist-info" > /dev/null 2>&1; then
    log "Clearing existing numpy dist-info entries (defensive)..."
    rm -rf "$PYSITE"/numpy-*.dist-info
fi

# Install order matters. Pip processes each `install` independently;
# later --upgrade calls can clobber earlier-resolved versions. We
# install in this order:
#   1. Framework: torch cu130 + NVIDIA + SageAttention (defines
#      tensor/CUDA stack)
#   2. Hub pin: huggingface_hub[hf_transfer] <1.0,>=0.34 (locks the
#      hub version that transformers 4.57+ requires)
#   3. App requirements LAST: ComfyUI + rs-nodes + rose-opt — these
#      transitively bring in transformers and friends; by being last,
#      they see hub already pinned and pip resolves transformers to a
#      version compatible with the pinned hub.

# 1. Framework
# Pick the right torch wheel for the host's CUDA driver. RunPod fleets
# have shifted to mostly CUDA 12.x driver hosts; installing cu130
# unconditionally would bootloop with "driver too old" on those.
# Detect once at bootstrap time and pick wisely. Restart never touches
# torch (startup.sh doesn't reinstall), so whichever wheel lands here
# stays.
TORCH_VARIANT="cu130"
if command -v nvidia-smi >/dev/null 2>&1; then
    _drv=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version:[[:space:]]*[0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$_drv" ] && [ "$_drv" -lt 13 ] 2>/dev/null; then
        TORCH_VARIANT="cu128"
        log "Host driver supports CUDA ${_drv}.x (< 13) — installing torch cu128"
    elif [ -n "$_drv" ]; then
        log "Host driver supports CUDA ${_drv}.x — installing torch cu130 (full Blackwell perf)"
    else
        log "nvidia-smi parse returned empty — defaulting to torch cu130"
    fi
fi

# Detect what's currently installed and only reinstall if it's the
# wrong variant. pip's --upgrade flag refuses to downgrade — if cu130
# is already installed (from a previous bootstrap iteration that
# crashed before completing), `pip install --upgrade cu128 torch`
# says "already satisfied" and leaves cu130 in place.
# Fix: explicitly uninstall when variant doesn't match, then install
# the right one.
_installed=$(python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null | tr -d '.')
_wanted="${TORCH_VARIANT#cu}"
if [ -n "$_installed" ] && [ "$_installed" != "$_wanted" ]; then
    log "Currently installed torch is cu${_installed} — uninstalling before fresh ${TORCH_VARIANT} install"
    pip uninstall -y torch torchvision torchaudio 2>&1 | tail -3 || true
fi

log "Installing PyTorch ${TORCH_VARIANT} wheels..."
pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
    torch torchvision torchaudio || log "WARN: ${TORCH_VARIANT} install failed"

# Verify torch can actually init CUDA. If it can't, we may have guessed
# the wrong wheel — try the other one.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    if [ "$TORCH_VARIANT" = "cu130" ]; then
        _other="cu128"
    else
        _other="cu130"
    fi
    log "${TORCH_VARIANT} can't init CUDA — falling back to ${_other}"
    pip uninstall -y torch torchvision torchaudio 2>&1 | tail -3 || true
    pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/${_other}" \
        torch torchvision torchaudio || log "WARN: ${_other} fallback install failed"
fi

log "Ensuring NVIDIA CUDA runtime libraries (NVRTC + cuDNN + cuBLAS)..."
pip install --no-cache-dir \
    nvidia-cuda-nvrtc nvidia-cuda-runtime \
    nvidia-cublas nvidia-cudnn || \
    log "WARN: CUDA runtime libs install failed"

# pip-installed NVIDIA libs go in venv site-packages/nvidia/*/lib/;
# PyTorch dlopen() needs LD_LIBRARY_PATH set so it can find them.
NV_LIB_PATHS=$(python -c "
import nvidia, os
r = os.path.dirname(nvidia.__file__)
print(':'.join(os.path.join(r, d, 'lib') for d in sorted(os.listdir(r))
               if os.path.isdir(os.path.join(r, d, 'lib'))))
" 2>/dev/null || echo "")
if [ -n "$NV_LIB_PATHS" ]; then
    log "NVIDIA lib paths: $NV_LIB_PATHS"
    export LD_LIBRARY_PATH="$NV_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
log "Installing SageAttention..."
pip install --no-cache-dir sageattention || log "WARN: SageAttention install failed"

# 2. hf-transfer + huggingface_hub pin (BEFORE app requirements)
log "Installing hf-transfer + huggingface_hub (pinned <1.0)..."
pip install --no-cache-dir "huggingface_hub[hf_transfer]<1.0,>=0.34" || \
    log "WARN: hf-transfer install failed"

# 3. App requirements LAST — see comment above for rationale.
log "Installing ComfyUI Python deps..."
pip install --no-cache-dir -r "$COMFY_DIR/requirements.txt" || \
    log "WARN: ComfyUI deps install failed"
log "Installing rs-nodes Python deps..."
[ -f "$RS_NODES_DIR/requirements.txt" ] && \
    pip install --no-cache-dir -r "$RS_NODES_DIR/requirements.txt" || \
    log "WARN: rs-nodes deps install failed"
log "Installing ROSE optimizer..."
pip install --no-cache-dir rose-opt || log "WARN: ROSE install failed"

# 4. Verification — every critical Python import must succeed before
# we declare bootstrap "done". If any fail, bootstrap exits non-zero
# and the marker files are NOT written. Container Start Command will
# re-run bootstrap on next boot, which is the right behavior — better
# than silently launching ComfyUI into a crash loop 30 minutes later.
log "Verifying core Python deps (numpy / torch / transformers / sageattention)..."
python - <<'PY' || { log "ERROR: Python dep verification failed. Bootstrap aborted; markers not written."; exit 1; }
import importlib, sys
ok = True
for mod in ["numpy", "torch", "torchvision", "transformers", "sageattention", "huggingface_hub"]:
    try:
        m = importlib.import_module(mod)
        print(f"  OK   {mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  FAIL {mod}: {type(e).__name__}: {e}", file=sys.stderr)
        ok = False
sys.exit(0 if ok else 1)
PY
log "All core deps verified."

# Write provision hash marker AFTER verification passes. Future startup.sh
# boots fast-path past pip when this marker matches the current hash.
PROVISION_MARKER="$WORKSPACE/.provision_hash"
PROV_HASH=$(cat "$COMFY_DIR/requirements.txt" "$RS_NODES_DIR/requirements.txt" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -n "$PROV_HASH" ]; then
    echo "$PROV_HASH" > "$PROVISION_MARKER"
    log "Wrote provision hash: $PROVISION_MARKER"
fi

# Phase 2.5 (rclone install) removed — user doesn't use it, the
# install was broken, and it was killing bootstrap with set -e on
# failure. B2 sync is handled by RunPod's built-in Cloud Sync.

# -----------------------------------------------------------------------------
# Phase 3 — Extra custom-node packs (workflow-specific)
# -----------------------------------------------------------------------------
banner "Phase 3/7  Custom node packs ($RS_NODE_PACKS)"
declare -A NODE_PACKS=(
    [vhs]="ComfyUI-VideoHelperSuite|https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
    [controlnet_aux]="comfyui_controlnet_aux|https://github.com/Fannovel16/comfyui_controlnet_aux.git"
    [essentials]="ComfyUI_essentials|https://github.com/cubiq/ComfyUI_essentials.git"
    [ltxvideo]="ComfyUI-LTXVideo|https://github.com/Lightricks/ComfyUI-LTXVideo.git"
    [seedvr2]="ComfyUI-SeedVR2_VideoUpscaler|https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
    [res4lyf]="RES4LYF|https://github.com/ClownsharkBatwing/RES4LYF.git"
)

CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"
for key in $RS_NODE_PACKS; do
    spec="${NODE_PACKS[$key]:-}"
    if [ -z "$spec" ]; then
        log "WARN: unknown pack '$key'; skipping (known: ${!NODE_PACKS[*]})"
        continue
    fi
    name="${spec%%|*}"
    url="${spec##*|}"
    target="$CUSTOM_NODES_DIR/$name"

    if [ -d "$target/.git" ]; then
        log "[$key] $name exists; pulling..."
        git -C "$target" pull --ff-only || log "  WARN: pull failed"
    else
        log "[$key] cloning $url"
        git clone "$url" "$target" || { log "  ERROR: clone failed"; continue; }
    fi
    [ -f "$target/requirements.txt" ] && \
        pip install --no-cache-dir -r "$target/requirements.txt" || \
        log "  WARN: pip install failed for $key"
    [ -f "$target/install.py" ] && \
        (cd "$target" && python install.py) || true
done

# -----------------------------------------------------------------------------
# Phase 4 — Model weights from HuggingFace via hf-transfer
# Default manifest: Koi_NoPeople IC-LoRA workflow (~57 GB total, bf16 dev).
#
# hf-transfer is HuggingFace's Rust-based parallel downloader — 50-150x
# faster than single-connection wget on long-haul routes (US<->EU was
# observed at 3 MB/s with wget vs 200-500 MB/s with hf-transfer).
# -----------------------------------------------------------------------------
banner "Phase 4/7  Model weights from HuggingFace (hf-transfer)"

# Ensure hf-transfer is available; install if missing.
# Pin huggingface_hub<1.0 because transformers (~4.57.x) declares
# huggingface-hub<1.0 as a strict requirement and refuses to import
# with hub 1.x present. hf_transfer support exists in both 0.x and
# 1.x, so 0.x keeps everyone happy.
if ! python -c "import hf_transfer" 2>/dev/null; then
    log "Installing hf-transfer + huggingface_hub (pinned <1.0)..."
    pip install --no-cache-dir "huggingface_hub[hf_transfer]<1.0,>=0.34" || \
        log "WARN: hf-transfer install failed; will fall back to curl"
fi

# Enable parallel-connection downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
HAVE_HF=$(command -v hf >/dev/null 2>&1 && echo 1 || echo 0)

# Pass HF_TOKEN to child processes via environment, NEVER via CLI args.
# CLI args (--token, --header=Bearer ...) appear in `ps -ef` and are
# visible to anyone with read access to /proc. The hf CLI reads HF_TOKEN
# from the environment by default, and so does huggingface_hub.
#
# AUTH HUGELY MATTERS for download speed. Without HF_TOKEN, HuggingFace
# rate-limits downloads to ~1-5 MB/s and bootstraps can take hours.
# With a token (read scope is enough) you get 100-500 MB/s on good
# peering. Set HF_TOKEN in your RunPod pod env vars — get a token at
# https://huggingface.co/settings/tokens.
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    log "HF_TOKEN detected — authenticated downloads enabled (fast)"
else
    log ""
    log "============================================================"
    log "  WARN: HF_TOKEN not set — downloads will be SLOW"
    log "  Anonymous HF is rate-limited (~1-5 MB/s, hours per model)"
    log ""
    log "  Set HF_TOKEN in your RunPod pod env vars to fix:"
    log "    1. Get token: https://huggingface.co/settings/tokens"
    log "    2. RunPod console → Edit Pod → Environment Variables"
    log "    3. Add HF_TOKEN=hf_xxxxxxxxxxxxx"
    log "    4. Restart container"
    log "============================================================"
    log ""
fi

# Manifest format: <subdir>|<filename>|<repo_id>|<repo_path>
# Splitting URL into (repo_id, repo_path) so we can use `hf download`
# directly without parsing the resolve URL.
#
# LTX checkpoint variants (both default ON, gate each independently):
#   RS_LTX_FP8=0   → skip fp8 dev (~29 GB)
#   RS_LTX_BF16=0  → skip bf16 dev (~44 GB)
# Most workflows need at least one; set whichever your workflow doesn't
# reference to 0 to halve the download time.
MODELS=(
    "text_encoders|gemma_3_12B_it_fp4_mixed.safetensors|Comfy-Org/ltx-2|split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
    "latent_upscale_models|ltx-2.3-spatial-upscaler-x2-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    "loras|ltx-2.3-22b-distilled-lora-384-1.1.safetensors|Lightricks/LTX-2.3|ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    "loras|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors|Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control|ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
)
if [ "${RS_LTX_BF16:-1}" = "1" ]; then
    MODELS+=("checkpoints|ltx-2.3-22b-dev.safetensors|Lightricks/LTX-2.3|ltx-2.3-22b-dev.safetensors")
fi
# fp8 checkpoint OFF by default — set RS_LTX_FP8=1 to enable.
# bf16 is the primary inference checkpoint; fp8 is a quantized alt.
if [ "${RS_LTX_FP8:-0}" = "1" ]; then
    MODELS+=("checkpoints|ltx-2.3-22b-dev-fp8.safetensors|Lightricks/LTX-2.3-fp8|ltx-2.3-22b-dev-fp8.safetensors")
fi

mkdir -p "$MODELS_ROOT"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r subdir name repo_id repo_path <<< "$entry"
    dir="$MODELS_ROOT/$subdir"
    file="$dir/$name"
    mkdir -p "$dir"
    if [ -s "$file" ]; then
        log "[ok]    $subdir/$name ($(stat -c%s "$file") bytes)"
        continue
    fi
    log "[fetch] $subdir/$name (repo=$repo_id)"
    if [ "$HAVE_HF" = "1" ]; then
        # Download to staging dir. On success, atomic-mv into place. On
        # FAILURE never touch $file — only clean up staging. This is the
        # critical invariant: a failed re-download must NEVER destroy an
        # existing good copy of the model.
        staging=$(mktemp -d -p /workspace 2>/dev/null || mktemp -d)
        if hf download "$repo_id" "$repo_path" \
                --local-dir "$staging"; then
            mv -f "$staging/$repo_path" "$file" 2>/dev/null || \
                cp -f "$staging/$repo_path" "$file"
            rm -rf "$staging"
            log "  done: $(stat -c%s "$file") bytes"
        else
            log "  ERROR: hf download failed for $name (existing $file preserved if any)"
            rm -rf "$staging"
            continue
        fi
    else
        # Fallback: curl. Download to $file.partial, mv into place on
        # success, leave existing $file alone on failure. Same invariant
        # as the hf path — failed download must never destroy good copy.
        url="https://huggingface.co/${repo_id}/resolve/main/${repo_path}"
        hdrfile=""
        curl_auth=()
        if [ -n "${HF_TOKEN:-}" ]; then
            hdrfile=$(mktemp)
            chmod 600 "$hdrfile"
            printf 'Authorization: Bearer %s\n' "$HF_TOKEN" > "$hdrfile"
            curl_auth=(-H "@$hdrfile")
        fi
        partial="$file.partial"
        if curl -L --fail --retry 3 -C - --progress-bar \
                "${curl_auth[@]}" \
                -o "$partial" "$url"; then
            mv -f "$partial" "$file"
            log "  done: $(stat -c%s "$file") bytes"
        else
            log "  ERROR: curl download failed for $name (existing $file preserved if any)"
            # Leave $partial in place so -C - on the next run can resume it.
            # NEVER touch $file.
        fi
        [ -n "$hdrfile" ] && rm -f "$hdrfile"
    fi
done

# -----------------------------------------------------------------------------
# Phase 5 — Ollama install + model pull
# -----------------------------------------------------------------------------
if [ "$RS_INSTALL_OLLAMA" = "1" ]; then
    banner "Phase 5/7  Ollama (model: $OLLAMA_MODEL)"
    if ! command -v ollama >/dev/null 2>&1; then
        # Install with verify + retry. Previously this was a single
        # `curl ... | sh` with no verification — when the install
        # silently failed (network glitch, CDN issue, gpg verify
        # failure), bootstrap continued without ollama in PATH and
        # the user only discovered it broken hours later.
        for _try in 1 2 3; do
            log "Installing Ollama (attempt ${_try}/3)..."
            curl -fsSL https://ollama.com/install.sh | sh || \
                log "  install attempt ${_try} returned non-zero"
            if command -v ollama >/dev/null 2>&1; then
                log "Ollama installed: $(ollama --version 2>&1 | head -1)"
                break
            fi
            log "  ollama binary not found after attempt ${_try}; retrying in 5s..."
            sleep 5
        done
        # Final check — emit a LOUD warning if still missing so it's
        # obvious in the log rather than buried.
        if ! command -v ollama >/dev/null 2>&1; then
            cat <<EOF >&2

============================================================================
==                                                                        ==
==   WARNING: Ollama install FAILED after 3 attempts.                     ==
==                                                                        ==
==   Bootstrap will continue but ollama-dependent nodes will not work.   ==
==   To install manually after bootstrap finishes, SSH in and run:        ==
==     curl -fsSL https://ollama.com/install.sh | sh                      ==
==                                                                        ==
============================================================================
EOF
            log "WARN: ollama install failed after 3 attempts — see banner above"
        fi
    else
        log "Ollama already installed: $(ollama --version 2>&1 | head -1)"
    fi
    mkdir -p "$OLLAMA_MODELS"

    # Validate GPU support. Official installer can drop the CPU-only
    # payload on RunPod (no cuda_v* lib dir). With a GPU present and no
    # cuda libs, force a reinstall so we get the CUDA build.
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        if ! ls /usr/local/lib/ollama/cuda_v* >/dev/null 2>&1; then
            log "ollama: GPU detected but cuda_v* libs missing — reinstalling for GPU support"
            pkill -9 -f "ollama serve" 2>/dev/null || true
            sleep 2
            rm -rf /usr/local/bin/ollama /usr/local/lib/ollama
            curl -fsSL https://ollama.com/install.sh | sh 2>&1 | sed 's/^/[ollama-reinstall] /' || true
            if ls /usr/local/lib/ollama/cuda_v* >/dev/null 2>&1; then
                log "ollama: GPU build installed"
            else
                log "ollama: WARN — reinstall completed but cuda_v* libs still missing; daemon will run on CPU"
            fi
        else
            log "ollama: GPU build verified"
        fi
    fi

    if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        log "Ollama server already running on $OLLAMA_HOST"
    else
        log "Starting Ollama server (OLLAMA_MODELS=$OLLAMA_MODELS, NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES)"
        nohup env OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" \
            NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES" \
            ollama serve > /workspace/ollama.log 2>&1 &
        for i in $(seq 1 30); do
            curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && break
            sleep 2
        done
        if ! curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
            log "WARN: Ollama server didn't come up. See /workspace/ollama.log"
        fi
    fi

    for model in $OLLAMA_MODEL; do
        if ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$model"; then
            log "$model already cached"
        else
            log "Pulling $model..."
            ollama pull "$model" || log "  WARN: pull failed for $model"
        fi
    done
else
    banner "Phase 5/7  Ollama (skipped, RS_INSTALL_OLLAMA=0)"
fi

# -----------------------------------------------------------------------------
# Phase 6 — Summary
# -----------------------------------------------------------------------------
banner "Phase 6/7  Summary"
log "ComfyUI:        $COMFY_DIR ($(git -C "$COMFY_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
log "rs-nodes:       $RS_NODES_DIR ($(git -C "$RS_NODES_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
log "Custom nodes installed:"
ls -1 "$CUSTOM_NODES_DIR" | sed 's/^/    /'
log "Models on volume:"
du -sh "$MODELS_ROOT"/* 2>/dev/null | sed 's/^/    /' || true
log "Disk usage on /workspace:"
df -h /workspace | tail -1 | sed 's/^/    /'

# Mark bootstrap complete so init.sh's fast path will pick startup.sh
# on subsequent boots. Without this marker init.sh re-runs bootstrap.sh
# (which is fine since it's idempotent, but slower than startup.sh).
touch "$WORKSPACE/.bootstrap_done"
log "Bootstrap complete marker written: $WORKSPACE/.bootstrap_done"

# -----------------------------------------------------------------------------
# Phase 7 — Launch ComfyUI
# -----------------------------------------------------------------------------
if [ "$RS_LAUNCH_COMFY" = "1" ]; then
    # --highvram keeps weights GPU-resident; huge speedup on a 96 GB
    # Blackwell card vs the default lazy offload. Override via
    # COMFY_EXTRA_ARGS env var.
    COMFY_EXTRA_ARGS="${COMFY_EXTRA_ARGS-}"
    if [ "$COMFY_EXTRA_ARGS" = "NONE" ] || [ "$COMFY_EXTRA_ARGS" = "none" ]; then
        COMFY_EXTRA_ARGS=""
    fi
    banner "Phase 7/7  Launching ComfyUI on 0.0.0.0:${PORT} (venv: $VENV, args: $COMFY_EXTRA_ARGS)"
    cd "$COMFY_DIR"
    exec "$VENV/bin/python" main.py --listen 0.0.0.0 --port "$PORT" $COMFY_EXTRA_ARGS
else
    banner "Phase 7/7  Launch skipped (RS_LAUNCH_COMFY=0)"
    log "Run manually:  cd $COMFY_DIR && python main.py --listen 0.0.0.0 --port $PORT"
fi
