#!/usr/bin/env bash
# rs-nodes pod-side bootstrap
# -----------------------------------------------------------------------------
# Goal: bring a freshly-booted RunPod container to a working ComfyUI on
# port 8188 with rs-nodes installed, idempotently. State lives on the
# network volume mounted at /workspace, so subsequent boots are fast
# (just a git pull + a venv check) and don't redownload models.
#
# Wire this script in as the pod template's "Container Start Command":
#   bash /workspace/startup.sh
# Place a copy at /workspace/startup.sh once during initial volume setup
# (see runpod/README.md for the one-time provisioning steps).

set -euo pipefail

WORKSPACE=/workspace
COMFY_DIR="$WORKSPACE/ComfyUI"
RS_NODES_DIR="$COMFY_DIR/custom_nodes/rs-nodes"
VENV="$WORKSPACE/.venv"
PORT="${COMFY_PORT:-8188}"
LOG_FILE="${COMFY_LOG:-/workspace/comfyui.log}"

# Ubuntu 24.04 (PEP 668) marks the system Python as externally-managed.
# We do all installs into a venv on the network volume instead, so
# state persists across container resets (no re-installing torch every
# boot) AND we sidestep PEP 668 entirely. The flag stays set as a
# fallback for any pip that escapes the venv (e.g. via sudo).
export PIP_BREAK_SYSTEM_PACKAGES=1

# Tee everything to a log file so external shells (e.g. launch.bat
# tailing from Windows) can see startup + ComfyUI output even when
# the script runs as the container's start command (no attached TTY).
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { printf '[startup] %s\n' "$*"; }

log "rs-nodes pod startup beginning at $(date -Is)"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# -----------------------------------------------------------------------------
# 0. DNS guard — some RunPod regions (observed: EU-CZ-1) ship containers
#    with an empty /etc/resolv.conf, so even `git clone github.com` fails.
#    /etc/resolv.conf lives on the container disk, not the volume, so this
#    has to run on every boot. We only inject if the existing file is empty
#    or missing; pods with working DNS are left alone.
# -----------------------------------------------------------------------------
if ! getent hosts github.com >/dev/null 2>&1; then
    log "DNS resolution broken; injecting public resolvers into /etc/resolv.conf"
    {
        echo "nameserver 8.8.8.8"
        echo "nameserver 1.1.1.1"
    } > /etc/resolv.conf || log "WARN: could not write /etc/resolv.conf"
fi

# -----------------------------------------------------------------------------
# 0.4. Ensure sshd has host keys and is running.
#      Container Start Command replaces RunPod's stock init, which is
#      what would normally generate /etc/ssh/ssh_host_*_key on first
#      boot. Without those keys sshd refuses to start ("no hostkeys
#      available -- exiting"). We persist the host keys to the volume
#      so SSH clients don't get spammed with "host key changed"
#      warnings on every container reset.
# -----------------------------------------------------------------------------
mkdir -p /workspace/.ssh/host_keys /etc/ssh
if compgen -G "/workspace/.ssh/host_keys/ssh_host_*" > /dev/null; then
    log "Restoring sshd host keys from /workspace/.ssh/host_keys"
    cp -f /workspace/.ssh/host_keys/ssh_host_* /etc/ssh/
    chmod 600 /etc/ssh/ssh_host_*_key 2>/dev/null || true
    chmod 644 /etc/ssh/ssh_host_*_key.pub 2>/dev/null || true
elif compgen -G "/etc/ssh/ssh_host_*" > /dev/null; then
    log "Persisting existing sshd host keys to /workspace/.ssh/host_keys"
    cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
else
    log "No sshd host keys anywhere; generating fresh set with ssh-keygen -A"
    ssh-keygen -A
    cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
fi

# Start (or restart) sshd. service may or may not exist depending on
# the base image; fall back to direct sshd invocation.
if command -v service >/dev/null 2>&1; then
    service ssh restart 2>&1 | sed 's/^/[sshd] /' || true
elif [ -x /usr/sbin/sshd ]; then
    pkill -f /usr/sbin/sshd 2>/dev/null || true
    /usr/sbin/sshd
    log "sshd started directly"
else
    log "WARN: no sshd binary found; SSH will be unavailable on this container"
fi

# -----------------------------------------------------------------------------
# 0.5. Persist SSH authorized_keys on the network volume.
#      RunPod's stock auto-injection only fires on first container
#      creation; template changes / re-deploys produce a fresh
#      container with empty /root/.ssh and force a manual re-add.
#      We mirror authorized_keys to /workspace/.ssh/ so subsequent
#      boots always have the key, regardless of what spawned the
#      container. First-boot-after-manual-add direction is also
#      covered: if /root/.ssh has a key but the volume doesn't, we
#      persist UP to the volume.
# -----------------------------------------------------------------------------
mkdir -p /workspace/.ssh /root/.ssh
chmod 700 /workspace/.ssh /root/.ssh
if [ -s /workspace/.ssh/authorized_keys ]; then
    log "Restoring SSH authorized_keys from /workspace/.ssh"
    cp /workspace/.ssh/authorized_keys /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
elif [ -s /root/.ssh/authorized_keys ]; then
    log "First-time persist of SSH authorized_keys -> /workspace/.ssh"
    cp /root/.ssh/authorized_keys /workspace/.ssh/authorized_keys
    chmod 600 /workspace/.ssh/authorized_keys
else
    log "WARN: no authorized_keys found anywhere — SSH key auth will fail"
fi

# -----------------------------------------------------------------------------
# 1+2. ComfyUI + rs-nodes — clone if missing, pull on every boot.
#      Capture pre/post HEADs so we can detect whether anything actually
#      changed. If nothing changed, the express path below skips every
#      dependency check and launches ComfyUI in seconds.
# -----------------------------------------------------------------------------
if [ ! -d "$COMFY_DIR/.git" ]; then
    log "Cloning ComfyUI into $COMFY_DIR ..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
    COMFY_PRE=""
else
    COMFY_PRE=$(git -C "$COMFY_DIR" rev-parse HEAD 2>/dev/null || echo "")
    log "ComfyUI: pulling latest..."
    git -C "$COMFY_DIR" pull --ff-only 2>&1 | sed 's/^/  /' || \
        log "WARN: ComfyUI pull failed; continuing with current revision"
fi
COMFY_POST=$(git -C "$COMFY_DIR" rev-parse HEAD 2>/dev/null || echo "")

if [ ! -d "$RS_NODES_DIR/.git" ]; then
    log "Cloning rs-nodes into $RS_NODES_DIR ..."
    git clone https://github.com/richservo/rs-nodes.git "$RS_NODES_DIR"
    RS_PRE=""
else
    RS_PRE=$(git -C "$RS_NODES_DIR" rev-parse HEAD 2>/dev/null || echo "")
    log "rs-nodes: pulling latest..."
    git -C "$RS_NODES_DIR" pull --ff-only 2>&1 | sed 's/^/  /' || \
        log "WARN: rs-nodes pull failed; continuing with current revision"
fi
RS_POST=$(git -C "$RS_NODES_DIR" rev-parse HEAD 2>/dev/null || echo "")
git -C "$RS_NODES_DIR" submodule update --init --recursive 2>&1 | sed 's/^/  /' || \
    log "WARN: submodule update failed"

# Mirror pod-side helper scripts so Container Start Command + manual runs
# find them at the documented paths even on a fresh container disk.
if [ -d "$RS_NODES_DIR/runpod" ]; then
    cp -f "$RS_NODES_DIR/runpod/"*.sh /workspace/ 2>/dev/null || true
    chmod +x /workspace/*.sh 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# Express path — git tells us the truth about "did anything change?"
# If neither ComfyUI nor rs-nodes moved AND all setup markers are
# present, there is literally nothing to do except launch ComfyUI.
# Skip every dependency check, every pip call, every ollama wait.
#
# Activate venv, compute LD_LIBRARY_PATH (env vars don't survive
# container restarts), spawn ollama in background, exec ComfyUI.
# -----------------------------------------------------------------------------
if [ -n "$COMFY_PRE" ] && [ -n "$RS_PRE" ] && \
   [ "$COMFY_PRE" = "$COMFY_POST" ] && [ "$RS_PRE" = "$RS_POST" ] && \
   [ -f "$VENV/bin/python" ] && \
   [ -f "$WORKSPACE/.framework_installed" ] && \
   [ -f "$WORKSPACE/.provision_hash" ] && \
   [ -f "$WORKSPACE/.ollama_ready" ]; then
    log "Express path: no git updates + all markers present. Launching ComfyUI immediately."
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"

    # LD_LIBRARY_PATH must be recomputed every boot (env vars don't persist).
    NV_LIB_PATHS=$(python -c "
import nvidia, os
r = os.path.dirname(nvidia.__file__)
print(':'.join(os.path.join(r, d, 'lib') for d in sorted(os.listdir(r))
               if os.path.isdir(os.path.join(r, d, 'lib'))))
" 2>/dev/null || echo "")
    [ -n "$NV_LIB_PATHS" ] && export LD_LIBRARY_PATH="$NV_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # Spawn ollama in background (best effort)
    if [ "${RS_INSTALL_OLLAMA:-1}" = "1" ] && command -v ollama >/dev/null 2>&1; then
        export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
        export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
        nohup env OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" \
            ollama serve >/workspace/ollama.log 2>&1 &
    fi

    cd "$COMFY_DIR"
    COMFY_EXTRA_ARGS="${COMFY_EXTRA_ARGS:---highvram}"
    log "Launching ComfyUI on 0.0.0.0:${PORT}  (express, args: $COMFY_EXTRA_ARGS)"
    exec "$VENV/bin/python" main.py --listen 0.0.0.0 --port "$PORT" $COMFY_EXTRA_ARGS "$@"
fi

log "Updates or missing markers detected — running full setup pass."

# -----------------------------------------------------------------------------
# 2.6. Persistent venv on the network volume
#      Without this, every container reset re-downloads + reinstalls
#      every pip package (torch alone is ~3 GB). The venv lives on
#      /workspace so it survives Stop+Start / migrate / terminate.
#      --system-site-packages lets us inherit the base image's CUDA /
#      system libs while letting us override Python packages (torch,
#      sageattention, etc.) with newer versions installed into the venv.
# -----------------------------------------------------------------------------
if [ ! -f "$VENV/bin/python" ]; then
    log "Creating venv at $VENV (one-time, ~30s)"
    python3 -m venv --system-site-packages "$VENV"
fi
log "Activating venv at $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
log "Python: $(which python)  ($(python --version 2>&1))"
log "Pip:    $(which pip)"

PYSITE="$VENV/lib/python3.12/site-packages"

# -----------------------------------------------------------------------------
# 2.7. Self-healing checks — run on EVERY boot regardless of fast path.
#      These catch corruption from partial pip installs (SSH drop / OOM mid-
#      install) and version drift from older bootstraps. Cheap if everything
#      is already correct (pip says "already satisfied" in <2s).
# -----------------------------------------------------------------------------

# Numpy metadata heal: a partial pip install can leave a numpy-X.Y.dist-info
# directory with malformed METADATA, so importlib.metadata.version("numpy")
# returns None, which then breaks transformers' dependency check at import.
# Detect + force-reinstall if broken.
if ! python -c "from importlib.metadata import version; v = version('numpy'); assert v and v != 'None'" 2>/dev/null; then
    log "numpy metadata broken or missing — force-reinstalling..."
    rm -rf "$PYSITE"/numpy-*.dist-info 2>/dev/null || true
    pip install --no-cache-dir --force-reinstall numpy || \
        log "WARN: numpy reinstall failed"
fi

# huggingface_hub version pin: transformers 4.57+ declares hub<1.0 as a
# strict requirement. Older bootstrap revs installed hf-transfer with
# `pip -U huggingface_hub[hf_transfer]` which pulled 1.x and broke
# transformers. Idempotent — pip says "Requirement already satisfied"
# in ~1s if version is already correct.
pip install --no-cache-dir "huggingface_hub[hf_transfer]<1.0,>=0.34" 2>&1 | tail -3

# -----------------------------------------------------------------------------
# 3. Python deps — fast path on warm boots.
#
# On a fresh volume, every pip install runs (slow first boot). On every
# subsequent boot, a marker file at /workspace/.provision_hash holds the
# sha256 of the requirements.txt files. If the hash matches, skip every
# pip install — saves 30-90s per boot. If requirements.txt changes
# (because rs-nodes adds a dep, etc.), the hash mismatches and pip runs
# normally to install the new pieces.
#
# To force a full re-provision (e.g. to pick up a new torch wheel from
# the cu130 index), delete the marker:
#     rm /workspace/.provision_hash
# Then restart the container (kill.bat or Stop+Start).
# -----------------------------------------------------------------------------
PROVISION_MARKER="$WORKSPACE/.provision_hash"
REQ_HASH=$(cat "$COMFY_DIR/requirements.txt" "$RS_NODES_DIR/requirements.txt" 2>/dev/null | sha256sum | cut -d' ' -f1)
STORED_HASH=""
[ -f "$PROVISION_MARKER" ] && STORED_HASH=$(cat "$PROVISION_MARKER" 2>/dev/null)

if [ -n "$REQ_HASH" ] && [ "$REQ_HASH" = "$STORED_HASH" ]; then
    FAST_PATH=1
    log "Provision marker matches — skipping pip install steps (fast path)"
else
    FAST_PATH=0
    if [ -z "$STORED_HASH" ]; then
        log "No provision marker — running full pip provisioning"
    else
        log "Requirements changed since last boot — re-running pip"
    fi
fi

# -----------------------------------------------------------------------------
# Framework deps (torch cu130 + NVIDIA libs + sageattention) — install ONCE
# per fresh volume, never again. Even an "Already satisfied" `pip install
# --upgrade torch` against the cu130 index takes 60-120s for the network
# round-trip; running it on every boot just because rs-nodes added a node
# burns minutes for no reason.
#
# Marker: /workspace/.framework_installed. Auto-detect existing good
# installs (writes marker without reinstall). To force a fresh framework
# install (e.g. moving to a newer torch wheel), delete the marker and
# restart.
# -----------------------------------------------------------------------------
FRAMEWORK_MARKER="$WORKSPACE/.framework_installed"
if [ ! -f "$FRAMEWORK_MARKER" ]; then
    # Detect: torch at cu130 + sageattention importable = framework is good,
    # just missing the marker (e.g. installed by older bootstrap). Skip the
    # reinstall and write the marker so next boot fast-paths.
    if python -c "
import sys, torch
import sageattention  # noqa
assert 'cu130' in torch.__version__, f'torch is {torch.__version__}, not cu130'
" 2>/dev/null; then
        log "Framework already installed (torch cu130 + sageattention detected); writing marker."
        touch "$FRAMEWORK_MARKER"
    else
        log "Installing framework stack (one-time, ~60-120s)..."
        log "  PyTorch cu130 wheels..."
        pip install --upgrade --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cu130 \
            torch torchvision torchaudio || \
            log "WARN: PyTorch cu130 upgrade failed; running on stock cu128"

        log "  NVIDIA CUDA runtime libraries (NVRTC + cuDNN + cuBLAS)..."
        pip install --no-cache-dir \
            nvidia-cuda-nvrtc \
            nvidia-cuda-runtime \
            nvidia-cublas \
            nvidia-cudnn || \
            log "WARN: CUDA runtime libs install failed; JIT kernels may crash"

        log "  SageAttention..."
        pip install --no-cache-dir sageattention || \
            log "WARN: SageAttention install failed; using stock PyTorch attention"

        touch "$FRAMEWORK_MARKER"
        log "Framework marker written: $FRAMEWORK_MARKER"
    fi
fi

# App requirements (ComfyUI + rs-nodes + rose-opt) — hash-gated. Re-run
# only when one of the requirements.txt files actually changes.
if [ "$FAST_PATH" != "1" ]; then
    log "Installing ComfyUI Python deps..."
    pip install --no-cache-dir -r "$COMFY_DIR/requirements.txt" || \
        log "WARN: ComfyUI deps install failed"

    log "Installing rs-nodes Python deps..."
    if [ -f "$RS_NODES_DIR/requirements.txt" ]; then
        pip install --no-cache-dir -r "$RS_NODES_DIR/requirements.txt" || \
            log "WARN: rs-nodes deps install failed"
    fi

    # ROSE optimizer — published as rose-opt, imported as rose_opt
    log "Ensuring ROSE optimizer..."
    pip install --no-cache-dir rose-opt || \
        log "WARN: ROSE install failed (only needed for training)"
fi

# LD_LIBRARY_PATH must be computed every boot — env vars don't persist
# across container restarts, even though the underlying nvidia/* lib
# dirs do. Stays outside the fast-path skip.
#
# pip-installed NVIDIA libs land in venv site-packages/nvidia/*/lib/.
# PyTorch's nvrtc dlopen() searches LD_LIBRARY_PATH at runtime, so
# every nvidia/* lib dir has to be visible. Compute and persist it
# into the venv's activate script so any shell that activates the
# venv (including the exec at the end of this script) inherits it.
NV_LIB_PATHS=$(python -c "
import nvidia, os
r = os.path.dirname(nvidia.__file__)
paths = []
for d in sorted(os.listdir(r)):
    lib_dir = os.path.join(r, d, 'lib')
    if os.path.isdir(lib_dir):
        paths.append(lib_dir)
print(':'.join(paths))
" 2>/dev/null || echo "")
if [ -n "$NV_LIB_PATHS" ]; then
    log "NVIDIA lib paths: $NV_LIB_PATHS"
    export LD_LIBRARY_PATH="$NV_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ "$FAST_PATH" != "1" ]; then
    log "Ensuring SageAttention..."
    pip install --no-cache-dir sageattention || \
        log "WARN: SageAttention install failed; using stock PyTorch attention"

    # All pip blocks succeeded — write the provision marker so next
    # boot can fast-path.
    if [ -n "$REQ_HASH" ]; then
        echo "$REQ_HASH" > "$PROVISION_MARKER"
        log "Wrote provision marker: $PROVISION_MARKER"
    fi
fi

# -----------------------------------------------------------------------------
# 4. Optional InsightFace pre-download (gated by env var so the dispatch
#    path that doesn't need it doesn't pay the bandwidth)
# -----------------------------------------------------------------------------
if [ "${RS_PREFETCH_INSIGHTFACE:-0}" = "1" ]; then
    log "Pre-downloading InsightFace antelopev2..."
    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])" || \
        log "WARN: InsightFace prefetch failed (will retry on first use)"
fi

# -----------------------------------------------------------------------------
# 5. Ollama install + serve (default ON — RSPromptFormatter requires it,
#    so it's load-bearing for the standard workflows). Set RS_INSTALL_OLLAMA=0
#    in pod env vars to skip if you ever want a slimmer boot.
# -----------------------------------------------------------------------------
if [ "${RS_INSTALL_OLLAMA:-1}" = "1" ]; then
    if ! command -v ollama >/dev/null 2>&1; then
        log "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh || \
            log "WARN: Ollama install failed"
    fi
    # Models live on the network volume so they survive pod terminate.
    export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
    export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
    mkdir -p "$OLLAMA_MODELS"
    log "Starting Ollama in the background..."
    nohup env OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" \
        ollama serve >/workspace/ollama.log 2>&1 &

    # Ollama models marker — only do first-time wait+pull if the marker
    # is missing AND the models aren't already on disk. Auto-detects
    # pre-existing model directories on the volume (from a previous
    # bootstrap that didn't write the marker) and writes the marker
    # without re-pulling.
    OLLAMA_READY_MARKER="$WORKSPACE/.ollama_ready"
    OLLAMA_MODEL_LIST="${OLLAMA_MODEL:-gemma4:31b gemma4:26b}"
    if [ ! -f "$OLLAMA_READY_MARKER" ]; then
        # Quick filesystem check: do the model blob dirs exist on the volume?
        # If so, no need to wait+pull; just write the marker.
        ALL_PRESENT=1
        for model in $OLLAMA_MODEL_LIST; do
            # ollama stores model manifest under ~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>
            name="${model%%:*}"
            tag="${model##*:}"
            if [ ! -f "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/$name/$tag" ]; then
                ALL_PRESENT=0
                break
            fi
        done
        if [ "$ALL_PRESENT" = "1" ]; then
            log "Ollama models already on volume; writing ready marker without pull."
            touch "$OLLAMA_READY_MARKER"
        else
            log "First-time ollama setup — waiting for server then pulling: $OLLAMA_MODEL_LIST"
            for i in $(seq 1 15); do
                curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1 && break
                sleep 2
            done
            for model in $OLLAMA_MODEL_LIST; do
                if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$model"; then
                    log "Pulling Ollama model: $model"
                    ollama pull "$model" || log "WARN: pull failed for $model"
                fi
            done
            touch "$OLLAMA_READY_MARKER"
            log "Wrote ollama ready marker."
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 6. Launch ComfyUI on the public port
# -----------------------------------------------------------------------------
# --highvram keeps loaded weights resident on the GPU instead of
# offloading to CPU between operations. On a 96 GB Blackwell card
# the LTX-2.3 22B fp8 (~29 GB) + gemma fp4 text encoder (~9 GB) +
# audio_vae easily fit; eliminating the CPU↔GPU ping-pong can be
# 2-5x faster after the first warmup. Override via COMFY_EXTRA_ARGS
# env var (e.g. "" to disable, or "--gpu-only" for max-aggressive).
COMFY_EXTRA_ARGS="${COMFY_EXTRA_ARGS:---highvram}"
log "Launching ComfyUI on 0.0.0.0:${PORT}  (venv: $VENV, args: $COMFY_EXTRA_ARGS)"
cd "$COMFY_DIR"
exec "$VENV/bin/python" main.py --listen 0.0.0.0 --port "$PORT" $COMFY_EXTRA_ARGS "$@"
