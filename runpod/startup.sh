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

# Idempotent: start ollama in a fully-detached session if it's not
# already running. Safe to call every time startup.sh runs.
#
# Why setsid + nohup + </dev/null: ollama serve must survive both
# (a) the parent shell exiting normally and (b) the parent shell being
# killed (e.g. user kills ComfyUI to trigger a respring). nohup alone
# only catches SIGHUP — process-group signals from a parent death can
# still reach the child. setsid creates a brand-new session, so ollama
# is in its own process group with no controlling terminal. </dev/null
# closes stdin in case anything blocks on reading from it.
ensure_ollama_running() {
    [ "${RS_INSTALL_OLLAMA:-1}" = "1" ] || return 0
    command -v ollama >/dev/null 2>&1 || return 0
    if pgrep -f "ollama serve" >/dev/null 2>&1; then
        return 0  # already running
    fi
    export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
    export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
    mkdir -p "$OLLAMA_MODELS"
    log "Ollama not running — starting fully-detached..."
    setsid nohup env OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" \
        ollama serve >>/workspace/ollama.log 2>&1 </dev/null &
    disown 2>/dev/null || true
}

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
# Torch / host-driver alignment — runs on EVERY boot regardless of
# fast-path / express-path. Catches RunPod scheduler roulette where
# /workspace was provisioned on one host's driver but the current pod
# landed on a different host:
#
#   cu130 torch on a CUDA 12.x driver  → torch.cuda.is_available() fails
#   cu128 torch on a CUDA 13.x driver  → works but loses Blackwell perf
#
# Bidirectional heal:
#   * Driver >= CUDA 13 AND torch is cu128 → UPGRADE to cu130 (Blackwell perf)
#   * Driver <  CUDA 13 AND torch is cu130 → DOWNGRADE to cu128 (so it runs)
#   * Driver matches torch → no-op
#
# cu130 is the preferred install; cu128 is fallback only. We always
# heal toward the optimal wheel for whichever host we landed on.
# -----------------------------------------------------------------------------
if [ -f "$VENV/bin/python" ]; then
    # Parse host driver's CUDA version
    _smi_header=$(nvidia-smi 2>/dev/null | grep -E 'CUDA Version' | head -1 || echo "")
    _drv_cuda=$(echo "$_smi_header" | grep -oE 'CUDA Version:[[:space:]]*[0-9]+' | grep -oE '[0-9]+$')
    # Installed torch version
    _torch_cuda=$("$VENV/bin/python" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null | tr -d '.')

    _want_torch=""
    if [ -n "$_drv_cuda" ]; then
        if [ "$_drv_cuda" -ge 13 ] 2>/dev/null; then
            _want_torch="130"
        else
            _want_torch="128"
        fi
    fi

    if [ -n "$_want_torch" ] && [ -n "$_torch_cuda" ] && [ "$_torch_cuda" != "$_want_torch" ]; then
        # Mismatch: heal in the appropriate direction.
        if [ "$_want_torch" = "130" ]; then
            log "Host supports CUDA ${_drv_cuda}.x but torch is cu${_torch_cuda} — upgrading to cu130 for full Blackwell perf..."
        else
            log "Host driver maxes at CUDA ${_drv_cuda}.x but torch is cu${_torch_cuda} — downgrading to cu128 so it actually runs..."
        fi
        "$VENV/bin/pip" install --upgrade --no-cache-dir \
            --index-url "https://download.pytorch.org/whl/cu${_want_torch}" \
            torch torchvision torchaudio || \
            log "  WARN: cu${_want_torch} install failed"
        if "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            log "  Torch CUDA confirmed working (cu${_want_torch} on CUDA ${_drv_cuda}.x driver)"
        else
            log "  WARN: torch still can't init CUDA after cu${_want_torch} install"
        fi
    elif [ -n "$_drv_cuda" ] && [ -n "$_torch_cuda" ]; then
        # Matched. Quick sanity check that torch actually initializes.
        if "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            log "Torch cu${_torch_cuda} aligned with CUDA ${_drv_cuda}.x driver — OK"
        else
            log "WARN: torch cu${_torch_cuda} matches driver CUDA ${_drv_cuda}.x but CUDA init still failed (other root cause)"
        fi
    fi
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

    # Ensure ollama is running (fully detached so it survives comfy
    # process death — e.g. user killing main.py to respring).
    ensure_ollama_running

    cd "$COMFY_DIR"
    COMFY_EXTRA_ARGS="${COMFY_EXTRA_ARGS-}"
# Sentinel: COMFY_EXTRA_ARGS=NONE / none means "no flags at all" for
# UIs that don't accept empty values. Translate to empty string so the
# exec line below adds no args.
if [ "$COMFY_EXTRA_ARGS" = "NONE" ] || [ "$COMFY_EXTRA_ARGS" = "none" ]; then
    COMFY_EXTRA_ARGS=""
fi
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
# 4. InsightFace stack (used by face-aware ComfyUI nodes — IPAdapter
#    FaceID, ReActor, several SAM3 face workflows, etc.)
#
#    Three parts, all idempotent — re-running this section costs ~2s
#    when everything is already in place:
#      (a) libgl1 / libglib2.0-0    system libs OpenCV pulls in;
#                                   without them `import cv2` (which
#                                   insightface does eagerly) explodes.
#      (b) insightface + onnxruntime-gpu  the Python packages.
#                                   Default `onnxruntime` is CPU only;
#                                   we explicitly install the GPU build
#                                   so face models can run on CUDA. The
#                                   `-gpu` wheel must match the CUDA
#                                   major version of torch — the pip
#                                   resolver picks the right one because
#                                   we're on torch cu130 by this point.
#      (c) Prefetch antelopev2      pre-downloads ~280 MB of face
#                                   models to ~/.insightface so the
#                                   first node load doesn't stall on a
#                                   silent CDN fetch. Failing here is
#                                   non-fatal — node will retry on use.
#
#    Was previously gated behind RS_PREFETCH_INSIGHTFACE=1, but skipping
#    the install left every pod broken until manually fixed. Always run.
# -----------------------------------------------------------------------------
log "Section 4: InsightFace stack"

# 4a. System libs (apt). libgl1 is the libGL.so.1 OpenCV needs;
# libglib2.0-0 provides libgthread which some cv2 builds also pull in.
# Skip if both are already present — fast (<200ms) when no-op.
if ! ldconfig -p | grep -q 'libGL.so.1' || ! ldconfig -p | grep -q 'libgthread-2.0'; then
    log "  installing libgl1 libglib2.0-0..."
    DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        libgl1 libglib2.0-0 >/dev/null 2>&1 || \
        log "  WARN: apt install of libgl1/libglib2.0-0 failed"
else
    log "  libgl1 + libglib2.0-0 already present"
fi

# 4b. Python packages — onnxruntime-gpu MUST install cleanly with the
# CUDA EP available. The default `pip install onnxruntime-gpu` ships
# the CUDA 12.x wheel by default (as of late 2024), but if the CPU
# `onnxruntime` was previously installed (sometimes pulled in
# transitively by other packages), it shadows the GPU build and the
# CUDA provider silently disappears. So:
#   1. Force-remove any existing onnxruntime (CPU or GPU) to start
#      from a clean slate. -y so we don't prompt; || true so the
#      first-boot case (neither installed) doesn't fail the script.
#   2. Install onnxruntime-gpu fresh.
#   3. Probe providers. If CUDAExecutionProvider is missing, retry
#      with the Azure CUDA-12 index URL (the canonical CUDA-12-only
#      wheel source — covers the case where PyPI's default wheel was
#      built against a CUDA version the host doesn't match).

# Step 1: clean slate.
log "  removing any existing onnxruntime installs (clean reinstall)..."
pip uninstall -y onnxruntime onnxruntime-gpu 2>&1 | tail -3 || true

# Step 2: install GPU wheel + insightface.
log "  installing insightface + onnxruntime-gpu (default PyPI wheel)..."
pip install --no-cache-dir insightface onnxruntime-gpu || \
    log "  WARN: insightface/onnxruntime-gpu install failed"

# Step 3: verify CUDA EP. If missing, retry from the CUDA-12 index.
if ! python -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()" 2>/dev/null; then
    log "  CUDAExecutionProvider missing after default install — retrying from CUDA-12 index..."
    pip uninstall -y onnxruntime-gpu 2>&1 | tail -3 || true
    pip install --no-cache-dir onnxruntime-gpu \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ || \
        log "  WARN: onnxruntime-gpu CUDA-12 index install failed"

    if ! python -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()" 2>/dev/null; then
        log "  WARN: CUDAExecutionProvider STILL missing after CUDA-12 retry."
        log "        Face detection will fall back to CPU (slow but functional)."
        log "        Run: python -c \"import onnxruntime; print(onnxruntime.get_available_providers())\""
        log "        to diagnose. Likely cause: torch CUDA version vs onnxruntime CUDA build mismatch."
    else
        log "  CUDAExecutionProvider now available (CUDA-12 wheel)."
    fi
else
    log "  CUDAExecutionProvider available (default wheel)."
fi

# 4c. Prefetch the antelopev2 face model bundle (~280 MB). Uses
# CPUExecutionProvider for the prefetch since the GPU session has a
# warmup cost we don't need here. Actual node usage picks GPU at
# runtime. If a prior prefetch left a partial/corrupt download in
# ~/.insightface/, blow it away first so re-runs heal.
INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-$HOME/.insightface}"
if [ -d "$INSIGHTFACE_HOME/models/antelopev2" ]; then
    # Validate by counting expected .onnx files — antelopev2 ships 5.
    _onnx_count=$(find "$INSIGHTFACE_HOME/models/antelopev2" -name "*.onnx" 2>/dev/null | wc -l)
    if [ "$_onnx_count" -lt 5 ]; then
        log "  partial antelopev2 install ($_onnx_count/5 onnx files) — wiping for re-download"
        rm -rf "$INSIGHTFACE_HOME/models/antelopev2"
    fi
fi
log "  prefetching antelopev2 face models..."
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])" 2>&1 | tail -5 || \
    log "  WARN: InsightFace prefetch failed (will retry on first use)"

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
    log "Starting Ollama (idempotent, fully detached)..."
    ensure_ollama_running

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
COMFY_EXTRA_ARGS="${COMFY_EXTRA_ARGS-}"
# Sentinel: COMFY_EXTRA_ARGS=NONE / none means "no flags at all" for
# UIs that don't accept empty values. Translate to empty string so the
# exec line below adds no args.
if [ "$COMFY_EXTRA_ARGS" = "NONE" ] || [ "$COMFY_EXTRA_ARGS" = "none" ]; then
    COMFY_EXTRA_ARGS=""
fi
log "Launching ComfyUI on 0.0.0.0:${PORT}  (venv: $VENV, args: $COMFY_EXTRA_ARGS)"
cd "$COMFY_DIR"
exec "$VENV/bin/python" main.py --listen 0.0.0.0 --port "$PORT" $COMFY_EXTRA_ARGS "$@"
