#!/usr/bin/env bash
# rs-nodes pod-side startup. Three modes selected by env vars:
#
#   SETUP=TRUE     → full re-bootstrap: fetch bootstrap.sh fresh from
#                    GitHub and exec it. Use on new pods or when you
#                    want a clean reprovision.
#   RS_NODES=TRUE  → git pull rs-nodes, mirror runpod/*.sh into
#                    /workspace via atomic rename, then launch ComfyUI.
#                    No other setup.
#   both unset/FALSE → ensure ollama is running, then exec ComfyUI.
#                    Nothing else. Target: very fast.
#
# DNS + sshd are handled in init.sh (always run), so this script can be
# truly lean in fast modes.
#
# Truthy values: TRUE / true / 1 / yes / on (case-insensitive).

set -euo pipefail

WORKSPACE=/workspace
COMFY_DIR="$WORKSPACE/ComfyUI"
RS_NODES_DIR="$COMFY_DIR/custom_nodes/rs-nodes"
VENV="$WORKSPACE/.venv"
PORT="${COMFY_PORT:-8188}"
LOG_FILE="${COMFY_LOG:-/workspace/comfyui.log}"
BOOTSTRAP_URL="https://raw.githubusercontent.com/richservo/rs-nodes/master/runpod/bootstrap.sh"

export PIP_BREAK_SYSTEM_PACKAGES=1

# Pin JIT + HF caches to the network volume so they survive container
# restarts. Default locations are under ~/.cache (container disk) and
# get wiped every restart, forcing Triton/Inductor/comfy_kitchen to
# recompile CUDA kernels (~2 min) and HF to re-download tokenizers and
# configs on every boot. Pinning to /workspace makes 2nd+ boots fast.
mkdir -p /workspace/.cache/triton \
         /workspace/.cache/torch_inductor \
         /workspace/.cache/huggingface
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/torch_inductor}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { printf '[startup] %s\n' "$*"; }

_is_truthy() {
    case "${1,,}" in
        true|1|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

ensure_ollama_running() {
    # Never propagate failure — ollama issues should not abort ComfyUI
    # launch. Errors are logged loudly; check the log to diagnose.
    _ensure_ollama_inner || log "ollama: continuing without ollama (RSPromptFormatter will fail)"
    return 0
}

_ensure_ollama_inner() {
    [ "${RS_INSTALL_OLLAMA:-1}" = "1" ] || { log "ollama: disabled via RS_INSTALL_OLLAMA=0"; return 0; }

    export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
    export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
    # RunPod containers can ship NVIDIA_VISIBLE_DEVICES=void, which hides
    # all GPUs from ollama serve and forces silent CPU fallback. Override
    # to 'all' (or NVIDIA_VISIBLE_DEVICES_OVERRIDE if user wants to pin).
    export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES_OVERRIDE:-all}"
    mkdir -p "$OLLAMA_MODELS"

    # 1. Install binary if missing. Official installer puts it at
    #    /usr/local/bin/ollama which lives on the CONTAINER disk and
    #    gets wiped every container restart, so this runs often.
    if ! command -v ollama >/dev/null 2>&1; then
        log "ollama: binary missing — installing..."
        local _try
        for _try in 1 2 3; do
            curl -fsSL https://ollama.com/install.sh | sh 2>&1 | sed 's/^/[ollama-install] /' || true
            if command -v ollama >/dev/null 2>&1; then
                log "ollama: installed ($(ollama --version 2>&1 | head -1))"
                break
            fi
            log "ollama: install attempt ${_try}/3 failed; retrying in 3s..."
            sleep 3
        done
        if ! command -v ollama >/dev/null 2>&1; then
            log "ollama: ERROR — install failed after 3 attempts."
            return 1
        fi
    else
        log "ollama: binary present ($(ollama --version 2>&1 | head -1))"
    fi

    # 1b. Cache-poison guard. Fast path: if cuda_v* libs are present, we
    #     skip the entire block in a single stat call with no log output.
    #     Slow path: GPU present but no cuda libs (because a previous pod
    #     mirrored a CPU-only install to /workspace and init.sh restored
    #     it here) -> wipe + reinstall, BOUNDED to 3 tries. If 3 retries
    #     still fail, log a clear warning and let ollama run on CPU rather
    #     than infinite-loop on a template where the installer can't
    #     detect the GPU at all (looking at you, A100 template).
    if ls /usr/local/lib/ollama/cuda_v* >/dev/null 2>&1 \
       || ls /usr/share/ollama/lib/cuda_v* >/dev/null 2>&1; then
        : # GPU build present — silent fast path
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        log "ollama: GPU detected but cuda_v* libs missing — wiping cache + reinstalling (3 tries max)"
        pkill -9 -f "ollama serve" 2>/dev/null || true
        sleep 2
        rm -rf /usr/local/bin/ollama /usr/local/lib/ollama /usr/share/ollama/lib
        rm -rf /workspace/.ollama-install 2>/dev/null || true
        local _try _ok=0
        for _try in 1 2 3; do
            curl -fsSL https://ollama.com/install.sh | sh 2>&1 | sed 's/^/[ollama-reinstall] /' || true
            if ls /usr/local/lib/ollama/cuda_v* >/dev/null 2>&1; then
                log "ollama: GPU build installed on attempt ${_try}"
                _ok=1
                break
            fi
            log "ollama: reinstall attempt ${_try}/3 still missing cuda libs; retrying in 3s..."
            sleep 3
        done
        if [ "$_ok" != "1" ]; then
            log "ollama: GIVE UP — installer can't produce GPU build on this template after 3 tries. Daemon will run on CPU."
        fi
    fi

    # 2. If already running, healthcheck before returning.
    if pgrep -f "ollama serve" >/dev/null 2>&1; then
        if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
            log "ollama: already running and healthy on ${OLLAMA_HOST}"
            return 0
        fi
        log "ollama: process exists but API unreachable — killing and restarting..."
        pkill -f "ollama serve" 2>/dev/null || true
        sleep 1
    fi

    # 3. Launch fully-detached so it survives ComfyUI death.
    log "ollama: starting (OLLAMA_MODELS=$OLLAMA_MODELS, OLLAMA_HOST=$OLLAMA_HOST, NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES)..."
    setsid nohup env OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" \
        NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES" \
        ollama serve >>/workspace/ollama.log 2>&1 </dev/null &
    disown 2>/dev/null || true

    # 4. Wait for the API to come up. Up to 30s.
    local _i
    for _i in $(seq 1 15); do
        if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
            log "ollama: API up after $((_i * 2))s on ${OLLAMA_HOST}"
            return 0
        fi
        sleep 2
    done

    log "ollama: ERROR — server didn't come up in 30s. Tail of /workspace/ollama.log:"
    tail -20 /workspace/ollama.log 2>&1 | sed 's/^/[ollama-log] /' || true
    return 1
}

launch_comfy() {
    if [ ! -f "$VENV/bin/python" ]; then
        log "ERROR: venv missing at $VENV — set SETUP=TRUE to provision."
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"

    NV_LIB_PATHS=$(python -c "
import nvidia, os
r = os.path.dirname(nvidia.__file__)
print(':'.join(os.path.join(r, d, 'lib') for d in sorted(os.listdir(r))
               if os.path.isdir(os.path.join(r, d, 'lib'))))
" 2>/dev/null || echo "")
    if [ -n "$NV_LIB_PATHS" ]; then
        export LD_LIBRARY_PATH="$NV_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    ensure_ollama_running

    cd "$COMFY_DIR"
    COMFY_EXTRA_ARGS="${COMFY_EXTRA_ARGS-}"
    if [ "$COMFY_EXTRA_ARGS" = "NONE" ] || [ "$COMFY_EXTRA_ARGS" = "none" ]; then
        COMFY_EXTRA_ARGS=""
    fi
    log "Launching ComfyUI on 0.0.0.0:${PORT}"
    exec "$VENV/bin/python" main.py --listen 0.0.0.0 --port "$PORT" $COMFY_EXTRA_ARGS "$@"
}

# Mirror rs-nodes/runpod/*.sh into /workspace using ATOMIC RENAME, not
# in-place overwrite. cp -f truncates the existing inode in place — if
# bash is mid-execution of /workspace/startup.sh, its next read picks
# up new bytes at the old offset and chokes with a mid-line syntax
# error. mv unlinks the old inode (still readable via bash's open fd)
# and points the name at a new inode = safe.
sync_scripts() {
    [ -d "$RS_NODES_DIR/runpod" ] || return 0
    local src tmp name
    for src in "$RS_NODES_DIR/runpod/"*.sh; do
        [ -f "$src" ] || continue
        name=$(basename "$src")
        tmp="/workspace/.${name}.tmp.$$"
        if cp -f "$src" "$tmp" 2>/dev/null; then
            chmod +x "$tmp" 2>/dev/null || true
            mv -f "$tmp" "/workspace/$name"
        else
            rm -f "$tmp" 2>/dev/null || true
        fi
    done
}

MODE_SETUP=FALSE
MODE_RS_NODES=FALSE
_is_truthy "${SETUP:-FALSE}" && MODE_SETUP=TRUE
_is_truthy "${RS_NODES:-FALSE}" && MODE_RS_NODES=TRUE
log "Mode: SETUP=$MODE_SETUP RS_NODES=$MODE_RS_NODES"

if [ "$MODE_SETUP" = "TRUE" ]; then
    log "SETUP=TRUE — fetching bootstrap.sh and running full provision."
    rm -f "$WORKSPACE/.bootstrap_done"
    curl -fsSL "$BOOTSTRAP_URL" -o "$WORKSPACE/bootstrap.sh"
    chmod +x "$WORKSPACE/bootstrap.sh"
    exec bash "$WORKSPACE/bootstrap.sh"
fi

if [ "$MODE_RS_NODES" = "TRUE" ]; then
    log "RS_NODES=TRUE — pulling rs-nodes + mirroring scripts."
    if [ -d "$RS_NODES_DIR/.git" ]; then
        git -C "$RS_NODES_DIR" pull --ff-only 2>&1 | sed 's/^/  /' || \
            log "WARN: rs-nodes pull failed"
        git -C "$RS_NODES_DIR" submodule update --init --recursive 2>&1 | sed 's/^/  /' || \
            log "WARN: submodule update failed"
        sync_scripts
    else
        log "WARN: $RS_NODES_DIR/.git missing — skipping pull"
    fi
fi

launch_comfy "$@"
