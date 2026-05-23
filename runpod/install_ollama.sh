#!/usr/bin/env bash
# rs-nodes pod-side: install Ollama and pull the LLM(s) used by
# RSPromptFormatter for caption / negative-prompt generation.
#
# Ollama state goes under /root/.ollama by default, which is on the
# CONTAINER disk and gets wiped on pod terminate. To keep pulled models
# across pod lifecycles we relocate $OLLAMA_MODELS to the network volume.
#
# Idempotent: safe to re-run; the install step is a no-op if Ollama is
# already on PATH, and `ollama pull` is a no-op for already-cached models.
#
# Usage:
#   bash /workspace/install_ollama.sh                    # default model
#   OLLAMA_MODEL="gemma4:31b" bash install_ollama.sh     # override
#   OLLAMA_MODEL="gemma4:31b gemma4:26b" ...             # multiple

set -euo pipefail

# Default pulls BOTH gemma4 variants for A/B comparison:
#   * gemma4:26b — MoE 26B-A4B, ~4B active per token, faster.
#   * gemma4:31b — dense, all 31B active per token, higher quality
#     ceiling.
# Both run easily on a 96 GB GPU; switch by typing the model name in
# the RSPromptFormatter widget. Override with OLLAMA_MODEL="gemma4:31b"
# to pull just one.
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:31b gemma4:26b}"
OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/.ollama/models}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
# Force NVIDIA_VISIBLE_DEVICES=all in case the container env hides GPUs
# (RunPod sometimes ships NVIDIA_VISIBLE_DEVICES=void).
NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES_OVERRIDE:-all}"

export OLLAMA_MODELS OLLAMA_HOST NVIDIA_VISIBLE_DEVICES

# DNS guard.
if ! getent hosts ollama.com >/dev/null 2>&1; then
    echo "[install_ollama] DNS broken; injecting public resolvers"
    { echo "nameserver 8.8.8.8"; echo "nameserver 1.1.1.1"; } > /etc/resolv.conf || true
fi

mkdir -p "$OLLAMA_MODELS"

# 1. Install the binary if missing.
if ! command -v ollama >/dev/null 2>&1; then
    echo "[install_ollama] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[install_ollama] Ollama already installed: $(ollama --version)"
fi

# 1b. GPU build validation. Fast path noop if cuda_v* present. Slow path
#     (GPU detected but cuda libs missing): wipe + reinstall, bounded to
#     3 tries so a broken template never infinite-loops.
if ls /usr/local/lib/ollama/cuda_v* >/dev/null 2>&1; then
    : # GPU build present
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "[install_ollama] GPU detected but cuda_v* libs missing — reinstalling (3 tries max)"
    pkill -9 -f "ollama serve" 2>/dev/null || true
    sleep 2
    rm -rf /usr/local/bin/ollama /usr/local/lib/ollama /workspace/.ollama-install
    _CUDA_OK=0
    for _it in 1 2 3; do
        curl -fsSL https://ollama.com/install.sh | sh
        if ls /usr/local/lib/ollama/cuda_v* >/dev/null 2>&1; then
            echo "[install_ollama] GPU build installed on attempt ${_it}"
            _CUDA_OK=1
            break
        fi
        echo "[install_ollama] attempt ${_it}/3 still missing cuda libs; retrying in 5s"
        sleep 5
    done
    [ "$_CUDA_OK" = "1" ] || echo "[install_ollama] GIVE UP — daemon will run on CPU on this template"
fi

# 2. Start (or reuse) the server in the background. We log to
#    /workspace/ollama.log so the log itself persists too.
if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "[install_ollama] Ollama server already running on $OLLAMA_HOST"
else
    echo "[install_ollama] Starting Ollama server on $OLLAMA_HOST (NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES)"
    nohup env OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="$OLLAMA_HOST" \
        NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES" \
        ollama serve > /workspace/ollama.log 2>&1 &

    # Wait up to 60s for it to come up.
    for i in $(seq 1 30); do
        if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
            echo "  server up after ${i}s"
            break
        fi
        sleep 2
    done

    if ! curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo "ERROR: Ollama server did not come up. Check /workspace/ollama.log"
        tail -40 /workspace/ollama.log || true
        exit 1
    fi
fi

# 3. Pull the requested model(s). Words split on whitespace so the env
#    var can carry multiple model names: OLLAMA_MODEL="gemma4:31b mistral:7b"
for model in $OLLAMA_MODEL; do
    echo "[install_ollama] Ensuring $model is present..."
    if ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$model"; then
        echo "  $model already cached"
    else
        ollama pull "$model"
    fi
done

echo ""
echo "[install_ollama] Done. Models on volume:"
ollama list
echo ""
echo "RSPromptFormatter url widget should be: http://localhost:11434"
echo "(Or http://${OLLAMA_HOST} from outside the pod's container.)"
