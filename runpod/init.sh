#!/usr/bin/env bash
# rs-nodes one-shot pod entry-point.
#
# Wired in as the pod template's Container Start Command via:
#   bash -c "(getent hosts github.com >/dev/null 2>&1 || { echo nameserver 8.8.8.8 > /etc/resolv.conf; echo nameserver 1.1.1.1 >> /etc/resolv.conf; }) && curl -fsSL https://raw.githubusercontent.com/richservo/rs-nodes/master/runpod/init.sh | bash"
#
# What it does:
#   * First boot on a fresh /workspace volume — fetch bootstrap.sh and
#     run the full provisioning (clone, deps, models, Ollama, launch).
#   * Every subsequent boot — fall through to /workspace/startup.sh
#     which runs idempotent re-checks and launches ComfyUI in seconds.
#
# Env vars (set in the pod template):
#   HF_TOKEN          — HuggingFace token for model downloads
#   OLLAMA_MODEL      — Ollama model(s), space-separated (default: gemma4:31b gemma4:26b)
#   RS_INSTALL_OLLAMA — "0" to skip Ollama (default "1")
#   RS_NODE_PACKS     — custom-node pack keys (default: full set)

set -e

WORKSPACE=/workspace
BOOTSTRAP_URL="https://raw.githubusercontent.com/richservo/rs-nodes/master/runpod/bootstrap.sh"
DONE_MARKER="$WORKSPACE/.bootstrap_done"

mkdir -p "$WORKSPACE"

# Honor RunPod's PUBLIC_KEY env-var convention on every boot. RunPod's
# stock init copies $PUBLIC_KEY into /root/.ssh/authorized_keys, but
# our init.sh replaces the stock start command so that hook never
# fires — meaning users who set the pubkey via env var (instead of
# account-level Settings -> SSH Public Keys) lose SSH access entirely.
# Re-implement the hook here, before the fast-path branch, so it runs
# whether we go straight to startup.sh or fall through to bootstrap.sh.
#
# Picks up ANY env var starting with PUBLIC_KEY — so PUBLIC_KEY plus
# PUBLIC_KEY_2, PUBLIC_KEY_BOB, etc. all get added. Lets you share a
# pod with another artist without touching account-level settings:
# just add their pubkey as another PUBLIC_KEY_* env var.
#
# Behavior is REBUILD, not append: every boot we reconstruct
# /root/.ssh/authorized_keys from the union of (whatever was already
# there — typically RunPod's account-level injection) + (all
# PUBLIC_KEY* env vars), dedupe, then mirror to /workspace
# unconditionally. This way:
#   * stale /workspace cache can't override fresh env-var changes
#     via bootstrap.sh Phase 0.5's restore-from-volume logic
#   * a PUBLIC_KEY env var change (rotate, add another artist) takes
#     effect on the next pod restart with no manual intervention
#   * account-level RunPod pubkeys (if any) survive
#   * Windows-paste artifacts (CRLF, trailing spaces) get sanitized
#     so sshd doesn't silently reject them
mkdir -p /root/.ssh /workspace/.ssh
chmod 700 /root/.ssh /workspace/.ssh

AK_TMP=$(mktemp)
# Start with whatever's already in /root/.ssh/authorized_keys —
# that's where RunPod's account-level Settings keys land (if the
# user is using that mechanism in addition to env vars).
[ -f /root/.ssh/authorized_keys ] && cat /root/.ssh/authorized_keys >> "$AK_TMP" || true
# Add every PUBLIC_KEY* env var, one per line, with line endings
# normalized (strip \r so a CRLF-pasted value doesn't end up with
# a literal \r in the comment, which sshd silently rejects).
for var in $(compgen -e | grep -E '^PUBLIC_KEY' || true); do
    key="${!var}"
    [ -z "$key" ] && continue
    # tr -d '\r' strips Windows CRs; sed trims trailing whitespace
    sanitized=$(printf '%s\n' "$key" | tr -d '\r' | sed 's/[[:space:]]*$//')
    [ -z "$sanitized" ] && continue
    echo "[init] Installing $var pubkey into authorized_keys"
    printf '%s\n' "$sanitized" >> "$AK_TMP"
done
# Dedupe (drop blank lines + exact duplicates, preserve order)
awk 'NF && !seen[$0]++' "$AK_TMP" > /root/.ssh/authorized_keys
rm -f "$AK_TMP"
chmod 600 /root/.ssh/authorized_keys

# ALWAYS mirror to the volume — bootstrap.sh Phase 0.5 restores
# authorized_keys from /workspace, so /workspace must reflect the
# fresh /root we just built. Without this, a stale /workspace copy
# from a previous bootstrap can wipe out the env-var keys.
cp /root/.ssh/authorized_keys /workspace/.ssh/authorized_keys
chmod 600 /workspace/.ssh/authorized_keys

echo "[init] authorized_keys has $(wc -l < /root/.ssh/authorized_keys) key(s)"

# GPU detection — if there's no usable GPU, almost certainly a CPU
# pod the user spun up for maintenance (updating /workspace/.venv,
# editing files, doing pip work) because GPU availability was tight.
# Bootstrap, the framework-installed check, and ComfyUI launch all
# depend on CUDA and would boot-loop the container on a CPU pod.
# Instead: make sure sshd is alive (so they can SSH in to do the
# maintenance they came for) and idle. Override with RS_FORCE_BOOTSTRAP=1
# if you really want to run the normal flow on a CPU pod anyway.
# -----------------------------------------------------------------------------
# B2 credentials from env vars -> /workspace/.rclone/rclone.conf
#
# Declarative: set B2_KEY_ID and B2_APP_KEY in pod env vars (RunPod
# console) and init.sh writes a valid rclone.conf on EVERY boot.
# Same pattern as PUBLIC_KEY_* env vars. No manual paste, no
# b2_setup.bat round-trip required.
#
# Bucket name is not put in rclone.conf (rclone remote stanzas don't
# need it — bucket lands in the path: `b2:<bucket>/datasets/...`).
# B2_BUCKET env var is read by b2_helpers.sh as a convenience default.
# -----------------------------------------------------------------------------
if [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_APP_KEY:-}" ]; then
    mkdir -p /workspace/.rclone
    chmod 700 /workspace/.rclone
    cat > /workspace/.rclone/rclone.conf <<EOF
[b2]
type = b2
account = $B2_KEY_ID
key = $B2_APP_KEY
hard_delete = false
EOF
    chmod 600 /workspace/.rclone/rclone.conf
    echo "[init] Wrote /workspace/.rclone/rclone.conf from B2_KEY_ID / B2_APP_KEY env vars"
fi

# -----------------------------------------------------------------------------
# sshd sftp subsystem — ALWAYS-ON, both maintenance + normal modes.
#
# Some minimal base images strip the sftp-server subsystem from
# sshd_config — breaks scp / sftp / rs-studio's file panel while
# interactive ssh still works. Patch in the Subsystem line and
# install the binary if missing, so file transfer is always
# available on any pod running this init.sh.
# -----------------------------------------------------------------------------
SSHD_CONFIG=/etc/ssh/sshd_config
if [ -f "$SSHD_CONFIG" ]; then
    if ! grep -qE '^[[:space:]]*Subsystem[[:space:]]+sftp' "$SSHD_CONFIG"; then
        SFTP_SERVER=""
        for p in /usr/lib/openssh/sftp-server \
                 /usr/libexec/openssh/sftp-server \
                 /usr/lib/sftp-server \
                 /usr/libexec/sftp-server; do
            if [ -x "$p" ]; then SFTP_SERVER="$p"; break; fi
        done
        if [ -z "$SFTP_SERVER" ] && command -v apt-get >/dev/null 2>&1; then
            echo "[init] Installing openssh-sftp-server (for sftp/scp)..."
            DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 || true
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-sftp-server >/dev/null 2>&1 || true
            for p in /usr/lib/openssh/sftp-server /usr/libexec/openssh/sftp-server; do
                if [ -x "$p" ]; then SFTP_SERVER="$p"; break; fi
            done
        fi
        if [ -n "$SFTP_SERVER" ]; then
            echo "[init] Enabling sftp subsystem in sshd_config: $SFTP_SERVER"
            echo "Subsystem sftp $SFTP_SERVER" >> "$SSHD_CONFIG"
        else
            echo "[init] WARN: sftp-server binary unavailable — file transfer won't work."
        fi
    fi
fi

# -----------------------------------------------------------------------------
# GPU check / maintenance mode
# -----------------------------------------------------------------------------
if [ "${RS_FORCE_BOOTSTRAP:-0}" != "1" ] && ! nvidia-smi >/dev/null 2>&1; then
    echo "[init] No GPU detected — entering MAINTENANCE MODE."
    echo "[init] Full pod access except ComfyUI / CUDA-dependent work:"
    echo "[init]   * SSH (interactive + scp + sftp)"
    echo "[init]   * Network volume read/write"
    echo "[init]   * rclone / B2 sync (if B2_KEY_ID + B2_APP_KEY env vars set)"
    echo "[init]   * pip / apt / git / everything else"
    echo "[init] Set RS_FORCE_BOOTSTRAP=1 to override and run normal bootstrap."

    # Generate or restore sshd host keys BEFORE starting sshd. Without
    # this, sshd exits immediately with "no hostkeys available --
    # exiting" on minimal base images (e.g. the NVIDIA CUDA runtime
    # image, which is common for CPU-fallback pods). Bootstrap.sh's
    # Phase 0.5 would normally do this; maintenance mode skips
    # bootstrap, so we replicate the host-key setup here.
    #
    # Host keys are persisted on the /workspace volume so they survive
    # container resets — SSH clients won't see "host key changed"
    # warnings on every reboot.
    mkdir -p /workspace/.ssh/host_keys /etc/ssh
    if compgen -G "/workspace/.ssh/host_keys/ssh_host_*" > /dev/null; then
        echo "[init] Restoring sshd host keys from /workspace/.ssh/host_keys"
        cp -f /workspace/.ssh/host_keys/ssh_host_* /etc/ssh/
    elif compgen -G "/etc/ssh/ssh_host_*" > /dev/null; then
        echo "[init] Persisting existing sshd host keys to /workspace/.ssh/host_keys"
        cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
    else
        echo "[init] Generating fresh sshd host keys (ssh-keygen -A)"
        ssh-keygen -A
        cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
    fi
    chmod 600 /etc/ssh/ssh_host_*_key 2>/dev/null || true
    chmod 644 /etc/ssh/ssh_host_*_key.pub 2>/dev/null || true

    # /run/sshd needs to exist for the daemon's privilege-separation
    # chroot. Minimal images sometimes don't have it.
    mkdir -p /run/sshd
    chmod 755 /run/sshd

    # Start sshd by invoking the daemon directly — `service ssh restart`
    # on some minimal images is a no-op (script exits 0 without
    # actually starting anything) and the previous `cmd | sed ||
    # fallback` form swallowed the exit code through the pipe, so a
    # silent service failure looked like success and the /usr/sbin/sshd
    # fallback never ran. Direct invocation is simpler and the exit
    # code is meaningful.
    pkill -x sshd 2>/dev/null || true
    sleep 0.2
    if [ -x /usr/sbin/sshd ]; then
        SSHD_ERR=$(/usr/sbin/sshd 2>&1)
        SSHD_RC=$?
        if [ "$SSHD_RC" -ne 0 ] || [ -n "$SSHD_ERR" ]; then
            echo "[init] sshd: $SSHD_ERR"
        fi
    else
        echo "[init] WARN: /usr/sbin/sshd not found — install openssh-server"
    fi

    # Verify sshd is actually listening before declaring success.
    # Up to 3s grace; sshd binds the listening socket synchronously
    # but daemonization adds a tiny race.
    for _i in 1 2 3 4 5 6; do
        if pgrep -x sshd >/dev/null 2>&1; then break; fi
        sleep 0.5
    done
    if pgrep -x sshd >/dev/null 2>&1; then
        echo "[init] sshd running (interactive ssh + scp + sftp available)"
    else
        echo "[init] WARN: sshd failed to start. Last error: $SSHD_ERR"
        echo "[init]       Common causes:"
        echo "[init]         * Host keys missing (should be generated above; check /etc/ssh/ssh_host_*)"
        echo "[init]         * /run/sshd permissions wrong (should be 755)"
        echo "[init]         * Port 22 already bound by something else (rare)"
    fi

    # Print the canonical connection info so the user can copy/paste
    # the right SSH line into rs-studio / launch.bat / wherever.
    # RUNPOD_* env vars are set by RunPod's container runtime. Missing
    # values usually mean the pod template doesn't expose TCP for that
    # port — in that case only the proxy form works for SSH (and so
    # scp/sftp also need to go through the proxy form).
    echo
    echo "[init] =========================================================="
    echo "[init]  Connection info for this pod"
    echo "[init] =========================================================="
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "[init]  Pod ID:        $RUNPOD_POD_ID"
    fi
    if [ -n "${RUNPOD_PUBLIC_IP:-}" ] && [ -n "${RUNPOD_TCP_PORT_22:-}" ]; then
        echo "[init]  TCP SSH:       ssh root@$RUNPOD_PUBLIC_IP -p $RUNPOD_TCP_PORT_22 -i ~/.ssh/id_ed25519"
        echo "[init]  TCP scp/sftp:  scp -P $RUNPOD_TCP_PORT_22 -i ~/.ssh/id_ed25519 <file> root@$RUNPOD_PUBLIC_IP:<remote>"
    else
        echo "[init]  TCP SSH:       NOT EXPOSED on this pod template."
        echo "[init]                 (RUNPOD_PUBLIC_IP / RUNPOD_TCP_PORT_22 not set —"
        echo "[init]                  enable TCP port 22 in pod settings to use scp/sftp."
        echo "[init]                  Without it, only proxy-form SSH works and rs-studio's"
        echo "[init]                  file panel will fail to upload.)"
    fi
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "[init]  Proxy SSH:     ssh ${RUNPOD_POD_ID}-${RUNPOD_POD_HOSTNAME:-$(hostname)}@ssh.runpod.io -i ~/.ssh/id_ed25519"
    fi
    echo "[init] =========================================================="
    echo
    echo "[init] Maintenance mode ready. Container will stay alive until you stop it."
    # Keep PID 1 alive so RunPod doesn't auto-restart the container.
    exec tail -f /dev/null
fi

# -----------------------------------------------------------------------------
# Normal-mode sshd start. Maintenance mode handles its own sshd above; the
# normal path needs sshd too so the user can SSH in regardless of which
# startup.sh fast-path runs after this. Host keys are persisted on the
# volume so SSH clients don't see "host key changed" warnings.
# -----------------------------------------------------------------------------
mkdir -p /workspace/.ssh/host_keys /etc/ssh /run/sshd
chmod 755 /run/sshd
if compgen -G "/workspace/.ssh/host_keys/ssh_host_*" > /dev/null; then
    cp -f /workspace/.ssh/host_keys/ssh_host_* /etc/ssh/
elif compgen -G "/etc/ssh/ssh_host_*" > /dev/null; then
    cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
else
    ssh-keygen -A
    cp -f /etc/ssh/ssh_host_* /workspace/.ssh/host_keys/
fi
chmod 600 /etc/ssh/ssh_host_*_key 2>/dev/null || true
chmod 644 /etc/ssh/ssh_host_*_key.pub 2>/dev/null || true

if ! pgrep -x sshd >/dev/null 2>&1; then
    if command -v service >/dev/null 2>&1; then
        service ssh start 2>&1 | sed 's/^/[sshd] /' || true
    elif [ -x /usr/sbin/sshd ]; then
        /usr/sbin/sshd
    fi
fi

# -----------------------------------------------------------------------------
# Ollama install + start. Lives here in init.sh (fetched fresh from GitHub
# every boot via the Container Start Command) so it can't go stale, runs on
# EVERY boot regardless of which startup.sh/bootstrap.sh path runs after.
#
# Binary lives at /workspace/.ollama-install (persistent network volume) so
# after first-boot install it never needs to be reinstalled — every later
# boot just symlinks and starts. ~5min first boot, ~3s every boot after.
#
# Skipped if RS_INSTALL_OLLAMA=0.
# -----------------------------------------------------------------------------
if [ "${RS_INSTALL_OLLAMA:-1}" = "1" ]; then
    # Persistence target on the network volume — survives container restarts.
    OLLAMA_PERSIST_DIR="/workspace/.ollama-install"
    OLLAMA_PERSIST_BIN="$OLLAMA_PERSIST_DIR/bin/ollama"
    OLLAMA_PERSIST_LIB="$OLLAMA_PERSIST_DIR/lib/ollama"
    # Install path the official installer ACTUALLY uses (it ignores
    # OLLAMA_INSTALL_DIR despite docs claiming otherwise on some versions).
    OLLAMA_SYS_BIN="/usr/local/bin/ollama"
    OLLAMA_SYS_LIB="/usr/local/lib/ollama"
    OLLAMA_HOST_VAL="${OLLAMA_HOST:-127.0.0.1:11434}"
    OLLAMA_MODELS_VAL="${OLLAMA_MODELS:-/workspace/.ollama/models}"
    # RunPod container env can ship NVIDIA_VISIBLE_DEVICES=void which hides
    # all GPUs from ollama serve and forces silent CPU fallback. Override
    # to 'all' so the daemon sees the GPU at load time.
    NVIDIA_VISIBLE_DEVICES_VAL="${NVIDIA_VISIBLE_DEVICES_OVERRIDE:-all}"

    # Strategy: try restore-from-volume first (fast, no network), fall back
    # to fresh install (slow, ~15s download). After fresh install, mirror
    # /usr/local → /workspace for next-boot restore.

    # 1. Restore from /workspace if we have a cached install. /usr/local
    #    gets wiped on every container restart, so we re-populate it from
    #    /workspace on each boot if possible.
    if [ -x "$OLLAMA_PERSIST_BIN" ] && [ ! -x "$OLLAMA_SYS_BIN" ]; then
        echo "[init] ollama: restoring cached install from $OLLAMA_PERSIST_DIR"
        mkdir -p /usr/local/bin /usr/local/lib
        cp -a "$OLLAMA_PERSIST_BIN" "$OLLAMA_SYS_BIN"
        [ -d "$OLLAMA_PERSIST_LIB" ] && cp -a "$OLLAMA_PERSIST_LIB" /usr/local/lib/ 2>/dev/null || true
        chmod +x "$OLLAMA_SYS_BIN"
    fi

    # 2. If still missing, run the official installer (it installs to
    #    /usr/local regardless of OLLAMA_INSTALL_DIR — accept this and
    #    mirror to /workspace after).
    if [ ! -x "$OLLAMA_SYS_BIN" ]; then
        echo "[init] ollama: not present — running official installer"
        OLLAMA_INSTALLED=0
        for _ot in 1 2 3; do
            if curl -fsSL --max-time 600 https://ollama.com/install.sh | \
               sh 2>&1 | sed 's/^/[ollama-install] /'; then
                if [ -x "$OLLAMA_SYS_BIN" ]; then
                    OLLAMA_INSTALLED=1
                    echo "[init] ollama: installed at $OLLAMA_SYS_BIN"
                    break
                fi
            fi
            echo "[init] ollama: install attempt ${_ot}/3 failed, retrying in 5s"
            sleep 5
        done
        if [ "$OLLAMA_INSTALLED" != "1" ]; then
            echo "[init] ollama: ERROR — install failed after 3 attempts"
        else
            # Mirror to /workspace so next boot can restore without
            # network. cp -a preserves perms/symlinks within the lib dir.
            echo "[init] ollama: mirroring install to $OLLAMA_PERSIST_DIR for next-boot reuse"
            mkdir -p "$OLLAMA_PERSIST_DIR/bin"
            cp -a "$OLLAMA_SYS_BIN" "$OLLAMA_PERSIST_BIN" 2>/dev/null || true
            if [ -d "$OLLAMA_SYS_LIB" ]; then
                mkdir -p "$OLLAMA_PERSIST_DIR/lib"
                cp -a "$OLLAMA_SYS_LIB" "$OLLAMA_PERSIST_DIR/lib/" 2>/dev/null || true
            fi
        fi
    else
        echo "[init] ollama: binary present at $OLLAMA_SYS_BIN"
    fi

    # 2b. Validate GPU support. Fast path: if cuda_v* libs are restored
    #     from /workspace or already in place, this is a single stat call
    #     and we move on. Slow path (GPU present but CPU-only install)
    #     triggers wipe + reinstall + re-mirror to /workspace, so the
    #     fix is permanent across reboots.
    if [ -x "$OLLAMA_SYS_BIN" ] && ls "$OLLAMA_SYS_LIB"/cuda_v* >/dev/null 2>&1; then
        : # GPU build already present — silent fast path
    elif [ -x "$OLLAMA_SYS_BIN" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "[init] ollama: GPU detected but cuda_v* libs missing — forcing fresh install"
        pkill -9 -f "ollama serve" 2>/dev/null || true
        sleep 2
        rm -rf "$OLLAMA_SYS_BIN" "$OLLAMA_SYS_LIB"
        rm -rf "$OLLAMA_PERSIST_BIN" "$OLLAMA_PERSIST_LIB" 2>/dev/null || true
        for _ot in 1 2 3; do
            curl -fsSL --max-time 600 https://ollama.com/install.sh | \
                sh 2>&1 | sed 's/^/[ollama-reinstall] /' || true
            if [ -x "$OLLAMA_SYS_BIN" ] && ls "$OLLAMA_SYS_LIB"/cuda_v* >/dev/null 2>&1; then
                echo "[init] ollama: GPU build installed"
                break
            fi
            echo "[init] ollama: reinstall attempt ${_ot}/3 still missing cuda libs; retrying in 5s"
            sleep 5
        done
        if [ -x "$OLLAMA_SYS_BIN" ] && ls "$OLLAMA_SYS_LIB"/cuda_v* >/dev/null 2>&1; then
            echo "[init] ollama: mirroring GPU install to $OLLAMA_PERSIST_DIR"
            mkdir -p "$OLLAMA_PERSIST_DIR/bin" "$OLLAMA_PERSIST_DIR/lib"
            cp -a "$OLLAMA_SYS_BIN" "$OLLAMA_PERSIST_BIN" 2>/dev/null || true
            cp -a "$OLLAMA_SYS_LIB" "$OLLAMA_PERSIST_DIR/lib/" 2>/dev/null || true
        else
            echo "[init] ollama: WARN — reinstall finished but cuda_v* libs still missing; daemon will run on CPU"
        fi
    fi

    # 3. Start the server if not already running. Fully detached so it
    #    survives ComfyUI dying.
    if [ -x "$OLLAMA_SYS_BIN" ] && ! pgrep -f "ollama serve" >/dev/null 2>&1; then
        mkdir -p "$OLLAMA_MODELS_VAL"
        echo "[init] ollama: starting serve on $OLLAMA_HOST_VAL (NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES_VAL)"
        setsid nohup env OLLAMA_MODELS="$OLLAMA_MODELS_VAL" OLLAMA_HOST="$OLLAMA_HOST_VAL" \
            NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES_VAL" \
            "$OLLAMA_SYS_BIN" serve >>/workspace/ollama.log 2>&1 </dev/null &
        disown 2>/dev/null || true
    fi

    # 4. Healthcheck — wait up to 20s for API.
    if [ -x "$OLLAMA_SYS_BIN" ]; then
        for _oi in 1 2 3 4 5 6 7 8 9 10; do
            if curl -fsS --max-time 1 "http://${OLLAMA_HOST_VAL}/api/tags" >/dev/null 2>&1; then
                echo "[init] ollama: API up on $OLLAMA_HOST_VAL"
                break
            fi
            sleep 2
        done
        if ! curl -fsS --max-time 1 "http://${OLLAMA_HOST_VAL}/api/tags" >/dev/null 2>&1; then
            echo "[init] ollama: WARN — server didn't respond after 20s; check /workspace/ollama.log"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Dispatch: ONLY SETUP=TRUE triggers a full bootstrap. Anything else
# (FALSE, unset, empty) runs startup.sh.
#
# Separate from bootstrap: if ComfyUI itself updated (via git pull or
# ComfyUI-Manager) and its requirements.txt now needs packages the venv
# doesn't have, we run JUST `pip install -r ComfyUI/requirements.txt`
# to heal. This is NOT the full bootstrap — no torch reinstall, no
# ollama model pulls, no model downloads. Just ComfyUI's own deps.
# Triggered only when ComfyUI's startup imports actually fail; healthy
# boots skip it entirely.
# -----------------------------------------------------------------------------
case "${SETUP,,}" in
    true|1|yes|on)
        echo "[init] SETUP=TRUE — fetching bootstrap.sh and running full provision"
        # Up-front warning if HF_TOKEN missing — bootstrap downloads ~50 GB
        # of weights, and unauthenticated HF caps at ~1-5 MB/s = many hours.
        # With a token, the same downloads take minutes. Warn BEFORE we
        # spend 10 minutes on pip just to crawl HF.
        if [ -z "${HF_TOKEN:-}" ]; then
            echo "[init] ============================================================"
            echo "[init]   WARN: HF_TOKEN not set — bootstrap model downloads will"
            echo "[init]   be RATE-LIMITED to ~1-5 MB/s by HuggingFace."
            echo "[init]   With auth, the same downloads run at 100-500 MB/s."
            echo "[init]"
            echo "[init]   To enable: get token at"
            echo "[init]     https://huggingface.co/settings/tokens"
            echo "[init]   then add HF_TOKEN to your pod env vars and restart."
            echo "[init] ============================================================"
        else
            echo "[init] HF_TOKEN detected — authenticated downloads enabled (fast)"
        fi
        curl -fsSL "$BOOTSTRAP_URL" -o "$WORKSPACE/bootstrap.sh"
        chmod +x "$WORKSPACE/bootstrap.sh"
        exec bash "$WORKSPACE/bootstrap.sh"
        ;;
esac

# ComfyUI requirements sync — ONLY fires when ComfyUI's imports actually
# fail. Healthy venv → import test passes in <1s → skip pip install.
# Broken venv → run pip install -r ComfyUI/requirements.txt (this is NOT
# the full bootstrap; nothing else runs).
COMFY_REQ="/workspace/ComfyUI/requirements.txt"
COMFY_VENV_PY="/workspace/.venv/bin/python"
COMFY_VENV_PIP="/workspace/.venv/bin/pip"
if [ -f "$COMFY_REQ" ] && [ -x "$COMFY_VENV_PY" ] && [ -x "$COMFY_VENV_PIP" ]; then
    if ! ( cd /workspace/ComfyUI && "$COMFY_VENV_PY" -c "
import sys
sys.path.insert(0, '/workspace/ComfyUI')
import comfy
import server
import comfyui_workflow_templates
import comfyui_frontend_package
" 2>/dev/null ); then
        echo "[init] ComfyUI imports broken — running pip install -r ComfyUI/requirements.txt to heal venv"
        echo "[init] (this is NOT the full bootstrap — only ComfyUI's deps)"
        if "$COMFY_VENV_PIP" install --no-cache-dir -r "$COMFY_REQ" 2>&1 | sed 's/^/[comfy-reqs] /'; then
            echo "[init] ComfyUI deps healed."
        else
            echo "[init] WARN: pip install returned non-zero; ComfyUI may not start cleanly."
        fi
    fi
fi

# SETUP not TRUE — run startup.sh. Period.
if [ -x "$WORKSPACE/startup.sh" ]; then
    exec bash "$WORKSPACE/startup.sh"
fi

# Genuinely fresh pod — /workspace/startup.sh doesn't exist yet, and
# user hasn't set SETUP=TRUE. Bail with a clear message instead of
# silently bootstrapping.
echo "[init] ERROR: /workspace/startup.sh missing and SETUP != TRUE."
echo "[init]        This pod has not been provisioned yet. Set SETUP=TRUE in pod env vars,"
echo "[init]        restart the container once, then change it back to FALSE/unset."
exit 1
