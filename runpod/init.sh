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

# -----------------------------------------------------------------------------
# CUDA driver compatibility gate — FIRST CHECK, before any expensive work.
#
# We previously had this check only in startup.sh, but on first boot
# bootstrap.sh runs first — and if bootstrap dies mid-flight (which it
# can for many reasons unrelated to drivers), the user never sees the
# driver-check message and ends up watching a 10+ minute bootstrap loop
# without knowing the underlying host is unusable.
#
# Run it here so a doomed pod fails within ~3 seconds of container boot,
# regardless of bootstrap state. Override with RS_SKIP_CUDA_CHECK=1 if
# you specifically want to use cu128 fallback (rare; explained in the
# banner).
# -----------------------------------------------------------------------------
if [ "${RS_SKIP_CUDA_CHECK:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
    _drv_cuda=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version:[[:space:]]*[0-9]+' | head -1 | grep -oE '[0-9]+$')
    _drv_cuda="${_drv_cuda:-0}"
    if [ "$_drv_cuda" -lt 13 ] 2>/dev/null; then
        cat <<EOF >&2

############################################################################
##                                                                        ##
##   FATAL: This pod's NVIDIA driver supports CUDA ${_drv_cuda}.x at most.            ##
##          Torch cu130 requires a host driver supporting CUDA 13.x       ##
##          (typically driver >= 555).                                    ##
##                                                                        ##
##   --> TERMINATE THIS POD AND DEPLOY ON A DIFFERENT HOST <--            ##
##                                                                        ##
##   Container-image CUDA version != host driver CUDA version.            ##
##   Even templates named "CUDA13" can land on hosts with 12.x drivers.   ##
##   Verify each new pod with: nvidia-smi | head -3                       ##
##                                                                        ##
##   Override for cu128 fallback (slower on Blackwell, NOT recommended):  ##
##     set RS_SKIP_CUDA_CHECK=1 in pod env vars and re-deploy             ##
##                                                                        ##
##   nvidia-smi reports: CUDA Version ${_drv_cuda}.x                                  ##
##                                                                        ##
############################################################################
EOF
        echo "[init] FATAL: driver supports CUDA ${_drv_cuda}.x, need >= 13. Aborting init." >&2
        # Keep container alive briefly so the user can SSH in if they want
        # to inspect, but mostly so this banner stays visible in the pod
        # log instead of disappearing into a fast-restart loop.
        sleep 60
        exit 1
    fi
    echo "[init] CUDA driver check: host supports CUDA ${_drv_cuda}.x (>= 13 required) — OK"
fi

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
# B2 sync handled by RunPod Cloud Sync (web UI)
#
# RunPod console -> pod -> Cloud Sync -> Backblaze B2 covers all
# bucket <-> /workspace transfers. The auto-mount / auto-pull /
# inotify-autosync infrastructure that used to live here has been
# removed in favor of that built-in feature.
# -----------------------------------------------------------------------------

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

# Subsequent-boot fast path: bootstrap fully completed at least once
# (marker written by bootstrap.sh at the end of Phase 6).
#
# We require the marker rather than just file-existence checks: a
# partial bootstrap interrupted mid-run (e.g. user edited pod config
# and triggered a silent reboot) leaves startup.sh and the rs-nodes
# clone in place but skips Phase 4 (model weights). The marker tells
# us provisioning actually finished, so it's safe to take the fast
# path.
if [ -f "$DONE_MARKER" ] && [ -x "$WORKSPACE/startup.sh" ]; then
    echo "[init] $DONE_MARKER present — running startup.sh (fast path)"
    exec bash "$WORKSPACE/startup.sh"
fi

# First-boot OR incomplete-bootstrap path: fetch and exec bootstrap.sh.
# bootstrap.sh is idempotent; re-running on a partial volume picks up
# from where the last run was killed.
if [ -x "$WORKSPACE/startup.sh" ]; then
    echo "[init] startup.sh present but bootstrap marker missing — re-running bootstrap.sh"
else
    echo "[init] First boot detected — fetching bootstrap.sh"
fi
curl -fsSL "$BOOTSTRAP_URL" -o "$WORKSPACE/bootstrap.sh"
chmod +x "$WORKSPACE/bootstrap.sh"
exec bash "$WORKSPACE/bootstrap.sh"
