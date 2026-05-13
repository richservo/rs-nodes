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
if [ "${RS_FORCE_BOOTSTRAP:-0}" != "1" ] && ! nvidia-smi >/dev/null 2>&1; then
    echo "[init] No GPU detected — entering MAINTENANCE MODE."
    echo "[init]   * authorized_keys already populated from PUBLIC_KEY* env vars"
    echo "[init]   * sshd starting"
    echo "[init]   * container will idle so you can SSH in"
    echo "[init]   * SET RS_FORCE_BOOTSTRAP=1 to override and run normal bootstrap"

    # Start sshd. Try the service manager first (works on Ubuntu base
    # images), fall back to invoking the daemon directly. -D would
    # keep it foreground but we want sshd backgrounded so this
    # script's tail -f at the end is the PID 1 keep-alive.
    if command -v service >/dev/null 2>&1; then
        service ssh restart 2>&1 | sed 's/^/[init] sshd: /' || /usr/sbin/sshd
    elif [ -x /usr/sbin/sshd ]; then
        pkill -f /usr/sbin/sshd 2>/dev/null || true
        /usr/sbin/sshd
    fi

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
