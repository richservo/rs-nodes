#!/usr/bin/env bash
# Install or upgrade rclone on the pod. Idempotent — safe to call on
# every bootstrap. Force-upgrades any version < 1.65 because that's
# the minimum version with `rclone serve nfs` (which we rely on for
# the B2-mount-without-FUSE fallback on RunPod's stripped CUDA
# templates).

set -e

# Minimum rclone major.minor — bump this when the codebase relies on
# a newer feature. 1.65 added `serve nfs`. 1.66 made it stable.
RCLONE_MIN_MAJOR=1
RCLONE_MIN_MINOR=65

needs_upgrade=1
current_ver=""
if command -v rclone >/dev/null 2>&1; then
    current_ver=$(rclone version 2>/dev/null | head -1 | awk '{print $2}')
    # Parse "v1.58.1" -> major=1 minor=58
    cur_major=$(echo "$current_ver" | sed -E 's/^v?([0-9]+)\..*/\1/')
    cur_minor=$(echo "$current_ver" | sed -E 's/^v?[0-9]+\.([0-9]+).*/\1/')
    if [ "${cur_major:-0}" -gt "$RCLONE_MIN_MAJOR" ] || \
       { [ "${cur_major:-0}" -eq "$RCLONE_MIN_MAJOR" ] && [ "${cur_minor:-0}" -ge "$RCLONE_MIN_MINOR" ]; }; then
        echo "[install_rclone] already at supported version: $current_ver (need >= ${RCLONE_MIN_MAJOR}.${RCLONE_MIN_MINOR})"
        exit 0
    fi
    echo "[install_rclone] installed version $current_ver is too old (need >= ${RCLONE_MIN_MAJOR}.${RCLONE_MIN_MINOR}) — upgrading"
fi

# rclone's install.sh handles arch detection (amd64 / arm64 / etc.)
# and drops a single statically-linked binary into /usr/bin/rclone.
# Running as root inside the container so no sudo needed; the script
# detects this and skips sudo automatically. The --beta flag isn't
# needed — serve nfs is stable in releases since 1.66.
echo "[install_rclone] Installing/upgrading rclone via official installer..."
curl -fsSL https://rclone.org/install.sh | bash

# Verify
new_ver=$(rclone --version 2>&1 | head -1 | awk '{print $2}')
echo "[install_rclone] Installed: $new_ver"

# Confirm `serve nfs` is now available — that's the actual feature
# this whole upgrade dance exists for.
if rclone serve --help 2>&1 | grep -qE '^\s+nfs\b'; then
    echo "[install_rclone] serve nfs: available"
else
    echo "[install_rclone] WARN: serve nfs NOT in this rclone build — B2 mount fallback will fail."
fi
