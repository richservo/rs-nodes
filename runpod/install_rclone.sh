#!/usr/bin/env bash
# Install rclone on the pod if it's not already present. Idempotent —
# safe to call on every bootstrap. rclone is the standard tool for
# syncing /workspace with Backblaze B2 / S3 / Wasabi / R2 / etc. — see
# runpod/b2_helpers.sh for the rs-nodes wrapper functions.

set -e

if command -v rclone >/dev/null 2>&1; then
    echo "[install_rclone] already installed: $(rclone --version 2>&1 | head -1)"
    exit 0
fi

echo "[install_rclone] Installing rclone..."

# rclone's install.sh handles arch detection (amd64 / arm64 / etc.)
# and drops a single statically-linked binary into /usr/bin/rclone.
# Running as root inside the container so no sudo needed; the script
# detects this and skips sudo automatically.
curl -fsSL https://rclone.org/install.sh | bash

echo "[install_rclone] Installed: $(rclone --version | head -1)"
