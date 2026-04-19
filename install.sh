#!/bin/bash
echo "============================================"
echo " RS Nodes - Installation"
echo "============================================"
echo

cd "$(dirname "$0")"

# Initialize LTX-2 submodule (required for LoRA training)
echo "[1/3] Initializing LTX-2 submodule..."
git submodule update --init
echo

# Install Python dependencies (won't touch torch/ComfyUI packages)
echo "[2/3] Installing Python dependencies..."
pip install -r requirements.txt
echo

# Install ROSE optimizer (stateless optimizer for LoRA training)
echo "[3/3] Installing ROSE optimizer..."
pip install git+https://github.com/MatthewK78/Rose
echo

echo "============================================"
echo " Installation complete!"
echo "============================================"
