@echo off
echo ============================================
echo  RS Nodes - Installation
echo ============================================
echo.

cd /d "%~dp0"

:: Initialize LTX-2 submodule (required for LoRA training)
echo [1/3] Initializing LTX-2 submodule...
git submodule update --init
echo.

:: Install Python dependencies (won't touch torch/ComfyUI packages)
echo [2/3] Installing Python dependencies...
pip install -r requirements.txt
echo.

:: Pre-download InsightFace antelopev2 face detection + recognition pack (~300MB)
echo [3/3] Pre-downloading InsightFace antelopev2 models...
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])" || echo   (download will retry on first use)
echo.

echo ============================================
echo  Installation complete!
echo ============================================
echo.
echo Speaker-attributed dialogue captioning (voice_refs_folder) uses
echo speechbrain ECAPA-TDNN, which auto-downloads on first use. No
echo HuggingFace token or terms acceptance required.
echo.
pause
