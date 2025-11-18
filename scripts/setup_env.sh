#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD/embeddings}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

echo "[*] Project dir: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"

# 1. Install uv if missing
if ! command -v uv &>/dev/null; then
  echo "[*] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[*] uv already installed."
  export PATH="$HOME/.local/bin:$PATH"
fi

uv --version || { echo "uv not found after install"; exit 1; }

# 2. Create / reuse venv
cd "$PROJECT_DIR"
if [ ! -d "$VENV_DIR" ]; then
  echo "[*] Creating venv at $VENV_DIR..."
  uv venv --python "$PYTHON_VERSION"
else
  echo "[*] Venv already exists at $VENV_DIR."
fi

# Activate venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# 3. Install PyTorch (CUDA 12.4 wheel by default)
echo "[*] Installing PyTorch..."
uv pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu124

# 4. Install the rest of the Python deps
echo "[*] Installing transformers/datasets/hf hub/etc..."
uv pip install \
  "transformers>=4.44" \
  "datasets>=2.20" \
  huggingface_hub \
  scikit-learn \
  tqdm

uv pip install hf_transfer
uv pip install scikit-learn

echo
echo "[✓] Environment ready."
echo "To use it now run:"
echo "  source $VENV_DIR/bin/activate"
