#!/usr/bin/env bash
set -euo pipefail

echo "=== Nimzo Engine Arch Linux Setup ==="
echo "Checking dependencies..."

if command -v pacman >/dev/null 2>&1; then
    echo "Arch Linux detected. Recommending system packages installation:"
    echo "  sudo pacman -S --needed base-devel git python zstd lm_sensors python-pytorch-opt"
    echo "Stockfish and cutechess are AUR-only (not in the official repos):"
    echo "  yay -S stockfish cutechess"
    echo "ordo (Elo rating tool) has no AUR package; build from source:"
    echo "  git clone https://github.com/michiguel/Ordo.git && cd Ordo && make && sudo install -Dm755 ordo /usr/local/bin/ordo"
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with system site packages..."
    python -m venv --system-site-packages .venv
fi

source .venv/bin/activate
pip install python-chess numpy pyyaml tqdm pytest zstandard customtkinter

echo "Setup complete. Verifying PyTorch and chess..."
python -c "import torch, chess; print('Torch version:', torch.__version__); print('CPU capability:', getattr(torch.backends.cpu, 'get_cpu_capability', lambda: 'AVX2')()); print('Threads:', torch.get_num_threads())"
