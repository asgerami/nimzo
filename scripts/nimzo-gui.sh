#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src

DEFAULT_CONFIG="configs/base.yaml"
DEFAULT_WEIGHTS="runs/base_v4/ckpt_150000.pt"

if [[ "$*" == *"--config"* ]]; then
  python -m nimzo.gui "$@"
else
  python -m nimzo.gui --config "$DEFAULT_CONFIG" --weights "$DEFAULT_WEIGHTS" "$@"
fi
