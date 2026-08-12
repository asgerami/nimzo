#!/usr/bin/env bash
set -euo pipefail

# Systemd wrapper script to prevent machine freezing and thermal throttling
PHYSICAL_CORES=$(lscpu -p | grep -v '^#' | cut -d, -f2 | sort -u | wc -l)
QUOTA=$(( (PHYSICAL_CORES - 1) * 100 ))

echo "Launching training with CPUQuota=${QUOTA}% and MemoryMax=5G"

export OMP_NUM_THREADS=$PHYSICAL_CORES
export MKL_NUM_THREADS=$PHYSICAL_CORES

if command -v systemd-run >/dev/null 2>&1; then
    systemd-run --user --scope \
        -p CPUQuota=${QUOTA}% \
        -p MemoryMax=5G \
        nice -n 19 chrt --batch 0 \
        python -m nimzo.train "$@"
else
    echo "[Warning] systemd-run not available, running directly with nice"
    nice -n 19 python -m nimzo.train "$@"
fi
