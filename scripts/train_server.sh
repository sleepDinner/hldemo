#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

NUM_GPUS="${NUM_GPUS:-2}"
CONFIG="${CONFIG:-configs/train_casia_manifest.yaml}"

torchrun --nproc_per_node="${NUM_GPUS}" -m tools.train --config "${CONFIG}"

