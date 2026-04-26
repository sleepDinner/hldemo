#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/ablation.yaml}"

python -m tools.run_ablation --config "${CONFIG}"

