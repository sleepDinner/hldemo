#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/test_cross_dataset_manifest.yaml}"

python -m tools.test --config "${CONFIG}"

