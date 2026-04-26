#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/test_cross_dataset.yaml}"

python -m tools.test --config "${CONFIG}"

