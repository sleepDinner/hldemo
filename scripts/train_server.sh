#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-2}"
CONFIG="${CONFIG:-configs/train_casia.yaml}"

torchrun --nproc_per_node="${NUM_GPUS}" -m tools.train --config "${CONFIG}"

