#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CONFIG_FILE=${CONFIG_FILE:-configs/e2e_rel_ovi6.yaml}
export DETECTOR_CKPT=${DETECTOR_CKPT:-./checkpoints/pretrained_detector/oiv6_det.pth}
export OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/transformer-pl-oiv6-sgdet-seed${SEED:-678}}
exec "$SCRIPT_DIR/../run.sh" test transformer sgdet "$@"
