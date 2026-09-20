#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <train|test> <motifs|vctree|transformer> <predcls|sgcls|sgdet> [CONFIG_OVERRIDES...]" >&2
  exit 2
fi

ACTION=$1
BACKBONE=$2
TASK=$3
shift 3

case "$ACTION" in
  train) ENTRYPOINT=tools/relation_train_net.py ;;
  test) ENTRYPOINT=tools/relation_test_net.py ;;
  *) echo "Unsupported action: $ACTION" >&2; exit 2 ;;
esac

case "$BACKBONE" in
  motifs)
    PREDICTOR=MotifPredictor_PL
    DEFAULT_WEIGHTS='[1, 2, 16]'
    DEFAULT_LR=0.01
    ;;
  vctree)
    PREDICTOR=VCTreePredictor_PL
    DEFAULT_WEIGHTS='[1, 4, 16]'
    DEFAULT_LR=0.01
    ;;
  transformer)
    PREDICTOR=TransformerPredictor_PL
    DEFAULT_WEIGHTS='[1, 4, 16]'
    DEFAULT_LR=0.001
    ;;
  *) echo "Unsupported backbone: $BACKBONE" >&2; exit 2 ;;
esac

case "$TASK" in
  predcls)
    USE_GT_BOX=True
    USE_GT_LABEL=True
    DEFAULT_TEST_BATCH=12
    ;;
  sgcls)
    USE_GT_BOX=True
    USE_GT_LABEL=False
    DEFAULT_TEST_BATCH=2
    ;;
  sgdet)
    USE_GT_BOX=False
    USE_GT_LABEL=False
    DEFAULT_TEST_BATCH=2
    ;;
  *) echo "Unsupported task: $TASK" >&2; exit 2 ;;
esac

GPUS=${GPUS:-0,1}
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NPROC=${NPROC:-${#GPU_IDS[@]}}
MASTER_PORT=${MASTER_PORT:-29500}
SEED=${SEED:-678}
CONFIG_FILE=${CONFIG_FILE:-configs/e2e_relation_X_101_32_8_FPN_1x.yaml}
GLOVE_DIR=${GLOVE_DIR:-./glove}
DETECTOR_CKPT=${DETECTOR_CKPT:-./checkpoints/pretrained_faster_rcnn/model_final.pth}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/${BACKBONE}-pl-${TASK}-seed${SEED}}
NUM_EXPERTS=${NUM_EXPERTS:-3}
EXPERT_MODE=${EXPERT_MODE:-hbt_b_t}
KNOWLEDGE_WEIGHTS=${KNOWLEDGE_WEIGHTS:-$DEFAULT_WEIGHTS}
BASE_LR=${BASE_LR:-$DEFAULT_LR}
TRAIN_BATCH=${TRAIN_BATCH:-12}
TEST_BATCH=${TEST_BATCH:-$DEFAULT_TEST_BATCH}
MAX_ITER=${MAX_ITER:-20000}
VAL_PERIOD=${VAL_PERIOD:-2000}
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-2000}
SYNC_GATHER=${SYNC_GATHER:-False}

export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

COMMON_ARGS=(
  --config-file "$CONFIG_FILE"
  --loss_option PLME_LOSS
  --num_experts "$NUM_EXPERTS"
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX "$USE_GT_BOX"
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL "$USE_GT_LABEL"
  MODEL.ROI_RELATION_HEAD.USE_RELATION_AWARE_GATING False
  MODEL.ROI_RELATION_HEAD.USE_PER_CLASS_CONTEXT_AWARE False
  MODEL.ROI_RELATION_HEAD.PREDICTOR "$PREDICTOR"
  MODEL.ROI_RELATION_HEAD.EXPERT_MODE "$EXPERT_MODE"
  MODEL.ROI_RELATION_HEAD.KNOWLEDGE_WEIGHTS "$KNOWLEDGE_WEIGHTS"
  TEST.IMS_PER_BATCH "$TEST_BATCH"
  DTYPE float16
  GLOVE_DIR "$GLOVE_DIR"
  MODEL.PRETRAINED_DETECTOR_CKPT "$DETECTOR_CKPT"
  OUTPUT_DIR "$OUTPUT_DIR"
)

ACTION_ARGS=()
if [[ "$ACTION" == train ]]; then
  ACTION_ARGS+=(--seed "$SEED")
  COMMON_ARGS+=(
    SOLVER.IMS_PER_BATCH "$TRAIN_BATCH"
    SOLVER.BASE_LR "$BASE_LR"
    SOLVER.MAX_ITER "$MAX_ITER"
    SOLVER.VAL_PERIOD "$VAL_PERIOD"
    SOLVER.CHECKPOINT_PERIOD "$CHECKPOINT_PERIOD"
    SOLVER.PRE_VAL False
  )
else
  # Object-based gathering preserves dataset indices across multi-GPU evaluation.
  COMMON_ARGS+=(TEST.RELATION.SYNC_GATHER "$SYNC_GATHER")
fi

echo "Action: $ACTION | Backbone: $BACKBONE | Task: $TASK"
echo "GPUs: $GPUS | Seed: $SEED | Output: $OUTPUT_DIR"
echo "Experts: $NUM_EXPERTS | Mode: $EXPERT_MODE | Voting weights: $KNOWLEDGE_WEIGHTS"

COMMAND=(
  python -m torch.distributed.launch
  --master_port "$MASTER_PORT"
  --nproc_per_node "$NPROC"
  "$ENTRYPOINT"
  "${ACTION_ARGS[@]}"
  "${COMMON_ARGS[@]}"
  "$@"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'Command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
exec "${COMMAND[@]}"
