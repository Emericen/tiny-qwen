#!/usr/bin/env bash
# Launch a training run on a fresh CUDA box (RunPod H100 is the target).
#
#   MODEL=Qwen/Qwen3.5-2B VS=station STEPS=300 bash rl/pod.sh
#
# MODE=colocate (default): vLLM runs inside the trainer process, one GPU.
# MODE=server: a separate vLLM server on the same (or another) GPU; the
# trainer pushes new weights over NCCL.
# On a two-GPU box set VLLM_GPU=0 TRAIN_GPU=1 to split them.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
VS=${VS:-station}
STEPS=${STEPS:-200}
GENS=${GENS:-8}
MODE=${MODE:-colocate}
VLLM_GPU=${VLLM_GPU:-0}
TRAIN_GPU=${TRAIN_GPU:-0}
OUT=${OUT:-runs/$(basename "$MODEL")-$VS}
EXTRA=${EXTRA:-}
mkdir -p "$(dirname "$OUT")"

pip install -q -r requirements.txt -r rl/requirements.txt
python -m rl.test_dealer

if [ "$MODE" = server ]; then
  CUDA_VISIBLE_DEVICES=$VLLM_GPU VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --logprobs-mode processed_logprobs --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}' \
    --gpu-memory-utilization 0.35 --max-model-len 4096 --port 8000 > "$OUT.vllm.log" 2>&1 &
  echo "waiting for vllm (log: $OUT.vllm.log)"
  until curl -sf localhost:8000/health > /dev/null; do sleep 5; done
fi

CUDA_VISIBLE_DEVICES=$TRAIN_GPU python -m rl.train \
  --model "$MODEL" --vs "$VS" --steps "$STEPS" --gens "$GENS" --vllm "$MODE" --output "$OUT" $EXTRA \
  2>&1 | tee "$OUT.train.log"
