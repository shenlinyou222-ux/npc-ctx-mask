#!/bin/bash
set -u
PY=/home/user/vllm-venv/bin/python
pkill -f 'served-model-name q17b-flex' 2>/dev/null || true
for i in $(seq 1 20); do pgrep -f 'served-model-name q17b-flex' >/dev/null || break; sleep 1; done
export NPC_VIS_BLOCKS=/home/user/_ab_scratch/npc_vis_blocks.json
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /home/user/_ab_scratch || exit 1
nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model /mnt/c/Users/user/npc-models-qwen3-1.7b \
  --served-model-name q17b-flex --host 127.0.0.1 --port 8001 \
  --max-model-len 16384 --gpu-memory-utilization 0.84 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 --dtype bfloat16 \
  --enforce-eager --enable-prefix-caching --attention-backend FLEX_ATTENTION \
  > /home/user/_ab_scratch/server_io.log 2>&1 &
echo "started $!"
for i in $(seq 1 90); do
  curl -s -m 2 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 && { echo "up ${i}s"; exit 0; }
  sleep 2
done
echo "failed"; tail -5 /home/user/_ab_scratch/server_io.log; exit 1
