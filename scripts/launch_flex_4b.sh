#!/bin/bash
# 4B AWQ 版（同样的掩码路径）
cd /home/user || exit 1
export NPC_VIS_BLOCKS=/home/user/_ab_scratch/npc_vis_blocks.json
export VLLM_USE_FLASHINFER_SAMPLER=0
exec /home/user/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/c/Users/user/npc-models-qwen3-4b-awq \
  --served-model-name q4b-flex \
  --host 127.0.0.1 --port 8001 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --enforce-eager \
  --enable-prefix-caching \
  --attention-backend FLEX_ATTENTION
