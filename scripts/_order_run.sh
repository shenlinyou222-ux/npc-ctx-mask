#!/bin/bash
set -u
PY=/home/user/vllm-venv/bin/python
R="/mnt/c/Users/user/Desktop/ds harness/npc-ctx-mask"
cd /home/user/_ab_scratch || exit 1

restart () {
  pkill -f 'served-model-name q17b-flex' || true
  for i in $(seq 1 30); do pgrep -f 'served-model-name q17b-flex' >/dev/null || break; sleep 1; done
  export NPC_VIS_BLOCKS=/home/user/_ab_scratch/npc_vis_blocks.json
  export VLLM_USE_FLASHINFER_SAMPLER=0
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model /mnt/c/Users/user/npc-models-qwen3-1.7b \
    --served-model-name q17b-flex --host 127.0.0.1 --port 8001 \
    --max-model-len 16384 --gpu-memory-utilization 0.84 \
    --max-num-seqs 1 --max-num-batched-tokens 8192 --dtype bfloat16 \
    --enforce-eager --enable-prefix-caching --attention-backend FLEX_ATTENTION \
    > /home/user/_ab_scratch/server_oi.log 2>&1 &
  for i in $(seq 1 90); do
    curl -s -m 2 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 && { echo "server up ${i}s"; return 0; }
    sleep 2
  done
  echo "server failed"; return 1
}

cp "$R/scripts/order_independence.py" .

echo "=== forward ==="
restart || exit 1
"$PY" order_independence.py forward 2>&1 | tail -16

echo "=== reverse ==="
restart || exit 1
"$PY" order_independence.py reverse 2>&1 | tail -16

echo "=== compare ==="
"$PY" - <<'PY'
import json
a = json.load(open('/home/user/_ab_scratch/order2_forward.json'))
b = json.load(open('/home/user/_ab_scratch/order2_reverse.json'))
same = 0
for k in a['answers']:
    s = a['answers'][k] == b['answers'][k]
    same += s
    print(('一致 ' if s else '不同 ') + k.ljust(16) + ' | 正序: ' + a['answers'][k][:34] + ' | 倒序: ' + b['answers'][k][:34])
print('一致 %d / %d' % (same, len(a['answers'])))
PY
