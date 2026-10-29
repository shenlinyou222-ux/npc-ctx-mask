#!/bin/bash
cd /home/user/_ab_scratch || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in 0 1 2 3 4 5 6 7; do
  timeout 400 /home/user/vllm-venv/bin/python wiki_test.py bf16 1 "$i" 2>/dev/null \
    | grep -vE 'Loading weights|UserWarning|pick' 
  sleep 1
done
