#!/bin/bash
cd /home/user/_ab_scratch || exit 1
for i in 0 1 2 3 4 5 6 7 8 9; do
  timeout 300 /home/user/vllm-venv/bin/python session_once.py "$i" 2>/dev/null | grep RESULT
  sleep 2
done
