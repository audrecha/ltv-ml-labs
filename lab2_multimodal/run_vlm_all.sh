#!/bin/bash
# Lab 2: zero-shot VLM runs on the 300 lab 1 test users (one model in memory at a time).
cd "$(dirname "$0")"
for m in mlx-community/Qwen2.5-VL-7B-Instruct-4bit mlx-community/gemma-3-4b-it-4bit; do
  for mode in image image_text; do
    name=$(basename $m)_${mode}_test
    .venv/bin/python run_vlm.py $m $mode results/vlm_$name.jsonl > results/vlm_$name.log 2>&1
  done
done
echo done > results/VLM_DONE
