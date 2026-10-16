#!/bin/bash
# Lab 1: all LLM x mode runs on the test set, sequentially (one model in memory at a time).
cd "$(dirname "$0")"
for m in mlx-community/Qwen2.5-7B-Instruct-4bit mlx-community/Meta-Llama-3.1-8B-Instruct-4bit mlx-community/Mistral-7B-Instruct-v0.3-4bit; do
  for mode in few zero; do
    name=$(basename $m)_${mode}_test
    .venv/bin/python run_llm.py $m $mode test results/$name.jsonl > results/$name.log 2>&1
  done
done
echo done > results/ALL_DONE
