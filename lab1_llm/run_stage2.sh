#!/bin/bash
# Lab 1, stage 2: best model on the big slice; prefix-cache check; few-shot sensitivity of the best model.
cd "$(dirname "$0")"
L=mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
.venv/bin/python run_llm.py $L zero big results/Meta-Llama-3.1-8B-Instruct-4bit_zero_big.jsonl > results/llama_zero_big.log 2>&1
.venv/bin/python run_llm.py $L few test results/check_prefix_cache.jsonl --prefix-cache > results/check_prefix_cache.log 2>&1
for s in fewshot_s1 fewshot_s2; do
  .venv/bin/python run_llm.py $L few test results/sens_llama_${s}.jsonl --prefix-cache --shots $s > results/sens_llama_${s}.log 2>&1
done
echo done > results/STAGE2_DONE
