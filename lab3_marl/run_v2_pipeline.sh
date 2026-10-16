#!/bin/bash
# Lab 3 v2 driver: after stage 1, run the 4 best MARL algorithms on valid plus single-agent PPO with 3M frames
# (stage 2), select the final algorithm on valid among them, run its ablations with 3M frames (stage 3).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/bin/python
until [ -f runs_v2/STAGE1_DONE ]; do sleep 30; done
$PY metrics3.py base lam0 lam05 > logs_v2/metrics_stage1.txt 2>&1
TOP=$($PY -c "import json; print(' '.join(json.load(open('results_v2/metrics3.json'))['top4_base']))")
echo "stage 2: $TOP ppo_single" > logs_v2/stage2_plan.txt
for a in $TOP ppo_single; do for s in 0 1 2 3 4; do echo "$a $s long"; done; done \
  | xargs -P 6 -L 1 bash -c '.venv/bin/python train.py $0 $1 --variant $2 --frames 3000000 > logs_v2/$2_$0_s$1.log 2>&1'
$PY metrics3.py base long lam0 lam05 > logs_v2/metrics_stage2.txt 2>&1
BEST=$($PY -c "import json; print(json.load(open('results_v2/metrics3.json'))['selection_long']['best'])")
echo "stage 3: $BEST" > logs_v2/stage3_plan.txt
for v in norisk nocost noconc; do for s in 0 1 2 3 4; do echo "$BEST $s $v"; done; done \
  | xargs -P 6 -L 1 bash -c '.venv/bin/python train.py $0 $1 --variant $2 --frames 3000000 > logs_v2/$2_$0_s$1.log 2>&1'
$PY metrics3.py base long lam0 lam05 norisk nocost noconc > logs_v2/metrics_final.txt 2>&1
mkdir -p figures_v2 && $PY analysis3.py > logs_v2/analysis.txt 2>&1
echo done > runs_v2/ALL_DONE
