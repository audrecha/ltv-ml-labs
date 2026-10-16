#!/bin/bash
# Lab 3 v2 (cash earns the T-bill yield), stage 1: main series (9 algorithms x 5 seeds, 1M frames) and the
# risk-penalty ablation (lam_var 0 and 0.5) for QMIX and MAPPO; 6 runs in parallel.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p logs_v2
{
  for a in mappo ippo masac isac qmix vdn iql maddpg ppo_single; do for s in 0 1 2 3 4; do echo "$a $s base"; done; done
  for v in lam0 lam05; do for a in qmix mappo; do for s in 0 1 2 3 4; do echo "$a $s $v"; done; done; done
} | xargs -P 6 -L 1 bash -c '.venv/bin/python train.py $0 $1 --variant $2 --frames 1000000 > logs_v2/$2_$0_s$1.log 2>&1'
echo done > runs_v2/STAGE1_DONE
