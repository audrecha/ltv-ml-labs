"""Lab 3: train one (algorithm, seed, variant) with BenchMARL and evaluate the policy step by step.

Usage: train.py <algo> <seed> [--variant base|norisk|nocost|noconc] [--frames N]
algo: mappo ippo masac isac qmix vdn iql maddpg (MARL) | ppo_single (one agent, continuous, same library)
Output: runs/<variant>/<algo>_s<seed>/ — BenchMARL logs (csv), eval_<split>.npz, meta.json.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
from benchmarl.algorithms import (IppoConfig, IqlConfig, IsacConfig, MaddpgConfig, MappoConfig, MasacConfig,
                                  QmixConfig, VdnConfig)
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from tensordict.nn import set_interaction_type
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from env import PortfolioEnv
from task import PortfolioTask, panel

HERE = pathlib.Path(__file__).resolve().parent
ALGOS = {'mappo': MappoConfig, 'ippo': IppoConfig, 'masac': MasacConfig, 'isac': IsacConfig, 'qmix': QmixConfig,
         'vdn': VdnConfig, 'iql': IqlConfig, 'maddpg': MaddpgConfig, 'ppo_single': MappoConfig}
CONTINUOUS = {'maddpg', 'ppo_single'}            # MADDPG supports continuous actions only; single PPO is continuous
VARIANTS = {'base': {}, 'long': {}, 'smoke': {}, 'norisk': {'risk_agent': False}, 'nocost': {'reward_cost': False},
            'noconc': {'lam_conc': 0.0}, 'lam0': {'lam_var': 0.0}, 'lam05': {'lam_var': 0.5},
            'lam0_nostop': {'lam_var': 0.0, 'dd_stop': None}}
STRESS = {'covid_2020': ('valid', '2020-02-14', '2020-06-26'), 'bear_2022': ('test', '2022-01-07', '2022-10-14'),
          'tariff_2025': ('test', '2025-02-14', '2025-05-30')}
HYPER = {'lr': 3e-4, 'gamma': 0.99, 'on_policy_n_envs_per_worker': 60, 'on_policy_collected_frames_per_batch': 6000,
         'on_policy_n_minibatch_iters': 10, 'on_policy_minibatch_size': 600, 'off_policy_n_envs_per_worker': 60,
         'off_policy_collected_frames_per_batch': 6000, 'off_policy_n_optimizer_steps': 200,
         'off_policy_train_batch_size': 256, 'off_policy_memory_size': 300_000, 'exploration_eps_init': 0.8,
         'exploration_eps_end': 0.02, 'mlp_cells': [128, 128]}


def neutral(env, k):
    """Neutral action of agent k: a sector agent gives both assets score 1, the risk agent sets exposure 1.0."""
    if env.continuous:
        return torch.tensor([1.0, 1.0]) if k == 4 else torch.tensor([0.0, 0.0])
    return torch.tensor(8 if k == 4 else 4)


def evaluate(experiment, env_kwargs, universe, split, continuous, replace_agent=None, cost_bps=None):
    """Deterministic pass over a whole split; returns per-step arrays.
    replace_agent: agent index whose action is replaced by the neutral one (counterfactual contribution)."""
    kw = dict(env_kwargs)
    if cost_bps is not None:
        kw['cost_bps'] = cost_bps
    env = PortfolioEnv(panel(universe), split=split, num_envs=1, continuous=continuous, **kw)
    policy = experiment.policy
    out = {k: [] for k in ['logret', 'turnover', 'cost', 'excess', 'violation', 'w', 'action', 'cash']}
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        for _ in range(env.episode_len):
            td = policy(td)
            if replace_agent is not None:
                act = td['agents', 'action'].clone()
                act[0, replace_agent] = neutral(env, replace_agent).to(act.dtype)
                td['agents', 'action'] = act
            out['action'].append(td['agents', 'action'][0].float().numpy())
            td = env.step(td)
            for k in ['logret', 'turnover', 'cost', 'excess', 'violation', 'w', 'cash']:
                out[k].append(env.info[k][0].float().numpy())
            td = step_mdp(td)
    res = {k: np.array(v) for k, v in out.items()}
    mask = panel(universe)['split'] == split
    res['dates'] = panel(universe)['dates'][mask][:env.episode_len].astype(str)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('algo')
    ap.add_argument('seed', type=int)
    ap.add_argument('--variant', default='base')
    ap.add_argument('--frames', type=int, default=1_000_000)
    ap.add_argument('--root', default='runs_v2')
    args = ap.parse_args()
    env_kwargs = dict(VARIANTS[args.variant])
    single = args.algo == 'ppo_single'
    if single:
        env_kwargs['single'] = True
    out = HERE / args.root / args.variant / f'{args.algo}_s{args.seed}'
    out.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_frames = args.frames
    cfg.loggers = ['csv']
    cfg.evaluation = False
    cfg.render = False
    cfg.create_json = False
    cfg.checkpoint_at_end = True
    cfg.exclude_buffer_from_checkpoint = True       # policy weights only: the replay buffer is not needed later
    cfg.save_folder = str(out)
    cfg.prefer_continuous_actions = args.algo in CONTINUOUS
    cfg.share_policy_params = True
    for k, v in HYPER.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.exploration_anneal_frames = args.frames // 2
    model = MlpConfig.get_from_yaml()
    model.num_cells = HYPER['mlp_cells']
    task = PortfolioTask.SECTORS.get_task(config={'universe': 'sectors', 'split': 'train', **env_kwargs})
    algo_cfg = ALGOS[args.algo].get_from_yaml()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    exp = Experiment(task=task, algorithm_config=algo_cfg, model_config=model, critic_model_config=model,
                     seed=args.seed, config=cfg)
    t0 = time.perf_counter()
    exp.run()
    train_s = time.perf_counter() - t0

    eval_kwargs = {k: v for k, v in env_kwargs.items()}
    cont = args.algo in CONTINUOUS
    t0 = time.perf_counter()
    for universe, split in [('sectors', 'valid'), ('sectors', 'test'), ('foodtech', 'foodtech')]:
        res = evaluate(exp, eval_kwargs, universe, split, cont)
        np.savez(out / f'eval_{split}.npz', **res)
    # stress: doubled transaction costs on the held-out test
    np.savez(out / 'eval_test_cost20.npz', **evaluate(exp, eval_kwargs, 'sectors', 'test', cont, cost_bps=20.0))
    # cooperation: each agent replaced by its neutral action (MARL only)
    if not single:
        n_agents = 4 + int(env_kwargs.get('risk_agent', True))
        for split in ['valid', 'test']:
            cf = {f'agent{k}': evaluate(exp, eval_kwargs, 'sectors', split, cont, replace_agent=k)['logret']
                  for k in range(n_agents)}
            np.savez(out / f'cf_{split}.npz', **cf)
    infer_s = time.perf_counter() - t0
    (out / 'meta.json').write_text(json.dumps({
        'algo': args.algo, 'seed': args.seed, 'variant': args.variant, 'frames': args.frames,
        'train_seconds': train_s, 'eval_seconds_3_splits': infer_s, 'continuous': cont, 'env_kwargs': env_kwargs,
        'hyper': HYPER, 'n_params_policy': int(sum(p.numel() for p in exp.policy.parameters())),
        'exp_folder': str(exp.folder_name)}, indent=2))
    print('done', args.algo, args.seed, args.variant, f'{train_s:.0f}s')


if __name__ == '__main__':
    main()
