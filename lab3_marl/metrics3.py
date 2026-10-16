"""Lab 3: metrics, seed aggregation, model selection on valid, tests against equal weight, stress and regimes.

Usage: metrics3.py [variant ...]   (default: base). Writes results/metrics3.json and results/curves.json.
"""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

from env import PortfolioEnv
from task import panel

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / os.environ.get('LAB3_RUNS', 'runs_v2')
RES = HERE / os.environ.get('LAB3_RESULTS', 'results_v2')
ALGOS = ['mappo', 'ippo', 'masac', 'isac', 'qmix', 'vdn', 'iql', 'maddpg', 'ppo_single']
MARL = ALGOS[:-1]
STRESS = {'covid_2020': ('valid', '2020-02-14', '2020-06-26'), 'bear_2022': ('test', '2022-01-07', '2022-10-14'),
          'tariff_2025': ('test', '2025-02-14', '2025-05-30')}
WEIGHTS = {'sharpe': 0.35, 'sortino': 0.20, 'max_dd': 0.20, 'cvar95': 0.10, 'turnover_per_year': 0.10,
           'sharpe_seed_std': 0.05}
HIGHER = {'sharpe': True, 'sortino': True, 'max_dd': True, 'cvar95': True, 'turnover_per_year': False,
          'sharpe_seed_std': False}       # max_dd and cvar95 are negative numbers: higher (closer to 0) is better
N_BOOT, BLOCK = 2000, 13


def perf(logret, turnover=None, cost=None, violation=None, w=None, rf=None):
    """Sharpe and Sortino on returns in excess of the cash yield (rf, weekly log), when it is given."""
    r = np.asarray(logret, float)
    x = r - (np.asarray(rf, float) if rf is not None else 0.0)
    years = len(r) / 52
    simple = np.expm1(r)
    cum = np.cumsum(r)
    dd = np.exp(cum - np.maximum.accumulate(np.maximum(cum, 0))) - 1
    xdown = np.sqrt(np.mean(np.minimum(x, 0) ** 2))
    tail = np.sort(simple)[:max(1, int(np.ceil(0.05 * len(r))))]
    out = {'weeks': len(r), 'ann_return': float(np.expm1(r.mean() * 52)), 'ann_vol': float(r.std() * np.sqrt(52)),
           'sharpe': float(x.mean() / x.std() * np.sqrt(52)) if x.std() > 0 else 0.0,
           'sortino': float(x.mean() / xdown * np.sqrt(52)) if xdown > 0 else 0.0,
           'excess_ann_return': float(x.mean() * 52),
           'max_dd': float(dd.min()), 'cvar95': float(tail.mean()), 'total_return': float(np.expm1(cum[-1]))}
    if turnover is not None:
        out['turnover_per_year'] = float(np.sum(turnover) / years)
        out['cost_per_year'] = float(np.sum(cost) / years)
        out['violation_share'] = float(np.mean(violation))
        out['mean_exposure'] = float(np.mean(np.sum(w, 1)))
    return out


def equal_weight(universe, split, cost_bps=10.0, risk_agent=True):
    env = PortfolioEnv(panel(universe), split=split, num_envs=1, cost_bps=cost_bps, risk_agent=risk_agent)
    td = env.reset()
    act = torch.tensor([[4, 4, 4, 4, 8][:env.n_agents]])
    rec = {k: [] for k in ['logret', 'turnover', 'cost', 'violation', 'w', 'cash']}
    for _ in range(env.episode_len):
        td['agents', 'action'] = act
        env.step(td)
        for k in rec:
            rec[k].append(env.info[k][0].float().numpy())
    rec = {k: np.array(v) for k, v in rec.items()}
    mask = panel(universe)['split'] == split
    rec['dates'] = panel(universe)['dates'][mask][:env.episode_len].astype(str)
    return rec


def load_run(path, name):
    d = np.load(path / f'{name}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files}


def block_bootstrap_diff(a, b, seed=0):
    """Paired circular block bootstrap of the weekly return difference and the Sharpe difference."""
    rng = np.random.default_rng(seed)
    n = len(a)
    starts = np.arange(n)
    dm, ds = [], []
    sh = lambda x: x.mean() / x.std() * np.sqrt(52)
    for _ in range(N_BOOT):
        idx = np.concatenate([(s + np.arange(BLOCK)) % n for s in rng.choice(starts, n // BLOCK + 1)])[:n]
        dm.append((a[idx] - b[idx]).mean() * 52)
        ds.append(sh(a[idx]) - sh(b[idx]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {'ann_return_diff': float((a - b).mean() * 52), 'ann_return_diff_ci95': ci(dm),
            'sharpe_diff': float(sh(a) - sh(b)), 'sharpe_diff_ci95': ci(ds),
            'p_sharpe_le_0': float(np.mean(np.array(ds) <= 0))}


def window(rec, a, b):
    m = (rec['dates'] >= a) & (rec['dates'] <= b)
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in rec.items()}


def summarize(variant):
    out = {'per_algo': {}, 'per_run': {}}
    for algo in ALGOS:
        runs = sorted((RUNS / variant).glob(f'{algo}_s*'))
        runs = [r for r in runs if (r / 'meta.json').exists()]
        if not runs:
            continue
        per = {}
        for r in runs:
            meta = json.loads((r / 'meta.json').read_text())
            m = {}
            for split in ['valid', 'test', 'foodtech', 'test_cost20']:
                e = load_run(r, f'eval_{split}')
                m[split] = perf(e['logret'], e['turnover'], e['cost'], e['violation'], e['w'], e['cash'])
            for sname, (split, a, b) in STRESS.items():
                e = window(load_run(r, f'eval_{split}'), a, b)
                m[sname] = perf(e['logret'], e['turnover'], e['cost'], e['violation'], e['w'], e['cash'])
            rf_split = {sp: load_run(r, f'eval_{sp}')['cash'] for sp in ['valid', 'test']}
            if (r / 'cf_valid.npz').exists():
                for split in ['valid', 'test']:
                    cf = load_run(r, f'cf_{split}')
                    full = m[split]
                    m[f'contribution_{split}'] = {
                        k: {'sharpe': full['sharpe'] - perf(v, rf=rf_split[split])['sharpe'],
                            'ann_return': full['ann_return'] - perf(v, rf=rf_split[split])['ann_return']}
                        for k, v in cf.items()}
            m['train_seconds'] = meta['train_seconds']
            m['n_params_policy'] = meta['n_params_policy']
            per[r.name] = m
        out['per_run'][algo] = per
        agg = {}
        for split in ['valid', 'test', 'foodtech', 'test_cost20', *STRESS]:
            keys = per[next(iter(per))][split].keys()
            agg[split] = {k: {'mean': float(np.mean([p[split][k] for p in per.values()])),
                              'std': float(np.std([p[split][k] for p in per.values()], ddof=1)) if len(per) > 1 else 0.0}
                          for k in keys}
        agg['seeds'] = len(per)
        agg['train_seconds'] = float(np.mean([p['train_seconds'] for p in per.values()]))
        agg['n_params_policy'] = int(np.mean([p['n_params_policy'] for p in per.values()]))
        if 'contribution_valid' in per[next(iter(per))]:
            for split in ['valid', 'test']:
                agents = per[next(iter(per))][f'contribution_{split}'].keys()
                agg[f'contribution_{split}'] = {
                    a: {k: float(np.mean([p[f'contribution_{split}'][a][k] for p in per.values()]))
                        for k in ['sharpe', 'ann_return']} for a in agents}
        out['per_algo'][algo] = agg
    return out


def select(base):
    """Admission + composite score on VALID (min-max over admitted algorithms)."""
    rows = {}
    for algo, a in base['per_algo'].items():
        v = a['valid']
        rows[algo] = {'sharpe': v['sharpe']['mean'], 'sortino': v['sortino']['mean'], 'max_dd': v['max_dd']['mean'],
                      'cvar95': v['cvar95']['mean'], 'turnover_per_year': v['turnover_per_year']['mean'],
                      'sharpe_seed_std': v['sharpe']['std'], 'violation_share': v['violation_share']['mean'],
                      'ann_return': v['ann_return']['mean']}
    admitted = {a: r for a, r in rows.items() if r['violation_share'] <= 0.01 and r['ann_return'] > 0}
    score = {}
    for a, r in admitted.items():
        s = 0.0
        for k, wgt in WEIGHTS.items():
            vals = [x[k] for x in admitted.values()]
            lo, hi = min(vals), max(vals)
            z = 0.5 if hi == lo else (r[k] - lo) / (hi - lo)
            s += wgt * (z if HIGHER[k] else 1 - z)
        score[a] = s
    marl = {a: v for a, v in score.items() if a in MARL}          # the single-agent PPO is a baseline, not a candidate
    best = max(marl, key=marl.get) if marl else None
    top = sorted(marl, key=marl.get, reverse=True)
    return {'rows_valid': rows, 'admitted': list(admitted), 'score': score, 'best': best, 'ranking': top}


def main():
    RES.mkdir(exist_ok=True)
    variants = sys.argv[1:] or ['base']
    result = {v: summarize(v) for v in variants}
    sel = select(result['base'])
    result['selection'] = sel
    result['top4_base'] = sel['ranking'][:4]
    final_variant = 'long' if 'long' in result and result['long']['per_algo'] else 'base'
    if final_variant == 'long':
        result['selection_long'] = select(result['long'])
        sel = result['selection_long']
    result['final_variant'] = final_variant
    base = result[final_variant]
    ew = {split: equal_weight(u, split) for u, split in [('sectors', 'valid'), ('sectors', 'test'),
                                                         ('foodtech', 'foodtech')]}
    ew['test_cost20'] = equal_weight('sectors', 'test', cost_bps=20.0)
    result['equal_weight'] = {s: perf(e['logret'], e['turnover'], e['cost'], e['violation'], e['w'], e['cash'])
                              for s, e in ew.items()}
    for sname, (split, a, b) in STRESS.items():
        e = window(ew[split], a, b)
        result['equal_weight'][sname] = perf(e['logret'], e['turnover'], e['cost'], e['violation'], e['w'], e['cash'])
    # Best policy vs equal weight: seed-averaged weekly returns, paired block bootstrap; and per-seed Sharpe.
    tests = {}
    for algo in [sel['best'], 'ppo_single'] + [a for a in MARL if a != sel['best']]:
        if algo is None or algo not in base['per_run']:
            continue
        tests[algo] = {}
        for split in ['valid', 'test', 'foodtech']:
            series = np.mean([load_run(RUNS / final_variant / r, f'eval_{split}')['logret'] for r in base['per_run'][algo]], 0)
            rf = ew[split]['cash']
            tests[algo][split] = block_bootstrap_diff(series - rf, ew[split]['logret'] - rf)
            tests[algo][split]['seeds_beating_ew_sharpe'] = int(sum(
                perf(load_run(RUNS / final_variant / r, f'eval_{split}')['logret'], rf=ew[split]['cash'])['sharpe']
                > result['equal_weight'][split]['sharpe']
                for r in base['per_run'][algo]))
    result['vs_equal_weight'] = tests
    # Equal weight scaled to the selected policy's average exposure (the rest in cash): the same risk level without learning.
    if sel['best']:
        scaled = {}
        for split in ['valid', 'test', 'foodtech']:
            expo = float(np.mean([load_run(RUNS / final_variant / r, f'eval_{split}')['w'].sum(1).mean()
                                  for r in base['per_run'][sel['best']]]))
            lr = np.log(expo * np.exp(ew[split]['logret']) + (1 - expo) * np.exp(ew[split]['cash']))
            scaled[split] = {'exposure': expo, **perf(lr, rf=ew[split]['cash'])}
        result['equal_weight_scaled_to_best_exposure'] = scaled
    # Regimes on test: weeks where the equal-weight 13-week return is negative and volatility is above median.
    if sel['best']:
        e = ew['test']
        r_ew = pd.Series(e['logret'])
        trend = r_ew.rolling(13, min_periods=4).sum().shift(1)
        vol = r_ew.rolling(13, min_periods=4).std().shift(1)
        stress = (trend < 0) & (vol > vol.median())
        pol = np.mean([load_run(RUNS / final_variant / r, 'eval_test')['logret'] for r in base['per_run'][sel['best']]], 0)
        diff = pol - e['logret']                      # cash yield cancels in the difference only when both hold it
        result['regimes_test'] = {
            'stress_weeks': int(stress.sum()), 'calm_weeks': int((~stress).sum()),
            'excess_ann_stress': float(diff[stress.to_numpy()].mean() * 52),
            'excess_ann_calm': float(diff[~stress.to_numpy()].mean() * 52)}
        roll = pd.Series(diff).rolling(26).sum()
        result['regimes_test']['worst_26w_excess'] = float(roll.min())
        result['regimes_test']['worst_26w_end'] = str(e['dates'][int(roll.idxmin())]) if roll.notna().any() else None
    # Learning curves (episode reward mean during collection), mean over seeds.
    curves = {}
    for variant in [v for v in ('base', 'long') if v in result]:
        curves[variant] = {}
        for algo in ALGOS:
            series = []
            for r in sorted((RUNS / variant).glob(f'{algo}_s*')):
                f = list(r.glob('*/*/scalars/collection_agents_reward_episode_reward_mean.csv'))
                if f:
                    series.append(pd.read_csv(f[0], header=None)[1].to_numpy())
            if series:
                n = min(len(s) for s in series)
                arr = np.array([s[:n] for s in series])
                curves[variant][algo] = {'mean': arr.mean(0).tolist(), 'std': arr.std(0).tolist(), 'frames_per_point': 6000}
    (RES / 'curves.json').write_text(json.dumps(curves))
    (RES / 'metrics3.json').write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    print('final variant:', final_variant, 'best on valid:', sel['best'], {k: round(v, 3) for k, v in sel['score'].items()})
    print('top4 at 1M:', result['top4_base'])
    for algo, a in base['per_algo'].items():
        v, t = a['valid'], a['test']
        print(f"{algo:<11} valid Sharpe {v['sharpe']['mean']:.2f}±{v['sharpe']['std']:.2f} ret {v['ann_return']['mean']:.3f} "
              f"DD {v['max_dd']['mean']:.3f} viol {v['violation_share']['mean']:.3f} turn {v['turnover_per_year']['mean']:.1f} | "
              f"test Sharpe {t['sharpe']['mean']:.2f}±{t['sharpe']['std']:.2f} ret {t['ann_return']['mean']:.3f}")
    print('EW', {s: round(m['sharpe'], 2) for s, m in result['equal_weight'].items()})


if __name__ == '__main__':
    main()
