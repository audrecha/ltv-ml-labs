"""Lab 1: metrics for LLM runs and tabular baselines on the same users.

Usage: evaluate.py <split> [results/*.jsonl ...]
An invalid LLM answer counts as a wrong prediction (label 'invalid') in every quality metric.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef)

HERE = pathlib.Path(__file__).resolve().parent
CLASSES = ['none', 'low', 'high']
PRICES = json.loads((HERE / 'prices.json').read_text()) if (HERE / 'prices.json').exists() else {}
N_BOOT = 1000


def to_class(y):
    return np.where(y <= 0, 'none', np.where(y <= 3, 'low', 'high'))


def quality(gold, pred, seed=0):
    gold, pred = np.asarray(gold), np.asarray(pred)
    macro = lambda g, p: f1_score(g, p, labels=CLASSES, average='macro', zero_division=0)
    rng = np.random.default_rng(seed)
    boots = [macro(gold[idx], pred[idx]) for idx in (rng.integers(0, len(gold), len(gold)) for _ in range(N_BOOT))]
    per_class = f1_score(gold, pred, labels=CLASSES, average=None, zero_division=0)
    order = {c: i for i, c in enumerate(CLASSES)}
    valid = np.array([p in order for p in pred])
    dist = np.array([abs(order[g] - order[p]) for g, p in zip(gold[valid], pred[valid])])
    return {
        'macro_f1': macro(gold, pred), 'macro_f1_ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        **{f'f1_{c}': float(v) for c, v in zip(CLASSES, per_class)},
        'mcc': float(matthews_corrcoef(gold, pred)), 'balanced_acc': float(balanced_accuracy_score(gold, pred)),
        'accuracy': float((gold == pred).mean()), 'gross_errors_none_high': int((dist == 2).sum()),
        'pred_share': {c: float((pred == c).mean()) for c in CLASSES},
    }


def llm_row(path):
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(recs).sort_values('row')
    pred = df['pred'].fillna('invalid').to_numpy()
    model = df['model'].iloc[0].split('/')[-1]
    shots = df['shots'].iloc[0] if 'shots' in df and isinstance(df['shots'].iloc[0], str) else 'fewshot'
    suffix = '' if shots == 'fewshot' else f' ({shots})'
    out = {'name': f"{model} {df['mode'].iloc[0]}-shot{suffix}", 'run': path.name, 'model': model, 'mode': df['mode'].iloc[0], 'n': len(df),
           'invalid_share': float(df['pred'].isna().mean()), 'invalid_kinds': df['error'].value_counts().to_dict(),
           **quality(df['gold'].to_numpy(), pred),
           'latency_mean_s': float(df['latency_s'].mean()), 'latency_p95_s': float(df['latency_s'].quantile(0.95)),
           'tokens_in_mean': float(df['tokens_in'].mean()), 'tokens_out_mean': float(df['tokens_out'].mean()),
           'confusion': confusion_matrix(df['gold'], pred, labels=CLASSES + ['invalid']).tolist()}
    for provider, price in PRICES.get(model, {}).items():
        out[f'usd_1000_{provider}'] = 1000 * (out['tokens_in_mean'] * price['in'] + out['tokens_out_mean'] * price['out']) / 1e6
    return out


def baseline_rows(split):
    frame = pd.read_csv(HERE / 'data' / f'{split}.csv')
    thr = json.loads((HERE / 'data' / 'thresholds.json').read_text())
    gold = to_class(frame['target_orders_90'].to_numpy())
    rows = []
    for name, col in [('catboost', 'pred_catboost'), ('naive', 'pred_naive')]:
        t1, t2 = thr[name]
        p = frame[col].to_numpy()
        pred = np.where(p < t1, 'none', np.where(p < t2, 'low', 'high'))
        label = {'catboost': 'CatBoost', 'naive': 'naive: orders_180 / 2'}[name]
        rows.append({'name': label, 'model': name, 'mode': 'thresholds on valid', 'n': len(frame), 'invalid_share': 0.0,
                     **quality(gold, pred), 'confusion': confusion_matrix(gold, pred, labels=CLASSES + ['invalid']).tolist()})
    return rows


def main():
    split = sys.argv[1]
    paths = [pathlib.Path(p) for p in sys.argv[2:]] or sorted((HERE / 'results').glob(f'*_{split}.jsonl'))
    rows = [llm_row(p) for p in paths] + baseline_rows(split)
    (HERE / 'results' / f'metrics_{split}.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    cols = ['name', 'n', 'invalid_share', 'macro_f1', 'f1_none', 'f1_low', 'f1_high', 'mcc', 'balanced_acc',
            'gross_errors_none_high', 'latency_mean_s', 'latency_p95_s', 'tokens_in_mean', 'usd_1000_deepinfra', 'usd_1000_openrouter']
    table = pd.DataFrame(rows).reindex(columns=cols)
    table['ci95'] = [f"{r['macro_f1_ci95'][0]:.3f}–{r['macro_f1_ci95'][1]:.3f}" for r in rows]
    print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
