"""Lab 1: prefix-cache equivalence check and few-shot sensitivity metrics (results/*.json for the report)."""
import json
import pathlib

import pandas as pd

from evaluate import llm_row

RES = pathlib.Path(__file__).resolve().parent / 'results'
load = lambda name: pd.DataFrame([json.loads(l) for l in open(RES / name)]).set_index('row')

plain, cached = load('Meta-Llama-3.1-8B-Instruct-4bit_few_test.jsonl'), load('check_prefix_cache.jsonl')
joined = plain.join(cached, rsuffix='_cache', how='inner')
check = {'rows': len(joined), 'same_class_share': float((joined.pred == joined.pred_cache).mean()),
         'same_raw_text_share': float((joined.raw == joined.raw_cache).mean()),
         'latency_plain_s': float(joined.latency_s.mean()), 'latency_cache_s': float(joined.latency_s_cache.mean())}
(RES / 'check_prefix_cache.json').write_text(json.dumps(check, indent=2))
print(check)

runs = ['Meta-Llama-3.1-8B-Instruct-4bit_zero_test.jsonl', 'Meta-Llama-3.1-8B-Instruct-4bit_few_test.jsonl',
        'sens_llama_fewshot_s1.jsonl', 'sens_llama_fewshot_s2.jsonl']
sens = [llm_row(RES / r) for r in runs]
(RES / 'metrics_sens.json').write_text(json.dumps(sens, indent=2, ensure_ascii=False))
for m in sens:
    print(m['name'], round(m['macro_f1'], 3), [round(x, 3) for x in m['macro_f1_ci95']], round(m['f1_none'], 3),
          m['gross_errors_none_high'], {k: round(v, 2) for k, v in m['pred_share'].items()}, m['invalid_share'],
          round(m['latency_mean_s'], 2))

# Paired bootstrap of macro-F1(CatBoost) - macro-F1(best LLM) on the same users.
import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from analysis import load as load_pairs  # noqa: E402

paired = {}
for split, run in [('test', 'Meta-Llama-3.1-8B-Instruct-4bit_zero_test.jsonl'),
                   ('big', 'Meta-Llama-3.1-8B-Instruct-4bit_zero_big.jsonl')]:
    frame = load_pairs(split, RES / run)
    gold, llm, cb = (frame[c].to_numpy() for c in ('gold', 'llm', 'cb'))
    f = lambda idx, p: f1_score(gold[idx], p[idx], labels=['none', 'low', 'high'], average='macro', zero_division=0)
    rng = np.random.default_rng(0)
    full = np.arange(len(gold))
    diffs = [f(i, cb) - f(i, llm) for i in (rng.integers(0, len(gold), len(gold)) for _ in range(1000))]
    paired[split] = {'diff': f(full, cb) - f(full, llm), 'ci95': [float(np.percentile(diffs, 2.5)),
                                                                  float(np.percentile(diffs, 97.5))],
                     'share_le_0': float(np.mean(np.array(diffs) <= 0))}
(RES / 'paired_catboost_vs_llm.json').write_text(json.dumps(paired, indent=2))
print(paired)
