"""Lab 2: metrics, confidence intervals, gain tests, missing-modality and later-period evaluation.

Writes results/metrics2.json. Run after models2.py and the VLM runs.
"""
import json
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
DATA, RES = HERE / 'data', HERE / 'results'
CLASSES = ['none', 'low', 'high']
N_BOOT, N_PERM = 1000, 2000
INFO = json.loads((RES / 'models.json').read_text())
UNI = ['uni_profile', 'uni_series', 'uni_context', 'uni_image', 'uni_text']
FUSION = ['early_numeric', 'early_all', 'late_stacking']
LLM_RUN = RES / 'llm_text_test.jsonl'  # lab 1 answers of the text-only LLM on the same 300 users
VLM_RUNS = {f'vlm_{m}_{mode}': RES / f'vlm_{m}_{mode}_test.jsonl'
            for m in ['Qwen2.5-VL-7B-Instruct-4bit', 'gemma-3-4b-it-4bit'] for mode in ['image', 'image_text']}


def macro(y, p):
    return f1_score(y, p, labels=[0, 1, 2], average='macro', zero_division=0)


def decide(prob, w):
    return (prob * np.asarray(w)).argmax(1)


def ece(y, prob, bins=10):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def boot_ci(y, p, rng):
    vals = [macro(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def metrics(y, p, prob=None, seed=0):
    rng = np.random.default_rng(seed)
    out = {'n': int(len(y)), 'macro_f1': float(macro(y, p)), 'macro_f1_ci95': boot_ci(y, p, rng),
           'mcc': float(matthews_corrcoef(y, p)),
           **{f'f1_{c}': float(v) for c, v in zip(CLASSES, f1_score(y, p, labels=[0, 1, 2], average=None,
                                                                     zero_division=0))}}
    if prob is not None:
        onehot = np.eye(3)[y]
        out.update({'auroc': float(roc_auc_score(y, prob, multi_class='ovr', average='macro')),
                    'brier': float(((prob - onehot) ** 2).sum(1).mean()), 'ece': ece(y, prob)})
    return out


def gain_test(y, p_a, p_b, seed=0):
    """macro-F1(a) - macro-F1(b) on the same users: bootstrap CI and paired permutation p-value (one-sided a > b)."""
    rng = np.random.default_rng(seed)
    diff = macro(y, p_a) - macro(y, p_b)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        boots.append(macro(y[i], p_a[i]) - macro(y[i], p_b[i]))
    perm = 0
    for _ in range(N_PERM):
        swap = rng.random(len(y)) < 0.5
        a, b = np.where(swap, p_b, p_a), np.where(swap, p_a, p_b)
        perm += (macro(y, a) - macro(y, b)) >= diff
    return {'diff': float(diff), 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'p_perm_one_sided': float((perm + 1) / (N_PERM + 1))}


def block_bootstrap(frame, preds, seed=0):
    """Later period: resample whole dates (blocks) with replacement."""
    rng = np.random.default_rng(seed)
    dates = sorted(frame['d'].unique())
    y = frame['cls'].to_numpy()
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(dates, len(dates))
        idx = np.concatenate([np.where(frame['d'].to_numpy() == d)[0] for d in pick])
        vals.append(macro(y[idx], preds[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load_prob(name, split):
    return np.load(RES / f'preds_{name}_{split}.npy')


def jsonl_preds(path, n=300):
    recs = {json.loads(line)['row']: json.loads(line) for line in open(path)}
    pred = np.array([CLASSES.index(recs[i]['pred']) if recs[i]['pred'] in CLASSES else -1 for i in range(n)])
    lat = np.array([recs[i]['latency_s'] for i in range(n)])
    tok_in = [recs[i].get('tokens_in') for i in range(n)]
    tok_out = [recs[i].get('tokens_out') for i in range(n)]
    return pred, lat, tok_in, tok_out


def main():
    test = pd.read_parquet(DATA / 'test.parquet')
    later = pd.read_parquet(DATA / 'later.parquet')
    y, yl = test['cls'].to_numpy(), later['cls'].to_numpy()
    sub = np.where(test['lab1'].to_numpy())[0][:300]
    out = {'test10k': {}, 'test300': {}, 'later': {}, 'gains': {}, 'missing': {}, 'later_by_date': {},
           'retail_later': {}, 'timing': {}}

    trainable = UNI + FUSION + [k for k in INFO if k.startswith('abl_')]
    for name in trainable:
        prob, w = load_prob(name, 'test'), INFO[name]['class_weights']
        p = decide(prob, w)
        out['test10k'][name] = metrics(y, p, prob)
        out['test300'][name] = metrics(y[sub], p[sub], prob[sub])
    for name, path in {**VLM_RUNS, 'llm_llama_text': LLM_RUN}.items():
        if not path.exists():
            continue
        pred, lat, tin, tout = jsonl_preds(path)
        valid = pred >= 0
        p = np.where(valid, pred, 3)  # an answer out of format is an error
        m = metrics(y[sub], np.where(p == 3, -1, p))
        m.update({'format_errors': int((~valid).sum()), 'latency_mean_s': float(lat.mean()),
                  'latency_p95_s': float(np.percentile(lat, 95)),
                  'tokens_in_mean': float(np.mean([t for t in tin if t is not None])) if any(tin) else None,
                  'tokens_out_mean': float(np.mean([t for t in tout if t is not None])) if any(tout) else None,
                  'pred_share': {c: float((p == k).mean()) for k, c in enumerate(CLASSES)}})
        out['test300'][name] = m

    # Best unimodal baseline is chosen on valid, not on test.
    best_uni = max(UNI, key=lambda m: INFO[m]['valid_macro_f1'])
    out['best_unimodal_by_valid'] = best_uni
    p_best = decide(load_prob(best_uni, 'test'), INFO[best_uni]['class_weights'])
    for name in FUSION:
        p = decide(load_prob(name, 'test'), INFO[name]['class_weights'])
        out['gains'][f'{name} vs {best_uni}'] = gain_test(y, p, p_best)
    p_full = decide(load_prob('early_all', 'test'), INFO['early_all']['class_weights'])
    p_num = decide(load_prob('early_numeric', 'test'), INFO['early_numeric']['class_weights'])
    out['gains']['early_all vs early_numeric'] = gain_test(y, p_full, p_num)

    for drop in ['profile', 'series', 'context', 'image', 'text']:
        prob = load_prob(f'missing_{drop}', 'test')
        out['missing'][drop] = metrics(y, decide(prob, INFO['early_all']['class_weights']), prob)

    # Later period (application): best fusion and baselines, per date, block bootstrap, retail segment.
    for name in UNI + FUSION:
        prob, w = load_prob(name, 'later'), INFO[name]['class_weights']
        p = decide(prob, w)
        m = metrics(yl, p, prob)
        m['macro_f1_ci95_block_by_date'] = block_bootstrap(later, p)
        out['later'][name] = m
        out['later_by_date'][name] = {d: float(macro(yl[g.index], p[g.index]))
                                      for d, g in later.reset_index(drop=True).groupby('d')}
        retail = (later['hist_shop_share_180'] >= 0.5).to_numpy()
        out['retail_later'][name] = metrics(yl[retail], p[retail], prob[retail])
    p_fuse = decide(load_prob('early_all', 'later'), INFO['early_all']['class_weights'])
    p_bl = decide(load_prob(best_uni, 'later'), INFO[best_uni]['class_weights'])
    out['gains']['later: early_all vs ' + best_uni] = gain_test(yl, p_fuse, p_bl)
    p_stack = decide(load_prob('late_stacking', 'later'), INFO['late_stacking']['class_weights'])
    out['gains']['later: late_stacking vs ' + best_uni] = gain_test(yl, p_stack, p_bl)
    # Zero-shot: does the second modality help? Same 300 users; an answer out of format counts as an error (-1).
    zs = {}
    for name, path in {**VLM_RUNS, 'llm_llama_text': LLM_RUN}.items():
        pred = jsonl_preds(path)[0]
        zs[name] = np.where(pred >= 0, pred, -1)
    g_it, g_i = 'vlm_gemma-3-4b-it-4bit_image_text', 'vlm_gemma-3-4b-it-4bit_image'
    out['gains']['zero-shot: gemma image+text vs llama text'] = gain_test(y[sub], zs[g_it], zs['llm_llama_text'])
    out['gains']['zero-shot: gemma image+text vs gemma image'] = gain_test(y[sub], zs[g_it], zs[g_i])
    for drop in ['profile', 'series', 'context', 'image', 'text']:
        prob = load_prob(f'missing_{drop}', 'later')
        out['missing'][f'later_{drop}'] = metrics(yl, decide(prob, INFO['early_all']['class_weights']), prob)

    # Render time of the chart (CPU), per 1000 users.
    from render_images import render
    rows = list(test.head(1000).itertuples())
    t = time.perf_counter()
    for r in rows:
        render(r)
    out['timing']['render_ms_per_user'] = (time.perf_counter() - t)
    (RES / 'metrics2.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    rows = [(k, v['macro_f1'], v['macro_f1_ci95'], v.get('auroc'), v.get('brier'), v.get('ece'))
            for k, v in out['test10k'].items()]
    for r in sorted(rows, key=lambda r: -r[1]):
        print(f'{r[0]:<18} F1 {r[1]:.3f} [{r[2][0]:.3f}-{r[2][1]:.3f}] AUROC {r[3]:.3f} Brier {r[4]:.3f} ECE {r[5]:.3f}')
    print('best unimodal (valid):', best_uni)
    for k, v in out['gains'].items():
        print(k, {kk: (np.round(vv, 4).tolist() if not isinstance(vv, float) else round(vv, 4)) for kk, vv in v.items()})
    for k, v in out['test300'].items():
        print('300', k, round(v['macro_f1'], 3), v.get('format_errors'))


if __name__ == '__main__':
    main()
