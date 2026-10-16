"""Lab 2: unimodal baselines, early and late fusion, ablations and missing-modality tests.

All trainable models: train = 60k (train dates, groups 4-19), early stopping / stacking / class weights on valid
(groups 0-3, train dates), test = 10k (groups 0-3, D = 2025-08-04), later = 7 later holdout dates.
Decision rule (same for every model): argmax(p * w), class weights w tuned on valid for macro-F1.
Writes results/preds_<model>_<split>.npy (probabilities), results/models.json (configuration and timings).
"""
import itertools
import json
import pathlib
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
DATA, RES = HERE / 'data', HERE / 'results'
META = json.loads((DATA / 'meta.json').read_text())
SPLITS = ['train', 'valid', 'test', 'later']
SEED = 42
CAT = ['dyn_last_region_group_name', 'dyn_last_region_id', 'dyn_tenure_first_source']


def load():
    frames = {s: pd.read_parquet(DATA / f'{s}.parquet') for s in SPLITS}
    emb = {s: {'image': np.load(DATA / f'emb_image_{s}.npy'), 'text': np.load(DATA / f'emb_text_{s}.npy')}
           for s in SPLITS}
    return frames, emb


def pca_block(emb, kind, k=32):
    pca = PCA(n_components=k, random_state=SEED).fit(emb['train'][kind])
    return {s: pd.DataFrame(pca.transform(emb[s][kind]), columns=[f'{kind}_pc{i}' for i in range(k)])
            for s in SPLITS}


def fit_catboost(frames, cols, extra=None, name=''):
    def x(s):
        out = frames[s][cols].reset_index(drop=True)
        if extra is not None:
            out = pd.concat([out, extra[s]], axis=1)
        return out
    cats = [c for c in cols if c in CAT]
    for s in SPLITS:
        for c in cats:
            frames[s][c] = frames[s][c].astype(str)
    model = CatBoostClassifier(loss_function='MultiClass', iterations=2000, learning_rate=0.06, depth=6,
                               random_seed=SEED, od_type='Iter', od_wait=100, verbose=False, thread_count=-1)
    t = time.perf_counter()
    model.fit(Pool(x('train'), frames['train']['cls'], cat_features=cats),
              eval_set=Pool(x('valid'), frames['valid']['cls'], cat_features=cats))
    fit_s = time.perf_counter() - t
    preds, infer = {}, {}
    for s in SPLITS[1:]:
        t = time.perf_counter()
        preds[s] = model.predict_proba(Pool(x(s), cat_features=cats))
        infer[s] = time.perf_counter() - t
    return model, preds, {'fit_s': fit_s, 'infer_s_test': infer['test'], 'iterations': model.get_best_iteration(),
                          'n_features': x('train').shape[1]}, x


def fit_logreg(emb, kind):
    scaler = StandardScaler().fit(emb['train'][kind])
    model = LogisticRegression(max_iter=2000, C=1.0)
    t = time.perf_counter()
    model.fit(scaler.transform(emb['train'][kind]), FR['train']['cls'])
    fit_s = time.perf_counter() - t
    preds, infer = {}, {}
    for s in SPLITS[1:]:
        t = time.perf_counter()
        preds[s] = model.predict_proba(scaler.transform(emb[s][kind]))
        infer[s] = time.perf_counter() - t
    return preds, {'fit_s': fit_s, 'infer_s_test': infer['test'], 'n_features': emb['train'][kind].shape[1]}


def class_weights(p_valid, y_valid):
    best = (-1, (1.0, 1.0, 1.0))
    grid = np.round(np.arange(0.5, 2.51, 0.1), 2)
    for w1, w2 in itertools.product(grid, grid):
        w = np.array([1.0, w1, w2])
        f = f1_score(y_valid, (p_valid * w).argmax(1), average='macro')
        if f > best[0]:
            best = (f, tuple(w))
    return list(best[1]), best[0]


def save(name, preds, info):
    w, f_valid = class_weights(preds['valid'], FR['valid']['cls'].to_numpy())
    for s, p in preds.items():
        np.save(RES / f'preds_{name}_{s}.npy', p)
    INFO[name] = {**info, 'class_weights': w, 'valid_macro_f1': f_valid}
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in INFO[name].items()}, flush=True)


def main():
    global FR, INFO
    RES.mkdir(exist_ok=True)
    FR, emb = load()
    INFO = {}
    profile, context, series = META['profile'], META['context'], META['series']
    img_pc, txt_pc = pca_block(emb, 'image'), pca_block(emb, 'text')

    # Unimodal baselines.
    for name, cols in [('uni_profile', profile), ('uni_series', series), ('uni_context', context)]:
        _, preds, info, _ = fit_catboost(FR, cols, name=name)
        save(name, preds, {**info, 'model': 'CatBoost', 'inputs': name.split('_', 1)[1]})
    for kind in ['image', 'text']:
        preds, info = fit_logreg(emb, kind)
        save(f'uni_{kind}', preds, {**info, 'model': 'LogReg on frozen embeddings', 'inputs': kind})

    # Early fusion.
    _, preds, info, _ = fit_catboost(FR, profile + series + context)
    save('early_numeric', preds, {**info, 'model': 'CatBoost', 'inputs': 'profile+series+context'})
    both = {s: pd.concat([img_pc[s], txt_pc[s]], axis=1) for s in SPLITS}
    model, preds, info, xfun = fit_catboost(FR, profile + series + context, extra=both)
    save('early_all', preds, {**info, 'model': 'CatBoost', 'inputs': 'profile+series+context+image_pca32+text_pca32'})

    # Ablations of the full early fusion: drop one modality and retrain.
    for drop, cols, extra in [
        ('profile', series + context, both), ('series', profile + context, both),
        ('context', profile + series, both),
        ('image', profile + series + context, {s: txt_pc[s] for s in SPLITS}),
        ('text', profile + series + context, {s: img_pc[s] for s in SPLITS})]:
        _, preds, info, _ = fit_catboost(FR, cols, extra=extra)
        save(f'abl_no_{drop}', preds, {**info, 'model': 'CatBoost', 'inputs': f'early_all without {drop}'})

    # Missing modality at test time (no retraining): the modality is blanked for the full early fusion.
    groups = {'profile': profile, 'series': series, 'context': context,
              'image': list(img_pc['test'].columns), 'text': list(txt_pc['test'].columns)}
    for drop, cols in groups.items():
        for s in ['test', 'later']:
            x = xfun(s)
            x[[c for c in cols if c not in CAT]] = np.nan
            for c in [c for c in cols if c in CAT]:
                x[c] = '__missing__'
            cats = [c for c in x.columns if c in CAT]
            np.save(RES / f'preds_missing_{drop}_{s}.npy', model.predict_proba(Pool(x, cat_features=cats)))
        INFO[f'missing_{drop}'] = {'model': 'early_all', 'inputs': f'{drop} blanked at test time',
                                   'class_weights': INFO['early_all']['class_weights']}

    # Late fusion: stacking of unimodal probabilities, trained on valid (unimodal models never saw valid).
    uni = ['uni_profile', 'uni_series', 'uni_context', 'uni_image', 'uni_text']
    stack = {s: np.hstack([np.log(np.clip(np.load(RES / f'preds_{m}_{s}.npy'), 1e-6, 1)) for m in uni])
             for s in ['valid', 'test', 'later']}
    meta = LogisticRegression(max_iter=2000, C=1.0).fit(stack['valid'], FR['valid']['cls'])
    preds = {s: meta.predict_proba(stack[s]) for s in ['valid', 'test', 'later']}
    # Class weights for stacking are tuned on valid too (in-sample for the meta-model; noted in the report).
    save('late_stacking', preds, {'model': 'LogReg stacking of 5 unimodal models', 'inputs': ','.join(uni)})

    (RES / 'models.json').write_text(json.dumps(INFO, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
