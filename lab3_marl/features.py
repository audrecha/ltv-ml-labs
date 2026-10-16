"""Lab 3: data audit and weekly feature panel built only from the past.

Universes:
- sectors: 8 SPDR sector ETFs (XLU and SPY are kept for reference only);
- foodtech: 8 listed food-delivery companies (application to the thesis-related sector, zero-shot).
A decision is taken at the close of the last trading day of week t with daily data up to that close; the reward
uses the return from that close to the close of week t+1. Feature scaling uses train-period statistics of the
sector universe only. Writes data/panel_<universe>.npz and data/audit.json.
"""
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RAW, OUT = HERE / 'data/raw', HERE / 'data'
SECTORS = ['XLK', 'XLY', 'XLF', 'XLI', 'XLE', 'XLB', 'XLP', 'XLV']          # agent k manages assets 2k, 2k+1
FOODTECH = ['DASH', 'UBER', 'CART', 'GRAB', 'DHER.DE', '3690.HK', 'ETERNAL.NS', 'HFG.DE']
SPLITS = {'train': ('1999-12-31', '2015-12-31'), 'valid': ('2016-01-01', '2020-12-31'),
          'test': ('2021-01-01', '2026-10-16')}
FEATURES = ['ret_1w', 'ret_4w', 'ret_12w', 'vol_4w', 'vol_ratio', 'volume_z', 'ma50_gap', 'dd_52w']


def load(ticker):
    df = pd.read_csv(RAW / f'{ticker}.csv', index_col=0, parse_dates=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
    if ticker == 'GRAB':                       # Yahoo history includes the SPAC period before the listing
        df = df[df.index >= '2021-12-02']
    return df


def audit(ticker, df, ref_index=None):
    ret = df['Close'].pct_change()
    gaps = df.index.to_series().diff().dt.days
    out = {'rows': len(df), 'first': str(df.index[0].date()), 'last': str(df.index[-1].date()),
           'duplicate_dates': int(df.index.duplicated().sum()), 'nan_rows': int(df.isna().any(axis=1).sum()),
           'nonpositive_close': int((df['Close'] <= 0).sum()), 'zero_volume_days': int((df['Volume'] <= 0).sum()),
           'abs_daily_return_gt_15pct': int((ret.abs() > 0.15).sum()), 'max_gap_days': int(gaps.max()),
           'high_lt_low_rows': int((df['High'] < df['Low']).sum())}
    if ref_index is not None:
        window = ref_index[(ref_index >= df.index[0]) & (ref_index <= df.index[-1])]
        out['missing_vs_reference_days'] = int(len(window.difference(df.index)))
    return out


def daily_features(df):
    c, v = df['Close'], df['Volume'].replace(0, np.nan)
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    f['ret_1w'] = np.log(c / c.shift(5))
    f['ret_4w'] = np.log(c / c.shift(20))
    f['ret_12w'] = np.log(c / c.shift(60))
    f['vol_4w'] = r.rolling(20).std() * np.sqrt(252)
    f['vol_ratio'] = r.rolling(20).std() / r.rolling(60).std()
    lv = np.log(v)
    # zero-volume days (holidays on foreign exchanges) are skipped instead of voiding the whole window
    f['volume_z'] = ((lv.rolling(20, min_periods=15).mean() - lv.rolling(120, min_periods=90).mean())
                     / lv.rolling(120, min_periods=90).std())
    f['ma50_gap'] = c / c.rolling(50).mean() - 1
    f['dd_52w'] = c / c.rolling(252, min_periods=120).max() - 1
    return f


def weekly_panel(tickers):
    """Weekly decision dates (last trading day of each week), features at that close, next-week log return."""
    feats, closes = {}, {}
    for t in tickers:
        df = load(t)
        f = daily_features(df)
        wk = f.groupby(pd.Grouper(freq='W-FRI')).last()       # last available daily row of the week
        cl = df['Close'].groupby(pd.Grouper(freq='W-FRI')).last()
        feats[t], closes[t] = wk, cl
    close = pd.DataFrame(closes).dropna()                       # weeks where every asset traded
    fwd = np.log(close.shift(-1) / close)                       # return realised AFTER the decision
    idx = close.index[:-1]
    X = np.stack([feats[t].reindex(idx)[FEATURES].to_numpy() for t in tickers], axis=1)   # [T, A, F]
    R = fwd.loc[idx, tickers].to_numpy()                                                    # [T, A]
    ok = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(R).any(axis=1)
    return idx[ok], X[ok], R[ok]


def cash_returns(idx):
    """Weekly log return of cash over the week after the decision: 13-week T-bill yield known at the decision."""
    irx = pd.read_csv(RAW / 'IRX.csv', index_col=0, parse_dates=True)['Close'].sort_index()
    y = irx.groupby(pd.Grouper(freq='W-FRI')).last().ffill().reindex(idx).ffill()
    return (np.log1p(y.to_numpy() / 100.0) / 52.0).astype(np.float32)


def main():
    spy = load('SPY')
    audits = {t: audit(t, load(t), spy.index) for t in SECTORS + ['XLU', 'SPY']}
    for t in FOODTECH:
        audits[t] = audit(t, load(t))
    # GRAB: Yahoo history starts before the 2021-12-02 listing (SPAC period) — keep only data after listing.
    grab = load('GRAB')
    raw_grab = pd.read_csv(RAW / 'GRAB.csv', index_col=0, parse_dates=True)
    audits['GRAB']['rows_before_listing_2021_12_02_removed'] = int((raw_grab.index < '2021-12-02').sum())
    audits['_replaced'] = {'SWIGGY.NS': 'listed 2024-11-13, features only from 2025-09 — replaced by HFG.DE'}
    audits['_not_available'] = {'TKWY.AS': 'Just Eat Takeaway — delisted after acquisition, no data',
                                'ROO.L': 'Deliveroo — delisted after acquisition, no data'}

    dates, X, R = weekly_panel(SECTORS)
    train = (dates >= SPLITS['train'][0]) & (dates <= SPLITS['train'][1])
    mu = np.nanmean(X[train].reshape(-1, X.shape[2]), axis=0)
    sd = np.nanstd(X[train].reshape(-1, X.shape[2]), axis=0)
    split = np.select([(dates >= a) & (dates <= b) for a, b in SPLITS.values()], list(SPLITS), default='none')
    np.savez(OUT / 'panel_sectors.npz', dates=dates.astype(str), X=np.clip((X - mu) / sd, -5, 5).astype(np.float32),
             R=R.astype(np.float32), C=cash_returns(dates), split=split, tickers=np.array(SECTORS), mu=mu, sd=sd)

    fd, fX, fR = weekly_panel(FOODTECH)
    keep = fd >= fd[0]                                         # first week with features for every company
    np.savez(OUT / 'panel_foodtech.npz', dates=fd[keep].astype(str),
             X=np.clip((fX[keep] - mu) / sd, -5, 5).astype(np.float32), R=fR[keep].astype(np.float32),
             C=cash_returns(fd[keep]),
             split=np.array(['foodtech'] * int(keep.sum())), tickers=np.array(FOODTECH), mu=mu, sd=sd)

    stats = {}
    for name, d, r in [('sectors', dates, R), ('foodtech', fd[keep], fR[keep])]:
        for s in (list(SPLITS) if name == 'sectors' else ['foodtech']):
            m = split == s if name == 'sectors' else np.ones(len(d), bool)
            ew = np.log(np.exp(r[m]).mean(axis=1))              # equal-weight weekly log return
            stats[f'{name}:{s}'] = {'weeks': int(m.sum()), 'first': str(d[m][0])[:10], 'last': str(d[m][-1])[:10],
                                    'ew_ann_return': float(np.exp(ew.mean() * 52) - 1),
                                    'ew_ann_vol': float(ew.std() * np.sqrt(52)),
                                    'asset_weekly_ret_mean': r[m].mean(0).round(4).tolist(),
                                    'asset_weekly_ret_std': r[m].std(0).round(4).tolist(),
                                    'asset_return_corr_mean': float(np.corrcoef(r[m].T)[np.triu_indices(r.shape[1], 1)].mean())}
    (OUT / 'audit.json').write_text(json.dumps({'tickers': audits, 'panels': stats, 'features': FEATURES,
                                                'splits': SPLITS}, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=1)[:3000])
    for t, a in audits.items():
        print(t, a)


if __name__ == '__main__':
    main()
