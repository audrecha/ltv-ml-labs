"""Lab 3: download open daily OHLCV from Yahoo Finance (yfinance), adjusted for splits and dividends.

Main universe: 8 SPDR sector ETFs (traded since 1998). Thesis-related sector: listed food-delivery companies.
Writes data/raw/<ticker>.csv and data/raw/download_log.json.
"""
import datetime
import json
import pathlib

import yfinance as yf

OUT = pathlib.Path(__file__).resolve().parent / 'data' / 'raw'
MAIN = ['XLK', 'XLY', 'XLF', 'XLI', 'XLE', 'XLB', 'XLP', 'XLV', 'XLU', 'SPY']
FOOD = ['DASH', 'UBER', 'CART', 'GRAB', 'DHER.DE', 'TKWY.AS', '3690.HK', 'ETERNAL.NS', 'SWIGGY.NS', 'DELIVEROO.L',
        'ROO.L', 'MELI']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = {'downloaded_at': datetime.datetime.now().isoformat(timespec='seconds'), 'source': 'Yahoo Finance via yfinance',
           'auto_adjust': True, 'tickers': {}}
    for t in MAIN + FOOD:
        df = yf.download(t, start='1998-12-01', end='2026-10-16', auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            log['tickers'][t] = {'rows': 0}
            print(t, 'empty')
            continue
        if hasattr(df.columns, 'levels'):
            df.columns = [c[0] for c in df.columns]
        df.to_csv(OUT / f'{t}.csv')
        log['tickers'][t] = {'rows': len(df), 'first': str(df.index[0].date()), 'last': str(df.index[-1].date())}
        print(t, len(df), df.index[0].date(), df.index[-1].date())
    (OUT / 'download_log.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
