"""Lab 2: frozen encoders — SigLIP image embeddings of the activity charts and e5 text embeddings of the profile text.

Usage: embed.py [split ...]  — writes data/emb_image_<split>.npy and data/emb_text_<split>.npy
"""
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
IMAGE_MODEL, TEXT_MODEL = 'google/siglip-base-patch16-224', 'intfloat/multilingual-e5-small'
BATCH = 128


def fmt(value, digits=0):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'unknown'
    return f'{value:.{digits}f}'


def describe(row) -> str:
    """The lab 1 profile text (run_llm.describe in lab 1), kept identical."""
    return '\n'.join([
        f'- orders in the last 30 days: {fmt(row.hist_orders_30)}',
        f'- orders in the last 90 days: {fmt(row.hist_orders_90)}',
        f'- orders in the last 180 days: {fmt(row.hist_orders_180)}',
        f'- orders 90-180 days ago: {fmt(row.hist_orders_90_180)}',
        f'- days since the last order: {fmt(row.dyn_days_since_last_order)}',
        f'- weeks with at least one order out of the last 26: {fmt(row.hist_active_weeks_180)}',
        f'- mean days between orders over 180 days: {fmt(row.hist_mean_gap_days_180, 1)}',
        f'- days since the first order ever: {fmt(row.dyn_tenure_days_ever)}',
        f'- average order value over 180 days, RUB: {fmt(row.hist_avg_check_180)}',
        f'- share of grocery (shop) orders over 180 days: {fmt(row.hist_shop_share_180, 2)}',
        f'- share of orders with a promo over 180 days: {fmt(row.hist_promo_share_180, 2)}',
        f'- days with app sessions in the last 30 days: {fmt(row.hist_sess_days_30)}',
        f'- days with app sessions in the last 90 days: {fmt(row.hist_sess_days_90)}',
        f'- days since the last app session: {fmt(row.dyn_days_since_last_session)}',
    ])


@torch.no_grad()
def image_embeddings(split, n):
    processor = AutoProcessor.from_pretrained(IMAGE_MODEL)
    model = AutoModel.from_pretrained(IMAGE_MODEL).to(DEVICE).eval()
    out = []
    for start in range(0, n, BATCH):
        images = [Image.open(HERE / 'data' / 'images' / split / f'{i}.png').convert('RGB')
                  for i in range(start, min(n, start + BATCH))]
        inputs = processor(images=images, return_tensors='pt').to(DEVICE)
        emb = model.get_image_features(**inputs)
        emb = emb.pooler_output if hasattr(emb, 'pooler_output') else emb
        out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def text_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).to(DEVICE).eval()
    out = []
    for start in range(0, len(texts), BATCH):
        batch = tokenizer(['query: ' + t for t in texts[start:start + BATCH]], padding=True, truncation=True,
                          max_length=256, return_tensors='pt').to(DEVICE)
        hidden = model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1)
        out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
    return np.concatenate(out)


def main():
    for split in sys.argv[1:] or ['test', 'later', 'valid', 'train']:
        frame = pd.read_parquet(HERE / 'data' / f'{split}.parquet')
        t = time.perf_counter()
        img = image_embeddings(split, len(frame))
        t_img = time.perf_counter() - t
        t = time.perf_counter()
        txt = text_embeddings([describe(r) for r in frame.itertuples()])
        t_txt = time.perf_counter() - t
        np.save(HERE / 'data' / f'emb_image_{split}.npy', img.astype(np.float32))
        np.save(HERE / 'data' / f'emb_text_{split}.npy', txt.astype(np.float32))
        print(split, img.shape, txt.shape, f'image {t_img:.0f}s text {t_txt:.0f}s', flush=True)


if __name__ == '__main__':
    main()
