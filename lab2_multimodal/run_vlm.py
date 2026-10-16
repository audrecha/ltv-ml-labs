"""Lab 2: zero-shot vision-language models on the activity chart (image) and chart + profile text (image_text).

Usage: run_vlm.py <model_repo> <image|image_text> <out.jsonl> [--rows lab1|all] [--split test]
Greedy decoding, one JSON answer per user, resumable. Only the 300 lab 1 users by default.
"""
import json
import pathlib
import re
import sys
import time

import pandas as pd
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from embed import describe

HERE = pathlib.Path(__file__).resolve().parent
CLASSES = ['none', 'low', 'high']
TASK = ('You are an analyst at a food delivery service. Predict how many delivered orders the user will make in '
        'the next 90 days. Classes: "none" = 0 orders, "low" = 1 to 3 orders, "high" = 4 or more orders. ')
CHART = ('The image shows the user\'s activity over the 180 days before the forecast date D (time runs left to '
         'right, D is on the right): orders per week, days with app sessions per week and spend per week in RUB. '
         'Each bar covers a time window; the number on the left is the top of the scale, a red mark means the value '
         'is above it. ')
ANSWER = 'Answer with JSON only, no other text, exactly in this format: {"class": "none|low|high"}'


def prompt_for(row, mode):
    text = TASK + CHART
    if mode == 'image_text':
        text += 'User profile:\n' + describe(row) + '\n'
    return text + ANSWER


def parse(text):
    match = re.search(r'\{.*?\}', text, flags=re.S)
    if not match:
        return None, 'no_json'
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 'bad_json'
    cls = obj.get('class') if isinstance(obj, dict) else None
    return (cls, None) if cls in CLASSES else (None, 'bad_class')


def main():
    repo, mode, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    args = sys.argv[4:]
    split = args[args.index('--split') + 1] if '--split' in args else 'test'
    rows_kind = args[args.index('--rows') + 1] if '--rows' in args else 'lab1'
    frame = pd.read_parquet(HERE / 'data' / f'{split}.parquet')
    idx = list(frame.index[frame['lab1']]) if rows_kind == 'lab1' else list(frame.index)
    done = set()
    if out.exists():
        done = {json.loads(line)['row'] for line in out.read_text().splitlines() if line.strip()}
    model, processor = load(repo)
    config = load_config(repo)
    with out.open('a') as sink:
        for i in idx:
            if i in done:
                continue
            row = next(frame.loc[[i]].itertuples())
            image = str(HERE / 'data' / 'images' / split / f'{i}.png')
            prompt = apply_chat_template(processor, config, prompt_for(row, mode), num_images=1)
            start = time.perf_counter()
            result = generate(model, processor, prompt, image=[image], max_tokens=40, temperature=0.0, verbose=False)
            latency = time.perf_counter() - start
            text = result.text if hasattr(result, 'text') else str(result)
            cls, error = parse(text)
            y = float(row.target_orders_90)
            rec = {'row': int(i), 'model': repo, 'mode': mode, 'split': split, 'y': y,
                   'gold': CLASSES[int(row.cls)], 'pred': cls, 'error': error, 'raw': text, 'latency_s': latency,
                   'tokens_in': getattr(result, 'prompt_tokens', None),
                   'tokens_out': getattr(result, 'generation_tokens', None)}
            sink.write(json.dumps(rec, ensure_ascii=False) + '\n')
            sink.flush()
            if len(done) % 25 == 0:
                print(i, cls, error, f'{latency:.2f}s', flush=True)
            done.add(i)


if __name__ == '__main__':
    main()
