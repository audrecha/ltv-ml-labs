"""Lab 1: LLM classification of a user's 90-day order count (none / low / high) with mlx-lm, locally.

Usage: run_llm.py <model_repo> <zero|few> <test|big> <out.jsonl> [--prefix-cache] [--shots fewshot_s1]
Greedy decoding (temperature 0), one JSON answer per user, resumable (skips rows already in out.jsonl).
--prefix-cache keeps the KV cache of the shared prompt prefix (instruction + few-shot examples) and only
processes each user's part; the prompt tokens are the same, only prefill time changes.
"""
import json
import math
import pathlib
import re
import sys
import time

import pandas as pd
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

DATA = pathlib.Path(__file__).resolve().parent / 'data'
CLASSES = ['none', 'low', 'high']
MAX_TOKENS = 40

SYSTEM = (
    'You are an analyst at a food delivery service. From a user\'s past activity, predict how many '
    'delivered orders the user will make in the next 90 days. Classes: "none" = 0 orders, '
    '"low" = 1 to 3 orders, "high" = 4 or more orders. '
    'Answer with JSON only, no other text, exactly in this format: {"class": "none|low|high"}'
)


def fmt(value, digits=0, suffix=''):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'unknown'
    return f'{value:.{digits}f}{suffix}'


def describe(row) -> str:
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


def to_class(y: float) -> str:
    return 'none' if y <= 0 else ('low' if y <= 3 else 'high')


def gold_answer(y: float) -> str:
    return json.dumps({'class': to_class(y)})


def build_messages(row, shots: pd.DataFrame | None):
    messages = [{'role': 'system', 'content': SYSTEM}]
    if shots is not None:
        for shot in shots.itertuples():
            messages.append({'role': 'user', 'content': 'User profile:\n' + describe(shot)})
            messages.append({'role': 'assistant', 'content': gold_answer(shot.target_orders_90)})
    messages.append({'role': 'user', 'content': 'User profile:\n' + describe(row)})
    return messages


def apply_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Mistral templates reject a system role: merge it into the first user turn.
        merged = [dict(m) for m in messages[1:]]
        merged[0]['content'] = messages[0]['content'] + '\n\n' + merged[0]['content']
        return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


def parse(text: str):
    """Return (class, error). Valid = a JSON object with class in CLASSES."""
    match = re.search(r'\{.*\}', text, flags=re.S)
    if not match:
        return None, 'no_json'
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 'bad_json'
    cls = obj.get('class') if isinstance(obj, dict) else None
    if cls not in CLASSES:
        return None, 'bad_class'
    return cls, None


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Tokenize like mlx_lm.generate does for a string prompt: no second BOS if the template already has one."""
    add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    return tokenizer.encode(prompt, add_special_tokens=add_special)


class PrefixCache:
    """KV cache of the token prefix shared by all prompts; trimmed back to the prefix after each call."""

    def __init__(self, model, tokenizer, sample_prompts):
        a, b = (encode_prompt(tokenizer, p) for p in sample_prompts)
        n = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) - 1
        self.prefix = a[:n]
        self.cache = make_prompt_cache(model)
        model(mx.array(self.prefix)[None], cache=self.cache)
        mx.eval([c.state for c in self.cache])

    def run(self, model, tokenizer, tokens):
        assert tokens[:len(self.prefix)] == self.prefix, 'prompt does not start with the cached prefix'
        text = generate(model, tokenizer, prompt=tokens[len(self.prefix):], max_tokens=MAX_TOKENS,
                        verbose=False, prompt_cache=self.cache)
        trim_prompt_cache(self.cache, self.cache[0].offset - len(self.prefix))
        return text


def main():
    repo, mode, split, out = sys.argv[1], sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
    use_cache = '--prefix-cache' in sys.argv[5:]
    frame = pd.read_csv(DATA / f'{split}.csv')
    args = sys.argv[5:]
    shots_name = args[args.index('--shots') + 1] if '--shots' in args else 'fewshot'
    shots = pd.read_csv(DATA / f'{shots_name}.csv') if mode == 'few' else None
    done = set()
    if out.exists():
        done = {json.loads(line)['row'] for line in out.read_text().splitlines() if line.strip()}
    model, tokenizer = load(repo)
    rows = list(frame.itertuples())
    prefix = None
    if use_cache:
        prefix = PrefixCache(model, tokenizer, [apply_template(tokenizer, build_messages(r, shots)) for r in rows[:2]])
    with out.open('a') as sink:
        for i, row in enumerate(rows):
            if i in done:
                continue
            prompt = apply_template(tokenizer, build_messages(row, shots))
            tokens = encode_prompt(tokenizer, prompt)
            n_in = len(tokens)
            start = time.perf_counter()
            if prefix is not None:
                text = prefix.run(model, tokenizer, tokens)
            else:
                text = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)
            latency = time.perf_counter() - start
            cls, error = parse(text)
            record = {'row': i, 'model': repo, 'mode': mode, 'split': split, 'y': float(row.target_orders_90),
                      'gold': to_class(row.target_orders_90), 'pred': cls, 'error': error,
                      'raw': text, 'tokens_in': n_in, 'tokens_out': len(tokenizer.encode(text, add_special_tokens=False)),
                      'latency_s': latency, 'prefix_cache': use_cache,
                      'shots': shots_name if shots is not None else None}
            sink.write(json.dumps(record, ensure_ascii=False) + '\n')
            sink.flush()
            if i % 25 == 0:
                print(i, cls, error, f'{latency:.2f}s', flush=True)


if __name__ == '__main__':
    main()
