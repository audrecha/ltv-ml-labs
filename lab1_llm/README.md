# Лабораторная работа 1. Большие языковые модели для сегментации пользователей по будущей ценности

Код запуска и оценки LLM в режимах zero-shot и few-shot: модель получает профиль пользователя текстом
и относит его к одному из трёх классов по числу заказов за следующие 90 дней (`none` — 0, `low` — 1–3, `high` — 4+).
Модели — Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 в 4-битной версии для MLX (Apple Silicon).

Данные в репозиторий не входят.

| Файл | Что делает |
| --- | --- |
| `download.py` | скачивает веса моделей с Hugging Face |
| `run_llm.py` | запускает модель по пользователям, разбирает JSON-ответ, пишет `results/*.jsonl` |
| `run_all.sh`, `run_stage2.sh` | все прогоны: 3 модели × 2 режима, большой срез, разные наборы примеров |
| `evaluate.py` | macro-F1 с бутстреп-интервалом, F1 по классам, MCC, доля ответов не по формату, задержка, цена API |
| `finalize.py` | проверка кэша промпта и парный бутстреп разницы с бейзлайном |

```
pip install mlx-lm pandas pyarrow scikit-learn
python download.py
python run_llm.py mlx-community/Meta-Llama-3.1-8B-Instruct-4bit zero test results/llama_zero_test.jsonl
python evaluate.py test
```

`run_llm.py` ожидает `data/<split>.csv` с колонкой `target_orders_90` и признаками, перечисленными в `describe()`.
