# Лабораторная работа 2. Мультимодальные модели для сегментации пользователей по будущей ценности

Код сравнения одномодальных моделей и схем слияния модальностей: табличный профиль, временной ряд активности,
контекст региона, картинка-график ряда и текстовое описание профиля. Классы — число заказов за 90 дней
(`none` — 0, `low` — 1–3, `high` — 4+).

Данные в репозиторий не входят.

| Файл | Что делает |
| --- | --- |
| `download.py` | скачивает SigLIP, multilingual-e5-small, Qwen2.5-VL-7B и Gemma-3-4B |
| `render_images.py` | рисует график активности пользователя (картинка-модальность) |
| `embed.py` | эмбеддинги картинок (SigLIP) и текста профиля (e5) |
| `models2.py` | одномодальные модели, раннее и позднее слияние, абляции, пропуск модальности на тесте |
| `run_vlm.py`, `run_vlm_all.sh` | визуально-языковые модели без обучения: картинка и картинка + текст |
| `evaluate2.py` | macro-F1, MCC, AUROC, Brier, ECE, бутстреп и перестановочный тест прироста |

```
pip install mlx-vlm torch transformers sentence-transformers catboost scikit-learn pandas pyarrow pillow
python download.py
python render_images.py test later valid train
python embed.py
python models2.py
bash run_vlm_all.sh
python evaluate2.py
```

Скрипты ожидают `data/<split>.parquet` (колонка класса `cls`) и `data/meta.json` со списками признаков
по модальностям: `profile`, `context`, `series`.
