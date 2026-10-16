"""Lab 2: image modality — a chart of the user's activity series over the 180 days before D.

Three panels on one 336x336 image, time runs left (180 days before D) to right (D):
orders per week, days with app sessions per week, spend (GMV) per week. Fixed scales for every user, values above
the cap are drawn at the cap with a mark, so images are comparable between users.
Usage: render_images.py [split ...]   — writes data/images/<split>/<row>.png
"""
import pathlib
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
SIZE = 336
FONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
SMALL = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)
PANELS = [  # (title, columns with (start_day, end_day) before D, cap, colour)
    ('orders per week', [('ts_orders_90_180', 180, 90), ('ts_orders_60_90', 90, 60), ('ts_orders_30_60', 60, 30),
                         ('ts_orders_7_30', 30, 7), ('ts_orders_0_7', 7, 0)], 2.0, (66, 110, 180)),
    ('app days per week', [('ts_sessdays_90_180', 180, 90), ('ts_sessdays_30_90', 90, 30),
                           ('ts_sessdays_7_30', 30, 7), ('ts_sessdays_0_7', 7, 0)], 7.0, (60, 150, 90)),
    ('spend per week, RUB', [('ts_gmv_90_180', 180, 90), ('ts_gmv_30_90', 90, 30), ('ts_gmv_0_30', 30, 0)],
     3000.0, (120, 120, 120)),
]
LEFT, RIGHT, TOP = 34, 8, 6
PANEL_H = (SIZE - TOP - 18) // 3


def render(row) -> Image.Image:
    img = Image.new('RGB', (SIZE, SIZE), 'white')
    draw = ImageDraw.Draw(img)
    width = SIZE - LEFT - RIGHT
    x_of = lambda day: LEFT + width * (180 - day) / 180
    for k, (title, cols, cap, colour) in enumerate(PANELS):
        y0 = TOP + k * PANEL_H
        base = y0 + PANEL_H - 6
        top = y0 + 16
        draw.text((LEFT, y0), title, fill='black', font=FONT)
        draw.line([(LEFT, base), (SIZE - RIGHT, base)], fill='black')
        draw.line([(LEFT, top), (LEFT, base)], fill='black')
        label = f'{cap:.0f}'
        draw.text((2, top - 6), label, fill='black', font=SMALL)
        draw.text((2, base - 10), '0', fill='black', font=SMALL)
        for col, start, end in cols:
            value = float(getattr(row, col))
            h = min(value, cap) / cap * (base - top)
            x1, x2 = x_of(start) + 1, x_of(end) - 1
            if h > 0:
                draw.rectangle([x1, base - h, x2, base], fill=colour)
            if value > cap:
                draw.text(((x1 + x2) / 2 - 3, top - 2), '^', fill='red', font=SMALL)
    axis_y = SIZE - 14
    for day in (180, 90, 60, 30, 0):
        x = x_of(day)
        draw.text((x - 8, axis_y), f'-{day}' if day else 'D', fill='black', font=SMALL)
    return img


def main():
    for split in sys.argv[1:] or ['test', 'later']:
        frame = pd.read_parquet(HERE / 'data' / f'{split}.parquet')
        out = HERE / 'data' / 'images' / split
        out.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(frame.itertuples()):
            render(row).save(out / f'{i}.png')
        print(split, len(frame), flush=True)


if __name__ == '__main__':
    main()
