import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import random


class Draw:
    # Загрузка шрифтов
    _font = [
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "assets/RoadNumbers2.0.ttf"), 105),
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "assets/RoadNumbers2.0.ttf"), 98),
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "assets/RoadNumbers2.0.ttf"), 80)
    ]

    _bg_white_long_v1 = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "assets/rus_white_long_lp.png")),(440, 95))
    _bg_white_long_v2 = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "assets/rus_white_long_lp_v2.png")),
                                   (440, 95))
    _bg_white_long_v3 = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "assets/rus_white_long_lp_without_borders.png")),
                                   (440, 95))
    _bg_white_short_v1 = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "assets/rus_white_lp.png")), (440, 95))
    _bg_white_short_v2 = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "assets/rus_white_lp_without_borders.png")),
                                    (440, 95))

    def __call__(self, plate, type="long"):
        if type == "long_v1":
            bg = self._bg_white_long_v1
            fg = self._draw_fg_long(plate)
        elif type == "long_v2":
            bg = self._bg_white_long_v2
            fg = self._draw_fg_long(plate)
        elif type == "long_v3":
            bg = self._bg_white_long_v3
            fg = self._draw_fg_long(plate)
        elif type == "short_v1":
            bg = self._bg_white_short_v1
            fg = self._draw_fg_short(plate)  # Отрисовка для короткого типа
        else:
            bg = self._bg_white_short_v2
            fg = self._draw_fg_short(plate)  # Отрисовка для короткого типа
        return cv2.cvtColor(cv2.bitwise_and(fg, bg), cv2.COLOR_BGR2RGB)

    def _draw_char(self, ch, size, padding_top):
        img = Image.new("RGB", (45, 95), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text(
            (0, padding_top), ch,
            fill=(0, 0, 0),
            font=self._font[size]
        )
        if img.width > 45:
            img = img.resize((45, 95))
        return np.array(img)

    def _draw_fg_long(self, plate):
        """
        Логика для длинного номера.
        """
        img = np.array(Image.new("RGB", (440, 95), (255, 255, 255)))
        offset = 26

        img[0:95, offset:offset + 45] = self._draw_char(plate[0], 1, 33)
        offset = offset + 44

        img[0:95, offset:offset + 45] = self._draw_char(plate[1], 0, 29)
        offset = offset + 40 + 8

        img[0:95, offset:offset + 45] = self._draw_char(plate[2], 0, 29)
        offset = offset + 40 + 8

        img[0:95, offset:offset + 45] = self._draw_char(plate[3], 0, 29)
        offset = offset + 49

        img[0:95, offset:offset + 45] = self._draw_char(plate[4], 1, 33)
        offset = offset + 41

        img[0:95, offset:offset + 45] = self._draw_char(plate[5], 1, 33)
        offset = offset + 61

        img[0:95, offset:offset + 45] = self._draw_char(plate[6], 2, 22)
        offset = offset + 34

        img[0:95, offset:offset + 45] = self._draw_char(plate[7], 2, 22)
        offset = offset + 34

        img[0:95, offset:offset + 45] = self._draw_char(plate[8], 2, 22)

        return img

    def _draw_fg_short(self, plate):
        """
        Логика для короткого номера.
        """
        img = np.array(Image.new("RGB", (440, 95), (255, 255, 255)))
        offset = 26  # Начальный отступ такой же

        # Отрисовка символов такая же, как в длинной версии
        img[0:95, offset:offset + 45] = self._draw_char(plate[0], 1, 33)
        offset = offset + 44

        img[0:95, offset:offset + 45] = self._draw_char(plate[1], 0, 29)
        offset = offset + 40 + 8

        img[0:95, offset:offset + 45] = self._draw_char(plate[2], 0, 29)
        offset = offset + 40 + 8

        img[0:95, offset:offset + 45] = self._draw_char(plate[3], 0, 29)
        offset = offset + 49

        img[0:95, offset:offset + 45] = self._draw_char(plate[4], 1, 33)
        offset = offset + 41

        img[0:95, offset:offset + 45] = self._draw_char(plate[5], 1, 33)
        offset = offset + 61

        # img[0:95, offset:offset + 45] = self._draw_char(plate[6], 2, 22)
        offset = offset + 34

        img[0:95, offset:offset + 45] = self._draw_char(plate[6], 2, 22)
        offset = offset + 34

        img[0:95, offset:offset + 45] = self._draw_char(plate[7], 2, 22)

        return img


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate a license plate with a random background.")
    parser.add_argument("plate", help="license plate number (default: A000AA00)", type=str, nargs="?",
                        default="B001EY196")
    parser.add_argument("--type", help="type of license plate (long or short)", type=str, default="long")
    args = parser.parse_args()

    draw = Draw()
    plate = draw(args.plate, type=args.type)

    if plate is not None:
        plt.imshow(plate)
        plt.show()
