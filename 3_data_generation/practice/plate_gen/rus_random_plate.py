import cv2
import random
import numpy as np
from tqdm import tqdm
import argparse
from functools import partial
# import mxnet as mx

import os
import re

import rus_white_short as rus_white_short
import rus_white_long as rus_white_long
import rus_yellow as rus_yellow
import rus_blue as rus_blue
import rus_black as rus_black
import rus_red_long as rus_red_long
import rus_red_short as rus_red_short


class Draw:
    _draw = [
        rus_white_short.Draw(),
        rus_white_long.Draw(),
        rus_yellow.Draw(),
        rus_blue.Draw(),
        rus_black.Draw(),
        rus_red_long.Draw(),
        rus_red_short.Draw()
    ]

    _chars_en = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                 "W", "X", "Y", "Z"]
    _chars_dip = ["A", "B", "C", "D", "E", "O", "H", "K", "M", "P", "T", "X", "Y"]
    _chars_rus = ["A", "B", "C", "E", "O", "H", "K", "M", "P", "T", "X", "Y"]
    _digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __call__(self):
        draw = self._draw[1]
        candidates = []
        if type(draw) == rus_white_short.Draw:
            candidates += [self._chars_rus] * 1
            candidates += [self._digits] * 3
            candidates += [self._chars_rus] * 2
            candidates += [self._digits] * 2  # Код региона из двух цифр для short
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label
        elif type(draw) == rus_white_long.Draw:
            candidates += [self._chars_rus] * 1
            candidates += [self._digits] * 3
            candidates += [self._chars_rus] * 2
            # Генерация кода региона в зависимости от типа
            rand = random.random()
            if rand < 0.2:
                candidates += [self._digits] * 2  # Две цифры для short
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, type="short_v1"), label
            elif rand >=0.2 and rand < 0.4:
                candidates += [self._digits] * 2  # Две цифры для short
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, type="short_v2"), label
            elif rand >= 0.4 and rand < 0.6:
                candidates += [self._digits] * 3  # Три цифры для long
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, type="long_v1"), label
            elif rand >= 0.6 and rand < 0.8:
                candidates += [self._digits] * 3  # Три цифры для long
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, type="long_v2"), label
            else:
                candidates += [self._digits] * 3  # Три цифры для long
                label = "".join([random.choice(c) for c in candidates])
                return draw(label, type="long_v3"), label
        elif type(draw) == rus_yellow.Draw:
            candidates += [self._chars_rus] * 2
            candidates += [self._digits] * 5
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label
        elif type(draw) == rus_blue.Draw:
            candidates += [self._chars_rus] * 1
            candidates += [self._digits] * 6
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label
        elif type(draw) == rus_black.Draw:
            candidates += [self._digits] * 4
            candidates += [self._chars_rus] * 2
            candidates += [self._digits] * 2
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label
        elif type(draw) == rus_red_long.Draw:
            candidates += [self._digits] * 3
            candidates += [self._chars_dip] * 1
            candidates += [self._digits] * 5
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label

        elif type(draw) == rus_red_short.Draw:
            candidates += [self._digits] * 3
            candidates += [self._chars_dip] * 2
            candidates += [self._digits] * 3
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label


def gauss_blur(image, level):
    return cv2.blur(image, (level * 2 + 1, level * 2 + 1))


def gauss_noise(image):
    for i in range(image.shape[2]):
        c = image[:, :, i]
        diff = 255 - c.max()
        noise = np.random.normal(0, random.randint(1, 6), c.shape)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = diff * noise
        image[:, :, i] = c + noise.astype(np.uint8)
    return image

def motion_blur(image):
    psf = np.zeros((50, 50, 3))
    psf = cv2.ellipse(psf, 
                    (25, 25), # center
                    (5, 0), # axes -- 22 for blur length, 0 for thin PSF 
                    random.random()*90, # angle of motion in degrees
                    0, 360, # ful ellipse, not an arc
                    (1, 1, 1), # white color
                    thickness=-1) # filled

    psf /= psf[:,:,0].sum() # normalize by sum of one channel 
                            # since channels are processed independently

    return cv2.filter2D(image, -1, psf)
    


def add_noise(img): 
  
    # Getting the dimensions of the image 
    row , col, _ = img.shape 
      
    # Randomly pick some pixels in the 
    # image for coloring them white 
    # Pick a random number between 300 and 10000 
    number_of_pixels = random.randint(300, 10000) 
    for i in range(number_of_pixels): 
        
        # Pick a random y coordinate 
        y_coord=random.randint(0, row - 1) 
          
        # Pick a random x coordinate 
        x_coord=random.randint(0, col - 1) 
          
        # Color that pixel to white 
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in 
    # the image for coloring them black 
    # Pick a random number between 300 and 10000 
    number_of_pixels = random.randint(300 , 10000) 
    for i in range(number_of_pixels): 
        
        # Pick a random y coordinate 
        y_coord=random.randint(0, row - 1) 
          
        # Pick a random x coordinate 
        x_coord=random.randint(0, col - 1) 
          
        # Color that pixel to black 
        img[y_coord][x_coord] = 0
          
    return img 


def fake_plate(additional_functions=[]):
    draw = Draw()
    plate, label = draw()
    
    for func, thres in additional_functions:
        if random.random() < thres:
            plate = func(plate)
    
    # Случайное применение смазывания
    # if smudge and random.random() < 0.2:
    #     plate = smudge(plate)

    # # Случайное применение размытия
    # if random.random() < 0.1:
    #     plate = gauss_blur(plate, 1)

    # Случайное применение шума
    # if random.random() < 0.3:
    #     plate = gauss_noise(plate)

    return plate, label #mx.nd.array(plate), label


class Smudginess:
    def __init__(self, smu="assets/smu.png"):
        self._smu = cv2.imread(str(smu))
        if self._smu is None:
            print(f"Smudge not found in path {smu}")

    def __call__(self, raw):
        if self._smu is None:
            return raw
        y = random.randint(0, self._smu.shape[0] - raw.shape[0])
        x = random.randint(0, self._smu.shape[1] - raw.shape[1])
        texture = self._smu[y:y + raw.shape[0], x:x + raw.shape[1]]
        return cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(raw), texture))


def main(
    num_plates: int,
    save_path: str,
    smudge_path: str,
    ):

    directory_out = f"{save_path}/"

    if not os.path.exists(directory_out):
        os.makedirs(directory_out)  # создаём директорию, если её нет

    mud = Smudginess(smudge_path)
    additional_functions = [
        [mud, 0.5],
        [partial(gauss_blur, level=1), 0.5],
        [gauss_noise, 0.5],
        [motion_blur, 0.5],
        [add_noise, 0.5]
    ]

    for i in tqdm(range(num_plates)):
        plate, label = fake_plate(additional_functions)  # передаём функциb смазывания

        cv2.imwrite(directory_out + label + '.jpg', cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a random russian car license plate.")
    parser.add_argument("-num", help="set the number of plates (default: 10)", type=int, default=10)
    parser.add_argument("-dst", help="Save path", type=str, default='./white_plates_v5')
    parser.add_argument("--smudge", help="Path of the smudge image", type=str, default='assets/smu.png')
    args = parser.parse_args()
    main(num_plates=args.num, save_path=args.dst, smudge_path=args.smudge)
