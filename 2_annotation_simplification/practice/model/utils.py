import os
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)

import pandas as pd
import threading
import urllib
import time
import logging
import io
from urllib.error import URLError, HTTPError


from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


from .settings import MAX_IMG_SIZE

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class PadCustom(transforms.Pad):
    """
    принимает на вход до какого размера нужно заппадить,
    а transforms.Pad на сколько нужно западдить
    """

    def forward(self, img: Union[Image.Image, torch.Tensor]):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.
        Returns:
            PIL Image or Tensor: Padded image.
        """
        if isinstance(img, torch.Tensor):
            img_size = img.size()[1:]
        else:
            img_size = img.size

        height_diff = self.padding[0] - img_size[0]
        width_diff = self.padding[1] - img_size[1]

        padding = (
            height_diff // 2,
            width_diff // 2,
            height_diff // 2 + height_diff % 2,
            width_diff // 2 + width_diff % 2,
        )

        return F.pad(img, padding, self.fill, self.padding_mode)


def get_preprocessor(pad_size=MAX_IMG_SIZE):
    preprocessor = transforms.Compose(
        [
            PadCustom(pad_size),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return preprocessor


def plot_imgs_with_labels(
    image_paths: List[str],
    labels_1: Tuple[List[Any], str] = None,
    labels_2: Tuple[List[Any], str] = None,
    labels_3: Tuple[List[Any], str] = None,
):
    plt.figure(figsize=(20, 20))

    n_subplots = len(image_paths)
    n_lines = int((n_subplots + 2) // 3)

    for i in range(n_subplots):
        plt.subplot(n_lines, 3, i + 1)
        img = mpimg.imread(image_paths[i])
        title = ""
        if labels_1 is not None:
            title += f"{labels_1[1]}: {labels_1[0][i]}"
        if labels_2 is not None:
            if isinstance(labels_2[0][i], float):
                labels_2[0][i] = round(labels_2[0][i], 4)
            title += f"\n{labels_2[1]}: {labels_2[0][i]}"

        if labels_3 is not None:
            if isinstance(labels_3[0][i], float):
                labels_3[0][i] = round(labels_3[0][i], 4)
            title += f"\n{labels_3[1]}: {labels_3[0][i]}"

        plt.title(title)
        plt.imshow(img)


def plot_sample(test_df, value="детская", column="label_pred", size=10):
    plot_df = test_df[test_df[column] == value]
    sample_size = min(plot_df.shape[0], size)

    sample_df = plot_df.sample(sample_size).reset_index()

    plot_imgs_with_labels(
        sample_df["img_path"],
        (sample_df["label"], "label"),
        (sample_df["label_pred"], "label_pred"),
        (sample_df["proba"], "proba"),
    )


def _load_images(
    urls: List[str], save_dir: str, max_retries: int = 3, timeout: int = 10
) -> List[str]:
    """Загружает изображения по URL-адресам в указанную директорию."""
    results = []
    errors = []

    def getter(url: str, dest: str):
        for attempt in range(max_retries):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                req = urllib.request.Request(url, headers=headers)

                with urllib.request.urlopen(req, timeout=timeout) as response:
                    with open(dest, "wb") as out_file:
                        out_file.write(response.read())

                results.append((url, dest))
                return

            except (URLError, HTTPError, ConnectionError, TimeoutError) as e:
                if attempt == max_retries - 1:
                    errors.append((url, str(e)))
                else:
                    time.sleep(2**attempt)
            except Exception as e:
                errors.append((url, f"Unexpected error: {str(e)}"))
                break

        results.append((url, dest)) if any(url == r[0] for r in results) else None

    os.makedirs(save_dir, exist_ok=True)

    threads = []
    for url in urls:
        filename = os.path.split(url)[-1]
        dest_path = os.path.join(save_dir, filename)

        if os.path.exists(dest_path):
            results.append((url, dest_path))
            continue

        t = threading.Thread(target=getter, args=(url, dest_path), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.05)

    for t in threads:
        t.join()

    image_paths = [dest for url, dest in results]

    if errors:
        logger.warning(f"Не загружено файлов: {len(errors)} из {len(urls)}")

    return image_paths


def load_images(urls: List[str], save_dir: str, max_threads_num: int = 10) -> List[str]:
    """
    Загружает изображения батчами.

    Args:
        urls: Список URL-адресов изображений
        save_dir: Директория для сохранения
        max_threads_num: Количество потоков для параллельной загрузки

    Returns:
        Список путей к загруженным файлам
    """
    if not urls:
        logger.warning("Список URL пуст")
        return []

    logger.info(f"Начало загрузки {len(urls)} изображений в {save_dir}")

    image_paths = []
    total_batches = (len(urls) + max_threads_num - 1) // max_threads_num

    for i in range(0, len(urls), max_threads_num):
        batch_num = i // max_threads_num + 1
        batch = urls[i : i + max_threads_num]

        if batch_num == 1 or batch_num == total_batches or batch_num % 50 == 0:
            logger.info(f"Батч {batch_num}/{total_batches} ({len(batch)} файлов)")

        new_paths = _load_images(batch, save_dir)
        image_paths.extend(new_paths)

        if i + max_threads_num < len(urls):
            time.sleep(1)

    success_rate = len(image_paths) / len(urls) * 100
    logger.info(
        f"Завершено. Успешно загружено: {len(image_paths)}/{len(urls)} ({success_rate:.1f}%)"
    )

    return image_paths


def load_or_download_df(
    source_path, working_path, images_dir, path_column="img_path", url_column="image"
):
    """
    Загружает DataFrame с путями, если он есть.
    Если нет — загружает исходный, скачивает картинки и сохраняет новый.
    """
    if os.path.exists(working_path):
        print(f"Найдён готовый файл: {working_path}")
        df = pd.read_csv(working_path)

        if path_column in df.columns and os.path.exists(df[path_column].iloc[0]):
            print("Пути валидны, картинки на месте.")
            return df
        else:
            print(
                "Файл есть, но картинки не найдены. Будет произведена повторная загрузка."
            )

    print(f"Загрузка исходного файла: {source_path}")
    df = pd.read_csv(source_path)

    print(f"Скачивание изображений в: {images_dir}...")
    df[path_column] = load_images(
        df[url_column].tolist(), images_dir, max_threads_num=30
    )

    print(f"Сохранение DataFrame с путями: {working_path}")
    df.to_csv(working_path, index=False)

    return df


def plot_imgs_with_labels_from_urls(
    image_urls: List[str],
    labels_1: Tuple[List[Any], str] = None,
    labels_2: Tuple[List[Any], str] = None,
    labels_3: Tuple[List[Any], str] = None,
):
    """Отображает изображения по URL с подписями."""
    plt.figure(figsize=(20, 20))

    n_subplots = len(image_urls)
    n_lines = int((n_subplots + 2) // 3)

    for i in range(n_subplots):
        plt.subplot(n_lines, 3, i + 1)

        try:
            # Загрузка изображения по URL
            response = urllib.request.urlopen(image_urls[i])
            img = Image.open(io.BytesIO(response.read()))
            plt.imshow(img)
        except Exception as e:
            # Если не удалось загрузить, показываем заглушку
            plt.text(
                0.5, 0.5, f"Error\n{str(e)[:20]}", ha="center", va="center", fontsize=8
            )
            plt.xlim(0, 1)
            plt.ylim(0, 1)

        # Формирование заголовка
        title = ""
        if labels_1 is not None:
            title += f"{labels_1[1]}: {labels_1[0][i]}"
        if labels_2 is not None:
            val_2 = labels_2[0][i]
            if isinstance(val_2, float):
                val_2 = round(val_2, 4)
            title += f"\n{labels_2[1]}: {val_2}"

        if labels_3 is not None:
            val_3 = labels_3[0][i]
            if isinstance(val_3, float):
                val_3 = round(val_3, 4)
            title += f"\n{labels_3[1]}: {val_3}"

        plt.title(title, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_sample_from_urls(
    test_df, value="детская", column="label_pred", size=10, url_column="image"
):
    """Отображает случайные примеры из DataFrame по URL."""
    plot_df = test_df[test_df[column] == value]
    sample_size = min(plot_df.shape[0], size)

    sample_df = plot_df.sample(sample_size).reset_index()

    plot_imgs_with_labels_from_urls(
        sample_df[url_column].tolist(),
        (sample_df["label"], "label"),
        (sample_df["label_pred"], "label_pred"),
        (sample_df["proba"], "proba"),
    )
