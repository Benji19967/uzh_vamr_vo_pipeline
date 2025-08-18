from functools import wraps
from pathlib import Path
from time import time
from typing import Any

import cv2
import numpy as np
from matplotlib import pyplot as plt

cv2: Any  # supresses pyright type hint errors

MAX_LENGTH_IMAGE = 1920


def read_img(filepath: Path) -> np.ndarray:
    return cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)  # type: ignore


def show_img(img: np.ndarray, title: str | None = None):
    plt.imshow(img, interpolation="nearest")
    plt.gray()
    if title:
        plt.title(title)
    plt.show()


def show_img_cv(img: np.ndarray):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def downsample_img(img: np.ndarray, width: int, height: int):
    inverse = False
    if height > width:
        width, height = height, width
        inverse = True

    while width > MAX_LENGTH_IMAGE:
        width //= 2
        height //= 2

    if inverse:
        return cv2.resize(img, (height, width))
    return cv2.resize(img, (width, height))


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap
