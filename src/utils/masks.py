import numpy as np


def compose_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Compose two masks into one:

    [0, 1, 0, 1, 0, 1, 1]
    [   0,    1,    1, 1]
    =
    [0, 0, 0, 1, 0, 1, 1]
    """

    result = mask1.copy()
    indices = np.where(mask1 == 1)[0]
    result[indices] = mask2
    return result
