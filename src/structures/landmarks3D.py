import numpy as np

from src.structures.base_point import ArrayXxN, BasePoints


class Landmarks3D(BasePoints):
    """
    (3xN) landmarks in 3D coordinates (x, y, z)
    """

    def __init__(self, array: ArrayXxN = np.empty((3, 0))) -> None:
        super().__init__(array)
