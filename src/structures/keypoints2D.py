from typing import Type, TypeVar

import numpy as np

from src.structures.base_point import BasePoints

T = TypeVar("T", bound="BaseKeypoints2D")


class BaseKeypoints2D(BasePoints):

    def to_cv2(self) -> np.ndarray:
        """
        Convert format of points to match what cv2 expects

        Args:
            self.array (np.ndarray): (2xN)

        Returns:
            np.ndarray: (Nx1x2)
        """
        arr = self.array.astype(np.float32)
        N = arr.shape[1]
        return arr.T.reshape((N, 1, 2))

    @classmethod
    def from_cv2(cls: Type[T], p_I: np.ndarray) -> T:
        """
        Args:
            p_I (np.ndarray): (Nx1x2)

        Returns:
            np.ndarray: (2xN)
        """
        if p_I is None:
            return cls(array=np.zeros((2, 0), dtype=np.int32))
        return cls(array=p_I.T.reshape(2, -1))


class Keypoints2D(BaseKeypoints2D):
    pass


class CandidateKeypoints2D(BaseKeypoints2D):
    pass
