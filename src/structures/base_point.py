from typing import Annotated, Literal, Type, TypeVar

import numpy as np
import numpy.typing as npt

ArrayXxN = Annotated[npt.NDArray[np.int32], Literal[2 | 3, "N"]]
ArrayBooleanN = Annotated[npt.NDArray[np.bool_], Literal["N"]]

T = TypeVar("T", bound="BasePoints")


class BasePoints:
    def __init__(self, array: ArrayXxN) -> None:
        self._array = array

    def keep(self, mask: ArrayBooleanN) -> None:
        if self._array.any():
            self._array = self._array[:, mask]

    def keep_last(self, n: int) -> None:
        """Keep the last n points"""
        self._array = self.array[:, -n:]

    def filtered(self: T, mask: ArrayBooleanN) -> T:
        if self._array.any():
            return self.__class__(self._array[:, mask])
        raise ValueError("Array not set")

    def add(self, points: "BasePoints" | ArrayXxN) -> None:
        if isinstance(points, BasePoints):
            self._array = np.c_[self.array, points.array]
        else:
            self._array = np.c_[self.array, points]

    @property
    def array(self) -> ArrayXxN:
        return self._array

    @property
    def array_hom(self) -> ArrayXxN:
        return np.r_[self._array, np.ones((1, self._array.shape[1]))]

    @property
    def count(self) -> int:
        return self.shape[1]

    @property
    def shape(self):
        return self._array.shape
