from enum import Enum
from pathlib import Path

import numpy as np


class Dataset(Enum):
    KITTI = 1
    MALAGA = 2
    PARKING = 3
    OTHER = 4


class Image:
    def __init__(
        self,
        img: np.ndarray,
        filepath: Path,
    ) -> None:
        self._img = img
        self._filepath = filepath

    @property
    def img(self) -> np.ndarray:
        return self._img

    @property
    def filepath(self) -> Path:
        return self._filepath

    def __str__(self) -> str:
        return f"{self.filepath}"


class DatasetImage(Image):

    def __init__(
        self,
        img: np.ndarray,
        dataset: Dataset | str,
        id: int,
        filepath: Path,
    ) -> None:
        super().__init__(img=img, filepath=filepath)
        self._dataset = dataset
        self._id = id

    @property
    def id(self) -> int:
        return self._id

    @property
    def dataset(self) -> Dataset:
        if isinstance(self._dataset, str):
            return Dataset[self._dataset.upper()]
        return self._dataset

    def __str__(self) -> str:
        return f"{self.dataset.name.lower()}_{self.filepath}"
