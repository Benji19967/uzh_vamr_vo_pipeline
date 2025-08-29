import numpy as np
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    INITIALIZATION_SECOND_IMAGE_ID_PARKING: int = 2
    INITIALIZATION_SECOND_IMAGE_ID_MALAGA: int = 2
    INITIALIZATION_SECOND_IMAGE_ID_KITTI: int = 3

    NUM_IMAGES_PARKING: int = 599
    NUM_IMAGES_MALAGA: int = 2121
    NUM_IMAGES_KITTI: int = 2761

    K_PARKING: np.ndarray = np.array(
        [
            [331.37, 0, 320],
            [0, 369.568, 240],
            [0, 0, 1],
        ]
    )
    K_MALAGA: np.ndarray = np.array(
        [
            [837.619011, 0, 522.434637],
            [0, 839.808333, 402.367400],
            [0, 0, 1],
        ]
    )
    K_KITTI: np.ndarray = np.array(
        [
            [707.0912, 0, 601.8873],
            [0, 707.0912, 183.1104],
            [0, 0, 1],
        ]
    )


settings = Settings()
