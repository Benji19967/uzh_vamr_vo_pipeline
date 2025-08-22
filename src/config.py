import numpy as np
from pydantic_settings import BaseSettings

POSTGRES_URL_FORMAT = "postgresql://{username}:{password}@{host}:{port}/{database_name}"


class Settings(BaseSettings):
    NUM_IMAGES_PARKING: int = 598
    NUM_IMAGES_MALAGA: int = 2121
    NUM_IMAGES_KITTI: int = 2760

    K_MALAGA: np.ndarray = np.array(
        [
            [837.619011, 0, 522.434637],
            [0, 839.808333, 402.367400],
            [0, 0, 1],
        ]
    )
    K_PARKING: np.ndarray = np.array(
        [
            [331.37, 0, 320],
            [0, 369.568, 240],
            [0, 0, 1],
        ]
    )
    K_KITTI: np.ndarray = np.array(
        [
            [721.5377, 0, 609.5593],
            [0, 721.5377, 172.8540],
            [0, 0, 1],
        ]
    )


settings = Settings()
