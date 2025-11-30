import numpy as np
from pydantic_settings import BaseSettings


class PlotSettings(BaseSettings):
    plot_keypoints: bool
    plot_landmarks: bool
    plot_tracking: bool
    plot_reprojection_errors: bool
    plot_scale_drift: bool
    plot_trajectory: bool


class ParkingSettings:
    initialization_second_image_id: int = 2
    num_images: int = 599
    k: np.ndarray = np.array(
        [
            [331.37, 0, 320],
            [0, 369.568, 240],
            [0, 0, 1],
        ]
    )


class MalagaSettings:
    initialization_second_image_id: int = 2
    num_images: int = 2121
    k: np.ndarray = np.array(
        [
            [837.619011, 0, 522.434637],
            [0, 839.808333, 402.367400],
            [0, 0, 1],
        ]
    )


class KittiSettings:
    initialization_second_image_id: int = 3
    num_images: int = 2761
    k: np.ndarray = np.array(
        [
            [707.0912, 0, 601.8873],
            [0, 707.0912, 183.1104],
            [0, 0, 1],
        ]
    )


class DatasetSettings:
    parking: ParkingSettings = ParkingSettings()
    malaga: MalagaSettings = MalagaSettings()
    kitti: KittiSettings = KittiSettings()


class Settings(BaseSettings):
    dataset: DatasetSettings = DatasetSettings()
    # plotting: PlotSettings = PlotSettings()


settings = Settings()
