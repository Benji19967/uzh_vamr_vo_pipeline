from enum import Enum
from typing import Annotated, List, Optional

import typer

from src.config import settings
from src.initialize import initialize
from src.utils.data_reader import KittiDataReader, MalagaDataReader, ParkingDataReader
from src.vo import run_vo


class Dataset(str, Enum):
    PARKING = "parking"
    MALAGA = "malaga"
    KITTI = "kitti"


class Plot(str, Enum):
    KEYPOINTS = "keypoints"
    LANDMARKS = "landmarks"
    TRACKING = "tracking"
    REPROJECTION_ERRORS = "reprojection-errors"
    SCALE_DRIFT = "scale-drift"
    TRAJECTORY = "trajectory"


def run(
    dataset: Annotated[Dataset, typer.Option()],
    plot: Annotated[List[Plot], typer.Option()] = [
        Plot.KEYPOINTS,
        Plot.LANDMARKS,
        Plot.REPROJECTION_ERRORS,
        Plot.SCALE_DRIFT,
        Plot.TRAJECTORY,
    ],
    num_images: Annotated[Optional[int], typer.Option()] = None,
) -> None:
    if dataset == Dataset.PARKING:
        DataReader = ParkingDataReader
        K = settings.K_PARKING
        NUM_IMAGES = settings.NUM_IMAGES_PARKING
        SECOND_IMAGE_ID = settings.INITIALIZATION_SECOND_IMAGE_ID_PARKING
    elif dataset == Dataset.MALAGA:
        DataReader = MalagaDataReader
        K = settings.K_MALAGA
        NUM_IMAGES = settings.NUM_IMAGES_MALAGA
        SECOND_IMAGE_ID = settings.INITIALIZATION_SECOND_IMAGE_ID_MALAGA
    elif dataset == Dataset.KITTI:
        DataReader = KittiDataReader
        K = settings.K_KITTI
        NUM_IMAGES = settings.NUM_IMAGES_KITTI
        SECOND_IMAGE_ID = settings.INITIALIZATION_SECOND_IMAGE_ID_KITTI

    NUM_IMAGES = num_images or NUM_IMAGES

    image_0 = DataReader.read_image(id=0)
    image_1 = DataReader.read_image(id=SECOND_IMAGE_ID)
    p1_I_keypoints, _, p_W_landmarks = initialize(image_0, image_1, K=K)

    plot_trajectory = Plot.TRAJECTORY in plot
    plot_scale_drift = Plot.SCALE_DRIFT in plot
    camera_positions_ground_truth = (
        DataReader.read_trajectory() if (plot_trajectory or plot_scale_drift) else None
    )
    images = DataReader.read_imgs(end_id=NUM_IMAGES)
    run_vo(
        images=images,
        p_I_keypoints_initial=p1_I_keypoints,
        p_W_landmarks_initial=p_W_landmarks,
        K=K,
        plot_keypoints=Plot.KEYPOINTS in plot,
        plot_landmarks=Plot.LANDMARKS in plot,
        plot_tracking=Plot.TRACKING in plot,
        plot_reprojection_errors=Plot.REPROJECTION_ERRORS in plot,
        plot_scale_drift=Plot.SCALE_DRIFT in plot,
        plot_trajectory=Plot.TRAJECTORY in plot,
        camera_positions_ground_truth=camera_positions_ground_truth,
    )
