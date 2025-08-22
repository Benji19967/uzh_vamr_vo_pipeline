from enum import Enum

import typer

from src.config import settings
from src.initialize import initialize
from src.utils.image_reader import KittiDataReader, MalagaDataReader, ParkingDataReader
from src.vo import run_vo

vo = typer.Typer(no_args_is_help=True)


class Dataset(str, Enum):
    PARKING = "parking"
    MALAGA = "malaga"
    KITTI = "kitti"


@vo.command(no_args_is_help=True)
def run(
    dataset: Dataset = typer.Option(
        ...,
        "--dataset",
    ),
) -> None:
    if dataset == Dataset.PARKING:
        DataReader = ParkingDataReader
        K = settings.K_PARKING
        NUM_IMAGES = settings.NUM_IMAGES_PARKING
        SECOND_IMAGE_ID = settings.SECOND_IMAGE_ID_PARKING
    elif dataset == Dataset.MALAGA:
        DataReader = MalagaDataReader
        K = settings.K_MALAGA
        NUM_IMAGES = settings.NUM_IMAGES_MALAGA
        SECOND_IMAGE_ID = settings.SECOND_IMAGE_ID_MALAGA
    elif dataset == Dataset.KITTI:
        DataReader = KittiDataReader
        K = settings.K_KITTI
        NUM_IMAGES = settings.NUM_IMAGES_KITTI
        SECOND_IMAGE_ID = settings.SECOND_IMAGE_ID_KITTI

    image_0 = DataReader.read_image(id=0)
    image_1 = DataReader.read_image(id=SECOND_IMAGE_ID)
    p1_I_keypoints, _, p_W_landmarks = initialize(image_0, image_1, K=K)

    images = DataReader.read_imgs(end_id=NUM_IMAGES)
    run_vo(
        images=images,
        p_I_keypoints_initial=p1_I_keypoints,
        p_W_landmarks_initial=p_W_landmarks,
        K=K,
    )
