import numpy as np

import src.vo as vo
from src.config import settings
from src.initialize import initialize
from src.utils.image_reader import KittiDataReader, MalagaDataReader, ParkingDataReader


def main() -> None:
    DataReader = ParkingDataReader
    K = settings.K_PARKING
    NUM_IMAGES = settings.NUM_IMAGES_PARKING

    # DataReader = MalagaDataReader
    # K = settings.K_MALAGA
    # NUM_IMAGES = settings.NUM_IMAGES_MALAGA

    image_0 = DataReader.read_image(id=0)
    image_1 = DataReader.read_image(id=2)
    p1_I_keypoints, _, p_W_landmarks = initialize(image_0, image_1, K=K)

    images = DataReader.read_imgs(end_id=NUM_IMAGES)
    vo.run_vo(
        images=images,
        p_I_keypoints_initial=p1_I_keypoints,
        p_W_landmarks_initial=p_W_landmarks,
        K=K,
    )


if __name__ == "__main__":
    main()
