import numpy as np

import src.vo as vo
from src.initialize import initialize
from src.utils.image_reader import KittiDataReader, MalagaDataReader, ParkingDataReader

NUM_IMAGES_PARKING = 598
NUM_IMAGES_MALAGA = 2121

K_MALAGA = np.array(
    [
        [837.619011, 0, 522.434637],
        [0, 839.808333, 402.367400],
        [0, 0, 1],
    ]
)
K_PARKING = np.array(
    [
        [331.37, 0, 320],
        [0, 369.568, 240],
        [0, 0, 1],
    ]
)


def main() -> None:
    DataReader = ParkingDataReader
    K = K_PARKING
    NUM_IMAGES = NUM_IMAGES_PARKING

    # DataReader = MalagaDataReader
    # K = K_MALAGA
    # NUM_IMAGES = NUM_IMAGES_MALAGA

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
