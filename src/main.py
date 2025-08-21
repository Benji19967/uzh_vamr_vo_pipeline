import numpy as np

import src.vo as vo
from src.initialize import initialize
from src.utils.image_reader import KittiDataReader, MalagaDataReader, ParkingDataReader

NUM_IMAGES_PARKING = 598

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
    image_0 = ParkingDataReader.read_image(id=0)
    image_1 = ParkingDataReader.read_image(id=2)
    p1_I_keypoints, _, p_W_landmarks = initialize(image_0, image_1, K=K_PARKING)

    images = ParkingDataReader.read_imgs(end_id=NUM_IMAGES_PARKING)
    vo.run_vo(
        images=images,
        p_I_keypoints_initial=p1_I_keypoints,
        p_W_landmarks_initial=p_W_landmarks,
        K=K_PARKING,
    )


if __name__ == "__main__":
    main()
