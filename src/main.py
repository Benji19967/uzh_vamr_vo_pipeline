import numpy as np

import tracking
from src.features import features_cv2
from src.features.descriptor import compute_descriptors, match_descriptors
from src.features.features import Keypoints
from src.initialize import initialize
from src.utils import utils
from src.utils.image_reader import KittiDataReader, MalagaDataReader, ParkingDataReader

K_Malaga = np.array(
    [
        [837.619011, 0, 522.434637],
        [0, 839.808333, 402.367400],
        [0, 0, 1],
    ]
)
K_Parking = np.array(
    [
        [331.37, 0, 320],
        [0, 369.568, 240],
        [0, 0, 1],
    ]
)


def demo_image_readers() -> None:
    KittiDataReader.show_image()
    MalagaDataReader.show_images(end_id=4)
    ParkingDataReader.show_image(id=5)

    image1 = KittiDataReader.read_image()
    image2 = MalagaDataReader.read_image(id=3)
    images = ParkingDataReader.read_images(start_id=100, end_id=105)

    for image in [image1, image2, *images]:
        utils.show_img(img=image.img)


def main() -> None:
    I_0 = ParkingDataReader.read_image(id=0)
    I_1 = ParkingDataReader.read_image(id=2)

    p1_P_keypoints, p2_P_keypoints, p_W_landmarks = initialize(
        I_0=I_0, I_1=I_1, K=K_Parking
    )
    images = ParkingDataReader.read_images(end_id=30)

    # img0 = images[0].img
    # p_P_keypoints0 = features_cv2.good_features_to_track(img=img0, max_features=200)
    # descriptors0 = compute_descriptors(img=img0, p_P_keypoints=p_P_keypoints0)
    # img1 = images[5].img
    # p_P_keypoints1 = features_cv2.good_features_to_track(img=img1, max_features=200)
    # descriptors1 = compute_descriptors(img=img1, p_P_keypoints=p_P_keypoints1)
    # matches = match_descriptors(
    #     query_descriptors=descriptors1, db_descriptors=descriptors0
    # )
    # print(matches)
    # print(matches.shape)
    # print(p_P_keypoints0.shape)
    # print(p_P_keypoints1.shape)

    # kp0 = Keypoints(image=images[0])
    # p_P_keypoints = kp0.select(num_keypoints=200)
    # p_P_keypoints[[0, 1]] = p_P_keypoints[[1, 0]]  # keypoints are (y, x)

    tracking.run_klt(
        images=images,
        p_P_keypoints_initial=p1_P_keypoints,
        p_W_landmarks_initial=p_W_landmarks,
        K=K_Parking,
    )


if __name__ == "__main__":
    # demo_image_readers()
    # init()
    main()
