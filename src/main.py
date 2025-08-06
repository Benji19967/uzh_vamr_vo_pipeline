import numpy as np

import tracking
from src.features import features_cv2
from src.features.descriptor import compute_descriptors
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


def init():
    # TODO: should be far enough apart but not too far...
    I_0 = ParkingDataReader.read_image(id=0)
    I_1 = ParkingDataReader.read_image(id=1)

    p1_P_keypoints, p2_P_keypoints, p_W = initialize(I_0=I_0, I_1=I_1, K=K_Parking)
    # print(p1_P_keypoints)
    # print(p2_P_keypoints)
    # print(p_W)

    # TODO: ransacLocalization

    return p1_P_keypoints, p2_P_keypoints, p_W


def main() -> None:
    p1_P_keypoints, p2_P_keypoints, p_W_landmarks = init()
    images = ParkingDataReader.read_images(end_id=30)

    img = images[0].img
    p_P_keypoints = features_cv2.good_features_to_track(img=img, max_features=100)
    descriptors = compute_descriptors(img=img, p_P_keypoints=p_P_keypoints)
    print(descriptors.shape)

    # kp0 = Keypoints(image=images[0])
    # p_P_keypoints = kp0.select(num_keypoints=200)
    # p_P_keypoints[[0, 1]] = p_P_keypoints[[1, 0]]  # keypoints are (y, x)

    # tracking.run_klt(
    #     images=images,
    #     p_P_keypoints_initial=p1_P_keypoints,
    #     p_W_landmarks_initial=p_W_landmarks,
    #     K=K_Parking,
    # )


if __name__ == "__main__":
    # demo_image_readers()
    # init()
    main()
