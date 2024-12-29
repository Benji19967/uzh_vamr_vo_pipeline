import numpy as np

from src.initialize import initialize
from src.utils import utils
from src.utils.image_reader import KittiDataReader, MalagaDataReader, ParkingDataReader


def demo_image_readers() -> None:
    KittiDataReader.show_image()
    MalagaDataReader.show_images(end_id=4)
    ParkingDataReader.show_image(id=5)

    image1 = KittiDataReader.read_image()
    image2 = MalagaDataReader.read_image(id=3)
    images = ParkingDataReader.read_images(start_id=100, end_id=105)

    for image in [image1, image2, *images]:
        utils.show_img(img=image.img)


def init() -> None:
    I_0 = ParkingDataReader.read_image(id=0)
    I_1 = ParkingDataReader.read_image(id=1)
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

    p1, p2, P = initialize(I_0=I_0, I_1=I_1, K=K_Parking)
    print(p1)
    print(p2)
    print(P)


def main() -> None:
    pass


if __name__ == "__main__":
    # demo_image_readers()
    init()
    main()
