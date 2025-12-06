from pathlib import Path

import cv2
import numpy as np

from src.structures.keypoints2D import Keypoints2D
from src.structures.pose import Pose

HERE = Path(__file__).parent
BA_DATA_FILENAME = HERE / ".." / ".." / "ba_data" / "ba_data.txt"


class BAExporter:

    def __init__(self) -> None:
        pass

    def write_header(self, num_keyframes: int) -> None:
        with open(BA_DATA_FILENAME, "w") as f:
            f.write(f"{num_keyframes}\n")

    def write(self, pose: Pose, landmarks, keypoints: Keypoints2D):
        rvec, _ = cv2.Rodrigues(pose.R)  # type: ignore
        tvec = pose.t
        with open(BA_DATA_FILENAME, "a+") as f:
            f.write(f"{landmarks.shape[1]}\n")
            np.savetxt(f, rvec)
            np.savetxt(f, tvec.T)
            np.savetxt(f, keypoints.array.T)
            np.savetxt(f, landmarks.array.T)
