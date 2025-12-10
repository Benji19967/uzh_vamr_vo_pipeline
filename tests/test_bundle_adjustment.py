import numpy as np

from src.structures.landmarks3D import Landmarks3D
from src.structures.pose import Pose


def test_bundle_adjustment():
    bundle_adjuster = BundleAdjuster()

    pose = Pose(np.eye(3), np.array([0, 0, 0]))
    landmarks = Landmarks3D(np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 3, 4, 5]]))

    bundle_adjuster.refine_pose_and_landmarks(poses, landmarks)
