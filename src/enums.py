from enum import Enum


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
