import numpy as np


class Pose:

    def __init__(self, R, t) -> None:
        self.R: np.ndarray = R  # (3,3) rotation matrix
        self.t: np.ndarray = t  # (3,) translation vector

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.eye(3), np.zeros(3))

    @property
    def T_C_W(self):
        return np.c_[self.R, self.t]
