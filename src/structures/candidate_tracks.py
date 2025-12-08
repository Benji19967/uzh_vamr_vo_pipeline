import numpy as np

from src.structures.keypoints2D import CandidateKeypoints2D
from src.structures.pose import Pose


class CandidateTracks:

    def __init__(self) -> None:
        self._current_coords = CandidateKeypoints2D(np.zeros((2, 0), dtype=np.int32))
        self._initial_coords = CandidateKeypoints2D(np.zeros((2, 0), dtype=np.int32))
        self._poses_at_initial_uv: list[Pose] = []

    def get_current_coords(self) -> CandidateKeypoints2D:
        return self._current_coords

    def get_initial_coords(self) -> CandidateKeypoints2D:
        return self._initial_coords

    def get_poses_at_intial_coords(self) -> list[Pose]:
        return self._poses_at_initial_uv

    def get_TCWs_at_intial_coords(self) -> np.ndarray:
        """
        Flatten:

        From (3, 4):

        r11 r12 r13 tx
        r21 r22 r23 ty
        r31 r32 r33 tz

        to (1, 12):

        r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
        """

        TCWs = [p.T_C_W.flatten() for p in self._poses_at_initial_uv]
        return np.column_stack(TCWs)

    def add_candidate_keypoints(
        self, candidate_keypoints: CandidateKeypoints2D, pose: Pose
    ) -> None:
        self._current_coords.add(candidate_keypoints)
        self._initial_coords.add(candidate_keypoints)
        self._poses_at_initial_uv.extend([pose] * candidate_keypoints.count)

    def update_tracks(
        self, new_coords: CandidateKeypoints2D, tracked_mask: np.ndarray
    ) -> None:
        self._current_coords = new_coords
        self.keep(tracked_mask)

    def keep(self, mask) -> None:
        self._current_coords.keep(mask)
        self._initial_coords.keep(mask)
        self._poses_at_initial_uv = np.array(self._poses_at_initial_uv)[mask].tolist()

    def keep_last(self, n: int) -> None:
        self._current_coords.keep_last(n)
        self._initial_coords.keep_last(n)
        self._poses_at_initial_uv = self._poses_at_initial_uv[-n:]
