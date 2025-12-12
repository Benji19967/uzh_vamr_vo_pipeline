from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src.mapping.reprojection_error import reprojection_errors_ba
from src.structures.landmark_tracks import LandmarkTracks
from src.structures.landmarks3D import Landmarks3D
from src.structures.pose import Pose

HERE = Path(__file__).parent
BA_DATA_FILENAME = HERE / ".." / "ba_data" / "ba_data.txt"


def fun(x, frame_ids, all_poses, all_landmarks, all_observations, K):
    """
    Compute residuals -- 2 for each observation (u_proj - u, v_proj - v)

    Args:
        - x np.ndarray((6*num_cameras) + (3*num_3d_points),)
        -

    Returns:
        - residuals np.ndarray(2*num_observations,)

    x contains the camera parameters and 3D point positions to optimize.

    typical x:
        x = [
            C0_R1,
            C0_R2,
            C0_R3,
            C0_t1,
            C0_t2,
            C0_t3,
            ...
            Cm_R1,
            Cm_R2,
            Cm_R3,
            Cm_t1,
            Cm_t2,
            Cm_t3,
            P0_X,
            P0_Y,
            P0_Z,
            ...
            Pn_X,
            Pn_Y,
            Pn_Z,
        ]
    """
    frame_ids_set = set(frame_ids)
    observations = [o for o in all_observations if o[0] in frame_ids_set]
    return reprojection_errors_ba(x, all_poses, all_landmarks, observations, K)


class BundleAdjuster:
    def __init__(self, landmark_tracks: LandmarkTracks) -> None:
        self._landmark_tracks = landmark_tracks

    def refine_poses_and_landmarks(
        self,
        frame_ids: list[int],
        K: np.ndarray,
    ):
        landmarks, poses = self._get_optimized_poses_and_landmarks(frame_ids, K)
        self._landmark_tracks.set_poses_by_ids(poses, frame_ids)
        self._landmark_tracks.set_landmarks_by_ids(landmarks, frame_ids)

    def _get_optimized_poses_and_landmarks(
        self,
        frame_ids: list[int],
        K: np.ndarray,
    ) -> tuple[Landmarks3D, list[Pose]]:
        frame_ids_mask = np.array(frame_ids)
        poses_to_optimize = self._landmark_tracks.get_poses_by_ids(ids=frame_ids)
        landmarks_to_optimize = self._landmark_tracks.get_active_landmarks().filtered(
            frame_ids_mask
        )

        x0 = np.array(
            [[pose.rvec.flatten(), pose.tvec.flatten()] for pose in poses_to_optimize]
        ).flatten()
        x0 = np.concatenate([x0, landmarks_to_optimize.array.reshape(-1)])

        res = least_squares(
            fun,
            x0,
            method="trf",
            args=(
                frame_ids,
                self._landmark_tracks._poses,
                self._landmark_tracks._landmarks,
                self._landmark_tracks.get_observations(),
                K,
            ),
        )
        optimized_x = res.x

        # Unpack poses
        poses_np = optimized_x[: 6 * len(frame_ids)]
        poses = []
        for i in range(len(frame_ids)):
            block = poses_np[i * 6 : (i + 1) * 6]
            rvec = block[:3]
            tvec = block[3:]
            pose = Pose.from_rvec_tvec(rvec, tvec)
            poses.append(pose)

        # Unpack landmarks
        landmarks_np = optimized_x[6 * len(frame_ids) :]
        landmarks = Landmarks3D(landmarks_np.reshape(3, -1))

        return landmarks, poses


def main():
    pass


if __name__ == "__main__":
    main()
