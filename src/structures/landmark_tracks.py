from collections import defaultdict
from typing import Iterable

import numpy as np

from src.structures.keypoints2D import Keypoints2D
from src.structures.landmarks3D import Landmarks3D
from src.structures.pose import Pose
from src.utils.masks import compose_masks


class LandmarkTracks:
    """
    Equivalent to MapPoints: keep track of points that are already triangulated and part of the map.
    """

    def __init__(
        self,
    ) -> None:
        self._landmarks = Landmarks3D()
        self._observations: dict[int, Keypoints2D] = defaultdict(Keypoints2D)
        self._observations_landmark_indexes: dict[int, np.ndarray] = {}
        self._poses: dict[int, Pose] = {}
        self._active_mask: np.ndarray = np.array([], dtype=bool)

    def add_landmarks(
        self, frame_id: int, landmarks: Landmarks3D, observations: Keypoints2D
    ) -> None:
        self._landmarks.add(landmarks)
        self._observations[frame_id].add(observations)
        self._active_mask = np.append(
            self._active_mask, np.ones(landmarks.count, dtype=bool)
        )
        self._observations_landmark_indexes[frame_id] = self._active_mask.nonzero()[0]

    def add_frame_observations(
        self, frame_id: int, observations: Keypoints2D, tracked_mask: np.ndarray
    ) -> None:
        self._observations[frame_id] = observations.filtered(tracked_mask)
        self._active_mask = compose_masks(self._active_mask, tracked_mask)
        self._observations_landmark_indexes[frame_id] = self._active_mask.nonzero()[0]

    def add_frame_pose(self, frame_id: int, pose: Pose) -> None:
        self._poses[frame_id] = pose

    @property
    def num_landmarks(self) -> int:
        return self.get_active_landmarks().count

    def keep(self, mask: np.ndarray) -> None:
        self._active_mask = compose_masks(self._active_mask, mask)
        last_frame = next(reversed(self._observations))
        self._observations[last_frame].keep(mask)

    def get_active_landmarks(self) -> Landmarks3D:
        return self._landmarks.filtered(self._active_mask)

    def set_landmarks_by_ids(self, landmarks: Landmarks3D, ids: list[int]) -> None:
        assert landmarks.count == len(ids)
        self._landmarks.array[:, ids] = landmarks.array

    def get_active_keypoints(self) -> Keypoints2D:
        last_frame = next(reversed(self._observations))
        return self._observations[last_frame]

    def get_poses_by_ids(self, ids: list[int]) -> list[Pose]:
        return [self._poses[id] for id in ids]

    def set_poses_by_ids(self, poses: list[Pose], ids: list[int]) -> None:
        assert len(poses) == len(ids)
        for pose, id in zip(poses, ids):
            self._poses[id] = pose

    def get_visible_landmarks(
        self, frame_ids: list[int]
    ) -> tuple[Landmarks3D, np.ndarray]:
        """
        Returns:
            - landmarks
            - visible_landmarks: np.ndarray(N, ): visible landmarks indexes
        """
        visible_landmarks = self._observations_landmark_indexes[frame_ids[0]]
        for frame_id in frame_ids[1:]:
            visible_landmarks = np.concatenate(
                (visible_landmarks, self._observations_landmark_indexes[frame_id])
            )
        visible_landmarks = np.unique(visible_landmarks)  # elements are sorted
        return (
            Landmarks3D(self._landmarks.array[:, visible_landmarks]),
            visible_landmarks,
        )

    def get_observations(
        self, frame_ids: Iterable[int] | None = None
    ) -> list[tuple[int, int, int, int]]:
        frame_ids_set = None if not frame_ids else set(frame_ids)
        obs = []
        for frame_id in self._observations:
            if frame_ids_set and frame_id not in frame_ids_set:
                continue
            observations = self._observations[frame_id]
            visible_landmarks = self._observations_landmark_indexes[frame_id]
            for uv, landmark_idx in zip(observations.array.T, visible_landmarks):
                obs.append((frame_id, landmark_idx, int(uv[0]), int(uv[1])))
        return obs
