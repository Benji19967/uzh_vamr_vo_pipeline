import numpy as np

from src.bundle_adjustment.ba import BundleAdjuster
from src.structures.keypoints2D import Keypoints2D
from src.structures.landmark_tracks import LandmarkTracks
from src.structures.landmarks3D import Landmarks3D
from src.structures.pose import Pose


def test_bundle_adjustment():
    landmarks = Landmarks3D(np.array([[0, 0], [1, 1], [2, 3]]))
    observations = Keypoints2D(np.array([[0, 0], [2, 4]]))
    pose = Pose(np.eye(3), np.array([0, 0, 0]))

    frame_id = 0
    landmark_tracks = LandmarkTracks()
    landmark_tracks.add_landmarks(
        frame_id=frame_id, landmarks=landmarks, observations=observations
    )
    landmark_tracks.add_frame_pose(frame_id, pose)

    frame_id = 1
    landmarks2 = Landmarks3D(np.array([[0, 0], [2, 2], [2, 3]]))
    observations2 = Keypoints2D(np.array([[0, 0], [6, 8]]))
    pose2 = Pose(np.eye(3), np.array([1, 1, 1]))

    landmark_tracks.add_landmarks(
        frame_id=frame_id, landmarks=landmarks2, observations=observations2
    )
    landmark_tracks.add_frame_pose(frame_id, pose2)

    bundle_adjuster = BundleAdjuster(landmark_tracks)
    frame_ids = [0, 1]
    K = np.eye(3)
    landmarks, poses = bundle_adjuster._get_optimized_poses_and_landmarks(frame_ids, K)

    print(landmarks.array, poses[0])

    # assert bundle_adjuster._landmark_tracks._landmarks
