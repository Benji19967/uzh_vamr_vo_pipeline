import numpy as np

from src.bundle_adjustment.ba import BundleAdjuster
from src.structures.keypoints2D import Keypoints2D
from src.structures.landmark_tracks import LandmarkTracks
from src.structures.landmarks3D import Landmarks3D
from src.structures.pose import Pose
from src.transformations.transformations import camera_to_pixel, world_to_camera


def test_bundle_adjustment():
    np.random.seed(42)

    K = np.eye(3)
    pose = Pose(np.eye(3), np.array([0, 0, 0]))
    landmarks = Landmarks3D(np.array([[0, 0], [1, 1], [2, 3]]))
    p_I = camera_to_pixel(world_to_camera(landmarks.array_hom, pose.T_C_W), K)
    observations = Keypoints2D(p_I)

    frame_id = 0
    landmark_tracks = LandmarkTracks()
    landmark_tracks.add_landmarks(
        frame_id=frame_id, landmarks=landmarks, observations=observations
    )
    landmark_tracks.add_frame_pose(frame_id, pose)

    frame_id = 1
    pose2 = Pose(np.eye(3), np.array([1, 1, 1]))
    landmarks2 = Landmarks3D(np.array([[0, 0], [2, 2], [2, 3]]))
    p_I2 = camera_to_pixel(world_to_camera(landmarks2.array_hom, pose2.T_C_W), K)
    observations2 = Keypoints2D(p_I2)

    landmark_tracks.add_landmarks(
        frame_id=frame_id, landmarks=landmarks2, observations=observations2
    )
    landmark_tracks.add_frame_pose(frame_id, pose2)

    # Add noise
    landmark_tracks_noise = LandmarkTracks()
    landmark_tracks_noise.add_landmarks(
        frame_id=0,
        landmarks=Landmarks3D(
            np.asarray(landmarks.array + np.random.randn(*landmarks.array.shape) * 0.01)
        ),
        observations=observations,
    )
    landmark_tracks_noise.add_frame_pose(
        frame_id=0,
        pose=Pose.from_rvec_tvec(
            rvec=pose.rvec.ravel() + np.random.randn(3) * 0.01,
            tvec=pose.tvec.ravel() + np.random.randn(3) * 0.01,
        ),
    )
    landmark_tracks_noise.add_landmarks(
        frame_id=1,
        landmarks=Landmarks3D(
            np.asarray(
                landmarks2.array + np.random.randn(*landmarks2.array.shape) * 0.01
            )
        ),
        observations=observations2,
    )
    landmark_tracks_noise.add_frame_pose(
        frame_id=1,
        pose=Pose.from_rvec_tvec(
            rvec=pose2.rvec.ravel() + np.random.randn(3) * 0.01,
            tvec=pose2.tvec.ravel() + np.random.randn(3) * 0.01,
        ),
    )

    bundle_adjuster = BundleAdjuster(landmark_tracks_noise, K)
    frame_ids = [1]
    landmarks_opt, poses_opt = bundle_adjuster._get_optimized_poses_and_landmarks(
        frame_ids
    )

    lm = landmark_tracks._landmarks.array
    lm_noise = landmark_tracks_noise._landmarks.array
    lm_opt = landmarks_opt.array

    # TODO: more robust test, where the second becomes smaller
    print(np.linalg.norm(lm - lm_noise))
    print(np.linalg.norm(lm - lm_opt))

    assert not np.array_equal(
        landmarks_opt.array, landmark_tracks_noise._landmarks.array
    )
    np.testing.assert_allclose(landmarks_opt.array, landmark_tracks._landmarks.array)
