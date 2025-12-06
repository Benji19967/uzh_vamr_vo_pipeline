import numpy as np

from src.structures.candidate_tracks import CandidateTracks
from src.structures.keypoints2D import CandidateKeypoints2D
from src.structures.pose import Pose


def test_add_candidate_tracks_init():
    candidate_keypoints = CandidateKeypoints2D(np.array([[0, 0, 0, 0], [1, 2, 3, 4]]))
    pose = Pose(np.eye(3), np.array([0, 0, 0]))

    candidate_tracks = CandidateTracks()
    candidate_tracks.add_candidate_keypoints(candidate_keypoints, pose)

    assert np.array_equal(
        candidate_tracks.get_current_coords().array, candidate_keypoints.array
    )
    assert np.array_equal(
        candidate_tracks.get_initial_coords().array, candidate_keypoints.array
    )
    assert candidate_tracks.get_poses_at_intial_coords() == [pose, pose, pose, pose]


def test_add_candidate_tracks_to_existing():
    candidate_keypoints = CandidateKeypoints2D(np.array([[0, 0, 0, 0], [1, 2, 3, 4]]))
    pose = Pose(np.eye(3), np.array([0, 0, 0]))

    candidate_tracks = CandidateTracks()
    candidate_tracks.add_candidate_keypoints(candidate_keypoints, pose)

    assert np.array_equal(
        candidate_tracks.get_current_coords().array, candidate_keypoints.array
    )
    assert np.array_equal(
        candidate_tracks.get_initial_coords().array, candidate_keypoints.array
    )
    assert candidate_tracks.get_poses_at_intial_coords() == [pose, pose, pose, pose]

    candidate_keypoints2 = CandidateKeypoints2D(np.array([[0, 0], [4, 6]]))
    pose2 = Pose(np.eye(3), np.array([1, 1, 1]))

    candidate_tracks.add_candidate_keypoints(candidate_keypoints2, pose2)

    expected_current_coords = np.c_[
        candidate_keypoints.array, candidate_keypoints2.array
    ]
    assert np.array_equal(
        candidate_tracks.get_current_coords().array, expected_current_coords
    )

    expected_initial_coords = np.c_[
        candidate_keypoints.array, candidate_keypoints2.array
    ]
    assert np.array_equal(
        candidate_tracks.get_initial_coords().array, expected_initial_coords
    )

    assert candidate_tracks.get_poses_at_intial_coords() == [
        pose,
        pose,
        pose,
        pose,
        pose2,
        pose2,
    ]


def test_candidate_keypoints_update_tracks():
    candidate_keypoints = CandidateKeypoints2D(np.array([[0, 0, 0, 0], [1, 2, 3, 4]]))
    pose = Pose(np.eye(3), np.array([0, 0, 0]))

    candidate_tracks = CandidateTracks()
    candidate_tracks.add_candidate_keypoints(candidate_keypoints, pose)

    candidate_keypoints2 = CandidateKeypoints2D(np.array([[0, 0, 0, 0], [2, 3, 4, 5]]))
    tracked_mask = np.array([True, False, True, False])
    candidate_tracks.update_tracks(
        new_coords=candidate_keypoints2, tracked_mask=tracked_mask
    )

    expected_current_coords = np.array([[0, 0], [2, 4]])
    assert np.array_equal(
        candidate_tracks.get_current_coords().array, expected_current_coords
    )

    expected_initial_coords = np.array([[0, 0], [1, 3]])
    assert np.array_equal(
        candidate_tracks.get_initial_coords().array, expected_initial_coords
    )

    assert candidate_tracks.get_poses_at_intial_coords() == [pose, pose]
