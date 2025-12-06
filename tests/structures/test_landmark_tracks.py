import numpy as np

from src.structures.keypoints2D import Keypoints2D
from src.structures.landmark_tracks import LandmarkTracks
from src.structures.landmarks3D import Landmarks3D


def test_add_landmarks_init():
    landmarks = Landmarks3D(np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 3, 4, 5]]))
    observations = Keypoints2D(np.array([[0, 0, 0, 0], [2, 4, 6, 9]]))

    landmark_tracks = LandmarkTracks()
    landmark_tracks.add_landmarks(
        frame_id=0, landmarks=landmarks, observations=observations
    )

    assert np.array_equal(
        landmark_tracks._active_mask, np.array([True, True, True, True])
    )
    assert np.array_equal(landmark_tracks.get_active_landmarks().array, landmarks.array)
    assert np.array_equal(
        landmark_tracks.get_active_keypoints().array, observations.array
    )
    assert landmark_tracks.get_observations() == [
        (0, 0, 0, 2),
        (0, 1, 0, 4),
        (0, 2, 0, 6),
        (0, 3, 0, 9),
    ]


def test_add_landmarks_to_existing():
    landmarks = Landmarks3D(np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 3, 4, 5]]))
    observations = Keypoints2D(np.array([[0, 0, 0, 0], [2, 4, 6, 9]]))

    landmark_tracks = LandmarkTracks()
    landmark_tracks.add_landmarks(
        frame_id=0, landmarks=landmarks, observations=observations
    )

    landmarks2 = Landmarks3D(np.array([[7, 7], [8, 8], [10, 11]]))
    observations2 = Keypoints2D(np.array([[88, 88], [91, 92]]))
    landmark_tracks.add_landmarks(
        frame_id=0, landmarks=landmarks2, observations=observations2
    )

    expected_landmarks = np.c_[landmarks.array, landmarks2.array]
    assert np.array_equal(
        landmark_tracks._active_mask, np.array([True, True, True, True, True, True])
    )
    assert np.array_equal(
        landmark_tracks.get_active_landmarks().array, expected_landmarks
    )
    expected_observations = np.c_[observations.array, observations2.array]
    assert np.array_equal(
        landmark_tracks.get_active_keypoints().array, expected_observations
    )
    assert landmark_tracks.get_observations() == [
        (0, 0, 0, 2),
        (0, 1, 0, 4),
        (0, 2, 0, 6),
        (0, 3, 0, 9),
        (0, 4, 88, 91),
        (0, 5, 88, 92),
    ]


def test_track_landmarks_add_frame_observations():
    landmarks = Landmarks3D(np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 3, 4, 5]]))
    observations = Keypoints2D(np.array([[0, 0, 0, 0], [2, 4, 6, 9]]))

    landmark_tracks = LandmarkTracks()
    landmark_tracks.add_landmarks(
        frame_id=0, landmarks=landmarks, observations=observations
    )

    tracked_mask = np.array([True, False, True, False])
    observations2 = Keypoints2D(np.array([[0, 0, 0, 0], [3, 0, 7, 0]]))
    landmark_tracks.add_frame_observations(
        frame_id=5, observations=observations2, tracked_mask=tracked_mask
    )
    assert np.array_equal(
        landmark_tracks._active_mask, np.array([True, False, True, False])
    )

    assert np.array_equal(
        landmark_tracks.get_active_landmarks().array, np.array([[0, 0], [1, 1], [2, 4]])
    )
    assert np.array_equal(
        landmark_tracks.get_active_keypoints().array, np.array([[0, 0], [3, 7]])
    )
    assert landmark_tracks.get_observations() == [
        (0, 0, 0, 2),
        (0, 1, 0, 4),
        (0, 2, 0, 6),
        (0, 3, 0, 9),
        (5, 0, 0, 3),
        (5, 2, 0, 7),
    ]

    tracked_mask = np.array([False, True])
    observations3 = Keypoints2D(np.array([[0, 0], [0, 8]]))
    landmark_tracks.add_frame_observations(
        frame_id=6, observations=observations3, tracked_mask=tracked_mask
    )
    assert np.array_equal(
        landmark_tracks._active_mask, np.array([False, False, True, False])
    )
    assert np.array_equal(
        landmark_tracks.get_active_landmarks().array, np.array([[0], [1], [4]])
    )
    assert np.array_equal(
        landmark_tracks.get_active_keypoints().array, np.array([[0], [8]])
    )
    assert landmark_tracks.get_observations() == [
        (0, 0, 0, 2),
        (0, 1, 0, 4),
        (0, 2, 0, 6),
        (0, 3, 0, 9),
        (5, 0, 0, 3),
        (5, 2, 0, 7),
        (6, 2, 0, 8),
    ]
