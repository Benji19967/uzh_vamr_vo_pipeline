import numpy as np

from src.klt import run_klt


def test_klt():
    image_0 = np.zeros((100, 100), dtype=np.uint8)
    image_1 = np.zeros((100, 100), dtype=np.uint8)
    image_0[50, 50] = 255  # Set a pixel to white in image_0
    image_1[52, 52] = 255  # Set a pixel to white in image_1
    image_0[20, 20] = 255  # Set a pixel to white in image_0
    image_1[21, 21] = 255  # Set a pixel to white in image_1

    p0_I_keypoints = np.array(
        [
            [50, 20, 87],
            [50, 20, 87],
        ]
    )
    p1_I_keypoints_expected = np.array(
        [
            [52, 21, -999],
            [52, 21, -999],
        ]
    )

    p1_I_keypoints, status = run_klt(image_0, image_1, p0_I_keypoints)

    assert np.allclose(p1_I_keypoints[:, :2], p1_I_keypoints_expected[:, :2], atol=1e-1)
    assert status.shape == (3,)
    assert np.allclose(status, np.array([True, True, False], dtype=bool))
