import numpy as np
from src.utils import points


def test_filter():
    p_W_hom = np.array(
        [
            [1, 9, 2],
            [3, 0, 3],
            [7, 1, 0],
            [1, 1, 1],
        ]
    )
    mask = np.array([True, True, False])
    assert np.array_equal(points.filter(p_W_hom, mask), p_W_hom[:, :2])
