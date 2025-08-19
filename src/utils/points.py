import numpy as np


def filter(points: np.ndarray, mask: np.ndarray):
    """
    Apply mask to points

    Args:
        - points  np.ndarray(Any, N)
        - mask    np.ndarray(N,)
    """

    if points.any():
        return points[:, mask]
    return points


# def filter_many(points: list[np.ndarray], mask: np.ndarray):
#     """
#     Apply mask to list of points
#
#     Args:
#         - points  list[np.ndarray(Any, N)]
#         - mask    np.ndarray(N,)
#     """
#     if isinstance(points, list):
#         masked = []
#         for pts in points:
#             if pts.any():
#                 masked.append(pts[:, mask])
#             else:
#                 masked.append(pts)
#         return masked
