# import numpy as np


# def match_keypoints(p_P_keypoints_0: np.ndarray, p_P_keypoints_1: np.ndarray, matches):
#     """
#     For each query_descriptor find the closest db_descriptor.
#     Use each db_descriptor only once.

#     Args:
#         p_P_keypoints_0 np.ndarray(R, N_0): keypoints 0
#         p_P_keypoints_1 np.ndarray(R, N_1): keypoints 1
#         matches np.ndarray(N_1):

#     Returns:
#         np.ndarray: (N_Q,)
#     """
#     I_0_keypoints = keypoints[0]
#     I_1_keypoints = keypoints[1]
#     I_1_indices = np.nonzero(matches >= 0)[0]
#     I_0_indices = matches[I_1_indices]

#     I_0_matched_keypoints = np.zeros((2, len(I_1_indices)))
#     I_1_matched_keypoints = np.zeros((2, len(I_1_indices)))

#     I_0_matched_keypoints[0:] = I_0_keypoints[0, I_0_indices]
#     I_0_matched_keypoints[1:] = I_0_keypoints[1, I_0_indices]
#     I_1_matched_keypoints[0:] = I_1_keypoints[0, I_1_indices]
#     I_1_matched_keypoints[1:] = I_1_keypoints[1, I_1_indices]

#     # kps[0].plot(I_0_matched_keypoints)
#     # kps[1].plot(I_1_matched_keypoints)

#     # Switch pixel coordinates from (y, x) to (x, y)
#     I_0_matched_keypoints[[0, 1], :] = I_0_matched_keypoints[[1, 0], :]
#     I_1_matched_keypoints[[0, 1], :] = I_1_matched_keypoints[[1, 0], :]

#     return I_0_matched_keypoints, I_1_matched_keypoints
