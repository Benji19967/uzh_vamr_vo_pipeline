import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist

from src.image import Image
from src.utils.utils import timing


class HarrisScores:
    CORNER_PATCH_SIZE = 9
    HARRIS_KAPPA = 0.08

    def __init__(self, image: Image) -> None:
        self._image = image
        self._scores: np.ndarray | None = None

    @property
    def img(self) -> np.ndarray:
        return self._image.img

    @property
    def scores(self) -> np.ndarray:
        if self._scores is None:
            self._scores = self.compute_scores()
        return self._scores

    @timing
    def compute_scores(
        self,
        patch_size: int = CORNER_PATCH_SIZE,
        kappa: float = HARRIS_KAPPA,
    ) -> np.ndarray:
        """
        Args:
            img (np.ndarray): input image
            patch_size (int): size of patch in which to look for corners
            kappa (int):

        Returns:
            np.ndarray: (img.shape) score for each pixel: how likely is the neighborhood a corner?
        """
        (
            Sum_Ixx,
            Sum_Iyy,
            Sum_Ixy,
            patch_radius,
        ) = self._compute_coefficients(img=self.img, patch_size=patch_size)

        trace = Sum_Ixx + Sum_Iyy
        determinant = Sum_Ixx * Sum_Iyy - Sum_Ixy**2

        scores = determinant - kappa * (trace**2)
        scores[scores < 0] = 0

        scores = self._pad_scores(scores=scores, patch_radius=patch_radius)

        return scores

    def plot(self) -> None:
        fig, axs = plt.subplots(1, 1, squeeze=False)
        axs[0, 0].imshow(self.img, cmap="gray")
        axs[0, 0].axis("off")
        axs[0, 0].imshow(self.scores)
        axs[0, 0].set_title("Harris Scores")
        axs[0, 0].axis("off")

        fig.tight_layout()
        plt.show()

    @classmethod
    def _compute_coefficients(
        cls, img: np.ndarray, patch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        Ix, Iy = cls._compute_derivatives(img=img)
        Ixx = Ix * Ix

        Iyy = Iy * Iy
        Ixy = Ix * Iy

        patch = np.ones([patch_size, patch_size])
        patch_radius = patch_size // 2
        Sum_Ixx = signal.convolve2d(Ixx, patch, mode="valid")
        Sum_Iyy = signal.convolve2d(Iyy, patch, mode="valid")
        Sum_Ixy = signal.convolve2d(Ixy, patch, mode="valid")

        return Sum_Ixx, Sum_Iyy, Sum_Ixy, patch_radius

    @classmethod
    def _compute_derivatives(cls, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Sobel_x = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
        Sobel_y = [
            [-1, -2, 1],
            [0, 0, 0],
            [1, 2, 1],
        ]

        Ix = signal.convolve2d(Sobel_x, img, mode="valid")
        Iy = signal.convolve2d(Sobel_y, img, mode="valid")

        return Ix, Iy

    @classmethod
    def _pad_scores(cls, scores: np.ndarray, patch_radius: int):
        return np.pad(
            scores,
            [
                (patch_radius + 1, patch_radius + 1),
                (patch_radius + 1, patch_radius + 1),
            ],
            mode="constant",
            constant_values=0,
        )


class Keypoints:

    NUM_KEYPOINTS = 200
    NONMAXIMUM_SUPRESSION_RADIUS = 8

    def __init__(self, image: Image, scores: np.ndarray | None = None) -> None:
        self._image = image
        self._scores = scores
        self._keypoints: np.ndarray | None = None

    @property
    def img(self) -> np.ndarray:
        return self._image.img

    @property
    def keypoints(self) -> np.ndarray:
        if self._keypoints is None:
            self._keypoints = self.select()
        return self._keypoints

    @property
    def scores(self) -> np.ndarray:
        if self._scores is None:
            self._scores = HarrisScores(image=self._image).scores
        return self._scores

    def select(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        nonmax_suppression_radius: int = NONMAXIMUM_SUPRESSION_RADIUS,
    ) -> np.ndarray:
        """
        Args:
            scores np.ndarray: corner measure for each pixel
            num_keypoints (int): num keypoints to select
            nonmax_suppression_radius (int): radius in which to only keep one keypoint

        Returns:
            np.ndarray: (2xN) and p=(y, x)
        """
        r = nonmax_suppression_radius

        # keep `num` keypoints, each has 2 coordinates (u, v)
        keypoints = np.zeros([2, num_keypoints])

        # scores with padding
        temp_scores = np.pad(
            self.scores, [(r, r), (r, r)], mode="constant", constant_values=0
        )

        for i in range(num_keypoints):
            # find max value in 2 array (i, j)
            kp = np.unravel_index(temp_scores.argmax(), temp_scores.shape)

            keypoints[:, i] = np.array(kp) - r  # why `- r`? Because of padding!

            # nonmaximum-suppression
            temp_scores[
                (kp[0] - r) : (kp[0] + r + 1), (kp[1] - r) : (kp[1] + r + 1)
            ] = 0

        self._keypoints = keypoints

        return keypoints

    def plot(self, keypoints: np.ndarray | None = None) -> None:
        plt.clf()
        plt.close()
        plt.imshow(self.img, cmap="gray")
        if keypoints is None:
            keypoints = self.keypoints
        plt.plot(keypoints[1, :], keypoints[0, :], "rx", linewidth=2)
        plt.axis("off")
        plt.show()


class Descriptors:

    DESCRIPTOR_RADIUS = 9
    MATCH_LAMBDA = 4

    def __init__(self, image: Image, keypoints: np.ndarray) -> None:
        self._image = image
        self._keypoints = keypoints
        self._descriptors: np.ndarray | None = None

    @property
    def img(self) -> np.ndarray:
        return self._image.img

    @property
    def descriptors(self) -> np.ndarray:
        if self._descriptors is None:
            self._descriptors = self.describe_keypoints()
        return self._descriptors

    def describe_keypoints(
        self, descriptor_radius: int = DESCRIPTOR_RADIUS
    ) -> np.ndarray:
        r = descriptor_radius
        N = self._keypoints.shape[1]

        # `(2 * r + 1) ** 2` is the number of pixels in a patch/descriptor
        descriptors = np.zeros([(2 * r + 1) ** 2, N])
        padded = np.pad(self.img, [(r, r), (r, r)], mode="constant", constant_values=0)

        for i in range(N):
            kp = self._keypoints[:, i].astype(int) + r  # `+r` to account for padding

            # store the the pixel intensities of the descriptors in a flattened way
            descriptors[:, i] = padded[
                (kp[0] - r) : (kp[0] + r + 1), (kp[1] - r) : (kp[1] + r + 1)
            ].flatten()

        return descriptors

    @classmethod
    def match(
        cls,
        query_descriptors: np.ndarray,
        db_descriptors: np.ndarray,
        match_lambda: int = MATCH_LAMBDA,
    ) -> np.ndarray:
        """
        For each query_descriptor find the closest db_descriptor.
        Use each db_descriptor only once.

        Args:
            query_descriptors (np.ndarray): descriptors at time t2
            db_descriptors (np.ndarray): descriptors at time t1
            match_lambda (int):

        Returns:
            np.ndarray: (1, len(query_descriptors))
        """
        # shape: (Q, D) -- in this case (200, 200)
        # distance from each query descriptor to each database descriptor
        dists = cdist(query_descriptors.T, db_descriptors.T, "euclidean")

        # shape: (200, 1)
        # for each query_descriptor, which db_descriptor (index) is closest (argmin)
        matches = np.argmin(dists, axis=1)

        # shape: (200, 1)
        # keep only distances that matched in `matches`
        dists = dists[np.arange(matches.shape[0]), matches]

        # scalar
        # min distance between any two descriptors across both sets
        min_non_zero_dist = dists.min()

        # keep only descriptors with small distance
        # adaptive threshold (because there should be at least one match)
        matches[dists >= match_lambda * min_non_zero_dist] = -1

        # remove double matches:
        # if a db_descriptor was assigned to several query_descriptors, keep only 1 match
        unique_matches = np.ones_like(matches) * -1
        _, unique_match_idxs = np.unique(matches, return_index=True)
        unique_matches[unique_match_idxs] = matches[unique_match_idxs]

        return unique_matches

    def plot(self) -> None:
        plt.clf()
        plt.close()
        fig, axs = plt.subplots(4, 4)
        patch_size = 2 * self.DESCRIPTOR_RADIUS + 1
        for i in range(16):
            axs[i // 4, i % 4].imshow(
                self.descriptors[:, i].reshape([patch_size, patch_size])
            )
            axs[i // 4, i % 4].axis("off")

        plt.show()

    @classmethod
    def plot_matches(
        cls,
        matches: np.ndarray,
        query_keypoints: np.ndarray,
        database_keypoints: np.ndarray,
    ):
        query_indices = np.nonzero(matches >= 0)[0]
        match_indices = matches[query_indices]

        x_from = query_keypoints[0, query_indices]
        x_to = database_keypoints[0, match_indices]
        y_from = query_keypoints[1, query_indices]
        y_to = database_keypoints[1, match_indices]

        for i in range(x_from.shape[0]):
            plt.plot([y_from[i], y_to[i]], [x_from[i], x_to[i]], "g-", linewidth=3)
        plt.show()
