import numpy as np
import numpy.typing as npt


def rotate(vector: npt.NDArray, radians: float) -> npt.NDArray:
    """
    Rotate vector by angle in radians.
    """
    return vector @ get_rotation_matrix(radians)


def get_rotation_matrix(radians: float) -> npt.NDArray:
    """
    Get rotation matrix for an angle in radians.
    """
    cos_theta, sin_theta = np.cos(radians), np.sin(radians)

    return np.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ]
    )
