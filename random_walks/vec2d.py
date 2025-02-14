import numpy as np
import numpy.typing as npt
import functools


def rotate(v: npt.NDArray, radians: float) -> npt.NDArray:
    """
    Rotate vector v by angle in radians.
    """
    return v.dot(get_rotation_matrix(radians))


@functools.lru_cache(maxsize=4)
def get_rotation_matrix(radians: float) -> npt.NDArray:
    """
    Get rotation matrix for an angle in radians.
    """
    cos_theta, sin_theta = np.cos(radians), np.sin(radians)

    return np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta],
    ])