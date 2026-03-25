import numpy as np
from numpy import typing as npt

from pygarp.core.models.commons import Eps


def pocket_writing(path: npt.NDArray[np.float64], filename, mode: str = "w"):
    closed_path = (
        path
        if np.allclose(path[0], path[1], atol=Eps.eps08)
        else np.vstack((path, path[0, :]))
    )

    P = closed_path[:, :3]
    V = closed_path[:, 3:6]

    new_vectors = P - V
    final_data = np.column_stack((P, new_vectors))

    with open(filename, "w") as f:
        np.savetxt(
            f,
            final_data,
            fmt=["%.3f", " %.3f", " %.3f", " %.3f", " %.3f", " %.3f"],
            delimiter=",",
        )

    print("ok")
