import warnings
import numpy as np
from dataclasses import dataclass
from typing import Union

CoordsDType = Union[int, float, np.number]
ArrayLike = Union[np.typing.ArrayLike, np.typing.NDArray]

def validate_3d_coordinates(coordinates):
    if not isinstance(coordinates, np.typing.ArrayLike):
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy")

    new_coord = np.asarray(coordinates).flatten()
    update = new_coord.base is coordinates.base
    if new_coord.shape != (3,):
        raise ValueError("Le coordinate devono avere dimensione (2,)")

    result = __ensure_valid_dtype(new_coord)
    return result if update or result.base is new_coord.base else None
def validate_2d_coordinates(coordinates):
    if not isinstance(coordinates, np.typing.ArrayLike):
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy")

    new_coord = np.asarray(coordinates).flatten()
    update = new_coord.base is coordinates.base
    if new_coord.shape != (2, ):
        raise ValueError("Le coordinate devono avere dimensione (2,)")

    result = __ensure_valid_dtype(new_coord)
    return result if update or result.base is new_coord.base else None
def validate_array_of_2d_coordinates(coordinates):
    if not isinstance(coordinates, np.typing.ArrayLike):
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy")

    new_coord = np.asarray(coordinates)
    update = new_coord.base is coordinates.base
    if new_coord.shape[1] != 2:
        raise ValueError("Le coordinate devono avere dimensione (n, 2)")

    result = __ensure_valid_dtype(new_coord)
    return result if update or result.base is new_coord.base else None
def validate_array_of_3d_coordinates(coordinates):
    if not isinstance(coordinates, np.typing.ArrayLike):
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy")

    new_coord = np.asarray(coordinates)
    update = new_coord.base is coordinates.base
    if new_coord.shape[1] != 3:
        raise ValueError("Le coordinate devono avere dimensione (n, 3)")

    result = __ensure_valid_dtype(new_coord)
    return result if update or result.base is new_coord.base else None
def __ensure_valid_dtype(array: np.typing.NDArray) -> np.typing.NDArray[CoordsDType] | None:
    if array.dtype != CoordsDType:
        raise TypeError("Tipo di coordinata non valida")

    if array.dtype != np.float64 or array.dtype != float:
        warnings.warn(f"Casting implicito da {array.dtype} a np.float64")
        return np.asarray(array, dtype=np.float64)

    return None

@dataclass
class Eps:
    __eps04: float = 1e-04
    __eps05: float = 1e-05
    __eps06: float = 1e-06
    __eps08: float = 1e-08
    __eps10: float = 1e-10
    __eps11: float = 1e-11
    __eps12: float = 1e-12
    __eps13: float = 1e-13
    __eps14: float = 1e-14
    __eps15: float = 1e-15
    __eps16: float = np.finfo(np.float64).eps

    @property
    def eps04(self):
        return self.__eps04
    @property
    def eps05(self):
        return self.__eps05
    @property
    def eps06(self):
        return self.__eps06
    @property
    def eps08(self):
        return self.__eps08
    @property
    def eps10(self):
        return self.__eps10
    @property
    def eps11(self):
        return self.__eps11
    @property
    def eps12(self):
        return self.__eps12
    @property
    def eps13(self):
        return self.__eps13
    @property
    def eps14(self):
        return self.__eps14
    @property
    def eps15(self):
        return self.__eps15
    @property
    def eps16(self):
        return self.__eps16