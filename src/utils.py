import warnings
import numpy as np
from dataclasses import dataclass
from typing import Union

import warnings
import numpy as np
from typing import Union
import numpy.typing as npt

CoordsDType = Union[int, float, np.number]
ArrayLike = Union[npt.ArrayLike, npt.NDArray]

def __ensure_valid_dtype(array: npt.NDArray) -> npt.NDArray:
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Tipo di coordinata non valida: deve essere un numero")

    if not np.issubdtype(array.dtype, np.floating):
        warnings.warn(f"Casting implicito da {array.dtype} a np.float64")
        return array.astype(np.float64)

    return array
def validate_3d_coordinates(coordinates):
    try:
        new_coord = np.asarray(coordinates).ravel()
    except Exception as e:
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy") from e

    if new_coord.shape != (3,):
        raise ValueError("Le coordinate devono avere dimensione (3,)") # Messaggio corretto

    result = __ensure_valid_dtype(new_coord)
    update = new_coord.base is getattr(coordinates, 'base', None)
    return result if update or result.base is getattr(new_coord, 'base', None) else None
def validate_2d_coordinates(coordinates):
    try:
        new_coord = np.asarray(coordinates).ravel()
    except Exception as e:
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy") from e

    if new_coord.shape != (2, ):
        raise ValueError("Le coordinate devono avere dimensione (2, )")

    result = __ensure_valid_dtype(new_coord)
    update = new_coord.base is getattr(coordinates, 'base', None)
    return result if update or result.base is getattr(new_coord, 'base', None) else None
def validate_array_of_2d_coordinates(coordinates):
    try:
        new_coord = np.asarray(coordinates)
    except Exception as e:
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy") from e

    if new_coord.ndim != 2 or new_coord.shape[1] != 2:
        raise ValueError("Le coordinate devono avere dimensione (n, 2)")

    result = __ensure_valid_dtype(new_coord)
    update = new_coord.base is getattr(coordinates, 'base', None)
    return result if update or result.base is getattr(new_coord, 'base', None) else None
def validate_array_of_3d_coordinates(coordinates):
    try:
        new_coord = np.asarray(coordinates)
    except Exception as e:
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy") from e

    if new_coord.ndim != 2 or new_coord.shape[1] != 3:
        raise ValueError("Le coordinate devono avere dimensione (n, 3)")

    result = __ensure_valid_dtype(new_coord)
    update = new_coord.base is getattr(coordinates, 'base', None)
    return result if update or result.base is getattr(new_coord, 'base', None) else None

@dataclass(frozen=True)
class EpsConfig:
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
Eps = EpsConfig()