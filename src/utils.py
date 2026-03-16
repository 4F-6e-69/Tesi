from dataclasses import dataclass

import warnings

import math
import numpy as np
import numpy.typing as npt
from typing import List, Literal, Union, Tuple

CoordDType = Union[int, float, np.number]
ArrayLike = Union[npt.ArrayLike, npt.NDArray]
Resets = Union[List[Literal["all", "length", "area", "bounds", "barycenter", "step", "closure"]], Literal["all", "length", "area", "bounds", "barycenter", "step", "closure"], None]
Ref = Literal["origin", "barycenter"]
DiscretizationMethod = Union[Literal["adaptive", "uniform"], None]

# Validator
def __validate_numeric_dtype(array: npt.NDArray) -> npt.NDArray[np.float64]:
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Tipo di coordinata non valida: deve essere un numero")

    if not np.issubdtype(array.dtype, np.floating):
        warnings.warn(f"Casting implicito da {array.dtype} a np.float64")
        return array.astype(np.float64)

    return array

def validate_2d_coordinates(coordinates):
    try:
        new_coord = np.asarray(coordinates).ravel()
    except Exception as e:
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy") from e

    if new_coord.shape != (2, ):
        raise ValueError("Le coordinate devono avere dimensione (2, )")

    result = __validate_numeric_dtype(new_coord)
    update = new_coord.base is getattr(coordinates, 'base', None)
    return result if update or result.base is getattr(new_coord, 'base', None) else None
def validate_array_of_2d_coordinates(coordinates):
    try:
        new_coord = np.asarray(coordinates)
    except Exception as e:
        raise TypeError("Le coordinate devono essere di tipo array-numpy o array-like-numpy") from e

    if new_coord.ndim != 2 or new_coord.shape[1] != 2:
        raise ValueError("Le coordinate devono avere dimensione (n, 2)")

    result = __validate_numeric_dtype(new_coord)
    update = new_coord.base is getattr(coordinates, 'base', None)
    return result if update or result.base is getattr(new_coord, 'base', None) else None

def _tolerated_mcd(numbers: npt.NDArray[np.float64], error: float) -> Tuple[float, npt.NDArray[np.float64]]:
    max_divisors = np.min(numbers)
    dividers = np.arange(2, max_divisors + 1).reshape(-1, 1)

    remains = numbers % dividers
    dists = np.minimum(remains, dividers - remains)
    errors = np.max(dists, axis=1)
    validation_mask = errors < error

    sure_divisors = dividers[validation_mask].flatten()
    if sure_divisors.size == 0:
        return 1.0, np.array([0.0, 1.0], dtype=np.float64)

    sorted_divisors = np.sort(sure_divisors)
    return float(sorted_divisors[-1]), sorted_divisors
def divisors(n: float) -> npt.NDArray[np.float64]:

    if np.isclose(n, 0, atol=Eps.eps12):
        return np.array([])
    if np.isclose(n, 1, atol=Eps.eps12):
        return np.array([1])

    i = np.arange(1, int(n ** 0.5) + 1)
    divs = i[n % i == 0]
    all_divs = np.unique(np.concatenate((divs, n // divs)))
    return all_divs
def _all_almost_divisors(number: float, error: float) -> npt.NDArray[np.float64]:
    _, divs = _tolerated_mcd(np.array([number], dtype=np.float64), error)
    return divs
def mult_divisors(ns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if ns.size == 0:
        return np.array([], dtype=np.float64)

    ns_int = np.asarray(ns, dtype=np.int64)
    overall_gcd = np.gcd.reduce(ns_int)

    if overall_gcd == 0:
        return np.array([], dtype=np.float64)

    overall_gcd = int(abs(overall_gcd))
    divs = set()
    limit = math.isqrt(overall_gcd)

    for i in range(1, limit + 1):
        if overall_gcd % i == 0:
            divs.add(i)
            divs.add(overall_gcd // i)

    return np.asarray(sorted(list(divs)), dtype=np.float64)

def filter_arrays_tolerance(numbers: npt.NDArray[np.float64], digit: int, sort: bool = False) -> npt.NDArray[np.float64]:
    if sort:
        max_val = numbers[-1]
    else:
        max_val = np.max(numbers)

    if 0.0 < max_val < 10.0 ** digit:
        warnings.warn("Cifre decimali per la tolleranza non valide. Ricalcolo l'ordine di grandezza.")
        digit = int(np.floor(np.log10(max_val)) - 2)

    rounded_array = np.round(numbers, -digit)
    return np.unique(rounded_array, axis=0)

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
    def eps04(self) -> float:
        return self.__eps04
    @property
    def eps05(self) -> float:
        return self.__eps05
    @property
    def eps06(self) -> float:
        return self.__eps06
    @property
    def eps08(self) -> float:
        return self.__eps08
    @property
    def eps10(self) -> float:
        return self.__eps10
    @property
    def eps11(self) -> float:
        return self.__eps11
    @property
    def eps12(self) -> float:
        return self.__eps12
    @property
    def eps13(self) -> float:
        return self.__eps13
    @property
    def eps14(self) -> float:
        return self.__eps14
    @property
    def eps15(self) -> float:
        return self.__eps15
    @property
    def eps16(self) -> float:
        return self.__eps16
Eps = EpsConfig()