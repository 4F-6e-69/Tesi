from dataclasses import dataclass

import numpy as np
import numpy.typing as nptyping
from typing import List, Literal, Union

CoordDType = Union[int, float, np.number]
ArrayLike = Union[nptyping.ArrayLike, nptyping.NDArray]
Resets = Union[List[Literal["all", "length", "area", "bounds", "barycenter"]], None]
Ref = Literal["origin", "barycenter"]

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