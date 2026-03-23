import warnings
from typing import Optional, Tuple, Self
from pydantic import BaseModel, Field, model_validator

import numpy as np
from numpy import typing as npt

from src.pygarp.core.models.commons import ArrayLike, FillType


def __validate_numeric_dtype(array: npt.NDArray) -> npt.NDArray[np.float64]:
    """
    Valida il tipo di dato di un array e lo converte in float64 se necessario.

    Args:
        array (npt.NDArray): L'array NumPy da validare.

    Returns:
        npt.NDArray[np.float64]: L'array garantito come tipo float64.

    Raises:
        TypeError: Se l'array non è di tipo numerico.

    Warns:
        UserWarning: Se viene effettuato un casting implicito da un tipo non floating point.
    """

    # Garantisce che l'input sia strettamente numerico
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Tipo di coordinata non valido: deve essere un numero")

    # Forza il cast a float64 (avvisando se il tipo originale non era float)
    if array.dtype != np.float64:
        if not np.issubdtype(array.dtype, np.floating):
            warnings.warn(f"Casting implicito da {array.dtype} a np.float64")
        return array.astype(np.float64)

    return array


def validate_nd_coordinates(coordinates: ArrayLike, n: int) -> npt.NDArray[np.float64]:
    """
    Valida e formatta una singola coordinata in uno spazio n-dimensionale.

    Args:
        coordinates (array_like): I valori delle coordinate (es. Lista, tupla o array NumPy).
        n (int): La dimensionalità attesa per la coordinata (es. 2 per 2D, 3 per 3D).

    Returns:
        npt.NDArray[np.float64]: Un array NumPy 1D contenente la coordinata validata.

    Raises:
        TypeError: Se l'input non è convertibile in un array NumPy.
        ValueError: Se `n` non è un intero strettamente positivo.
        ValueError: Se la dimensione dell'array risultante non corrisponde a `(n, )`. (n righe o colonne)
    """

    # Converte l'input in un array NumPy 1D (flattening)
    try:
        new_coord = np.asarray(coordinates).ravel()
    except Exception as e:
        raise TypeError(
            "Le coordinate devono essere di tipo array-numpy o array-like"
        ) from e

    # Valida il parametro della dimensionalità attesa
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Numero di coordinate non valido: deve essere un intero > 0")

    # Verifica che il numero di coordinate corrisponda alla dimensione 'n' richiesta
    if new_coord.shape != (n,):
        raise ValueError(f"Le coordinate devono avere dimensione ({n}, )")

    # Valida il tipo di dato e applica il cast a float64
    return __validate_numeric_dtype(new_coord)


def validate_array_of_nd_coordinates(coordinates, n: int):
    """
    Valida e formatta un array di coordinate in uno spazio n-dimensionale.

    Si aspetta un input che rappresenti una matrice 2D dove ogni riga è un punto e
    ogni colonna è una dimensione.

    Args:
        coordinates (array_like): L'insieme di coordinate da validare.
        n (int): La dimensionalità di ogni singola coordinata.

    Returns:
        npt.NDArray[np.float64]: Un array NumPy 2D di shape (m, n) con le coordinate validate.

    Raises:
        TypeError: Se l'input non è convertibile in un array NumPy.
        ValueError: Se `n` non è un intero strettamente positivo.
        ValueError: Se il numero di colonne non corrisponde a `n`.
    """

    # Converte l'input in un array NumPy 2D (m x r) (dove r potrebbe essere diverso dalle colonne attese n)
    try:
        new_coord = np.asarray(coordinates)
    except Exception as e:
        raise TypeError(
            "Le coordinate devono essere di tipo array-numpy o array-like"
        ) from e

    # Valida il parametro della dimensionalità attesa
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Numero di coordinate non valido: deve essere un intero > 0")

    # Verifica che il numero di coordinate corrisponda alla dimensione 'n' richiesta
    if new_coord.ndim != n or new_coord.shape[1] != n:
        raise ValueError(f"L'array di coordinate deve avere dimensione (m, {n})")

    # Valida il tipo di dato e applica il cast a float64
    return __validate_numeric_dtype(new_coord)


class JobConfig(BaseModel):
    # Geometria
    shape: Optional[str] = None
    radius: Optional[float] = Field(None, gt=0)
    width: Optional[float] = Field(None, gt=0)
    height: Optional[float] = Field(None, gt=0)
    side: Optional[float] = Field(None, gt=0)
    path_in: Optional[str] = None

    # Orientamento
    origin: Tuple[float, float, float] = (0, 0, 0)
    x_axis: Tuple[float, float, float] = (1, 0, 0)
    y_axis: Tuple[float, float, float] = (0, 1, 0)
    z_axis: Tuple[float, float, float] = (0, 0, 1)

    # Pocketing
    outline: bool = True
    fill: Optional[FillType] = None
    fill_direction: float = 0.0

    # Scarfing Concentrico
    concentric: bool = False
    c_offset: Optional[float] = Field(None, gt=0)
    c_cycles: int = Field(1, ge=1)

    # Scarfing Ricorsivo
    recursive: bool = False
    r_offset: float = 0.5
    r_cycles: int = Field(1, ge=1)

    # Input / Output
    path_out: Optional[str] = None
    job_path: Optional[str] = None
    job_out: Optional[str] = None

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        if self.shape == "circle" and self.radius is None:
            raise ValueError(
                "Il raggio (--radius) è obbligatorio per la forma 'circle'"
            )

        if self.shape == "rectangle" and (self.width is None or self.height is None):
            raise ValueError(
                "Larghezza (--width) e Altezza (--height) sono obbligatorie per 'rectangle'"
            )

        if self.shape == "regular-polygon" and self.side is None:
            raise ValueError("Il lato (--side) è obbligatorio per 'regular-polygon'")

        if self.shape in ["shape", "spline"] and self.path_in is None:
            nome_forma = "spline" if self.shape == "spline" else "forma generica"
            raise ValueError(
                f"Per {nome_forma} va specificato il percorso del file (--path-in)"
            )

        return self
