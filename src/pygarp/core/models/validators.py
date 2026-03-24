import os.path
import warnings

from typing import Optional, Tuple, Self, Any
from pydantic import BaseModel, Field, model_validator

import numpy as np
from numpy import typing as npt

from pygarp.core.models.commons import ArrayLike, FillType, Eps


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


class ShapeConfig(BaseModel):
    shape: str
    path_point: Optional[str] = None
    control_points: Any = None

    width: Optional[float] = Field(default=None, gt=Eps.eps04)
    height: Optional[float] = Field(default=None, gt=Eps.eps04)

    side: Optional[float] = Field(default=None, gt=Eps.eps04)
    n: Optional[int] = Field(default=None, ge=3, le=12)

    radius: Optional[float] = Field(default=None, gt=Eps.eps04)

    center: Optional[Tuple[float, float]] = (0.0, 0.0)

    @model_validator(mode="after")
    def validate_shape(self):
        if self.shape in ("shape", "spline"):

            if self.path_point is None:
                raise ValueError(
                    f"La forma '{self.shape}' richiede il percorso del file (path_point)."
                )

            if not os.path.isfile(self.path_point):
                raise ValueError(f"Directory o file non valido: {self.path_point}")

            with open(os.path.abspath(self.path_point), "r") as f:
                file_txt_lines = f.readlines()

            if len(file_txt_lines) < 3:
                raise ValueError("Troppi pochi punti nel file di controllo.")

            if self.shape == "spline" and len(file_txt_lines) > 5000:
                raise ValueError("Interpolazione troppo costosa (più di 5000 punti).")

            points = []
            for line in file_txt_lines:
                if not line.strip():
                    continue

                coords = line.split(",")
                temp_coord = []

                for coord in coords:
                    try:
                        temp_coord.append(float(coord))
                    except ValueError:
                        raise ValueError(
                            "File contenente punti di controllo non valido: non tutti i valori sono numerici."
                        )

                points.append(np.asarray(temp_coord, dtype=np.float64))

            self.control_points = points

            self.width = None
            self.height = None
            self.radius = None
            self.side = None
            self.n = None
            self.center = None

        elif self.shape == "circle":
            if self.radius is None:
                raise ValueError(
                    "Un cerchio necessita di specificare il raggio (radius)."
                )

            self.radius = None

        elif self.shape == "rectangle":
            if self.width is None:
                raise ValueError(
                    "Un rettangolo necessita di specificare la base (width)."
                )
            if self.height is None:
                raise ValueError(
                    "Un rettangolo necessita di specificare l'altezza (height)."
                )

            self.radius = None
            self.side = None
            self.n = None

        elif self.shape == "regular-polygon":
            if self.n is None:
                raise ValueError(
                    "Un poligono regolare necessita del numero di lati (n)."
                )
            if self.side is None:
                raise ValueError(
                    "Un poligono regolare necessita della dimensione del lato (side)."
                )

            self.side = None
            self.n = None

        else:
            raise ValueError(f"Forma '{self.shape}' non supportata.")

        if self.shape in ("circle", "rectangle", "regular-polygon"):
            if self.center is None:
                raise ValueError(
                    f"La forma '{self.shape}' necessita di un centro specificato."
                )

        return


class SpaceConfig(BaseModel):
    space_type: str

    origin: Optional[Tuple[float, float, float]] = Field(default=None)
    x: Tuple[float, float, float] = Field(default=(1.0, 0.0, 0.0))
    y: Tuple[float, float, float] = Field(default=(0.0, 1.0, 0.0))
    z: Tuple[float, float, float] = Field(default=(0.0, 0.0, 1.0))

    @model_validator(mode="after")
    def validate_origin(self):
        if (
            self.x == self.origin
            or self.y == self.origin
            or self.z == self.origin
            or self.x == self.y
            or self.x == self.z
            or self.y == self.z
        ):
            raise ValueError(
                "I vettori degli assi x, y e z non possono essere identici tra loro."
            )
        return self


class ScarfingConfig(BaseModel):
    pocket_type: str

    outline: bool = Field(default=True)
    outline_style: Optional[str] = Field(default=None)
    fill: bool = Field(default=True)
    fill_style: Optional[str] = Field(default=None)
    fill_dir: Optional[float] = Field(default=0.0)
    fill_spacing: Optional[float] = Field(default=10.0, gt=Eps.eps04)

    concentric: Optional[bool] = Field(default=False)
    c_offset: Optional[float] = Field(default=None, gt=Eps.eps04)
    c_offset_0: Optional[float] = Field(default=None, gt=Eps.eps04)
    c_cycle: Optional[int] = Field(default=None, ge=1)

    recursive: Optional[bool] = Field(default=False)
    r_offset: Optional[float] = Field(default=None, gt=Eps.eps04)
    r_cycle: Optional[int] = Field(default=None, gt=1.0)
    z_off: Optional[float] = Field(default=None, gt=Eps.eps04)

    @model_validator(mode="after")
    def validate_pocket(self):
        if not self.outline and not self.fill:
            raise ValueError(
                "Devi abilitare almeno una lavorazione tra 'outline' o 'fill'."
            )

        if self.outline:
            if self.outline_style not in ["step", "gradient"]:
                raise ValueError(
                    "Se outline è abilito devi specificare il tipo di contorno"
                )

        if self.fill:
            if self.fill_style is None:
                raise ValueError(
                    "Se 'fill' è abilitato, devi specificare un 'fill_style'."
                )

            if self.fill_style not in ["grid", "rect"]:
                raise ValueError(
                    f"Stile di riempimento '{self.fill_style}' non valido. Usa 'grid' o 'rect'."
                )

            if self.fill_dir is not None:
                self.fill_dir = self.fill_dir % 360.0
                if abs(self.fill_dir) < Eps.eps04:
                    self.fill_dir = 0.0

            if self.concentric:
                if self.c_offset is None or self.c_cycle is None:
                    raise ValueError(
                        "Se 'concentric' è True, 'c_offset' e 'c_cycle' sono obbligatori."
                    )
            else:
                self.c_cycle = None
                self.c_offset = None

            if self.recursive:
                if self.r_offset is None or self.r_cycle is None:
                    raise ValueError(
                        "Se 'recursive' è True, 'r_offset' e 'r_cycle' sono obbligatori."
                    )

                if self.z_off is None:
                    raise ValueError(
                        "Le lavorazioni ricorsive richiedono uno step in Z ('z_off')."
                    )

                if self.pocket_type not in ["gradient", "step"]:
                    raise ValueError(
                        "Il tipo di tasca ('pocket_type') deve essere 'gradient' o 'step'."
                    )
            else:
                self.r_offset = None
                self.r_cycle = None
                self.z_off = None
                self.pocket_type = "None"

        else:
            self.fill_style = None
            self.fill_dir = None
            self.concentric = False
            self.c_offset = None
            self.c_cycle = None
            self.recursive = False
            self.r_offset = None
            self.r_cycle = None
            self.z_off = None
            self.pocket_type = "None"

        return self


class RobotConfig(BaseModel):
    gamma: float = Field(default=5.0)

    exit_quote: float = Field(default=50.0)
