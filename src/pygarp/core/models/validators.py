import os.path
import warnings

from typing import Optional, Tuple, Any, cast
from pydantic import BaseModel, Field, model_validator

import numpy as np
from numpy import typing as npt

from pygarp.core.models.commons import ArrayLike, Eps, TransformationRef, ShapeType


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
        coordinates (array_like): I valori delle coordinate (es. Lista, tuple o array NumPy).
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
    if new_coord.ndim != 2 or new_coord.shape[1] != n:
        raise ValueError(f"L'array di coordinate deve avere dimensione (m, {n})")

    # Valida il tipo di dato e applica il cast a float64
    return __validate_numeric_dtype(new_coord)


class ShapeConfig(BaseModel):
    shape_type: ShapeType

    control_points_path: Optional[str] = None
    control_points: Any = None
    assume_sort: bool = False

    width: Optional[float] = Field(default=None, gt=Eps.eps04)
    height: Optional[float] = Field(default=None, gt=Eps.eps04)
    side_length: Optional[float] = Field(default=None, gt=Eps.eps04)
    n_sides: Optional[int] = Field(default=None, ge=3, le=12)
    radius: Optional[float] = Field(default=None, gt=Eps.eps04)

    center: Tuple[float, float] = Field(default=(0.0, 0.0))
    origin: Tuple[float, float] = Field(default=(0.0, 0.0))
    eps: Optional[float] = Field(default=None, gt=Eps.eps14)

    @model_validator(mode="after")
    def validate_shape(self):
        if self.shape_type in ("shape", "spline"):
            if self.control_points_path is None:
                raise ValueError(
                    f"La forma '{self.shape_type}' richiede il percorso del file (path_point)."
                )

            if not os.path.isfile(self.control_points_path):
                raise ValueError(
                    f"File o directory non trovato: {self.control_points_path}"
                )

            # Legge il file ignorando direttamente le righe vuote o contenenti solo spazi
            with open(os.path.abspath(self.control_points_path), "r") as f:
                file_txt_lines = [line.strip() for line in f if line.strip()]

            if len(file_txt_lines) < 3:
                raise ValueError(
                    "Troppi pochi punti nel file di controllo (minimo richiesto: 3)."
                )

            if self.shape_type == "spline" and len(file_txt_lines) > 5000:
                raise ValueError(
                    "Interpolazione troppo costosa (superato il limite di 5000 punti)."
                )

            points = []
            for line in file_txt_lines:
                try:
                    coords = [float(coord) for coord in line.split(",")]
                    points.append(coords)
                except ValueError:
                    raise ValueError(
                        "File non valido: non tutti i valori delle coordinate sono numerici."
                    )

            self.control_points = np.array(points, dtype=np.float64)

            # Pulizia sicura
            self.width = self.height = self.radius = self.side_length = self.n_sides = (
                None
            )

        # --- CIRCLE ---
        elif self.shape_type == "circle":
            if self.radius is None:
                raise ValueError(
                    "Un cerchio necessita di specificare il raggio (radius)."
                )
            # Pulizia sicura
            self.width = self.height = self.side_length = self.n_sides = None

        # --- RECTANGLE ---
        elif self.shape_type == "rectangle":
            if self.width is None or self.height is None:
                raise ValueError(
                    "Un rettangolo necessita di specificare sia la base (width) che l'altezza (height)."
                )
            # Pulizia sicura
            self.radius = self.side_length = self.n_sides = None

        # --- REGULAR POLYGON ---
        elif self.shape_type == "regular_polygon":
            if self.n_sides is None or self.side_length is None:
                raise ValueError(
                    "Un poligono regolare necessita del numero di lati (n) e della dimensione del lato (side)."
                )
            # Pulizia sicura
            self.radius = self.width = self.height = None

        return self


class SpaceConfig(BaseModel):
    space_strategy: Optional[str] = Field(default=None)

    # Vettori di base (mantengono i default standard spaziali)
    x_axis: Optional[Tuple[float, float, float]] = Field(default=(1.0, 0.0, 0.0))
    y_axis: Optional[Tuple[float, float, float]] = Field(default=(0.0, 1.0, 0.0))
    z_axis: Optional[Tuple[float, float, float]] = Field(default=(0.0, 0.0, 1.0))

    # Origine e punti di riferimento.
    # Default a None per permettere controlli condizionali affidabili.
    origin: Optional[Tuple[float, float, float]] = Field(default=None)
    x_hint: Optional[Tuple[float, float, float]] = Field(default=None)
    y_hint: Optional[Tuple[float, float, float]] = Field(default=None)
    p_hint: Optional[Tuple[float, float, float]] = Field(default=None)

    @model_validator(mode="after")
    def determine_strategy(self):
        # Se la strategia è stata passata esplicitamente, controlliamo che abbia i requisiti
        if self.space_strategy is not None:
            if self.space_strategy == "OPP":
                if self.origin is None or self.x_hint is None or self.p_hint is None:
                    raise ValueError(
                        "Dati mancanti: la strategia 'OPP' richiede 'origin', 'x_hint' e 'p_hint'."
                    )

            elif self.space_strategy == "ONC":
                if self.origin is None or self.x_hint is None or self.z_axis is None:
                    raise ValueError(
                        "Dati mancanti: la strategia 'ONC' richiede 'origin', 'x_hint' e 'z'."
                    )

            elif self.space_strategy == "XYP":
                if (
                    self.x_axis is None
                    or self.x_hint is None
                    or self.y_axis is None
                    or self.y_hint is None
                ):
                    raise ValueError(
                        "Dati mancanti: la strategia 'XYP' richiede 'x', 'x_hint', 'y' e 'y_hint'."
                    )

            else:
                if self.space_strategy != "DFT":
                    # Gestione di strategie inserite a mano non valide
                    warnings.warn(
                        f"Strategia '{self.space_strategy}' non riconosciuta. Fallback automatico su 'DFT'."
                    )
                    self.space_strategy = "DFT"

            return self

        # 1. Strategie basate sull'Origine (ONC e OPP)
        if self.origin is not None and self.x_hint is not None:
            # OPP ha la priorità se viene passato un punto specifico (p_hint)
            if self.p_hint is not None:
                self.space_strategy = "OPP"
                self.validate_unique_vectors(self.origin, self.p_hint, self.x_hint)

            # Fallback su ONC usando l'asse z (che ha un valore di default)
            elif self.z_axis is not None:
                self.space_strategy = "ONC"
                self.validate_unique_vectors(self.origin, self.z_axis, self.x_hint)
            else:
                raise ValueError(
                    "Dati incompleti: per definire l'origine è richiesto anche 'p_hint' (OPP) o 'z' (ONC)."
                )

        # 2. Strategia basata sugli assi e i relativi hint (XYP)
        elif all(
            v is not None for v in [self.x_axis, self.x_hint, self.y_axis, self.y_hint]
        ):
            self.space_strategy = "XYP"
            self.validate_unique_vectors(
                self.x_axis, self.y_axis, self.x_hint, self.y_hint
            )

        # 3. Nessun match possibile
        else:
            raise ValueError(
                "Impossibile dedurre la strategia spaziale: la combinazione dei vettori/punti forniti non è sufficiente."
            )

        return self

    @staticmethod
    def validate_unique_vectors(*vectors):
        """
        Validazione iniziale blanda: si assicura semplicemente che non ci siano
        vettori o punti di riferimento esattamente coincidenti.
        """
        # Filtriamo eventuali valori None passati per errore
        valid_vecs = [v for v in vectors if v is not None]

        # Un Set rimuove i duplicati. Se la lunghezza scende, ci sono vettori identici.
        if len(set(valid_vecs)) < len(valid_vecs):
            raise ValueError(
                "Conflitto geometrico: i vettori o i punti chiave usati per generare la strategia non possono coincidere tra loro."
            )


class TransformConfig(BaseModel):
    translation: Optional[Tuple[float, float]] = Field(default=None)
    rotation: Optional[float] = Field(default=None)
    scale: Optional[Tuple[float, float]] = Field(default=None)

    ref: Optional[TransformationRef] = Field(default=None)

    @model_validator(mode="after")
    def validate_transformation(self):
        # Evita scale nulle o negative (devono essere maggiori della tolleranza)
        if self.scale is not None:
            if self.scale[0] < Eps.eps04 or self.scale[1] < Eps.eps04:
                raise ValueError(
                    "I fattori di scala (x, y) devono essere positivi e superiori alla tolleranza."
                )

        # Normalizza la rotazione limitandola all'angolo giro (es. 370 -> 10, -10 -> 350)
        if self.rotation is not None:
            self.rotation = self.rotation % 360.0

        if (self.rotation is not None or self.scale is not None) and self.ref is None:
            warnings.warn(
                "Punto di riferimento (ref) non specificato per la trasformazione. "
                "Verrà applicato il fallback automatico su 'origin'."
            )
            self.ref = cast(TransformationRef, "origin")

        return self


class EditConfig(BaseModel):
    x: bool = False
    y: bool = False
    z: bool = False
    x2y: bool = False


class ScarfingConfig(BaseModel):
    apply_outline: bool = Field(default=True)
    outline_mode: Optional[str] = Field(default=None)
    apply_infill: bool = Field(default=True)
    infill_mode: Optional[str] = Field(default=None)
    infill_dir: Optional[float] = Field(default=0.0)
    infill_spacing: Optional[float] = Field(default=10.0, gt=Eps.eps04)

    apply_concentric: Optional[bool] = Field(default=False)
    concentric_spacing: Optional[float] = Field(default=None, gt=Eps.eps04)
    concentric_initial_offset: Optional[float] = Field(default=None, gt=-Eps.eps13)
    concentric_cycles: Optional[int] = Field(default=None, ge=1)

    apply_recursive: Optional[bool] = Field(default=False)
    pocket_offset: Optional[float] = Field(default=None, gt=Eps.eps04)
    pocket_cycles: Optional[int] = Field(default=None, gt=1.0)
    z_step: Optional[float] = Field(default=None, gt=Eps.eps04)

    @model_validator(mode="after")
    def validate_pocket(self):
        if not self.apply_outline and not self.apply_infill:
            raise ValueError(
                "Devi abilitare almeno una lavorazione tra 'outline' o 'fill'."
            )

        if self.apply_outline:
            if self.outline_mode not in ["step", "gradient"]:
                raise ValueError(
                    "Se outline è abilito devi specificare il tipo di contorno"
                )

        if self.apply_infill:
            if self.infill_mode is None:
                raise ValueError(
                    "Se 'fill' è abilitato, devi specificare un 'fill_style'."
                )

            if self.infill_mode not in ["grid", "rect"]:
                raise ValueError(
                    f"Stile di riempimento '{self.infill_mode}' non valido. Usa 'grid' o 'rect'."
                )

            if self.infill_dir is not None:
                self.infill_dir = self.infill_dir % 360.0
                if abs(self.infill_dir) < Eps.eps04:
                    self.infill_dir = 0.0

            if self.apply_concentric:
                if self.concentric_spacing is None or self.concentric_cycles is None:
                    raise ValueError(
                        "Se 'concentric' è True, 'c_offset' e 'c_cycle' sono obbligatori."
                    )
            else:
                self.concentric_cycles = None
                self.concentric_spacing = None

            if self.apply_recursive:
                if self.pocket_offset is None or self.pocket_cycles is None:
                    raise ValueError(
                        "Se 'recursive' è True, 'r_offset' e 'r_cycle' sono obbligatori."
                    )

                if self.z_step is None:
                    raise ValueError(
                        "Le lavorazioni ricorsive richiedono uno step in Z ('z_off')."
                    )

                if self.outline_mode not in ["gradient", "step"]:
                    raise ValueError(
                        "Il tipo di tasca ('pocket_type') deve essere 'gradient' o 'step'."
                    )
            else:
                self.pocket_offset = None
                self.pocket_cycles = None
                self.z_step = None
                self.outline_mode = "None"

        else:
            self.infill_mode = None
            self.infill_dir = None
            self.apply_concentric = False
            self.concentric_spacing = None
            self.concentric_cycles = None
            self.apply_recursive = False
            self.pocket_offset = None
            self.pocket_cycles = None
            self.z_step = None
            self.outline_mode = "None"

        return self


class RobotConfig(BaseModel):
    gamma: float = 5.0
    exit_quote: float = 50.0
