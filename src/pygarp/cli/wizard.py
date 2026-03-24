import questionary

import os

from typing import Tuple
from typing_extensions import get_args

from pygarp.core.models.commons import ShapeType, FillType


def is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False


def is_positive_float(val: str) -> str | bool:
    try:
        val = float(val)
        if val > 0:
            return True
        return "Errore: Inserisci un numero maggiore di 0!"
    except ValueError:
        return "Errore: Inserisci un numero valido (es. 10.5)!"


def is_positive_integer(val: str) -> bool | str:
    try:
        if int(val) >= 1:
            return True
        return "Errore: Il numero di cicli deve essere almeno 1!"
    except ValueError:
        return "Errore: Inserisci un numero intero valido (senza virgola)!"


def is_tuple_3d(val: str) -> bool | str:
    try:
        part = val.split(",")
        if len(part) != 3:
            return (
                "Devi inserire esattamente 3 valori separati da virgola (es. 1.0, 0, 0)"
            )
        for p in part:
            float(p.strip())
        return True
    except ValueError:
        return "I valori devono essere numeri validi (es. 1.5, -2, 0.0)"


def validate_side(val_str: str) -> bool | str:
    try:
        val = int(val_str)
        if 3 <= val <= 12:
            return True
        return "Errore: Il numero di lati deve essere compreso tra 3 e 12!"
    except ValueError:
        return "Errore: Inserisci un numero intero valido (senza virgola)!"


def validate_existing_path(path: str) -> bool | str:
    if os.path.isfile(path):
        return True
    return "Errore: Il file specificato non esiste o il percorso è errato!"


def validate_new_file_json(path: str) -> bool | str:
    if os.path.isfile(path):
        return "Errore: Il nome specificato è già in uso!"
    if path.strip().split(".")[-1] == "json":
        return True
    return "Errore: il file di salvataggio deve avere estensione .json"


def ask_tuple(message: str, default_value: str) -> Tuple[float, float, float]:
    risposta = questionary.text(
        message,
        default=default_value,
        validate=is_tuple_3d,
    ).ask()

    if risposta is None:
        raise KeyboardInterrupt

    parti = risposta.split(",")
    return float(parti[0].strip()), float(parti[1].strip()), float(parti[2].strip())


def shape_query() -> dict:
    dictionary: dict = {}

    print("\n--- GEOMETRIA ---")
    shape = questionary.select(
        "Tipo di forma:", choices=list(get_args(ShapeType))
    ).ask()

    if shape is None:
        raise KeyboardInterrupt

    dictionary["shape"] = shape

    if shape == "circle":
        radius_str = questionary.text("Raggio:", validate=is_positive_float).ask()
        if radius_str is None:
            raise KeyboardInterrupt
        dictionary["radius"] = float(radius_str)

        dictionary["center"] = ask_tuple("Centro:", default_value="0.0, 0.0, 0.0")

    elif shape == "rectangle":
        width_str = questionary.text("Base:", validate=is_positive_float).ask()
        if width_str is None:
            raise KeyboardInterrupt

        height_str = questionary.text("Altezza:", validate=is_positive_float).ask()
        if height_str is None:
            raise KeyboardInterrupt

        dictionary["width"] = float(width_str)
        dictionary["height"] = float(height_str)

    elif shape == "regular-polygon":
        sides_str = questionary.text(
            "Numero di lati (3-12):", default="6", validate=validate_side
        ).ask()
        if sides_str is None:
            raise KeyboardInterrupt

        side_str = questionary.text("Lato:", validate=is_positive_float).ask()
        if side_str is None:
            raise KeyboardInterrupt

        dictionary["num_of_sides"] = int(sides_str)
        dictionary["side"] = float(side_str)

    elif shape in ["shape", "spline"]:
        path_in = questionary.path(
            "Percorso del file da cui importare i punti di controllo (usa TAB per auto-completamento):",
            validate=validate_existing_path,
        ).ask()

        if path_in is None:
            raise KeyboardInterrupt

        dictionary["path_in"] = path_in

    return dictionary


def orientation_query():
    dictionary: dict = {}

    print("\n--- ORIENTAMENTO ---")
    orientation = questionary.select(
        "Piano di lavoro:", choices=["default", "offset", "custom"]
    ).ask()
    if orientation is None:
        raise KeyboardInterrupt

    if orientation in ["default", "offset"]:
        dictionary["x_axis"] = (1.0, 0.0, 0.0)
        dictionary["y_axis"] = (0.0, 1.0, 0.0)
        dictionary["z_axis"] = (0.0, 0.0, 1.0)

        x0, y0, z0 = 0.0, 0.0, 0.0

        if orientation == "offset":
            valida_numero = (
                lambda text: is_float(text)
                or "Errore: Inserisci un numero valido (es. 1.5, -2)!"
            )

            x0 = float(
                questionary.text(
                    "X offset:", default="0.0", validate=valida_numero
                ).ask()
            )
            y0 = float(
                questionary.text(
                    "Y offset:", default="0.0", validate=valida_numero
                ).ask()
            )
            z0 = float(
                questionary.text(
                    "Z offset:", default="0.0", validate=valida_numero
                ).ask()
            )

        dictionary["origin"] = (x0, y0, z0)

    else:
        print("\n--- Configurazione Custom Assi e Origine ---")

        dictionary["origin"] = ask_tuple(
            "Origine (x, y, z):", default_value="0.0, 0.0, 0.0"
        )
        dictionary["x_axis"] = ask_tuple(
            "Vettore Asse X (x, y, z):", default_value="1.0, 0.0, 0.0"
        )
        dictionary["y_axis"] = ask_tuple(
            "Vettore Asse Y (x, y, z):", default_value="0.0, 1.0, 0.0"
        )
        dictionary["z_axis"] = ask_tuple(
            "Vettore Asse Z (x, y, z):", default_value="0.0, 0.0, 1.0"
        )

    return dictionary


def pocket_query() -> dict:
    dictionary: dict = {}
    print("\n--- STRATEGIA DI PATHING E SCAVO ---")

    outline = questionary.confirm(
        "Abilitare il percorso del contorno esterno (Outline)?",
        default=True,
        auto_enter=False,
    ).ask()
    if outline is None:
        raise KeyboardInterrupt
    dictionary["outline"] = outline

    fill_choices = ["Nessuno"] + list(get_args(FillType))
    fill_select = questionary.select(
        "Strategia di riempimento della tasca:", choices=fill_choices
    ).ask()
    if fill_select is None:
        raise KeyboardInterrupt

    if fill_select == "Nessuno":
        dictionary["fill"] = None
    else:
        dictionary["fill"] = fill_select
        dir_str = questionary.text(
            "Direzione delle linee di riempimento in gradi (°):",
            default="0.0",
            validate=is_float,
        ).ask()
        if dir_str is None:
            raise KeyboardInterrupt
        dictionary["fill_direction"] = float(dir_str)

        print("\n--- SCARFING ---")
        concentric = questionary.confirm(
            "Eseguire passate concentriche prima del riempimento?",
            default=False,
            auto_enter=False,
        ).ask()

        if concentric is None:
            raise KeyboardInterrupt
        dictionary["concentric"] = concentric

        if concentric:
            c_offset_str = questionary.text(
                "Distanza tra le passate concentriche (mm):", validate=is_positive_float
            ).ask()
            if c_offset_str is None:
                raise KeyboardInterrupt
            dictionary["c_offset"] = float(c_offset_str)

            c_cycles_str = questionary.text(
                "Numero di ripetizioni del ciclo concentrico:",
                default="1",
                validate=is_positive_integer,
            ).ask()
            if c_cycles_str is None:
                raise KeyboardInterrupt
            dictionary["c_cycles"] = int(c_cycles_str)

        # 3. SCARFING RICORSIVO
        recursive = questionary.confirm(
            "Reiterare il percorso della tasca in modo ricorsivo?",
            default=False,
            auto_enter=False,
        ).ask()
        if recursive is None:
            raise KeyboardInterrupt
        dictionary["recursive"] = recursive

        if recursive:
            r_offset_str = questionary.text(
                "Offset tra ogni passata ricorsiva (mm):",
                default="0.5",
                validate=is_positive_float,
            ).ask()
            if r_offset_str is None:
                raise KeyboardInterrupt
            dictionary["r_offset"] = float(r_offset_str)

            r_cycles_str = questionary.text(
                "Numero di cicli ricorsivi:", default="1", validate=is_positive_integer
            ).ask()
            if r_cycles_str is None:
                raise KeyboardInterrupt
            dictionary["r_cycles"] = int(r_cycles_str)

    return dictionary


def io_query() -> dict:
    dictionary: dict = {}
    print("\n--- ESPORTAZIONE E SALVATAGGIO (I/O) ---")

    save_csv = questionary.confirm(
        "Vuoi esportare le coordinate finali in un file CSV?",
        default=False,
        auto_enter=False,
    ).ask()

    if save_csv is None:
        raise KeyboardInterrupt

    if save_csv:
        path_out = questionary.path(
            "Percorso in cui salvare il CSV (es. output/percorso.csv):",
            default="percorso_generato.csv",
        ).ask()

        if path_out is None:
            raise KeyboardInterrupt
        dictionary["path_out"] = path_out
    else:
        dictionary["path_out"] = None

    save_job = questionary.confirm(
        "Vuoi salvare questa configurazione in un file JSON per riutilizzarla in futuro?",
        default=False,
        auto_enter=False,
    ).ask()

    if save_job is None:
        raise KeyboardInterrupt

    if save_job:
        job_out = questionary.path(
            "Percorso in cui salvare il file JSON (es. lavorazioni/job_01.json):",
            default="configurazione_job.json",
            validate=validate_new_file_json,
        ).ask()

        if job_out is None:
            raise KeyboardInterrupt
        dictionary["job_out"] = job_out
    else:
        dictionary["job_out"] = None

    return dictionary


def chiedi_configurazione_completa():
    config: dict = {}

    config_shape = shape_query()
    config_orientation = orientation_query()
    config_pocket = pocket_query()
    config_io = io_query()

    return config | config_shape | config_orientation | config_pocket | config_io
