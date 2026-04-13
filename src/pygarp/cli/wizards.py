import questionary
from questionary import Choice

import os
import sys


def _validate_float(testo):
    try:
        float(testo)
        return True
    except ValueError:
        return "Errore: Coordinata numerica non valida."


def _validate_positive_float(testo):
    try:
        val = float(testo)
        return (
            True if val > 0 else "Errore: La dimensione deve essere maggiore di zero."
        )
    except ValueError:
        return "Errore: Inserire un valore numerico valido (usare il punto per i decimali)."


def _validate_positive_int(testo):
    try:
        val = int(testo)
        return (
            True
            if val > 0
            else "Errore: Il valore deve essere un intero maggiore di zero."
        )
    except ValueError:
        return "Errore: Inserire un numero intero valido (senza decimali)."


def _validate_tuple_2d(testo):
    parti = testo.split(",")
    if len(parti) != 2:
        return "Errore: Inserire esattamente due valori separati da una virgola (es. 10.5, 20.0)."
    try:
        float(parti[0].strip())
        float(parti[1].strip())
        return True
    except ValueError:
        return "Errore: Entrambi i valori devono essere numerici."


def _validate_tuple_3d(testo):
    """Verifica che l'input contenga esattamente 3 valori numerici separati da virgola."""
    parti = testo.split(",")
    if len(parti) != 3:
        return "Errore: Inserire esattamente tre coordinate (X, Y, Z) separate da virgola (es. 10.0, 0.0, -5.5)."
    try:
        # Tenta il cast a float per ogni elemento
        float(parti[0].strip())
        float(parti[1].strip())
        float(parti[2].strip())
        return True
    except ValueError:
        return "Errore: Tutte le coordinate devono essere valori numerici."


def _validate_polygon_sides(testo):
    try:
        val = int(testo)
        return (
            True if val >= 3 else "Errore: Un poligono deve avere un minimo di 3 lati."
        )
    except ValueError:
        return "Errore: Inserire un numero intero."


def _validate_file(testo):
    if not os.path.exists(testo):
        return "Errore: Il percorso specificato non esiste."
    if not os.path.isfile(testo):
        return "Errore: Il percorso punta a una directory. Specificare un file."
    # if not testo.lower().endswith('.csv'):
    #     return "Errore: Formato non valido. Richiesto file con estensione .csv."
    return True


shape_questions = [
    # --- Selezione Geometria Base ---
    {
        "type": "select",
        "name": "shape_type",
        "message": "Definire la tipologia di geometria per la lavorazione:",
        "choices": [
            Choice("Forma Generica (Generic Shape)", value="shape"),
            Choice("Poligono Regolare (Regular Polygon)", value="regular_polygon"),
            Choice("Spline Chiusa (Closed Spline)", value="spline"),
            Choice("Rettangolo (Rectangle)", value="rectangle"),
            Choice("Cerchio (Circle)", value="circle"),
        ],
    },
    # --- Parametri Specifici: SHAPE e SPLINE ---
    {
        "type": "path",
        "name": "control_points_path",
        "message": "Percorso del file CSV contenente i punti di controllo (X, Y):",
        "when": lambda answers: answers.get("shape_type") in ["shape", "spline"],
        "validate": _validate_file,
    },
    # --- Parametri Specifici: RETTANGOLO ---
    {
        "type": "text",
        "name": "width",
        "message": "Dimensione Base [X]:",
        "when": lambda answers: answers.get("shape_type") == "rectangle",
        "validate": _validate_positive_float,
    },
    {
        "type": "text",
        "name": "height",
        "message": "Dimensione Altezza [Y]:",
        "when": lambda answers: answers.get("shape_type") == "rectangle",
        "validate": _validate_positive_float,
    },
    # --- Parametri Specifici: CERCHIO ---
    {
        "type": "text",
        "name": "radius",
        "message": "Valore del Raggio:",
        "when": lambda answers: answers.get("shape_type") == "circle",
        "validate": _validate_positive_float,
    },
    # --- Parametri Specifici: POLIGONO REGOLARE ---
    {
        "type": "text",
        "name": "n_sides",
        "message": "Numero di lati (N):",
        "when": lambda answers: answers.get("shape_type") == "regular_polygon",
        "validate": _validate_polygon_sides,
    },
    {
        "type": "text",
        "name": "side_length",
        "message": "Lunghezza del singolo lato:",
        "when": lambda answers: answers.get("shape_type") == "regular_polygon",
        "validate": _validate_positive_float,
    },
    # --- Dati di Posizionamento (Comuni a Rect, Circ, Poly) ---
    {
        "type": "text",
        "name": "center",
        "message": "Coordinate (X, Y) separate da virgola:",
        "when": lambda answers: answers.get("shape_type")
        in ["rectangle", "circle", "regular_polygon"],
        "default": "0.0, 0.0",
        "validate": _validate_tuple_2d,
    },
]
space_questions = [
    # --- Impostazione Sistema di Riferimento ---
    {
        "type": "select",
        "name": "space_strategy",
        "message": "Definire il sistema di riferimento spaziale (Frame) per la lavorazione:",
        "choices": [
            Choice(
                "DFT: Default (Sistema di coordinate globale assoluto)", value="DFT"
            ),
            Choice("OPP: Origine, Punto su asse X, Punto su piano XY", value="OPP"),
            Choice(
                "ONC: Origine, Punto su asse X, Vettore normale (Asse Z)", value="ONC"
            ),
            Choice(
                "XYP: Definizione esplicita tramite vettori e punti per asse X e asse Y",
                value="XYP",
            ),
        ],
    },
    # --- Origine ---
    {
        "type": "text",
        "name": "origin_point",
        "message": "Coordinate dell'origine (X, Y, Z):",
        "when": lambda answers: answers.get("space_type") in ["OPP", "ONC"],
        "default": "0.0, 0.0, 0.0",
        "validate": _validate_tuple_3d,
    },
    # --- Definizione Asse X ---
    {
        "type": "text",
        "name": "x_axis",
        "message": "Vettore direzione asse X (X, Y, Z):",
        "when": lambda answers: answers.get("space_type") == "XYP",
        "default": "1.0, 0.0, 0.0",
        "validate": _validate_tuple_3d,
    },
    {
        "type": "text",
        "name": "x_hint",
        "message": "Coordinate punto di passaggio asse X (X, Y, Z):",
        "when": lambda answers: answers.get("space_type") in ["OPP", "ONC", "XYP"],
        "default": lambda answers: (
            "1.0, 0.0, 0.0"
            if answers.get("space_type") in ["OPP", "ONC"]
            else "0.0, 0.0, 0.0"
        ),
        "validate": _validate_tuple_3d,
    },
    # --- Definizione Asse Y ---
    {
        "type": "text",
        "name": "y_axis",
        "message": "Vettore direzione asse Y (X, Y, Z):",
        "when": lambda answers: answers.get("space_type") == "XYP",
        "default": "0.0, 1.0, 0.0",
        "validate": _validate_tuple_3d,
    },
    {
        "type": "text",
        "name": "y_hint",
        "message": "Coordinate punto di passaggio asse Y (X, Y, Z):",
        "when": lambda answers: answers.get("space_type")
        == "XYP",  # 🛠️ BUG FIX: Era impostato per tutti i metodi
        "default": "0.0, 0.0, 0.0",
        "validate": _validate_tuple_3d,
    },
    # --- Terzo elemento per chiudere il frame (Z / Piano XY) ---
    {
        "type": "text",
        "name": "z_axis",
        "message": "Vettore normale asse Z (X, Y, Z):",
        "when": lambda answers: answers.get("space_type") == "ONC",
        "default": "0.0, 0.0, 1.0",
        "validate": _validate_tuple_3d,
    },
    {
        "type": "text",
        "name": "p_hint",
        "message": "Coordinate punto ausiliario sul piano XY (X, Y, Z):",
        "when": lambda answers: answers.get("space_type") == "OPP",
        "default": "0.0, 1.0, 0.0",
        "validate": _validate_tuple_3d,
    },
]
outline_questions = [
    # --- 1. Abilitazione Contorno (Booleana) ---
    {
        "type": "select",
        "name": "apply_outline",
        "message": "Abilitare la lavorazione di contorno (outline)?",
        "choices": [Choice("No", value=False), Choice("Sì", value=True)],
    },
    # --- 2. Modalità Contorno (Step / Gradient) ---
    {
        "type": "select",
        "name": "outline_mode",
        "message": "Definire la modalità di transizione per il contorno:",
        "when": lambda answers: answers.get("apply_outline", False),
        "choices": [
            Choice("Step (Transizione netta)", value="step"),
            Choice("Gradient (Transizione graduale)", value="gradient"),
        ],
    },
]
infill_questions = [
    # --- 1. Lavorazione Infill Base ---
    {
        "type": "select",
        "name": "apply_infill",
        "message": "Abilitare la lavorazione di riempimento interno (infill)?",
        "choices": [Choice("No", value=False), Choice("Sì", value=True)],
    },
    {
        "type": "select",
        "name": "infill_mode",
        "message": "Pattern di transizione per il riempimento interno:",
        "when": lambda answers: answers.get("apply_infill", False),
        "choices": [
            Choice("Lineare (Transizione netta)", value="linear"),
            Choice("Griglia (Transizione graduale)", value="grid"),
        ],
    },
    {
        "type": "text",
        "name": "infill_dir",
        "message": "Angolo di inclinazione delle linee di riempimento (gradi rispetto all'orizzontale):",
        "when": lambda answers: answers.get("apply_infill", False),
        "default": "0.0",
        "validate": _validate_float,
    },
    {
        "type": "text",
        "name": "infill_spacing",
        "message": "Spaziatura tra le linee di riempimento:",
        "when": lambda answers: answers.get("apply_infill", False),
        "validate": _validate_positive_float,
    },
    # --- 2. Linee Concentriche ---
    {
        "type": "select",
        "name": "apply_concentric",
        "message": "Abilitare la generazione di linee concentriche?",
        "when": lambda answers: answers.get("apply_infill", False),
        "choices": [Choice("No", value=False), Choice("Sì", value=True)],
        "default": False,
    },
    {
        "type": "text",
        "name": "concentric_initial_offset",
        "message": "Offset iniziale (float):",
        "when": lambda answers: answers.get("apply_concentric", False),
        "default": "0.0",
        "validate": _validate_float,
    },
    {
        "type": "text",
        "name": "concentric_cycles",
        "message": "Numero di cicli concentrici da eseguire (intero):",
        "when": lambda answers: answers.get("apply_concentric", False),
        "validate": _validate_positive_int,
    },
    {
        "type": "text",
        "name": "concentric_spacing",
        "message": "Spaziatura tra i cicli concentrici (intero):",
        "when": lambda answers: answers.get("apply_concentric", False),
        "validate": _validate_positive_int,
    },
    # --- 3. Tasche Ricorsive (Recursive Pockets) ---
    {
        "type": "select",
        "name": "apply_recursive",
        "message": "Abilitare la lavorazione a tasche ricorsive (recursive pockets)?",
        "when": lambda answers: answers.get("apply_infill", False),
        "choices": [Choice("No", value=False), Choice("Sì", value=True)],
        "default": False,
    },
    {
        "type": "text",
        "name": "pocket_offset",
        "message": "Offset tra ogni tasca ricorsiva (float):",
        "when": lambda answers: answers.get("apply_recursive", False),
        "validate": _validate_positive_float,
    },
    {
        "type": "text",
        "name": "pocket_cycles",
        "message": "Numero totale di tasche/cicli (intero):",
        "when": lambda answers: answers.get("apply_recursive", False),
        "validate": _validate_positive_int,
    },
    {
        "type": "text",
        "name": "z_step",
        "message": "Incremento in discesa lungo l'asse Z (float):",
        "when": lambda answers: answers.get("apply_recursive", False),
        "validate": _validate_positive_float,  # Tipicamente Z step down è inteso come valore positivo da sottrarre nel codice
    },
]
save_questions = [
    # --- Esportazione Dati CSV ---
    {
        "type": "select",
        "name": "save_csv",
        "message": "Esportare i dati di simulazione in formato CSV?",
        "choices": [Choice("No", value=False), Choice("Sì", value=True)],
    },
    {
        "type": "path",
        "name": "csv_path",
        "message": "Directory di destinazione per l'export CSV:",
        "when": lambda answers: answers.get("save_csv", False),
        "only_directories": True,
        "validate": lambda testo: (
            True
            if os.path.isdir(testo)
            else "Errore: Directory non valida o inesistente."
        ),
    },
    {
        "type": "text",
        "name": "csv_filename",
        "message": "Nome del file CSV (senza estensione):",
        "when": lambda answers: answers.get("save_csv", False),
        "default": "sim_export",
        "validate": lambda testo: len(testo.strip()) > 0
        or "Il nome del file è obbligatorio.",
    },
    # --- Esportazione Configurazione JSON ---
    {
        "type": "select",
        "name": "save_config",
        "message": "Esportare i parametri di configurazione in formato JSON?",
        "choices": [Choice("No", value=False), Choice("Sì", value=True)],
    },
    {
        "type": "select",
        "name": "custom_input",
        "message": "Utilizzare directory e nome file differenti rispetto all'export CSV?",
        "when": lambda answers: answers.get("save_config", False)
        and answers.get("save_csv", False),
        "choices": [Choice("No", value=False), Choice("Sì", value=True)],
    },
    {
        "type": "path",
        "name": "json_path",
        "message": "Directory di destinazione per il file JSON:",
        "when": lambda answers: answers.get("save_config", False)
        and (not answers.get("save_csv", False) or answers.get("custom_input", False)),
        "default": lambda answers: answers.get("csv_path", ""),
        "only_directories": True,
        "validate": lambda testo: (
            True
            if os.path.isdir(testo)
            else "Errore: Directory non valida o inesistente."
        ),
    },
    {
        "type": "text",
        "name": "json_filename",
        "message": "Nome del file JSON (senza estensione):",
        "when": lambda answers: answers.get("save_config", False)
        and (not answers.get("save_csv", False) or answers.get("custom_input", False)),
        "default": lambda answers: answers.get("csv_filename", "sim_export"),
        "validate": lambda testo: len(testo.strip()) > 0
        or "Il nome del file è obbligatorio.",
    },
    # --- Opzioni Avanzate ---
    {
        "type": "select",
        "name": "save_indi",
        "message": "Salvare le singole tratte di lavorazione come file indipendenti?",
        "choices": [Choice("No", value=False), Choice("Sì", value=True)],
        "when": lambda answers: answers.get("save_config", False),
    },
]


def check_abort(answers):
    if not answers:
        print("\n❌ Operazione annullata dall'operatore.")
        sys.exit(0)


def parse_raw_data(raw_data):
    parsed = raw_data.copy()

    float_keys = [
        "width",
        "height",
        "radius",
        "side_length",
        "infill_dir",
        "infill_spacing",
        "concentric_initial_offset",
        "pocket_offset",
        "z_step",
    ]
    int_keys = ["n_sides", "concentric_cycles", "concentric_spacing", "pocket_cycles"]
    tuple_2d_keys = ["center"]
    tuple_3d_keys = [
        "origin_point",
        "x_axis",
        "x_hint",
        "y_axis",
        "y_hint",
        "z_axis",
        "p_hint",
    ]

    for key, value in parsed.items():
        if value is None or value == "" or isinstance(value, bool):
            continue

        if key in float_keys:
            parsed[key] = float(value)
        elif key in int_keys:
            parsed[key] = int(value)
        elif key in tuple_2d_keys:
            parsed[key] = tuple(float(x.strip()) for x in value.split(","))
        elif key in tuple_3d_keys:
            parsed[key] = tuple(float(x.strip()) for x in value.split(","))

    return parsed


def ask_shape(convert: bool = False):
    output = questionary.prompt(shape_questions)
    check_abort(output)
    return output if not convert else parse_raw_data(output)


def ask_space(convert: bool = False):
    output = questionary.prompt(space_questions)
    check_abort(output)
    return output if not convert else parse_raw_data(output)


def ask_pocket_details(convert: bool = False):
    while True:
        out_answers = questionary.prompt(outline_questions)
        check_abort(out_answers)

        inf_answers = questionary.prompt(infill_questions)
        check_abort(inf_answers)

        output = out_answers | inf_answers

        if not output.get("apply_outline", False) and not output.get(
            "apply_infill", False
        ):
            questionary.print(
                "\n[!] Errore: È necessario abilitare almeno una lavorazione (Outline o Infill). Riprovare.\n",
                style="bold italic fg:red",
            )
            continue

        return output if not convert else parse_raw_data(output)


def ask_saving_info(convert: bool = False):
    output = questionary.prompt(save_questions)
    check_abort(output)
    return output if not convert else parse_raw_data(output)


def ask_pocket():
    shape_data = ask_shape()
    space_data = ask_space()
    pocket_data = ask_pocket_details()
    saving_data = ask_saving_info()

    raw_output = space_data | shape_data | pocket_data | saving_data
    return parse_raw_data(raw_output)
