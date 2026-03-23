from typing import Optional
from typing import Tuple

import typer

from pygarp.cli.wizard import chiedi_configurazione_completa
from pygarp.core.models.commons import ShapeType, FillType
#from pygarp.core.models.validators import JobConfig

app = typer.Typer()


@app.command()
def main(
    # --- GEOMETRIA BASE ---
    shape: Optional[ShapeType] = typer.Option(
        None,
        "--shape",
        "-s",
        help="Tipo di forma da generare.",
        rich_help_panel="1. Geometria Base",
    ),
    radius: Optional[float] = typer.Option(
        None,
        "--radius",
        "-r",
        help="Raggio (solo per 'cerchio').",
        rich_help_panel="1. Geometria Base",
    ),
    side: Optional[float] = typer.Option(
        None,
        "--side",
        "-l",
        help="Lato (solo per 'quadrato').",
        rich_help_panel="1. Geometria Base",
    ),
    width: Optional[float] = typer.Option(
        None,
        "--width",
        "-W",
        help="Larghezza (solo per 'rettangolo').",
        rich_help_panel="1. Geometria Base",
    ),
    height: Optional[float] = typer.Option(
        None,
        "--height",
        "-H",
        help="Altezza (solo per 'rettangolo').",
        rich_help_panel="1. Geometria Base",
    ),
    path_in: Optional[str] = typer.Option(
        None,
        "--path-in",
        "-I",
        help="Percorso file per spline o forme predefinite.",
        rich_help_panel="1. Geometria Base",
    ),
    # --- ORIENTAMENTO 3D ---
    origin: Tuple[float, float, float] = typer.Option(
        (0, 0, 0),
        "--origin",
        "-O",
        help="Origine del piano di lavoro.",
        rich_help_panel="2. Orientamento 3D",
    ),
    x_axis: Tuple[float, float, float] = typer.Option(
        (1, 0, 0),
        "--x-axis",
        help="Vettore asse X.",
        rich_help_panel="2. Orientamento 3D",
    ),
    y_axis: Tuple[float, float, float] = typer.Option(
        (0, 1, 0),
        "--y-axis",
        help="Vettore asse Y.",
        rich_help_panel="2. Orientamento 3D",
    ),
    z_axis: Tuple[float, float, float] = typer.Option(
        (0, 0, 1),
        "--z-axis",
        help="Vettore asse Z (normale).",
        rich_help_panel="2. Orientamento 3D",
    ),
    # --- STRATEGIA DI SCAVO ---
    outline: bool = typer.Option(
        True,
        "--outline/--no-outline",
        help="Abilita/Disabilita il contorno.",
        rich_help_panel="3. Strategia di Scavo",
    ),
    fill: FillType = typer.Option(
        "grid",
        "--fill-type",
        "-T",
        help="Strategia di riempimento.",
        rich_help_panel="3. Strategia di Scavo",
    ),
    fill_direction: float = typer.Option(
        0,
        "--fill-dir",
        "-D",
        help="Direzione linee riempimento in gradi °",
        rich_help_panel="3. Strategia di Scavo",
    ),
    # --- PARAMETRI SCARFING (CONCENTRICO E RICORSIVO) ---
    concentric: bool = typer.Option(
        False,
        "--concentric/--no-concentric",
        "-c",
        help="Passate concentriche pre-riempimento.",
        rich_help_panel="4. Parametri Scarfing",
    ),
    c_offset: Optional[float] = typer.Option(
        None,
        "--c-offset",
        help="Distanza tra passate concentriche.",
        rich_help_panel="4. Parametri Scarfing",
    ),
    c_cycles: int = typer.Option(
        1,
        "--c-cycles",
        help="Ripetizioni ciclo concentrico.",
        rich_help_panel="4. Parametri Scarfing",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive/--no-recursive",
        "-R",
        help="Reitera il percorso in modo ricorsivo.",
        rich_help_panel="4. Parametri Scarfing",
    ),
    r_offset: float = typer.Option(
        0.5,
        "--r-offset",
        "-o",
        help="Offset (mm) tra passate ricorsive.",
        rich_help_panel="4. Parametri Scarfing",
    ),
    r_cycles: int = typer.Option(
        1,
        "--r-cycles",
        "-C",
        help="Numero di cicli ricorsivi.",
        rich_help_panel="4. Parametri Scarfing",
    ),
    # --- I/O ---
    path_out: Optional[str] = typer.Option(
        None, "--path-out", "-P", help="File CSV di output.", rich_help_panel="I/O"
    ),
    job_path: Optional[str] = typer.Option(
        None,
        "--job",
        "-j",
        help="Percorso al file JSON contenente i parametri della lavorazione.",
        rich_help_panel="I/O",
    ),
    job_out: Optional[str] = typer.Option(
        None,
        "--job-out",
        "-J",
        help="Percorso dove salvare il file JSON della sessione corrente.",
        rich_help_panel="I/O",
    ),
):
    wizard_mode = shape is None and job_path is None
    if wizard_mode:
        jpb_config = chiedi_configurazione_completa()
    else:
        jpb_config = {}

if __name__ == "__main__":
    app()