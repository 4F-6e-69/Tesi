import typer
from pathlib import Path

from matplotlib import pyplot as plt

import numpy as np

from pygarp.core.models.validators import ShapeConfig, SpaceConfig, ScarfingConfig, RobotConfig
from pygarp.core.orchestrators.main import execute_pocketing_job
from pygarp.core.workers.writer import pocket_writing
'''
x: 800-1100
y: -50-180
'''
spline_dict = {
    "shape": "regular_polygon",
    "n": 6,
    "side": 72.4,
}
space_dict = {
    "space_type": "default",
    "strategy": "DFT",
    "origin": [750, 350, 30],
    "x_hint": [751, 350, 30],
    "p_hint": [750, 351, 30],
}
scarf_dict = {
    "pocket_type": "step",

    "outline": True,
    "outline_style": "step",

    "fill": False,
    "fill_style": "grid",
    "fill_dir": 45.0,
    "fill_spacing": 8.0,

    "concentric": False,
    "c_offset": 7.0,
    "c_cycle": 5,
    "c_offset_0": 0,


    "recursive": False,
    "r_offset": 8.0,
    "r_cycle": 2,
    "z_off": 1
}
robot_dict = {
    "gamma": -5.0,
    "exit_quote": 50
}

shape_config = ShapeConfig(**spline_dict)
space_config = SpaceConfig(**space_dict)
scarf_config = ScarfingConfig(**scarf_dict)
robot_config = RobotConfig(**robot_dict)

app = typer.Typer(
    name="PocketGen",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False
)


def plot_positions_and_orientations(filename: str, arrow_length: float = 5.0):
    """
    Legge un file CSV/TXT di 6 colonne (X, Y, Z, U, V, W) e genera un plot 3D
    mostrando il percorso e l'orientamento in ogni punto.
    """
    try:
        data = np.loadtxt(filename, delimiter=',')

        # Verifica che ci siano abbastanza colonne (almeno 6)
        if data.shape[1] < 6:
            print("Errore: Il file deve contenere almeno 6 colonne (Posizione 3D + Vettore 3D).")
            return

        # Verifica che ci siano abbastanza colonne
        if data.shape[1] < 6:
            print("Errore: Il file deve contenere almeno 6 colonne (Posizione 3D + Vettore 3D).")
            return

        # Estrai Posizioni (X, Y, Z)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        print(x.size)

        # Estrai Vettori di Orientamento (U, V, W)
        u = x - data[:, 3]
        v = y - data[:, 4]
        w = z - data[:, 5]

        # Configura il Plot 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Disegna la linea continua del percorso
        ax.plot(x, y, z, color='gray', linestyle='--', label='Traiettoria', alpha=0.5)
        # 2. Disegna i punti di posizione (il tuo TCP)
        ax.scatter(x, y, z, color='blue', s=15, label='Posizioni (TCP)')

        ax.scatter(u, v, w, color='red', s=15)

        # Estetica del grafico
        ax.set_title(f"Visualizzazione Toolpath: {filename}")
        ax.set_xlabel("Asse X")
        ax.set_ylabel("Asse Y")
        ax.set_zlabel("Asse Z")

        # Rende gli assi proporzionali (utile in 3D per non deformare la figura)
        ax.set_box_aspect([1, 1, 1])

        ax.legend()
        plt.show()

    except FileNotFoundError:
        print(f"Errore: Il file '{filename}' non esiste.")
    except Exception as e:
        print(f"Si è verificato un errore durante il plot: {e}")

@app.command("calc")
def main():
    result = execute_pocketing_job(shape_config, space_config, scarf_config, robot_config)
    filename = str(Path(__file__).resolve().parent.parent.parent.parent / "tests/output/esagono_con_4x3.csv")
    pocket_writing(result, filename)
    plot_positions_and_orientations(filename)

if __name__ == "__main__":
    app()