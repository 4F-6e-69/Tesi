from typing import Optional, List, Literal

import numpy as np
import warnings
import pyvista as pv

from src.utils import Eps, ArrayLike

UpdatesList = Optional[List[Literal["x", "y", "n", "o", "all"]]]

class WorkingSpace:
    def __init__(self, origin=(0, 0, 0), x_axis=(1, 0, 0), y_axis=(0, 1, 0), z_axis=(0, 0, 1), _skip=False):
        self._origin = np.asarray(origin, dtype=np.float64).flatten()

        x_raw = np.asarray(x_axis, dtype=np.float64).flatten()
        y_raw = np.asarray(y_axis, dtype=np.float64).flatten()
        z_raw = np.asarray(z_axis, dtype=np.float64).flatten()

        norm_x = np.linalg.norm(x_raw)
        norm_y = np.linalg.norm(y_raw)
        norm_z = np.linalg.norm(z_raw)

        if norm_x < Eps.eps08 or norm_y < Eps.eps08 or norm_z < Eps.eps08:
            raise ValueError('Gli assi forniti non possono essere vettori nulli (lunghezza pari a zero).')

        x = x_raw / norm_x
        y = y_raw / norm_y
        z = z_raw / norm_z

        if not _skip and not WorkingSpace.check_axes(x, y, z, Eps.eps08):
            raise ValueError('Gli assi forniti (x_axis, y_axis, z_axis) non sono reciprocamente ortogonali.')

        self._x_axis = x
        self._y_axis = y
        self._normal = z

        self._R = np.stack([x, y, z])
        self._interface = None
        self.__interface_element = {}
        self.__interface_updates = {}

    @classmethod
    def new_space_from_three_points(cls, origin: ArrayLike, x_hint: ArrayLike, plane_hint: ArrayLike) -> 'WorkingSpace':
        o = np.asarray(origin, dtype=np.float64).flatten()
        xh = np.asarray(x_hint, dtype=np.float64).flatten()
        ph = np.asarray(plane_hint, dtype=np.float64).flatten()

        v1 = xh - o
        v2 = ph - o
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if len_v1 < Eps.eps08 or len_v2 < Eps.eps08:
            raise ValueError("Impossibile definire gli assi: 'x_hint' o 'plane_hint' coincidono con l'origine.")

        ux = v1 / len_v1

        n_raw = np.cross(ux, v2)
        n_norm = np.linalg.norm(n_raw)

        if n_norm < Eps.eps08:
            raise ValueError("Impossibile definire un piano: i tre punti forniti sono collineari.")

        n = n_raw / n_norm

        uy_raw = np.cross(n, ux)
        uy = uy_raw / np.linalg.norm(uy_raw)

        if cls.check_axes(ux, uy, n, Eps.eps08):
            return cls(origin=o, x_axis=ux, y_axis=uy, z_axis=n, _skip=True)

        raise RuntimeError("Errore fatale interno: gli assi generati dai tre punti non risultano ortogonali.")
    @classmethod
    def new_space_from_normal(cls, origin: ArrayLike, normal: ArrayLike, x_hint: ArrayLike) -> 'WorkingSpace':
        o = np.asarray(origin, dtype=np.float64).flatten()
        n_raw = np.asarray(normal, dtype=np.float64).flatten()
        xh = np.asarray(x_hint, dtype=np.float64).flatten()

        n_len = np.linalg.norm(n_raw)
        if n_len < Eps.eps08:
            raise ValueError("Impossibile definire il piano: la normale fornita è un vettore nullo.")
        n = n_raw / n_len

        x_raw = xh - o
        x_len = np.linalg.norm(x_raw)
        if x_len < Eps.eps08:
            raise ValueError("Impossibile definire l'orientamento: 'x_hint' coincide con l'origine.")
        ux_hint = x_raw / x_len

        y_raw = np.cross(n, ux_hint)
        y_len = np.linalg.norm(y_raw)
        if y_len < Eps.eps08:
            raise ValueError("Impossibile definire l'orientamento: 'x_hint' è perfettamente parallelo alla normale.")

        uy = y_raw / y_len

        x_ortho_raw = np.cross(uy, n)
        ux = x_ortho_raw / np.linalg.norm(x_ortho_raw)

        if cls.check_axes(ux, uy, n, Eps.eps08):
            return cls(origin=o, x_axis=ux, y_axis=uy, z_axis=n, _skip=True)

        raise RuntimeError("Errore fatale interno: gli assi generati non risultano ortogonali.")
    @classmethod
    def new_space_from_two_straight(cls, x_axis: ArrayLike, x_hint: ArrayLike, y_axis: ArrayLike, y_hint: ArrayLike, verbose: bool = False) -> 'WorkingSpace':
        x_raw = np.asarray(x_axis, dtype=np.float64).flatten()
        y_raw = np.asarray(y_axis, dtype=np.float64).flatten()
        xh = np.asarray(x_hint, dtype=np.float64).flatten()
        yh = np.asarray(y_hint, dtype=np.float64).flatten()

        x_len = np.linalg.norm(x_raw)
        y_len = np.linalg.norm(y_raw)

        if x_len < Eps.eps08 or y_len < Eps.eps08:
            raise ValueError("I vettori di direzione 'x_axis' o 'y_axis' non sono validi (lunghezza nulla).")

        x_dir = x_raw / x_len
        y_dir = y_raw / y_len

        cross_uv = np.cross(x_dir, y_dir)
        norm_cross = np.linalg.norm(cross_uv)

        if norm_cross < Eps.eps08:
            dp = yh - xh

            rejection = dp - np.dot(dp, x_dir) * x_dir
            dist_between_lines = np.linalg.norm(rejection)

            if dist_between_lines < Eps.eps08:
                raise ValueError("Le rette fornite sono collineari (coincidenti): impossibile definire un piano unico.")

            n_raw = np.cross(x_dir, dp)
            n = n_raw / np.linalg.norm(n_raw)

            center = xh
            final_ux = x_dir
            final_uy = np.cross(n, final_ux)

        else:
            if np.allclose(xh, yh, atol=WorkingSpace.EPS_12):
                center = xh
            else:
                a = np.stack([x_dir, -y_dir], axis=1)
                b = yh - xh

                t_s, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
                t, s_param = t_s

                if verbose:
                    print(f" > Parametri calcolati: t={t:.4f}, s={s_param:.4f}")

                point_on_ux = xh + t * x_dir
                point_on_uy = yh + s_param * y_dir

                dist_mismatch = np.linalg.norm(point_on_ux - point_on_uy)

                if dist_mismatch > Eps.eps06:
                    raise ValueError(f"Le rette sono sghembe e non complanari. Distanza minima: {dist_mismatch:.2e}")

                center = point_on_ux + (point_on_uy - point_on_ux) / 2.0

            n = cross_uv / norm_cross
            final_ux = x_dir
            final_uy = np.cross(n, final_ux)

        if cls.check_axes(final_ux, final_uy, n, Eps.eps06):
            return cls(origin=center, x_axis=final_ux, y_axis=final_uy, z_axis=n, _skip=True)

        raise RuntimeError("Errore fatale interno: gli assi generati non risultano ortogonali.")

    @property
    def x_axis(self) -> np.typing.NDArray[np.float64]:
        if self._x_axis is None:
            raise AttributeError('L\'asse X (x_axis) non è stato inizializzato. (ha valore None)')
        return self._x_axis
    @property
    def y_axis(self) -> np.typing.NDArray[np.float64]:
        if self._y_axis is None:
            raise AttributeError('L\'asse Y (y_axis) non è stato inizializzato. (ha valore None)')
        return self._y_axis
    @property
    def z_axis(self) -> np.typing.NDArray[np.float64]:
        if self._normal is None:
            raise AttributeError('L\'asse Z (z_axis) non è stato inizializzato. (ha valore None)')
        return self._normal
    @property
    def normal(self) -> np.typing.NDArray[np.float64]:
        if self._normal is None:
            raise AttributeError('La normale (normal) non è stata inizializzata. (ha valore None)')
        return self._normal
    @property
    def space_matrix(self) -> np.typing.NDArray[np.float64]:
        if self._R is None:
            raise AttributeError('La matrice dello spazio (space_matrix / R) non è stata inizializzata. (ha valore None)')
        return self._R

    @property
    def space_transposed_matrix(self) -> np.typing.NDArray[np.float64]:
        if self._R is None:
            raise AttributeError('Impossibile calcolare la trasposta: la matrice dello spazio non è inizializzata. (ha valore None)')
        return np.ascontiguousarray(self._R.T)
    @property
    def axes(self) -> np.typing.NDArray[np.float64]:
        if WorkingSpace.check_axes(self.x_axis, self.y_axis, self._normal, self.tolerance):
            return np.asarray([self.x_axis, self.y_axis, self.z_axis], dtype=np.float64)

        raise RuntimeError('Stato corrotto: x_axis, y_axis e z_axis non sono più mutuamente ortogonali.')
    @property
    def origin(self) -> np.typing.NDArray[np.float64]:
        if self._origin is None:
            raise AttributeError('L\'origine (origin) non è stata inizializzata. (ha valore None)')
        return self._origin

    @property
    def interface(self):
        if self._interface is None:
            self.create_cad_interface(save=True)
        return self._interface
    def __update_interface(self, updates: UpdatesList = None):
        if updates is None or self._interface is None or self._interface.renderer is None:
            return

        def safe_remove(act_name: str):
            if act_name in self._interface.actors:
                self._interface.remove_actor(act_name)

        for target in updates:
            fine_params = {
                'scale': 60,
                'tip_radius': 0.08,  # Punta più sottile (default è 0.25)
                'tip_length': 0.25,  # Punta più corta e proporzionata
                'shaft_radius': 0.025,  # Asta molto sottile (stile linea)
                'shaft_resolution': 20,  # Più liscio
                'tip_resolution': 20  # Più liscio
            }
            if target == "all":
                for name in ["CustomPlane", "vettore_normale", "vettore_x", "vettore_y"]:
                    safe_remove(name)

                custom_plane = pv.Plane(center=self.origin, direction=self.normal, i_size=100, j_size=100,
                                        i_resolution=100, j_resolution=100)
                self._interface.add_mesh(custom_plane, color="lightgrey", show_edges=True, edge_color="blue",
                                         opacity=0.1,
                                         name="CustomPlane")

                n = pv.Arrow(start=self.origin, direction=self.normal, **fine_params)
                self._interface.add_mesh(n, color="blue", name="vettore_normale", smooth_shading=True)

                x = pv.Arrow(start=self.origin, direction=self.x_axis, **fine_params)
                self._interface.add_mesh(x, color="red", name="vettore_x", smooth_shading=True)

                y = pv.Arrow(start=self.origin, direction=self.y_axis, **fine_params)
                self._interface.add_mesh(y, color="green", name="vettore_y", smooth_shading=True)
                return

            elif target == "n":
                safe_remove("vettore_normale")
                n = pv.Arrow(start=self.origin, direction=self.normal, **fine_params)
                self._interface.add_mesh(n, color="blue", name="vettore_normale", smooth_shading=True)

            elif target == "x":
                safe_remove("vettore_x")
                x = pv.Arrow(start=self.origin, direction=self.x_axis, **fine_params)
                self._interface.add_mesh(x, color="red", name="vettore_x", smooth_shading=True)

            elif target == "y":
                safe_remove("vettore_y")
                y = pv.Arrow(start=self.origin, direction=self.y_axis, **fine_params)
                self._interface.add_mesh(y, color="green", name="vettore_y", smooth_shading=True)

    def global_to_local(self, points: ArrayLike) -> np.typing.NDArray[np.float64]:
        points_array = np.asarray(points, dtype=np.float64)
        shifted_points = points_array - self.origin
        return shifted_points @ self.space_transposed_matrix
    def local_to_global(self, points: ArrayLike) -> np.typing.NDArray[np.float64]:
        local_points = np.asarray(points, dtype=np.float64)
        rotated_points = local_points @ self.space_matrix
        return rotated_points + self.origin
    def space_to_local(self, points: ArrayLike, **kwargs) -> np.typing.NDArray[np.float64]:
        other_space = kwargs.get('space', None)

        if other_space is not None:
            if not isinstance(other_space, WorkingSpace):
                raise TypeError("L'argomento 'space' deve essere un'istanza di WorkingSpace.")
            if not other_space.is_valid():
                raise ValueError("Lo spazio di partenza ('space') risulta corrotto o non valido.")

            global_points = other_space.local_to_global(points)
            return self.global_to_local(global_points)

        other_matrix = kwargs.get('matrix', None)
        other_center = kwargs.get('center', None)

        if other_matrix is None and other_center is None:
            raise ValueError("Parametri insufficienti: devi fornire 'space', oppure 'matrix' e/o 'center'.")

        if other_matrix is None:
            other_matrix = np.eye(3, dtype=np.float64)
        else:
            if not WorkingSpace.validate_matrix(other_matrix):
                warnings.warn('La matrice fornita (other_matrix) introduce deformazioni (non è ortonormale).')

        if other_center is None:
            other_center = np.zeros(3, dtype=np.float64)
        else:
            other_center = np.asarray(other_center, dtype=np.float64)

        points_array = np.asarray(points, dtype=np.float64)
        global_points = points_array @ other_matrix + other_center
        return self.global_to_local(global_points)

    def invert_x(self):
        self._x_axis = -self.x_axis
        self.__update_matrix()
        self.__update_interface(["x"])
    def invert_y(self):
        self._y_axis = -self.y_axis
        self.__update_matrix()
        self.__update_interface(["y"])
    def invert_z(self):
        self._normal = -self.normal
        self.__update_matrix()
        self.__update_interface(["n"])
    def invert_normal(self):
        self._normal = -self.normal
        self.__update_matrix()
        self.__update_interface(["n"])
    def invert_space(self):
        self._x_axis = -self.x_axis
        self._y_axis = -self.y_axis
        self._normal = -self.normal

        self.__update_matrix()
        self.__update_interface(["x", "y", "n"])
    def invert_plane(self):
        self._x_axis = -self.x_axis
        self._y_axis = -self.y_axis

        self.__update_matrix()
        self.__update_interface(["x", "y"])
    def exchange_plane(self):
        temp_x = self.x_axis
        self._x_axis = self.y_axis
        self._y_axis = temp_x

        self.__update_matrix()
        self.__update_interface(["x", "y"])
    def __update_matrix(self):
        self._R = np.stack([
            self.x_axis,
            self.y_axis,
            self.normal
        ])

    @staticmethod
    def check_axes(ux: np.typing.NDArray[np.float64], uy: np.typing.NDArray[np.float64],
                   uz: np.typing.NDArray[np.float64], tolerance):
        if not isinstance(ux, np.ndarray) or not isinstance(uy, np.ndarray) or not isinstance(uz, np.ndarray):
            raise TypeError('ux, uy e uz devono essere oggetti np.ndarray.')

        if np.shape(ux) != (3,) or np.shape(uy) != (3,) or np.shape(uz) != (3,):
            raise ValueError('Gli assi ux, uy e uz devono avere tutti forma (3,).')

        space_matrix = np.stack([ux, uy, uz], dtype=np.float64)
        is_orthogonal = np.allclose(space_matrix.T @ space_matrix, np.eye(3), atol=tolerance)
        if not is_orthogonal:
            return False

        return True

    def is_valid(self) -> bool:
        if not isinstance(self.origin, np.ndarray) or not isinstance(self.x_axis, np.ndarray) or not isinstance(
                self.y_axis, np.ndarray) or not isinstance(self.z_axis, np.ndarray) or not isinstance(self.normal,
                np.ndarray) or not isinstance(self.space_matrix, np.ndarray) or not isinstance(
                self.space_transposed_matrix, np.ndarray):
            raise TypeError('Uno o più assi o matrici non sono del tipo corretto (richiesto np.ndarray).')

        check_matrix = np.stack([self.x_axis, self.y_axis, self.normal], dtype=np.float64)
        check_traspose = check_matrix.T

        if not np.allclose(check_matrix, self.space_matrix, Eps.eps08) or not np.allclose(check_traspose,
                                                                                               self.space_transposed_matrix,
                                                                                               Eps.eps08):
            raise ValueError('Inconsistenza interna: la space_matrix non corrisponde agli assi x, y, z attuali.')

        if not WorkingSpace.validate_matrix(self.space_matrix):
            warnings.warn(
                'Il sistema di riferimento introduce deformazioni (la matrice non è perfettamente ortonormale).')

        if not np.isclose(np.linalg.det(self.space_matrix), 1, Eps.eps08):
            warnings.warn(
                'Il sistema di riferimento non è destrorso puro (potrebbe contenere una riflessione degli assi).')

        return True

    @staticmethod
    def validate_space(space):
        if not isinstance(space, WorkingSpace):
            raise TypeError('L\'oggetto fornito non è un\'istanza valida di WorkingSpace.')

        space.is_valid()
    @staticmethod
    def validate_matrix(space_matrix: np.typing.NDArray[np.float64]):
        if not isinstance(space_matrix, np.ndarray):
            raise TypeError('space_matrix deve essere un oggetto np.ndarray.')

        if space_matrix.shape != (3, 3):
            raise ValueError(f'La matrice deve essere 3x3, trovata forma {space_matrix.shape}.')

        is_orthogonal = np.allclose(space_matrix.T @ space_matrix, np.eye(3), atol=Eps.eps08)
        if not is_orthogonal:
            return False

        return True

    def add_element(self, element_name: str, element_type: str, element_color: str, **kwargs):
        if element_type == "point":
            centro = kwargs["center"]
            self.__interface_updates[element_name] = {"type": element_type, "center": centro, "color": element_color}
            return
        if element_type == "arrow":
            start = kwargs["start"]
            end = kwargs["end"]
            self.__interface_updates[element_name] = {"type": element_type, "start": start, "end": end, "color": element_color}
            return
    def add_points(self, name: str, points: ArrayLike, color: str):
        for p in points:
            current_name = f"{name}-{p}"
            self.add_element(current_name, "point", color, center=p)
    def add_directions(self,name: str, color: str, start: ArrayLike, end: ArrayLike):
        if len(start) is not len(end):
            raise AttributeError()

        for index in range(len(start)):
            current_name = f"{name}-{start[index]}"
            self.add_element(current_name, "arrow", color, start=start[index], end=end[index])

    def create_cad_interface(self, save: Optional[bool] = False):
        pv.set_plot_theme("document")
        plotter = pv.Plotter(title="CAD Sanding Robot - Ultra Lean")
        plotter.enable_parallel_projection()

        state = {'last_scale': 0.0, 'last_step': 200, 'base_scale': 0.0}
        keys_down = set()
        last_action_performed = False

        plotter.add_mesh(pv.Line([-500, 0, 0], [500, 0, 0]), color="red")
        plotter.add_mesh(pv.Line([0, -500, 0], [0, 500, 0]), color="green")
        plotter.add_mesh(pv.Line([0, 0, -500], [0, 0, 500]), color="blue")

        testo_zoom = plotter.add_text("Zoom: 1.00x", position='upper_left', font_size=10)

        res_init = int(1000 / state['last_step'])
        grid_init = pv.Plane(i_size=1000, j_size=1000, i_resolution=res_init, j_resolution=res_init)
        plotter.add_mesh(grid_init, color="lightgrey", show_edges=True,
                         edge_color="blue", opacity=0.1, name="Dynamic Grid")


        custom_plane = pv.Plane(center=self.origin, direction=self.normal, i_size=100, j_size=100, i_resolution=100, j_resolution=100)
        plotter.add_mesh(custom_plane, color="lightgrey", show_edges=True, edge_color="blue", opacity=0.1, name="CustomPlane")
        fine_params = {
            'scale': 60,
            'tip_radius': 0.08,  # Punta più sottile (default è 0.25)
            'tip_length': 0.25,  # Punta più corta e proporzionata
            'shaft_radius': 0.025,  # Asta molto sottile (stile linea)
            'shaft_resolution': 20,  # Più liscio
            'tip_resolution': 20  # Più liscio
        }
        direction_params = {
            'scale': 5,
            'tip_radius': 0.08,  # Punta più sottile (default è 0.25)
            'tip_length': 0.25,  # Punta più corta e proporzionata
            'shaft_radius': 0.025,  # Asta molto sottile (stile linea)
            'shaft_resolution': 20,  # Più liscio
            'tip_resolution': 20  # Più liscio
        }

        n = pv.Arrow(start=self.origin, direction=self.normal, **fine_params)
        plotter.add_mesh(n, color="blue", name="vettore_normale", smooth_shading=True)

        x = pv.Arrow(start=self.origin, direction=self.x_axis, **fine_params)
        plotter.add_mesh(x, color="red", name="vettore_x", smooth_shading=True)

        y = pv.Arrow(start=self.origin, direction=self.y_axis, **fine_params)
        plotter.add_mesh(y, color="green", name="vettore_y", smooth_shading=True)

        def add_element_to_scene(element, name):
            if element["type"] == "point":
                new_point = pv.Sphere(center=element["center"])
                plotter.add_mesh(new_point, color=element["color"], show_edges=True, edge_color=f"dark{element["color"]}", name=name)

            elif element["type"] == "arrow":
                new_arrow = pv.Arrow(start=element["start"], direction=element["end"], **direction_params)
                plotter.add_mesh(new_arrow, color=element["color"], name=name, smooth_shading=True)

        def update_axis_labels(step):
            ticks = np.arange(-500, 501, step)
            n_points = len(ticks)

            axes_config = [
                (0, "red", "x_labels"),
                (1, "green", "y_labels"),
                (2, "blue", "z_labels")
            ]
            for axis_idx, color, label_name in axes_config:
                pts = np.zeros((n_points, 3))
                pts[:, axis_idx] = ticks

                plotter.add_point_labels(
                    pts, [str(v) for v in ticks],
                    name=label_name,
                    font_size=7,
                    show_points=False,
                    text_color=color,
                    shape=None,
                    background_opacity=0,
                    always_visible=True
                )
        def update_scene(*args):
            scale = plotter.camera.parallel_scale
            if state['last_scale'] != 0:
                diff_relativa = abs(scale - state['last_scale']) / state['last_scale']
                if diff_relativa < 0.005:
                    return

            state['last_scale'] = scale

            zoom_factor = state['base_scale'] / scale if state['base_scale'] != 0 else 1.0
            testo_zoom.set_text(2, f"Zoom: {zoom_factor:.1f}x")
            if scale > 400:
                step = 200
            elif scale > 200:
                step = 100
            else:
                step = 50

            if step != state['last_step']:
                state['last_step'] = step
                res = int(1000 / step)
                grid_floor = pv.Plane(i_size=1000, j_size=1000, i_resolution=res, j_resolution=res)
                plotter.add_mesh(grid_floor, color="lightgrey", show_edges=True,
                                 edge_color="blue", opacity=0.1, name="Dynamic Grid")
                update_axis_labels(step)

        plotter.view_isometric()
        plotter.reset_camera()
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.position = (1000, 1000, 1000)
        plotter.camera.zoom(0.8)

        state['last_scale'] = plotter.camera.parallel_scale
        state['base_scale'] = state['last_scale']
        plotter.iren.add_observer("ModifiedEvent", update_scene)
        plotter.add_axes(line_width=3, cone_radius=0.5, shaft_length=0.7, label_size=(0.05, 0.05))

        def on_press(obj, event):
            key = obj.GetKeySym().lower()
            keys_down.add(key)

            if last_action_performed:
                return

            if "i" in keys_down:
                if "a" in keys_down:
                    self.invert_space()
                elif "n" in keys_down:
                    self.invert_normal()
                elif "x" in keys_down:
                    self.invert_x()
                elif "y" in keys_down:
                    self.invert_y()
                elif "p" in keys_down:
                    self.invert_plane()
                else:
                    return
            elif "p" in keys_down:
                self.exchange_plane()
            elif "u" in keys_down:
                if len(self.__interface_updates) > 0:
                    for up in self.__interface_updates:
                        add_element_to_scene(self.__interface_updates[up], up)

                    self.__interface_element = self.__interface_element | self.__interface_updates
                    self.__interface_updates = {}
                else:
                    return
        def on_release(obj, event):
            key = obj.GetKeySym().lower()
            keys_down.discard(key)

            if not keys_down:
                last_action_performed = False

        plotter.iren.add_observer("KeyPressEvent", on_press)
        plotter.iren.add_observer("KeyReleaseEvent", on_release)

        update_axis_labels(200)

        if save:
            self._interface = plotter
        return plotter