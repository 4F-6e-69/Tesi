import numpy as np
from numpy import typing as npt

from shapely.geometry import Polygon

from pygarp.core.models.shapes.closed_spline import ClosedSpline

from pygarp.core.models.validators import (
    ShapeConfig,
    SpaceConfig,
    ScarfingConfig,
    RobotConfig,
)

from pygarp.core.orchestrators.shape import generate_shape
from pygarp.core.orchestrators.space import generate_virtual_space
from pygarp.core.workers.pocket_infill import (
    linear_rect_fill_path,
    calc_linear_intersection,
)
from pygarp.core.workers.pocket_infill import (
    concentric_plot_fill_path,
    calc_concentric_shapes,
)

from pygarp.core.workers.pocket_outline import (
    calc_step_outline,
    calc_gradient_outline,
    discretize_points,
)


def execute_pocketing_job(
    shape_config: ShapeConfig,
    space_config: SpaceConfig,
    scarfing_config: ScarfingConfig,
    robot_config: RobotConfig,
) -> npt.NDArray[np.float64]:
    shape = generate_shape(shape_config)
    space = generate_virtual_space(space_config)

    cust_step = 5
    local_outline_path = None
    if scarfing_config.outline:
        if scarfing_config.outline_style == "step":
            local_outline_path = calc_step_outline(
                shape.discretize(custom_step=cust_step), robot_config.gamma
            )
        else:
            local_outline_path = calc_gradient_outline(
                shape.discretize(), robot_config.gamma
            )

    fill_blocks = []
    if scarfing_config.fill:
        print("Filling")
        last_polygon = shape.polygon

        # 1. GESTIONE CONCENTRICO
        if scarfing_config.concentric:
            shapes = calc_concentric_shapes(
                shape.polygon,
                scarfing_config.c_cycle,
                scarfing_config.c_offset,
                scarfing_config.c_offset_0,
                first=False,
                flatter=True,
            )
            if shapes:
                concentric_fill_sape = calc_gradient_outline(
                    concentric_plot_fill_path(
                        shapes,
                        cust_step,
                        robot_config.exit_quote,
                        isinstance(shape, ClosedSpline),
                    ),
                    robot_config.gamma,
                )
                last_polygon = shapes[-1]
                # Controllo di sicurezza prima dell'append
                if concentric_fill_sape is not None and len(concentric_fill_sape) > 0:
                    fill_blocks.append(concentric_fill_sape)

        # 2. GESTIONE FILL BASE
        base_fill = _fill_pocket(scarfing_config, last_polygon, robot_config)
        print(f"Punti del riempimento: {base_fill.shape[0]}")
        # Controllo di sicurezza: appendo solo se c'è effettivamente un percorso
        if base_fill is not None and base_fill.shape[0] > 0:
            fill_blocks.append(base_fill)

        # 3. GESTIONE RICORSIVA
        if scarfing_config.recursive:
            stepped_layers = calc_concentric_shapes(
                last_polygon,
                scarfing_config.r_cycle-1,
                scarfing_config.r_offset,
                scarfing_config.r_offset,
                first=False,
                flatter=True,
            )

            for index, current_layer in enumerate(stepped_layers):
                profondita_z = (index + 1) * scarfing_config.z_off

                coords = np.column_stack(current_layer.exterior.xy)
                outline_grezzo = discretize_points(coords, shape.max_step / 20)

                if scarfing_config.outline:
                    grad_outline = (
                        calc_gradient_outline(outline_grezzo, robot_config.gamma)
                        if scarfing_config.outline_style == "gradient"
                        else calc_step_outline(outline_grezzo, robot_config.gamma)
                    )

                    off = np.asarray([0, 0, robot_config.exit_quote, 0, 0, robot_config.exit_quote], dtype=np.float64)
                    gradino_outline = np.vstack((grad_outline[0] + off, grad_outline, grad_outline[-1] + off))
                else: 
                    grad_outline = None
                    
                gradino_fill = _fill_pocket(
                    scarfing_config, current_layer, robot_config
                )

                if gradino_fill is not None and len(gradino_fill) > 0:
                    if grad_outline is not None: 
                        gradino_completo = np.vstack((gradino_fill, gradino_outline))
                    else:
                        gradino_completo = gradino_fill.copy()
                else:
                    gradino_completo = gradino_outline.copy()

                # Abbassiamo solo la posizione Z (colonna 2)
                gradino_completo[:, 2] -= profondita_z
                gradino_completo[:, 5] -= profondita_z

                # ELIMINATO: gradino_completo[:, 5] -= profondita_z (Non alterare l'orientamento!)

                if len(gradino_completo) > 0:
                    fill_blocks.append(gradino_completo)

        # Ricostruiamo il percorso di riempimento solo se la lista contiene qualcosa
    local_fill_path = np.vstack(fill_blocks) if len(fill_blocks) > 0 else None
    # --- IL RESTO DEL CODICE RIMANE INVARIATO ---
    if local_outline_path is None and local_fill_path is None:
        return np.asarray([], dtype=np.float64)

    if local_outline_path is None:
        local_path = local_fill_path
    elif local_fill_path is None:
        local_path = local_outline_path
    else:
        local_path = np.vstack((local_fill_path, local_outline_path))

    return np.hstack(
        (
            space.local_to_global(local_path[:, :3]),
            space.local_to_global(local_path[:, 3:]),
        )
    )


def _fill_pocket(
    scarfing_config: ScarfingConfig, last_polygon: Polygon, robot_config: RobotConfig
) -> npt.NDArray[np.float64]:
    block = []
    if scarfing_config.fill_style == "grid":
        local_fill_path_horizontal = calc_gradient_outline(
            linear_rect_fill_path(
                calc_linear_intersection(
                    last_polygon,
                    float(scarfing_config.fill_dir % 360),
                    scarfing_config.fill_spacing,
                ),
                robot_config.exit_quote,
            ),
            -robot_config.gamma,
        )
        local_fill_path_vertical = calc_gradient_outline(
            linear_rect_fill_path(
                calc_linear_intersection(
                    last_polygon,
                    float((scarfing_config.fill_dir + 90) % 360),
                    scarfing_config.fill_spacing,
                ),
                robot_config.exit_quote,
            ),
            -robot_config.gamma,
        )

        local_linear_fill_path = np.vstack(
            (local_fill_path_horizontal, local_fill_path_vertical)
        )
        block.append(local_linear_fill_path)

    elif scarfing_config.fill_style == "rect":
        local_linear_fill_path = calc_gradient_outline(
            linear_rect_fill_path(
                calc_linear_intersection(
                    last_polygon,
                    float(scarfing_config.fill_dir % 360),
                    scarfing_config.fill_spacing,
                ),
                robot_config.exit_quote,
            ),
            -robot_config.gamma,
        )
        block.append(local_linear_fill_path)
    
    else: 
        pass

    return np.vstack(block) if block else np.empty((0, 3))
