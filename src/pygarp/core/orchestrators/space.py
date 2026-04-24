import numpy as np

from pygarp.core.models.validators import SpaceConfig, EditConfig
from pygarp.core.models.virtual_space.core import WorkingSpace

"Define a coordinate space based on the origin, point X, and a secondary point within the OX plane"
"Define a coordinate system based on the origin, point X, and the normal to the OX plane."
"Define a coordinate system using X/Y axes and points x/y."


def generate_default(o, x, y, z) -> WorkingSpace:
    return WorkingSpace(o, x, y, z)


def generate_default_space():
    return WorkingSpace(np.zeros(3, dtype=np.float64), (1, 0, 0), (0, 1, 0), (0, 0, 1))


def generate_virtual_space(space_config: SpaceConfig) -> WorkingSpace:
    if space_config.strategy == "OPP":
        return WorkingSpace.new_space_from_three_points(
            space_config.origin,
            space_config.x_hint,
            space_config.p_hint,
        )
    elif space_config.strategy == "ONC":
        return WorkingSpace.new_space_from_normal(
            space_config.origin,
            space_config.x_hint,
            space_config.z,
        )
    elif space_config.strategy == "XYP":
        return WorkingSpace.new_space_from_two_straight(
            space_config.x,
            space_config.x_hint,
            space_config.y,
            space_config.y_hint,
        )
    elif space_config.strategy == "DFT":
        return generate_default_space()

    else:
        raise ValueError("Strategia di definizione del piano non supportata")


def edit_virtual_space(edit_config: EditConfig, space: WorkingSpace) -> WorkingSpace:
    if edit_config.x:
        space.invert_x()
    if edit_config.y:
        space.invert_y()
    if edit_config.z:
        space.invert_z()
    if edit_config.x2y:
        space.exchange_plane()

    return space
