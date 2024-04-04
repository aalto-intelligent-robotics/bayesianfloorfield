from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from mod.grid import Grid
from mod.models import BayesianDiscreteDirectional, DiscreteDirectional
from mod.occupancy import OccupancyMap


def polar2cart(theta: float, r: float) -> tuple[float, float]:
    z = r * np.exp(1j * theta)
    return np.real(z), np.imag(z)


def show_all(
    grid: Grid,
    occ: Optional[OccupancyMap] = None,
    occ_overlay: bool = False,
    dpi: int = 100,
) -> None:
    show_raw(grid, dpi)
    if (
        grid.model == DiscreteDirectional
        or grid.model == BayesianDiscreteDirectional
    ):
        show_discrete_directional(grid, occ, occ_overlay, dpi)
    plt.show()


def show_raw(grid: Grid, dpi: int = 100) -> None:
    plt.figure(dpi=dpi)

    for cell in grid.cells.values():
        plt.plot(cell.data["x"], cell.data["y"], "r,")


def show_discrete_directional(
    grid: Grid,
    occ: Optional[OccupancyMap] = None,
    occ_overlay: bool = False,
    dpi: int = 100,
    save_name: Optional[str] = None,
) -> None:
    plt.figure(dpi=dpi)
    X = []
    Y = []
    U = []
    V = []
    for key, cell in grid.cells.items():
        assert isinstance(cell, DiscreteDirectional)
        normalized_bins = cell.bins / np.max(cell.bins)
        X.append([key.column] * 8)
        Y.append([key.row] * 8)
        U.append(
            [
                polar2cart(cell.directions[i], normalized_bins[i])[0]
                for i in range(8)
            ]
        )
        V.append(
            [
                polar2cart(cell.directions[i], normalized_bins[i])[1]
                for i in range(8)
            ]
        )
    if occ and occ_overlay:
        show_occupancy(occ)
    plt.quiver(X, Y, U, V, scale_units="xy", scale=2, minshaft=2, minlength=0)
    if save_name:
        plt.savefig(save_name + ".eps", format="eps")


def show_occupancy(occ: OccupancyMap) -> None:
    r = occ.resolution
    o = occ.origin
    sz = occ.map.size
    extent = (o[0], o[0] + sz[0] * r, o[1], o[1] + sz[1] * r)
    plt.imshow(occ.map, extent=extent, cmap="gray")
