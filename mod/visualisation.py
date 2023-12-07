from typing import Optional

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

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
) -> None:
    plt.figure(dpi=dpi)
    X = []
    Y = []
    U = []
    V = []
    C = []
    P = []
    for key, cell in grid.cells.items():
        assert isinstance(cell, DiscreteDirectional)
        for i, bin in enumerate(cell.bins):
            if bin.probability > 0:
                X.append(key.column)
                Y.append(key.row)
                u, v = polar2cart(cell.directions[i], bin.probability)
                U.append(u)
                V.append(v)
                C.append(bin.probability)
                P.append(cell.probability)
    P = list(np.array(P) / max(P))
    norm = Normalize()
    norm.autoscale(C)
    colormap = colormaps["cividis"]
    if occ and occ_overlay:
        show_occupancy(occ)
    plt.quiver(
        X,
        Y,
        U,
        V,
        color=colormap(norm(C)),
        angles="xy",
        scale_units="xy",
        scale=1,
        minshaft=2,
        # alpha=P,
    )


def show_occupancy(occ: OccupancyMap) -> None:
    r = occ.resolution
    o = occ.origin
    sz = occ.map.size
    extent = (o[0], o[0] + sz[0] * r, o[1], o[1] + sz[1] * r)
    plt.imshow(occ.map, extent=extent, cmap="gray")
