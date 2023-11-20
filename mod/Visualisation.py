import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from . import Models

# TODO add logging and manual
# TODO fix colormap depending on the probability


def polar2cart(theta, r):
    z = r * np.exp(1j * theta)
    return np.real(z), np.imag(z)


class MapVisualisation:
    def __init__(self, mod=None, occ=None):
        self.mod = mod
        self.occ = occ
        self.vis_map = []

    def show(self, occ_overlay=False, dpi=100):
        self.__show_raw(dpi)
        if (
            self.mod.model == Models.DiscreteDirectional
            or self.mod.model == Models.BayesianDiscreteDirectional
        ):
            self.__show_discrete_directional(occ_overlay, dpi)

        plt.show()

    def __show_raw(self, dpi=100):
        plt.figure(dpi=dpi)

        for row_id in range(self.mod.dimensions[0]):
            for col_id in range(self.mod.dimensions[1]):
                key = (row_id, col_id)
                if key in self.mod.cells:
                    plt.plot(
                        self.mod.cells[key].data["x"],
                        self.mod.cells[key].data["y"],
                        "r.",
                    )

    def __show_discrete_directional(self, occ_overlay=False, dpi=100):
        plt.figure(dpi=dpi)
        X = []
        Y = []
        U = []
        V = []
        C = []
        P = []
        for key, cell in self.mod.cells.items():
            for b in cell.bins:
                if cell.bins[b]["probability"] > 0:
                    X.append(key[1])
                    Y.append(key[0])
                    u, v = polar2cart(b, cell.bins[b]["probability"])
                    U.append(u)
                    V.append(v)
                    C.append(cell.bins[b]["probability"])
                    P.append(cell.probability)
        P = list(np.array(P) / max(P))
        norm = Normalize()
        norm.autoscale(C)
        colormap = cm.cividis
        if occ_overlay:
            self.__show_occupancy()
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

    def __show_occupancy(self):
        if self.occ is not None:
            r = self.occ.resolution
            o = self.occ.origin
            sz = self.occ.map.size
            extent = (o[0], o[0] + sz[0] * r, o[1], o[1] + sz[1] * r)
            plt.imshow(self.occ.map, extent=extent, cmap="gray")
