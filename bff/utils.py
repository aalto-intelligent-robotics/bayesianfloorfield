import logging
from contextlib import nullcontext
from typing import Optional, Sequence, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms.functional import resize
from tqdm import tqdm, trange

from bff.nets import PeopleFlow
from mod.occupancy import OccupancyMap
from mod.utils import Direction

logger = logging.getLogger(__name__)

RowColumnPair = tuple[int, int]
DataPoint = tuple[torch.Tensor, torch.Tensor]
Loss = Union[torch.nn.MSELoss, torch.nn.KLDivLoss]


def plot_dir(
    occupancy: OccupancyMap,
    dynamics: np.ndarray,
    dir: Direction,
    dpi: int = 300,
    cmap: str = "inferno",
) -> None:
    """Plots the specied directional probabilities extracted from a dynamic
    map.

    Args:
        occupancy (OccupancyMap): The occupancy map to overlay on top of the
        dynamics.
        dynamics (np.ndarray): An array containing the 8-directional dynamics.
        dir (Direction): The direction for which the dynamics should be
        plotted.
        dpi (int, optional): The dots-per-inch (pixel per inch) for the created
        plot. Default is 300.
        cmap (str, optional): The colormap used to map normalized data values
        to RGB colors. Default is "inferno".
    """
    binary_map = occupancy.binary_map
    plt.figure(dpi=dpi)
    plt.title(f"Direction: {dir.name}")
    sz = occupancy.binary_map.size
    extent = 0, sz[0], 0, sz[1]
    plt.imshow(
        dynamics[..., dir.value], vmin=0, vmax=1, cmap=cmap, extent=extent
    )
    plt.imshow(
        np.ma.masked_where(np.array(binary_map) < 0.5, binary_map),
        vmin=0,
        vmax=1,
        cmap="gray",
        interpolation="none",
        extent=extent,
    )


def plot_quivers(
    occupancy: np.ndarray,
    dynamics: np.ndarray,
    scale: int = 1,
    window_size: Optional[int] = None,
    center: Optional[RowColumnPair] = None,
    normalize: bool = True,
    dpi: int = 300,
) -> None:
    """Plots the eight-direction people dynamics as quivers/arrow plots over
    the occupancy map.

    Args:
        occupancy (np.ndarray): The occupancy grid as a 2D numpy array.
        dynamics (np.ndarray): An array containing the 8-directional dynamics.
        scale (int, optional): Scaling factor used for reducing arrow density.
        Default is 1.
        window_size (int, optional): The size of the window to be plotted. If
        provided, a center must also be provided.
        center (RowColumnPair, optional): The center of the window to be
        plotted. If provided, a window size must also be provided.
        normalize (bool, optional): If True, the arrow scales are normalized.
        Default is True.
        dpi (int, optional): The dots-per-inch (pixel per inch) for the plot.
        Default is 300.
    """
    sz_occ = occupancy.shape
    sz_dyn = dynamics.shape
    assert sz_occ[0] // scale == sz_dyn[0] and sz_occ[1] // scale == sz_dyn[1]
    assert (window_size is None and center is None) or (
        window_size is not None and center is not None
    )
    assert sz_dyn[2] == 8

    if window_size is not None and center is not None:
        w_occ = Window(window_size)
        left, top, right, bottom = w_occ.corners(
            center, bounds=(0, sz_occ[0], 0, sz_occ[1])
        )
        w_dyn = Window(window_size // scale)
        center_dyn = (center[0] // scale, center[1] // scale)
        left_d, top_d, right_d, bottom_d = w_dyn.corners(
            center_dyn, bounds=(0, sz_dyn[0], 0, sz_dyn[1])
        )
        occ = occupancy[top:bottom, left:right]
        dyn = dynamics[top_d:bottom_d, left_d:right_d, ...]
        YX = np.mgrid[0 : window_size // scale, 0 : window_size // scale]
        extent = (
            -0.5,
            window_size // scale - 0.5,
            -0.5,
            window_size // scale - 0.5,
        )
    else:
        occ = occupancy
        dyn = dynamics
        YX = np.mgrid[0 : dyn.shape[0], 0 : dyn.shape[1]]
        extent = (-0.5, dyn.shape[0] - 0.5, -0.5, dyn.shape[1] - 0.5)
    dyn = dyn.reshape((-1, 8))
    if normalize:
        dyn = scale_quivers(dyn)

    plt.figure(dpi=dpi)
    Y: list[list[int]] = [[y] * 8 for y in YX[0].flatten()]
    X: list[list[int]] = [[x] * 8 for x in YX[1].flatten()]
    u = [d.u for d in Direction]
    v = [d.v for d in Direction]
    U = dyn * u
    V = dyn * v
    plt.quiver(X, Y, U, V, units="dots", minshaft=2, scale_units="xy", scale=2)
    plt.imshow(
        1 - occ,
        vmin=0,
        vmax=1,
        cmap="gray",
        interpolation="none",
        extent=extent,
    )


def scale_quivers(d: np.ndarray) -> np.ndarray:
    """Scales the quivers by normalizing the arrow lengths."""
    max = np.amax(d, axis=1)
    ret = np.expand_dims(np.where(max != 0, max, 1), axis=1)
    return d / ret


def estimate_dynamics(
    net: PeopleFlow,
    occupancy: Union[OccupancyMap, np.ndarray],
    scale: int = 1,
    net_scale: int = 1,
    # TODO: crop: Optional[Sequence[int]] = None,  # (left, top, right, bottom)
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Estimates the dynamics of people flow in a given occupancy.
    The function applies the network on windows of the provided occupancy to
    estimate the people flow dynamics in the region. It handles both
    OccupancyMap objects and numpy arrays representing occupancy. By default,
    if a scale factor greater than 1 is provided and the occupancy is an
    OccupancyMap, it will rescale the occupancy binary map accordingly before
    applying the network.

    Args:
        net (PeopleFlow): PeopleFlow network used for estimating the dynamics.
        occupancy (Union[OccupancyMap, np.ndarray]): Occupancy space to
        estimate the dynamics upon.
        scale (int, optional): Scale down factor for the occupancy map. This
        parameter is ignored when occupancy is a numpy array. Default is 1.
        net_scale (int, optional): Scale factor for the network window size.
        Default is 1.
        batch_size (int, optional): Number of patch samples to take from each
        batch. Default is 4.
        device (torch.device, optional): The device to move the network model
        and data to for computation. If not given, automatically set to GPU if
        available, else CPU.

    Returns:
        np.ndarray: An numpy array containing the estimated dynamics of people
        flow.
    """
    window = Window(net.window_size * net_scale)
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    if isinstance(occupancy, OccupancyMap):
        if scale > 1:
            occupancy_size = occupancy.binary_map.size
            padded_width = (-occupancy_size[0] % scale) + occupancy_size[0]
            padded_height = (-occupancy_size[1] % scale) + occupancy_size[1]
            bin_image = Image.new(
                occupancy.binary_map.mode, (padded_width, padded_height)
            )
            bin_image.paste(occupancy.binary_map, (0, 0))
            bin_map = np.array(
                bin_image.resize(
                    (bin_image.size[0] // scale, bin_image.size[1] // scale),
                    Image.ANTIALIAS,
                )
            )
        else:
            bin_map = np.array(occupancy.binary_map)
    else:
        if scale > 1:
            logger.warning(
                "Parameter scale is ignored when occupancy is a numpy array."
            )
        bin_map = occupancy
    h, w = bin_map.shape
    channels = 1
    padded_map = np.pad(bin_map, window.pad_amount)
    input = torch.from_numpy(padded_map)
    del bin_map, padded_map

    kh, kw = window.size, window.size  # kernel size
    dh, dw = 1, 1  # stride
    patches = input.unfold(0, kh, dh).unfold(1, kw, dw)
    ph, pw = patches.shape[0:2]
    patches_scaled = torch.empty(
        (ph * pw, channels, net.window_size, net.window_size),
        dtype=torch.float,
    )
    for row_i, patch in enumerate(tqdm(patches)):
        patches_scaled[row_i * pw : (row_i + 1) * pw, 0] = (
            resize(
                patch, size=(net.window_size, net.window_size), antialias=True
            )
            / 255.0
        )
    del input, patches

    num_pixels = patches_scaled.shape[0]
    empty_patch = torch.zeros(
        (1, 1, net.window_size, net.window_size), dtype=torch.float
    )
    empty_i = (patches_scaled == empty_patch).all(dim=3).all(dim=2).squeeze()
    nonempty_i = torch.logical_not(empty_i)
    nonempty_patches = patches_scaled[nonempty_i]
    del patches_scaled

    with torch.no_grad():
        nonempty_centers = torch.empty(
            (nonempty_patches.shape[0], net.out_channels)
        )
        empty_center: torch.Tensor = net(empty_patch.to(device=device))
        for i in trange(0, nonempty_patches.shape[0], batch_size):
            end = min(i + batch_size, nonempty_patches.shape[0])
            nonempty_centers[i:end] = net(
                nonempty_patches[i:end, ...].to(device=device)
            )
        del nonempty_patches, empty_patch

        centers = torch.empty((num_pixels, net.out_channels))
        centers[nonempty_i] = nonempty_centers.detach().cpu()
        centers[empty_i] = empty_center.detach().cpu()
        del nonempty_centers, empty_center, nonempty_i, empty_i
        map = centers.view((h, w, net.out_channels))
        del centers

    return map.numpy()


class Trainer:
    def __init__(
        self,
        net: PeopleFlow,
        optimizer: Optimizer,
        trainloader: DataLoader,
        valloader: Optional[DataLoader] = None,
        criterion: Loss = torch.nn.MSELoss(),
        device: torch.device = torch.device("cpu"),
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        self.device = device
        self.net = net
        self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.writer = writer
        self.train_epochs = 0

    def _step(
        self,
        data: DataPoint,
        training: bool,
    ) -> float:
        inputs, groundtruth = (
            data[0].to(self.device, dtype=torch.float),
            data[1].to(self.device, dtype=torch.float),
        )

        # zero the parameter gradients if training
        if training:
            self.optimizer.zero_grad()

        # forward
        outputs = self.net(inputs)
        if isinstance(self.criterion, torch.nn.KLDivLoss):
            outputs = F.log_softmax(outputs, dim=1)
        loss = self.criterion(outputs, groundtruth)

        # backward + optimize if training
        if training:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def _epoch(self, training: bool) -> float:
        if training:
            dataloader = self.trainloader
            cm = nullcontext()
        else:
            assert self.valloader is not None
            dataloader = self.valloader
            cm = torch.no_grad()  # type: ignore

        total_loss = 0.0
        for i, data in enumerate(dataloader):
            with cm:
                loss = self._step(data, training)
            total_loss += loss
            # logging statistics
            if self.writer and training:
                self.writer.add_scalar(
                    "loss/training",
                    loss,
                    self.train_epochs * len(dataloader) + i,
                )
        if training:
            self.train_epochs += 1
        return total_loss

    def training_epoch(self) -> float:
        self.net.train()
        return self._epoch(training=True)

    def validation_epoch(self) -> float:
        self.net.eval()
        return self._epoch(training=False)

    def train(self, epochs: int) -> None:
        prev_epochs = self.train_epochs
        if prev_epochs:
            logger.info(f"Recovering training from epoch {prev_epochs}")
        for epoch in range(epochs):
            train_loss = self.training_epoch()
            avg_train_loss = train_loss / len(self.trainloader)
            if self.valloader is not None:
                val_loss = self.validation_epoch()
                avg_val_loss = val_loss / len(self.valloader)
            else:
                avg_val_loss = np.nan

            # logging statistics
            if self.writer:
                self.writer.add_scalars(
                    "loss/epochs",
                    {"training": avg_train_loss, "validation": avg_val_loss},
                    prev_epochs + epoch + 1,
                )
            logger.info(
                f"[{prev_epochs + epoch + 1}] LOSS train {avg_train_loss:.3f},"
                f" validation {avg_val_loss:.3f}"
            )

    def save(self, path: str) -> None:
        torch.save(
            {
                "epoch": self.train_epochs,
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_epochs = checkpoint["epoch"]
        logger.debug(f"Loaded trainer checkpoint at epoch {self.train_epochs}")


class Window:
    """A class representing a window on a 2-dimensional grid."""

    def __init__(self, size: int) -> None:
        """Init `Window` class

        Args:
            size (int): The size of the square window, represented as the
            length of one side.
        """
        self.size = size

    @property
    def half_size(self) -> int:
        """Returns the integer division of the window size by 2"""
        return self.size // 2

    @property
    def center(self) -> RowColumnPair:
        """Returns the coordinate of the window center"""
        return (self.half_size, self.half_size)

    @property
    def pad_amount(self) -> Sequence[int]:
        """Returns the number of rows and columns to pad around the window"""
        return (self.half_size, self.half_size + self.size % 2 - 1)

    def corners(
        self, center: RowColumnPair, bounds: Optional[Sequence[int]] = None
    ) -> Sequence[int]:
        """Returns the corners of a window centered on the center.

        Args:
            center (RowColumnPair): The center coordinates of the window.
            bounds (Sequence[int], optional): If given, the provided
            corners will not exceed these bounds. Bounds should be given as
            (min_row, max_row, min_col, max_col). Defaults to None.
        Returns:
            Sequence[int]: The corner coordinates as (left, top, right, bottom)
        """
        left, top, right, bottom = (
            center[1] - self.half_size,  # left
            center[0] - self.half_size,  # top
            center[1] + self.half_size + self.size % 2,  # right
            center[0] + self.half_size + self.size % 2,  # bottom
        )
        if bounds:
            assert len(bounds) == 4
            min_row, max_row, min_col, max_col = bounds
            left = max(min_col, left)
            top = max(min_row, top)
            right = min(max_col, right)
            bottom = min(max_row, bottom)
        return (left, top, right, bottom)

    def indeces(
        self, center: RowColumnPair, bounds: Optional[Sequence[int]] = None
    ) -> Set[RowColumnPair]:
        """Generates the indices encompassed by the window centered in center.

        Args:
            center (RowColumnPair): The center coordinates of the window.
            bounds (Sequence[int], optional): If given, the provided indeces
            will not exceed these bounds. Bounds should be given as
            (min_row, max_row, min_col, max_col). Defaults to None.

        Returns:
            Set[RowColumnPair]: A set of row and column pairs representing each
            cell position within the window.
        """
        left, top, right, bottom = self.corners(center, bounds)
        indeces = {
            (row, col)
            for row in range(top, bottom)
            for col in range(left, right)
        }
        return indeces
