import logging
from contextlib import nullcontext
from enum import IntEnum
from typing import Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from mod.OccupancyMap import OccupancyMap
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

from deepflow.nets import PeopleFlow

logger = logging.getLogger(__name__)

RowColumnPair = Tuple[int, int]
DataPoint = Tuple[torch.Tensor, torch.Tensor]
Loss = Union[torch.nn.MSELoss, torch.nn.KLDivLoss]


class Direction(IntEnum):
    E = 0
    NE = 1
    N = 2
    NW = 3
    W = 4
    SW = 5
    S = 6
    SE = 7

    def rad(self) -> float:
        return self.value * 2 * np.pi / 8

    def u(self) -> float:
        return np.cos(self.rad())

    def v(self) -> float:
        return np.sin(self.rad())

    def uv(self) -> tuple[float, float]:
        return (self.u(), self.v())


def plot_dir(
    occupancy: OccupancyMap,
    dynamics: np.ndarray,
    dir: Direction,
    dpi: int = 300,
    cmap: str = "hot",
):
    binary_map = occupancy.binary_map
    plt.figure(dpi=dpi)
    plt.title(f"Direction: {dir.name}")
    plt.imshow(dynamics[..., dir.value], vmin=0, vmax=1, cmap=cmap)
    plt.imshow(
        np.ma.masked_where(np.array(binary_map) < 255, binary_map),
        vmin=0,
        vmax=255,
        cmap="gray",
        interpolation="none",
    )


def plot_quivers(
    occupancy: np.ndarray,
    dynamics: np.ndarray,
    enter_dir: Optional[Direction] = None,
    window_size: Optional[int] = None,
    center: Optional[RowColumnPair] = None,
    normalize: bool = True,
    dpi: int = 300,
):
    sz = dynamics.shape
    assert occupancy.shape == sz[0:2]
    assert (window_size is None and center is None) or (
        window_size is not None and center is not None
    )
    if enter_dir is None:
        assert sz[2] == 8
    else:
        e = enter_dir.value * 8
        dynamics = dynamics[..., e : e + 8]

    if window_size is not None and center is not None:
        w = Window(window_size)
        left, top, right, bottom = w.corners(
            center, bounds=(0, sz[0], 0, sz[1])
        )
        occ = occupancy[top:bottom, left:right]
        dyn = dynamics[top:bottom, left:right, ...]
    else:
        occ = occupancy
        dyn = dynamics
    dyn = dyn.reshape((-1, 8))
    if normalize:
        dyn = scale_quivers(dyn)

    plt.figure(dpi=dpi)
    YX = np.mgrid[0 : occ.shape[0], 0 : occ.shape[1]]
    Y: list[list[int]] = [[y] * 8 for y in YX[0].flatten()]
    X: list[list[int]] = [[x] * 8 for x in YX[1].flatten()]
    u = [d.u() for d in Direction]
    v = [d.v() for d in Direction]
    U = dyn * u
    V = dyn * v
    plt.quiver(X, Y, U, V, units="dots", minshaft=2, scale_units="xy", scale=2)
    plt.imshow(
        np.ma.masked_where(occ < 255, 255 - occ),
        vmin=0,
        vmax=255,
        cmap="gray",
        interpolation="none",
    )


def scale_quivers(d: np.ndarray) -> np.ndarray:
    max = np.amax(d, axis=1)
    ret = np.expand_dims(np.where(max != 0, max, 1), axis=1)
    return d / ret


def random_input(
    size: int = 32,
    p_occupied: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    w = Window(size)
    a = torch.rand((1, 1, size, size), device=device) < p_occupied
    a[0, 0, w.center[0], w.center[1]] = 0  # ensure center is empty
    return a.type(torch.float)


def estimate_dynamics(
    net: PeopleFlow,
    occupancy: OccupancyMap,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    window = Window(net.window_size)
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    bin_map = np.array(occupancy.binary_map)
    h, w = bin_map.shape
    channels = 1
    padded_map = np.pad(bin_map, window.pad_amount)
    input = torch.from_numpy(np.expand_dims(padded_map, axis=(0, 1)))
    del bin_map, padded_map

    kh, kw = window.size, window.size  # kernel size
    dh, dw = 1, 1  # stride
    patches = input.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(-1, channels, kh, kw)
    del input

    num_pixels = patches.shape[0]
    empty_patch = torch.zeros(
        (1, 1, net.window_size, net.window_size), dtype=torch.uint8
    )
    empty_i = (patches == empty_patch).all(dim=3).all(dim=2).squeeze()
    nonempty_i = torch.logical_not(empty_i)
    nonempty_patches = patches[nonempty_i]
    del patches

    with torch.no_grad():
        nonempty_centers = torch.empty(
            (nonempty_patches.shape[0], net.out_channels)
        )
        empty_center = net(empty_patch.to(device, dtype=torch.float))[
            :, :, window.center[0], window.center[1]
        ]
        for i in trange(0, nonempty_patches.shape[0], batch_size):
            end = min(i + batch_size, nonempty_patches.shape[0])
            batch = nonempty_patches[i:end, ...].to(device, dtype=torch.float)
            output = net(batch)
            nonempty_centers[i:end] = output[
                :, :, window.center[0], window.center[1]
            ]
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
        self.cr, self.cc = Window(net.window_size).center

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
        loss = self.criterion(outputs[..., self.cr, self.cc], groundtruth)

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
            cm = torch.no_grad()

        total_loss = 0.0
        for i, data in enumerate(dataloader, 0):
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
    def __init__(self, size: int) -> None:
        self.size = size

    @property
    def half_size(self) -> int:
        return self.size // 2

    @property
    def center(self) -> RowColumnPair:
        return (self.half_size, self.half_size)

    @property
    def pad_amount(self) -> Sequence[int]:
        return (self.half_size, self.half_size + self.size % 2 - 1)

    def corners(
        self, center: RowColumnPair, bounds: Optional[Sequence[int]] = None
    ) -> Sequence[int]:
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
        left, top, right, bottom = self.corners(center, bounds)
        indeces = {
            (row, col)
            for row in range(top, bottom)
            for col in range(left, right)
        }
        return indeces
