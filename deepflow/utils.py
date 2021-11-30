import logging
from contextlib import nullcontext
from typing import Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from mod.OccupancyMap import OccupancyMap
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

from deepflow.nets import PeopleFlow

RowColumnPair = Tuple[int, int]
DataPoint = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Loss = Union[torch.nn.MSELoss, torch.nn.KLDivLoss]

logger = logging.getLogger(__name__)


def estimate_dynamics(
    net: PeopleFlow,
    occupancy: OccupancyMap,
    window_size: int = 32,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> np.ndarray:

    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    bin_map = np.array(occupancy.binary_map)
    h, w = bin_map.shape
    channels = 1
    if window_size % 2:  # odd window
        pad_size = (window_size // 2, window_size // 2)
    else:  # even window
        pad_size = (window_size // 2 - 1, window_size // 2)
    padded_map = np.pad(bin_map, pad_size)
    input = torch.from_numpy(np.expand_dims(padded_map, axis=(0, 1)))

    kh, kw = window_size, window_size  # kernel size
    dh, dw = 1, 1  # stride

    patches = input.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(-1, channels, kh, kw)
    with torch.no_grad():
        centers = torch.empty((patches.shape[0], net.out_channels))
        for i in trange(0, patches.shape[0], batch_size):
            end = min(i + batch_size, patches.shape[0])
            batch = patches[i:end, ...].to(device, dtype=torch.float)
            output = net(batch)
            centers[i:end] = output[:, :, pad_size[0], pad_size[0]]
        map = centers.view((h, w, net.out_channels))

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
        inputs, groundtruth, _ = (
            data[0].to(self.device, dtype=torch.float),
            data[1].to(self.device, dtype=torch.float),
            data[2].to(self.device, dtype=torch.bool),
        )

        # zero the parameter gradients if training
        if training:
            self.optimizer.zero_grad()

        # forward
        outputs = self.net(inputs)
        loss = self.criterion(outputs, groundtruth.permute(0, 3, 1, 2))

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

    def corners(self, center: RowColumnPair) -> Sequence[int]:
        return (
            center[0] - self.half_size,  # left
            center[1] - self.half_size,  # top
            center[0] + self.half_size + self.size % 2,  # right
            center[1] + self.half_size + self.size % 2,  # bottom
        )

    def indeces(
        self, center: RowColumnPair, bounds: Optional[Sequence[int]] = None
    ) -> Set[RowColumnPair]:
        indeces = {
            (center[0] + row, center[1] + column)
            for row in range(-self.half_size, self.half_size + self.size % 2)
            for column in range(
                -self.half_size, self.half_size + self.size % 2
            )
        }
        if bounds:
            assert len(bounds) == 4
            min_row, max_row, min_col, max_col = (
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
            )
            indeces = {
                (
                    min(max(i[0], min_row), max_row),
                    min(max(i[1], min_col), max_col),
                )
                for i in indeces
            }
        return indeces
