# %% Some magic
# ! %load_ext autoreload
# ! %autoreload 2
# %% Imports

import logging
import pickle
import sys
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from directionalflow.data import (
    DiscreteDirectionalDataset,
    RandomHorizontalFlipPeopleFlow,
    RandomRotationPeopleFlow,
    RandomVerticalFlipPeopleFlow,
)
from directionalflow.nets import DiscreteDirectional
from directionalflow.utils import Trainer
from mod import grid, models
from mod.occupancy import OccupancyMap
from mod.visualisation import show_all

logging.basicConfig(level=logging.INFO)

# %% Network and dataset setup

sys.modules["Grid"] = grid
sys.modules["Models"] = models

# Change BASE_PATH to the folder where data and models are located
BASE_PATH = Path("/mnt/hdd/datasets/ATC/")

MAP_METADATA = BASE_PATH / "localization_grid.yaml"
GRID_TRAIN_DATA = (
    BASE_PATH
    / "models"
    / "bayes"
    / "20121114"
    / "discrete_directional_20121114_3121209.p"
)
GRID_TEST_DATA = (
    BASE_PATH
    / "models"
    / "bayes"
    / "20121118"
    / "discrete_directional_20121118_8533469.p"
)

NET_EPOCHS = 120
NET_WINDOW_SIZE = 64
NET_SCALE_FACTOR = 20
NET_BATCH_SIZE = 32

TRAINER_PREVIOUS_EPOCHS = 0  # to continue previous training

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PLOT_DPI = 800

occupancy = OccupancyMap.from_yaml(MAP_METADATA)
grid_train: grid.Grid = pickle.load(open(GRID_TRAIN_DATA, "rb"))
grid_test: grid.Grid = pickle.load(open(GRID_TEST_DATA, "rb"))

show_all(grid_train, occupancy, occ_overlay=True, dpi=PLOT_DPI)

# transform = None
transform = transforms.Compose(
    [
        RandomRotationPeopleFlow(),
        RandomHorizontalFlipPeopleFlow(),
        RandomVerticalFlipPeopleFlow(),
    ]
)

net_id_string = (
    f"_w{NET_WINDOW_SIZE}_s{NET_SCALE_FACTOR}"
    f"{'_t' if transform is not None else ''}"
)

trainset = DiscreteDirectionalDataset(
    occupancy=occupancy,
    dynamics=grid_train,
    window_size=NET_WINDOW_SIZE,
    scale=NET_SCALE_FACTOR,
    transform=transform,
)
valset = DiscreteDirectionalDataset(
    occupancy=occupancy,
    dynamics=grid_test,
    window_size=NET_WINDOW_SIZE,
    scale=NET_SCALE_FACTOR,
)

net = DiscreteDirectional(NET_WINDOW_SIZE)

# %% Training context

trainloader = DataLoader(
    trainset, batch_size=NET_BATCH_SIZE, shuffle=True, num_workers=2
)

valloader = DataLoader(
    valset, batch_size=NET_BATCH_SIZE, shuffle=False, num_workers=2
)

criterion = torch.nn.MSELoss()
# criterion = torch.nn.KLDivLoss(reduction="batchmean")
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(comment=net_id_string)

trainer = Trainer(
    net=net,
    trainloader=trainloader,
    valloader=valloader,
    optimizer=optimizer,
    criterion=criterion,
    device=DEVICE,
    writer=writer,
)

# %% Load network weights if continuing previous training

if TRAINER_PREVIOUS_EPOCHS:
    path = f"models/people_net{net_id_string}_{TRAINER_PREVIOUS_EPOCHS}.pth"
    trainer.load(path)

# %% Train

trainer.train(epochs=NET_EPOCHS)

# %% Save network weights

path = f"models/people_net{net_id_string}_{trainer.train_epochs}.pth"
trainer.save(path)
