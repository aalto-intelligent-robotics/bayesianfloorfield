# %% Some magic
# ! %load_ext autoreload
# ! %autoreload 2
# %% Imports

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

from directionalflow.data import (
    DiscreteDirectionalDataset,
    RandomHorizontalFlipPeopleFlow,
    RandomRotationPeopleFlow,
    RandomVerticalFlipPeopleFlow,
)
from directionalflow.nets import DiscreteDirectional
from directionalflow.utils import (
    Direction,
    Trainer,
    estimate_dynamics,
    plot_dir,
    plot_quivers,
    random_input,
)
from mod import grid, models
from mod.occupancy import OccupancyMap
from mod.visualisation import show_all

logging.basicConfig(level=logging.INFO)

# %% Network and dataset setup

sys.modules["Grid"] = grid
sys.modules["Models"] = models

# Change BASE_PATH to the folder where data and models are located
BASE_PATH = Path("/home/francesco/deep-flow/data/ATC/")
BASE_PATH_SYNTH = Path("/home/francesco/deep-flow/data/Rex/")

MAP_METADATA = BASE_PATH / "localization_grid.yaml"
MAP_METADATA_SYNTH = BASE_PATH_SYNTH / "randomenvimg_1_border.yaml"
GRID_TRAIN_DATA = BASE_PATH / "models" / "discrete_directional.p"
GRID_TRAIN_SYNTH_DATA = (
    BASE_PATH_SYNTH / "discrete_directional_randomenv_1_x5.p"
)
GRID_TEST_DATA = BASE_PATH / "models" / "discrete_directional_2.p"

occ = OccupancyMap.from_yaml(MAP_METADATA)
occ_synth = OccupancyMap.from_yaml(MAP_METADATA_SYNTH)
dyn_train: grid.Grid = pickle.load(open(GRID_TRAIN_DATA, "rb"))
dyn_train_synth: grid.Grid = pickle.load(open(GRID_TRAIN_SYNTH_DATA, "rb"))
dyn_test: grid.Grid = pickle.load(open(GRID_TEST_DATA, "rb"))

show_all(dyn_train, occ, occ_overlay=True)

# %%
window_size = 64
scale = 8
scale_synth = 8 / 5

# transform = None
transform = transforms.Compose(
    [
        RandomRotationPeopleFlow(),
        RandomHorizontalFlipPeopleFlow(),
        RandomVerticalFlipPeopleFlow(),
    ]
)

id_string = (
    f"_w{window_size}_s{scale}_synthatc{'_t' if transform is not None else ''}"
)

trainset = DiscreteDirectionalDataset(
    occupancy=occ,
    dynamics=dyn_train,
    window_size=window_size,
    scale=scale,
    transform=transform,
)
trainset_synth = DiscreteDirectionalDataset(
    occupancy=occ_synth,
    dynamics=dyn_train_synth,
    window_size=window_size,
    scale=scale_synth,
    transform=transform,
)
valset = DiscreteDirectionalDataset(
    occupancy=occ, dynamics=dyn_test, window_size=window_size, scale=scale
)

trainset_combined: ConcatDataset = ConcatDataset([trainset, trainset_synth])
trainset_weights = [len(trainset_combined) / len(trainset)] * len(trainset) + [
    len(trainset_combined) / len(trainset_synth)
] * len(trainset_synth)
weighted_sampler = WeightedRandomSampler(
    weights=trainset_weights, num_samples=len(trainset_weights)
)

net = DiscreteDirectional(window_size)

# %% Training context

batch_size = 32

trainloader = DataLoader(
    trainset_combined,
    batch_size=batch_size,
    # shuffle=True,
    sampler=weighted_sampler,
    num_workers=2,
)

valloader = DataLoader(
    valset, batch_size=batch_size, shuffle=False, num_workers=2
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.MSELoss()
# criterion = torch.nn.KLDivLoss(reduction="batchmean")
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(comment=id_string)

trainer = Trainer(
    net=net,
    trainloader=trainloader,
    valloader=valloader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    writer=writer,
)

# %% Show sample input

plt.imshow(next(iter(trainloader))[0][0][0], cmap="gray")
plt.plot(32, 32, "o", markersize=1)

# %% Train

trainer.train(epochs=100)

# %% Save network weights

path = f"models/people_net{id_string}_{trainer.train_epochs}.pth"
trainer.save(path)

# %% Load network weights
# epochs = trainer.train_epochs
epochs = 100

path = f"models/people_net{id_string}_{epochs}.pth"
trainer.load(path)

# %% Visualize a groundtruth

image, gt = trainset.get_by_center((40, 20))
outputs = np.zeros((window_size, window_size, 8))
outputs[window_size // 2, window_size // 2, :] = gt

plot_quivers(image[0] >= 1 / scale, outputs, dpi=1000)

# %% Visualize a sample output

image, _ = valset.get_by_center((40, 20))

outputs = estimate_dynamics(net, image[0], device=device, batch_size=32)

plot_quivers(image[0] >= 1 / scale, outputs, dpi=1000)

# %% Visualize output on random input

inputs = (
    random_input(size=32, p_occupied=0.1, device=device)
    .cpu()
    .numpy()[0, 0, ...]
)
outputs = estimate_dynamics(net, inputs, device=device, batch_size=32)

plot_quivers(inputs, outputs, dpi=1000)

# %% Build the full dynamic map

dyn_map = estimate_dynamics(
    net, occ, device=device, batch_size=100, scale=scale
)

# %% Save full dynamic map
np.save(f"maps/map{id_string}.npy", dyn_map)

# %% Load a saved full dynamic map
dyn_map = np.load(f"maps/map{id_string}.npy")

# %% Visualize

plot_dir(occ, dyn_map, Direction.NW)
plot_dir(occ, dyn_map, Direction.NE)
plot_dir(occ, dyn_map, Direction.SW)
plot_dir(occ, dyn_map, Direction.SE)

# %% Visualize quiver

center = (400, 730)
w = 64
plot_quivers(
    np.array(occ.binary_map) >= 1 / scale,
    dyn_map,
    scale=scale,
    dpi=1000,
    center=center,
    window_size=w * 4,
)

# %%
