# %% Imports

import logging
import pickle
import sys

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from deepflow.data import (
    DiscreteDirectionalDataset,
    RandomHorizontalFlipPeopleFlow,
    RandomVerticalFlipPeopleFlow,
)
from deepflow.nets import DiscreteDirectional
from deepflow.utils import (
    Direction,
    Trainer,
    estimate_dynamics,
    plot_dir,
    plot_quivers,
    random_input,
)
from mod import Grid, Helpers, Models
from mod.OccupancyMap import OccupancyMap
from mod.Visualisation import MapVisualisation

logging.basicConfig(level=logging.INFO)

# %% Network and dataset setup

sys.modules["Grid"] = Grid
sys.modules["Models"] = Models

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


_, local = Helpers.get_local_settings(
    json_path="mod/config/local_settings.json",
    schema_path="mod/config/local_settings_schema.json",
)
occ_path = local["dataset_folder"] + "localization_grid.yaml"
train_path = local["pickle_folder"] + "discrete_directional.p"
test_path = local["pickle_folder"] + "discrete_directional_2_small.p"

occ = OccupancyMap.from_yaml(occ_path)
dyn_train: Grid.Grid = pickle.load(open(train_path, "rb"))
dyn_test: Grid.Grid = pickle.load(open(test_path, "rb"))

MapVisualisation(dyn_train, occ).show(occ_overlay=True)

# %%
window_size = 64

transform = transforms.Compose(
    [RandomHorizontalFlipPeopleFlow(), RandomVerticalFlipPeopleFlow()]
)

trainset = DiscreteDirectionalDataset(
    occupancy=occ,
    dynamics=dyn_train,
    window_size=window_size,
    transform=transform,
)
valset = DiscreteDirectionalDataset(
    occupancy=occ, dynamics=dyn_test, window_size=window_size
)

net = DiscreteDirectional(window_size)

# %% Training context

batch_size = 32

trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

valloader = DataLoader(
    valset, batch_size=batch_size, shuffle=False, num_workers=2
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


trainer = Trainer(
    net=net,
    trainloader=trainloader,
    valloader=valloader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    writer=writer,
)

# %% Train

trainer.train(epochs=100)

# %% Save network weights

path = "./people_net.pth"
trainer.save(path)

# %% Load network weights

path = "./people_net.pth"
trainer.load(path)

# %% Visualize output on random input

inputs = (
    random_input(size=32, p_occupied=0.1, device=device)
    .cpu()
    .numpy()[0, 0, ...]
)
outputs = estimate_dynamics(net, inputs, device=device, batch_size=32)

plot_quivers(inputs * 255, outputs, dpi=1000)

# %% Build the full dynamic map

dyn_map = estimate_dynamics(net, occ, device=device, batch_size=100)

# %% Save full dynamic map
np.save("map.npy", dyn_map)

# %% Load a saved full dynamic map
dyn_map = np.load("map.npy")

# %% Visualize

plot_dir(occ, dyn_map, Direction.N)
plot_dir(occ, dyn_map, Direction.E)
plot_dir(occ, dyn_map, Direction.S)
plot_dir(occ, dyn_map, Direction.W)

# %% Visualize quiver

center = (200, 530)
w = 64
plot_quivers(
    np.array(occ.binary_map), dyn_map, dpi=1000, center=center, window_size=w
)

# %%
