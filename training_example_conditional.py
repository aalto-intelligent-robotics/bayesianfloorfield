# %% Some magic
# ! %load_ext autoreload
# ! %autoreload 2
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
    ConditionalDirectionalDataset,
    RandomHorizontalFlipPeopleFlow,
    RandomRotationPeopleFlow,
    RandomVerticalFlipPeopleFlow,
)
from deepflow.nets import ConditionalDiscreteDirectional
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

_, local = Helpers.get_local_settings(
    json_path="mod/config/local_settings.json",
    schema_path="mod/config/local_settings_schema.json",
)
occ_path = local["dataset_folder"] + "localization_grid.yaml"
train_path = local["pickle_folder"] + "conditional_directional.p"
test_path = local["pickle_folder"] + "conditional_directional_2.p"

occ = OccupancyMap.from_yaml(occ_path)
dyn_train: Grid.Grid = pickle.load(open(train_path, "rb"))
dyn_test: Grid.Grid = pickle.load(open(test_path, "rb"))

MapVisualisation(dyn_train, occ).show(occ_overlay=True)

# %%
window_size = 64
scale = 16

# transform = None
transform = transforms.Compose(
    [
        RandomRotationPeopleFlow(),
        RandomHorizontalFlipPeopleFlow(),
        RandomVerticalFlipPeopleFlow(),
    ]
)

id_string = (
    f"_cond_w{window_size}_s{scale}{'_t' if transform is not None else ''}"
)

trainset = ConditionalDirectionalDataset(
    occupancy=occ,
    dynamics=dyn_train,
    window_size=window_size,
    scale=scale,
    transform=transform,
)
valset = ConditionalDirectionalDataset(
    occupancy=occ, dynamics=dyn_test, window_size=window_size, scale=scale
)

net = ConditionalDiscreteDirectional(window_size)

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

# %% Train

trainer.train(epochs=100)

# %% Save network weights

path = f"./people_net{id_string}_{trainer.train_epochs}.pth"
trainer.save(path)

# %% Load network weights
# epochs = trainer.train_epochs
epochs = 100

path = f"./people_net{id_string}_{epochs}.pth"
trainer.load(path)

# %% Visualize a groundtruth

image, gt = trainset.get_by_center((40, 20))
outputs = np.zeros((window_size, window_size, 8))

enter_dir = Direction.E
exit_offset = enter_dir * 8
outputs[window_size // 2, window_size // 2, :] = gt[
    exit_offset : exit_offset + 8
]

plot_quivers(image[0] >= 1 / scale, outputs, dpi=1000)

# %% Visualize a sample output

image, _ = valset.get_by_center((40, 20))

outputs = estimate_dynamics(net, image[0], device=device, batch_size=32)

enter_dir = Direction.E
exit_offset = enter_dir * 8
plot_quivers(
    image[0] >= 1 / scale,
    outputs[:, :, exit_offset : exit_offset + 8],
    dpi=1000,
)

# %% Visualize output on random input

inputs = (
    random_input(size=32, p_occupied=0.1, device=device)
    .cpu()
    .numpy()[0, 0, ...]
)
outputs = estimate_dynamics(net, inputs, device=device, batch_size=32)

enter_dir = Direction.E
exit_offset = enter_dir * 8
plot_quivers(inputs, outputs[:, :, exit_offset : exit_offset + 8], dpi=1000)

# %% Build the full dynamic map

dyn_map = estimate_dynamics(net, occ, device=device, batch_size=100)

# %% Save full dynamic map
np.save(f"map{id_string}.npy", dyn_map)

# %% Load a saved full dynamic map
dyn_map = np.load(f"map{id_string}.npy")

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
