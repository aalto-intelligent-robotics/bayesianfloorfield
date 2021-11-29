# %% Imports

import pickle
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from deepflow.data import DiscreteDirectionalDataset
from deepflow.nets import DiscreteDirectional
from deepflow.utils import Trainer, estimate_dynamics
from mod import Grid, Helpers, Models
from mod.OccupancyMap import OccupancyMap
from mod.Visualisation import MapVisualisation

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

trainset = DiscreteDirectionalDataset(occ, dyn_train)
valset = DiscreteDirectionalDataset(occ, dyn_test)

net = DiscreteDirectional()

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
    # valloader=valloader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    writer=writer,
)

# %% Train

trainer.train(epochs=5)

# %% Save network weights

path = "./people_net.pth"
torch.save(trainer.net, path)

# %% Build the full dynamic map

dyn_map = estimate_dynamics(net, occ)
