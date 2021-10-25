# %% Imports

import pickle
import sys

import torch
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from deepflow.data import DiscreteDirectionalDataset
from deepflow.models import DiscreteDirectional
from mod import Grid, Helpers, Models
from mod.OccupancyMap import OccupancyMap
from mod.Visualisation import MapVisualisation

# %% Network and dataset setup

sys.modules["Grid"] = Grid
sys.modules["Models"] = Models

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

net = DiscreteDirectional()

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
testset = DiscreteDirectionalDataset(occ, dyn_test)

batch_size = 4

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

# %% Enable CUDA (if available)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)

# %% Train

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

testiter = iter(testloader)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, masks = (
            data[0].to(device, dtype=torch.float),
            data[1].to(device, dtype=torch.float),
            data[2].to(device, dtype=torch.bool),
        )

        data_test = next(testiter)
        inputs_test, labels_test, masks_test = (
            data_test[0].to(device, dtype=torch.float),
            data_test[1].to(device, dtype=torch.float),
            data_test[2].to(device, dtype=torch.bool),
        )

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.unsqueeze(1))
        loss = criterion(outputs, labels.permute(0, 3, 1, 2))
        loss.backward()
        optimizer.step()

        outputs_test = net(inputs_test.unsqueeze(1))
        loss_test = criterion(outputs_test, labels_test.permute(0, 3, 1, 2))

        # print statistics
        running_loss += loss.item()
        writer.add_scalars(
            "loss/training",
            {"training": loss.item(), "validation": loss_test.item()},
            i,
        )
        # writer.add_scalar("loss/testing", loss_test.item(), i)
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(
                "[%d, %5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / 2000)
            )
            running_loss = 0.0

print("Finished Training")

# %% Save network weights

PATH = "./people_net.pth"
torch.save(net.state_dict(), PATH)
