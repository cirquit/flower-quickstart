import argparse
import warnings
from collections import OrderedDict

import flwr as fl
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import wandb
import fire

import models
import shared

warnings.filterwarnings("ignore", category=UserWarning)

def train(model, trainloader, epochs):
    """Train the model on the training set."""
    device = shared.get_device()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epoch_loss = 0
    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        wandb.log({"epoch_loss": epoch_loss / len(trainloader.dataset)})
        epoch_loss = 0

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model: nn.Module, trainloader: DataLoader, testloader: DataLoader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        super().__init__()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = shared.numpy_to_state_dict(model=self.model, parameters=parameters)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

def main():
    model = models.Net().to(shared.get_device())
    trainloader, testloader = shared.get_cifar10_dataloader(
        partitioners={"train": wandb.config.client_count},
        test_train_split=wandb.config.test_train_split,
        node_id=wandb.config.node_id)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(model=model, trainloader=trainloader, testloader=testloader),
    )
    wandb.finish()

def setup_wandb(
        group_name: str,
        node_id: int,
        client_count: int,
        test_train_split: float = 0.2, # 80/20 train/test
        project_name: str = "flower-quickstart") -> None:

    wandb.init(
        project=project_name,
        group=group_name,
        name=f"{group_name}-client-{node_id}",
        config={
            "node_id": node_id,
            "client_count": client_count,
            "test_train_split": test_train_split
        }
    )

if __name__ == '__main__':
    fire.Fire(setup_wandb)
    main()
