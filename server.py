from typing import List, Tuple, Optional, Dict
from collections import OrderedDict
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import flwr as fl
from flwr.common import Metrics

import wandb
import fire
from tqdm import tqdm
import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from torchvision.transforms import Compose, Normalize, ToTensor

import models
import shared

class SplitLRFedAvg(fl.server.strategy.FedAvg):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

def get_evaluate_fn(model: nn.Module):

    _, testloader = shared.get_cifar10_dataloader(
        partitioners={"test": 1},
        test_train_split=1.0
    )

    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """ """
        model.load_state_dict(
            shared.numpy_to_state_dict(
                model=model,
                parameters=parameters),
            strict=True)
        loss, accuracy = shared.test(model=model, testloader=testloader)
        wandb.log({"full_test_accuracy": accuracy})
        return loss, {"full_test_accuracy": accuracy}
    return evaluate

def main():

    device = shared.get_device()
    model = models.Net().to(device)
    strategy = SplitLRFedAvg(
            evaluate_fn=get_evaluate_fn(model=model)
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=wandb.config.num_rounds),
        strategy=strategy,
    )
    wandb.finish()

def setup_wandb(
        group_name: str,
        num_rounds: int,
        project_name: str = "flower-quickstart"
    ) -> None:

    wandb.init(
        project=project_name,
        group=group_name,
        name=f"{group_name}-server",
        config={
            "num_rounds": num_rounds
        }
    )

if __name__ == '__main__':
   fire.Fire(setup_wandb)
   main()
