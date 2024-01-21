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
    Metrics
)

import flwr as fl
from flwr.server.client_manager import ClientManager, ClientProxy
from logging import INFO, DEBUG
from flwr.common.logger import log

import wandb
import fire
from tqdm import tqdm
import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from torchvision.transforms import Compose, Normalize, ToTensor
import random
import copy
import models
import shared

class SplitLRFedAvg(fl.server.strategy.FedAvg):

    def __init__(
        self,
        client_percentage: float,
        lr_factor: float,
        force_global_split: bool = False,
        force_local_split: bool = False,
        **kwargs
    ):
        """ Splits the clients (deterministically) into two groups by `client_percentage`
        and saves this split in `client_map`
        Group `client_percentage`: Learning rate will be multiplied by `pow(lr_factor, server_rounds)` 
        Group `1-client_percentage`: Constant client-side learning rate

        The remaining functionality is kept the same, on_fit_config can still be applied.

        WARNING: Due to clients being sampled, some may be selected further in the training.
            Therefore their lr_factor is defined as pow(lr_factor, server_rounds) so all clients that are 
            doing adaptive lr have the same lr.

        WARNING: Probabilistic group associtation works well with many clients, but can be catastrophic
        with less clients. `force_global_split` tries to adhere to the client_percentage mathematically based on
        ALL previously seen clients to generate the `client_percentage` split. 

        WARNING: Due to clients dropping out and new coming in, we may have enough clients previously saved to
        adhere by the `client_percentage`, but have not in a single round. `force_local_split` ignores earlier
        assignments and achieves the split with in every round. Overrides `force_global_split`
        """
        self.client_map: OrderedDict[int, bool] = {}
        self.client_percentage = client_percentage
        self.lr_factor = lr_factor
        self.force_global_split = force_global_split
        self.force_local_split = force_local_split
        super().__init__(**kwargs)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        configA = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            configA = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        def get_var_lr_percentage():
            if len(self.client_map) == 0: return 0.0
            return sum(self.client_map.values()) / len(self.client_map)

        # forgets earlier client assignments to focus only on current ones
        if self.force_local_split:
            self.client_map = dict()

        for client in clients:
            if not client.cid in self.client_map:
                if self.force_global_split:
                    self.client_map[client.cid] = get_var_lr_percentage() < self.client_percentage
                else:
                    self.client_map[client.cid] = random.random() < self.client_percentage

        configB = copy.deepcopy(configA)
        configA["lr_factor"] = pow(self.lr_factor, server_round)
        configB["lr_factor"] = 1.0
        
        update_lr_fitin = FitIns(parameters, configA)
        const_lr_fitin = FitIns(parameters, configB)
        
        client_fitins = []
        for client in clients:
            if self.client_map[client.cid]:
                client_fitins.append((client, update_lr_fitin))
            else:
                client_fitins.append((client, const_lr_fitin))

        log(INFO, f"Variable LR #clients: {sum(self.client_map.values())}")
        log(INFO, f"Fixed    LR #clients: {len(self.client_map) - sum(self.client_map.values())}")
        if len(self.client_map) != 0:
            log(INFO, f"Var. C.   Percentage: {sum(self.client_map.values()) / len(self.client_map) *100}%") 

        return client_fitins

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
        """Evaluate the resulting model on the central server"""
        model.load_state_dict(
            shared.numpy_to_state_dict(
                model=model,
                parameters=parameters),
            strict=True)
        loss, accuracy = shared.test(model=model, testloader=testloader)
        wandb.log({"central_test_accuracy": accuracy})
        return loss, {"central_test_accuracy": accuracy}
    return evaluate

def main():

    device = shared.get_device()
    model = models.Net().to(device)
    strategy = SplitLRFedAvg(
            client_percentage=0.5,
            lr_factor=0.99,
            force_global_split=True,
            force_local_split=False,
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
