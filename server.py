from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

import wandb
import fire

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy = sum(accuracies) / sum(examples)

    wandb.log({"accuracy": accuracy})
    return {"accuracy": accuracy}

def main():
    # Define strategy
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=wandb.config.num_rounds),
        strategy=strategy,
    )
    wandb.finish()

def setup_wandb(
        group_name: str,
        project_name: str = "flower-quickstart") -> None:
    wandb.init(
        project=project_name,
        group=group_name,
        name=f"{group_name}-server",
        config={
            "num_rounds": 3
        }
    )

if __name__ == '__main__':
   fire.Fire(setup_wandb)
   main()
