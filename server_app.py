"""Project01: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from project01.task import Net, get_weights, set_weights, test, get_transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
import json


def get_evaluate_fn(testloader, device):
    """Return a callback that evaluate the global model accuracy"""
    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using the test dataset not used for train"""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        return loss, {"Centralized_accuracy": accuracy} 
    
    return evaluate


def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    for _, m in metrics:
        print(m)

    return {}


def on_fit_config(server_round: int) -> Metrics:
    """Adjusts learning rate based on current round."""
    lr = 0.01
    if server_round > 5:
        lr = 0.005
    if server_round > 15:
        lr = 0.003 
    if server_round > 25:
        lr = 0.001      
    return {"lr": lr}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    # Loop trough all metrics received compute accuracies x examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Return weighted average accuracy
    return {"accuracy": sum(accuracies) / total_examples}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Load global set
    testset = load_dataset("Falah/Alzheimer_MRI")["test"] #zalando-datasets/fashion_mnist
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)



    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
