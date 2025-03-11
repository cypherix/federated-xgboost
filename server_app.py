"""federated_image_classification: Server script for federated learning with DenseNet + XGBoost."""

import argparse
import logging
from typing import Dict

import flwr as fl
from flwr.common import Parameters
from flwr.server.strategy import FedXgbBagging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("Server")

# def evaluate_metrics_aggregation(eval_metrics):
#     """Return an aggregated metric for evaluation."""
#     if not eval_metrics:
#         return {}
    
#     total_num = sum([num for num, _ in eval_metrics])
    
#     # Get all possible metrics
#     all_metrics = {}
#     for _, metrics in eval_metrics:
#         for key in metrics.keys():
#             all_metrics[key] = 0
    
#     # Aggregate each metric
#     for key in all_metrics.keys():
#         all_metrics[key] = (
#             sum([metrics.get(key, 0) * num for num, metrics in eval_metrics]) / total_num
#         )
    
#     return all_metrics
def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric for evaluation."""
    if not eval_metrics:
        return {}
    
    total_num = sum([num for num, _ in eval_metrics])
    
    # Get all possible metrics
    all_metrics = {}
    for _, metrics in eval_metrics:
        for key in metrics.keys():
            all_metrics[key] = 0
    
    # Aggregate each metric
    for key in all_metrics.keys():
        all_metrics[key] = (
            sum([metrics.get(key, 0) * num for num, metrics in eval_metrics]) / total_num
        )
    
    return all_metrics


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def start_server(server_address: str, num_rounds: int, fraction_fit: float, fraction_evaluate: float):
    """Start the Flower server."""
    
    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])
    
    # Define strategy
    strategy = FedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
    )
    
    # Start Flower server
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    logger.info(f"Starting server at {server_address}")
    logger.info(f"Server will run for {num_rounds} rounds")
    logger.info(f"Fraction of clients used for fitting: {fraction_fit}")
    logger.info(f"Fraction of clients used for evaluation: {fraction_evaluate}")
    
    # Start server and run federated learning
    fl.server.start_server(
        server_address=server_address,
        config=server_config,
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated XGBoost Server")
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="server address (IP:port)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="number of federated learning rounds",
    )
    parser.add_argument(
        "--fraction-fit",
        type=float,
        default=1.0,
        help="fraction of clients to use for fitting in each round",
    )
    parser.add_argument(
        "--fraction-evaluate",
        type=float,
        default=1.0,
        help="fraction of clients to use for evaluation in each round",
    )
    
    args = parser.parse_args()
    
    start_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
    )