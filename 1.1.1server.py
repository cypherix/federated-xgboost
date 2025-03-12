# server.py

"""federated_image_classification: Server script for asynchronous federated learning with Ensemble + XGBoost."""
# version: 1.1.1
import argparse
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
import flwr as fl
from flwr.common import Parameters
from flwr.server.strategy import FedXgbBagging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("Server")

from client import load_data, extract_ensemble_features, get_ensemble_feature_extractor
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


def global_evaluation(parameters: Parameters, validation_data: Tuple[xgb.DMatrix, int]) -> Dict[str, float]:
    """Evaluate the global model on the validation set."""
    valid_dmatrix, num_val = validation_data
    # Load global model
    bst = xgb.Booster()
    bst.load_model(bytearray(parameters.tensors[0]))
    
    # Get predictions
    predictions = bst.predict(valid_dmatrix)
    predicted_labels = np.round(predictions)  # Round probabilities to get binary predictions
    true_labels = valid_dmatrix.get_label()
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    logger.info(f"Global Validation Accuracy: {accuracy:.4f}")
    
    # Calculate AUC
    eval_results = bst.eval_set(
        evals=[(valid_dmatrix, "valid")],
        iteration=bst.num_boosted_rounds() - 1,
    )
    metrics_str = eval_results.split("\t")[1]
    metric_name = metrics_str.split(":")[0]
    metric_value = round(float(metrics_str.split(":")[1]), 4)
    logger.info(f"Global Validation {metric_name}: {metric_value}")
    
    return {
        "accuracy": accuracy,
        metric_name: metric_value,
    }


def start_server(server_address: str, num_rounds: int, fraction_fit: float, fraction_evaluate: float, validation_data_dir: str):
    """Start the Flower server."""
    
    # Load validation data
    logger.info("Loading validation data...")
    valid_dmatrix, _, num_val, _ = load_data(validation_data_dir, feature_selection=True)
    validation_data = (valid_dmatrix, num_val)
    
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
    
    # Store global metrics
    global_metrics = []
    
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
    parser.add_argument(
        "--validation-data-dir",
        type=str,
        required=True,
        help="directory with validation image data",
    )
    
    args = parser.parse_args()
    
    start_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        validation_data_dir=args.validation_data_dir,
    )