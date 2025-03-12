# server.py

"""federated_image_classification: Server script for asynchronous federated learning with Ensemble + XGBoost."""
# version: 1.1.2
import argparse
import logging
import os
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
import flwr as fl
from flwr.common import Parameters
from flwr.server.strategy import FedXgbBagging
from flwr.server.client_manager import ClientManager

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("Server")

# Import necessary functions from client
from client import load_data, extract_ensemble_features, get_ensemble_feature_extractor

class ExtendedFedXgbBagging(FedXgbBagging):
    """Extended version of FedXgbBagging with global evaluation capability."""
    
    def __init__(
        self,
        *args,
        global_evaluation_fn=None,
        global_data=None,
        metrics_output_path=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.global_evaluation_fn = global_evaluation_fn
        self.global_data = global_data
        self.metrics_output_path = metrics_output_path
        self.round_metrics = []
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model updates from clients and perform global evaluation."""
        # Call the parent's aggregate_fit method
        aggregated_params = super().aggregate_fit(server_round, results, failures)
        
        # Perform global evaluation if function is provided
        if aggregated_params is not None and self.global_evaluation_fn is not None and self.global_data is not None:
            logger.info(f"Performing global evaluation after round {server_round}")
            
            # Fix: Convert aggregated_params to the expected format for global_evaluation_fn
            if isinstance(aggregated_params, tuple):
                params_bytes = aggregated_params[0]
                params = Parameters(tensor_type="", tensors=[params_bytes])
            else:
                params = aggregated_params
                
            metrics = self.global_evaluation_fn(params, self.global_data)
            
            # Store metrics with round number
            round_metrics = {"round": server_round, **metrics}
            self.round_metrics.append(round_metrics)
            
        return aggregated_params
    
    def _save_metrics(self):
        """Save metrics to a JSON file."""
        if not self.metrics_output_path:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.metrics_output_path), exist_ok=True)
        
        # Save metrics to file
        with open(self.metrics_output_path, 'w') as f:
            json.dump(self.round_metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {self.metrics_output_path}")


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
    """Return a configuration with global rounds."""
    config = {
        "global_round": str(rnd),
    }
    return config


def global_evaluation(parameters: Parameters, validation_data: Tuple[xgb.DMatrix, int]) -> Dict[str, float]:
    """Evaluate the global model on the validation set."""
    valid_dmatrix, num_val = validation_data
    
    # Load global model
    bst = xgb.Booster()
    try:
        # Try to load the model - parameters might be in different formats
        if hasattr(parameters, 'tensors'):
            bst.load_model(bytearray(parameters.tensors[0]))
        else:
            # If parameters is not a Parameters object but a bytes-like object
            bst.load_model(bytearray(parameters))
    except Exception as e:
        logger.error(f"Error loading model for evaluation: {e}")
        logger.error(f"Type of parameters: {type(parameters)}")
        return {"error": "Failed to load model for evaluation"}
    
    # Get predictions
    predictions = bst.predict(valid_dmatrix)
    predicted_labels = np.round(predictions)  # Round probabilities to get binary predictions
    true_labels = valid_dmatrix.get_label()
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    logger.info(f"Global Validation Accuracy: {accuracy:.4f}")
    
    # Calculate confusion matrix
    tn = np.sum((predicted_labels == 0) & (true_labels == 0))
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))
    tp = np.sum((predicted_labels == 1) & (true_labels == 1))
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"Global Validation Precision: {precision:.4f}")
    logger.info(f"Global Validation Recall: {recall:.4f}")
    logger.info(f"Global Validation F1 Score: {f1_score:.4f}")
    
    # Calculate ROC AUC
    try:
        eval_results = bst.eval_set(
            evals=[(valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        metrics_str = eval_results.split("\t")[1]
        metric_name = metrics_str.split(":")[0]
        metric_value = round(float(metrics_str.split(":")[1]), 4)
        logger.info(f"Global Validation {metric_name}: {metric_value}")
    except Exception as e:
        logger.warning(f"Could not calculate additional metrics: {e}")
        metric_name = "auc"
        metric_value = 0.0
    
    # Get feature importance
    try:
        importance = bst.get_score(importance_type='gain')
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 important features:")
        for feature, score in top_features:
            logger.info(f"  {feature}: {score:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate feature importance: {e}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        metric_name: metric_value,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def start_server(
    server_address: str, 
    num_rounds: int, 
    fraction_fit: float, 
    fraction_evaluate: float, 
    validation_data_dir: str,
    global_eval_dir: str = None,
    metrics_output_path: str = None
):
    """Start the Flower server with global evaluation capabilities."""
    
    # Load validation data
    logger.info("Loading validation data...")
    valid_dmatrix, _, num_val, _ = load_data(validation_data_dir, feature_selection=True)
    validation_data = (valid_dmatrix, num_val)
    
    # Load global evaluation data if provided
    global_data = None
    if global_eval_dir:
        logger.info(f"Loading global evaluation data from {global_eval_dir}...")
        try:
            global_dmatrix, _, global_num, _ = load_data(
                global_eval_dir, 
                batch_size=32, 
                test_split=0.0,  # Use all data for evaluation
                feature_selection=True
            )
            global_data = (global_dmatrix, global_num)
            logger.info(f"Loaded global evaluation dataset with {global_num} samples")
        except Exception as e:
            logger.error(f"Failed to load global evaluation data: {e}")
            logger.warning("Continuing without global evaluation dataset")
    
    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])
    
    # Define strategy
    strategy = ExtendedFedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
        global_evaluation_fn=global_evaluation,
        global_data=global_data or validation_data,  # Use validation data if global data not provided
        metrics_output_path=metrics_output_path
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
    parser.add_argument(
        "--validation-data-dir",
        type=str,
        required=True,
        help="directory with validation image data",
    )
    parser.add_argument(
        "--global-eval-dir",
        type=str,
        default=None,
        help="directory with global evaluation image data (if different from validation data)",
    )
    parser.add_argument(
        "--metrics-output-path",
        type=str,
        default="metrics/global_metrics.json",
        help="path to save metrics JSON file",
    )
    
    args = parser.parse_args()
    
    start_server(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        validation_data_dir=args.validation_data_dir,
        global_eval_dir=args.global_eval_dir,
        metrics_output_path=args.metrics_output_path,
    )