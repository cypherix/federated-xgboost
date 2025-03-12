"""federated_image_classification: Client script for federated learning with DenseNet + XGBoost."""
# version: 1.0
import argparse
import logging
import os
import warnings
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import xgboost as xgb
import numpy as np
import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("Client")


class ImageDataset(Dataset):
    """Custom dataset for loading images from directories."""
    
    def __init__(self, data_dir, transform=None):
        """Initialize dataset with images from data_dir."""
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ["normal", "infected"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        logger.info(f"Loaded dataset with {len(self.samples)} images")
        logger.info(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self):
        """Get the class distribution in the dataset."""
        distribution = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            if class_name not in distribution:
                distribution[class_name] = 0
            distribution[class_name] += 1
        return distribution
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def extract_features(model, dataloader, device):
    """Extract features using a pre-trained DenseNet model."""
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    
    return np.vstack(features), np.concatenate(labels)


def load_data(data_dir: str, batch_size: int = 32, test_split: float = 0.2, seed: int = 42) -> Tuple:
    """Load and preprocess data, extract features using DenseNet."""
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset
    logger.info(f"Loading data from {data_dir}")
    full_dataset = ImageDataset(data_dir, transform=transform)
    
    # Split dataset
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - test_size
    
    # Use torch's random_split
    torch.manual_seed(seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train set: {train_size} samples, Validation set: {test_size} samples")
    
    # Get feature extractor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    feature_extractor = models.densenet121(pretrained=True)
    feature_extractor.classifier = torch.nn.Identity()  # Replace classifier with identity
    feature_extractor = feature_extractor.to(device)
    
    # Extract features
    logger.info("Extracting features using DenseNet...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    valid_features, valid_labels = extract_features(feature_extractor, valid_loader, device)
    
    logger.info(f"Extracted features - Train: {train_features.shape}, Valid: {valid_features.shape}")
    
    # Create DMatrix objects for XGBoost
    train_dmatrix = xgb.DMatrix(train_features, label=train_labels)
    valid_dmatrix = xgb.DMatrix(valid_features, label=valid_labels)
    
    return train_dmatrix, valid_dmatrix, train_size, test_size


class XGBoostClient(fl.client.Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for server aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        logger.info(f"Fitting on {self.num_train} examples")
        
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            logger.info("First round: training new model")
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            logger.info(f"Round {global_round}: updating global model")
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)
        
        logger.info(f"Fit completed. Model size: {len(local_model_bytes)} bytes")

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    # def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
    #     logger.info(f"Evaluating on {self.num_val} examples")
        
    #     # Load global model
    #     bst = xgb.Booster(params=self.params)
    #     para_b = bytearray(ins.parameters.tensors[0])
    #     bst.load_model(para_b)

    #     # Run evaluation
    #     eval_results = bst.eval_set(
    #         evals=[(self.valid_dmatrix, "valid")],
    #         iteration=bst.num_boosted_rounds() - 1,
    #     )
        
    #     # Extract metrics - looking for AUC for binary classification
    #     metrics_str = eval_results.split("\t")[1]
    #     metric_name = metrics_str.split(":")[0]
    #     metric_value = round(float(metrics_str.split(":")[1]), 4)
        
    #     logger.info(f"Evaluation result: {metric_name}={metric_value}")

    #     return EvaluateRes(
    #         status=Status(
    #             code=Code.OK,
    #             message="OK",
    #         ),
    #         loss=0.0,
    #         num_examples=self.num_val,
    #         metrics={metric_name: metric_value},
    #     )
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        logger.info(f"Evaluating on {self.num_val} examples")
        
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Get predictions
        predictions = bst.predict(self.valid_dmatrix)
        predicted_labels = np.round(predictions)  # Round probabilities to get binary predictions
        true_labels = self.valid_dmatrix.get_label()

        # Calculate accuracy
        accuracy = np.mean(predicted_labels == true_labels)
        logger.info(f"Accuracy: {accuracy:.4f}")

        # Calculate AUC (existing logic)
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        metrics_str = eval_results.split("\t")[1]
        metric_name = metrics_str.split(":")[0]
        metric_value = round(float(metrics_str.split(":")[1]), 4)
        logger.info(f"Evaluation result: {metric_name}={metric_value}")

        # Return both accuracy and AUC
        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,  # You can also calculate loss if needed
            num_examples=self.num_val,
            metrics={
                "accuracy": accuracy,
                metric_name: metric_value,  # Existing metric (e.g., AUC)
            },
        )


def get_xgboost_params() -> Dict:
    """Get XGBoost parameters."""
    return {
        "objective": "binary:logistic",
        "eta": 0.1,
        "max_depth": 6, 
        "eval_metric": "auc",
        "nthread": 16,
        "num_parallel_tree": 1,
        "subsample": 0.8,
        "tree_method": "hist",
    }


def main(data_dir: str, server_address: str, num_local_rounds: int):
    """Main function to start the client."""
    
    # Load and preprocess data
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        data_dir=data_dir,
        batch_size=32,
        test_split=0.2,
    )
    
    # Get XGBoost parameters
    params = get_xgboost_params()
    
    # Create client
    client = XGBoostClient(
        train_dmatrix=train_dmatrix,
        valid_dmatrix=valid_dmatrix,
        num_train=num_train,
        num_val=num_val,
        num_local_round=num_local_rounds,
        params=params,
    )
    
    # Start client
    logger.info(f"Starting client, connecting to server at {server_address}")
    fl.client.start_client(server_address=server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated XGBoost Client")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="directory with image data",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="server address (IP:port)",
    )
    parser.add_argument(
        "--num-local-rounds",
        type=int,
        default=1,
        help="number of local training rounds",
    )
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        server_address=args.server_address,
        num_local_rounds=args.num_local_rounds,
    )