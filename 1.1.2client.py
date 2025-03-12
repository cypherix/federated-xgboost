# client.py

"""federated_image_classification: Client script for asynchronous federated learning with Ensemble + XGBoost."""
# version: 1.1.2
import argparse
import logging
import os
import warnings
import threading
import time
from typing import Tuple, Dict, List, Optional

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
from sklearn.feature_selection import SelectKBest, f_classif

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


class BelugaWhaleOptimizationAlgorithm:
    """Implementation of Beluga Whale Optimization Algorithm for feature selection."""
    
    def __init__(self, num_features, population_size=10, max_iterations=20, a_max=2.0, a_min=0.0):
        """Initialize BWOA parameters."""
        self.num_features = num_features
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.a_max = a_max
        self.a_min = a_min
        
    def _fitness_function(self, X, y, selected_features):
        """Calculate fitness based on classification accuracy with selected features."""
        if np.sum(selected_features) == 0:
            return 0.0  # No features selected
        
        # Use only selected features
        X_selected = X[:, selected_features == 1]
        
        # Use a small subset for quick evaluation
        sample_idx = np.random.choice(len(y), min(500, len(y)), replace=False)
        X_sample = X_selected[sample_idx]
        y_sample = y[sample_idx]
        
        # Simple cross-validation with XGBoost
        from sklearn.model_selection import cross_val_score
        from xgboost import XGBClassifier
        
        clf = XGBClassifier(n_estimators=10, max_depth=3)
        scores = cross_val_score(clf, X_sample, y_sample, cv=3, scoring='accuracy')
        
        # Balance accuracy with number of features
        feature_ratio = np.sum(selected_features) / len(selected_features)
        accuracy = np.mean(scores)
        
        # Higher fitness for higher accuracy with fewer features
        return accuracy * (1 - 0.1 * feature_ratio)
    
    def select_features(self, X, y):
        """Apply BWOA to select features from X for predicting y."""
        # Initialize population randomly (binary vectors)
        population = np.random.randint(0, 2, size=(self.population_size, self.num_features))
        
        # Track best solution
        best_solution = None
        best_fitness = -1
        
        for iteration in range(self.max_iterations):
            # Linear decrease of a parameter
            a = self.a_max - (self.a_max - self.a_min) * (iteration / self.max_iterations)
            
            for i in range(self.population_size):
                # Calculate fitness for each whale
                current_fitness = self._fitness_function(X, y, population[i])
                
                # Update best solution if needed
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_solution = population[i].copy()
            
            # Update each whale's position
            for i in range(self.population_size):
                # Random parameters
                r = np.random.random()
                A = 2 * a * r - a
                C = 2 * r
                l = np.random.uniform(-1, 1)
                p = np.random.random()
                
                if p < 0.5:
                    # Encircling prey or spiral update
                    if abs(A) < 1:
                        # Encircling prey
                        D = np.abs(C * best_solution - population[i])
                        new_position = best_solution - A * D
                    else:
                        # Search for prey (exploration)
                        random_whale = population[np.random.randint(0, self.population_size)]
                        D = np.abs(C * random_whale - population[i])
                        new_position = random_whale - A * D
                else:
                    # Spiral update
                    D = np.abs(best_solution - population[i])
                    new_position = D * np.exp(l) * np.cos(2 * np.pi * l) + best_solution
                
                # Convert to binary
                sigmoid = 1 / (1 + np.exp(-new_position))
                population[i] = (sigmoid > np.random.random(self.num_features)).astype(int)
        
        # Return indices of selected features
        return np.where(best_solution == 1)[0]


class FeatureSelector:
    """Feature selection wrapper that can use different algorithms."""
    
    def __init__(self, method="bwoa", k=100):
        """Initialize feature selector with specified method."""
        self.method = method
        self.k = k  # Number of features to select if using SelectKBest
        self.selected_indices = None
    
    def fit_transform(self, X, y):
        """Fit the feature selector and transform the features."""
        if self.method == "bwoa":
            # Apply Beluga Whale Optimization Algorithm
            bwoa = BelugaWhaleOptimizationAlgorithm(X.shape[1])
            self.selected_indices = bwoa.select_features(X, y)
        elif self.method == "kbest":
            # Use SelectKBest as a faster alternative
            selector = SelectKBest(f_classif, k=self.k)
            selector.fit(X, y)
            self.selected_indices = np.where(selector.get_support())[0]
        else:
            # No feature selection
            self.selected_indices = np.arange(X.shape[1])
        
        return X[:, self.selected_indices]
    
    def transform(self, X):
        """Transform features using previously determined indices."""
        if self.selected_indices is None:
            return X
        return X[:, self.selected_indices]


def get_ensemble_feature_extractor():
    """Create an ensemble of pretrained models for feature extraction."""
    # Load pre-trained models with frozen parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # DenseNet
    densenet = models.densenet121(pretrained=True)
    densenet.classifier = torch.nn.Identity()
    for param in densenet.parameters():
        param.requires_grad = False
    
    # MobileNet
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.classifier = torch.nn.Identity()
    for param in mobilenet.parameters():
        param.requires_grad = False
    
    # EfficientNet
    efficientnet = models.efficientnet_b0(pretrained=True)
    efficientnet.classifier = torch.nn.Identity()
    for param in efficientnet.parameters():
        param.requires_grad = False
    
    # Move models to device
    densenet = densenet.to(device)
    mobilenet = mobilenet.to(device)
    efficientnet = efficientnet.to(device)
    
    return {
        "densenet": densenet,
        "mobilenet": mobilenet,
        "efficientnet": efficientnet
    }


def save_model_with_timestamp(model, base_dir="extracted_features"):
    """Save model with timestamp in the specified directory."""
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create model filename with timestamp
    model_filename = f"model_{timestamp}.json"
    features_filename = f"features_{timestamp}.npy"
    
    # Full paths
    model_path = os.path.join(base_dir, model_filename)
    features_path = os.path.join(base_dir, features_filename)
    
    # Save model
    model_bytes = model.save_raw("json")
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    
    # Get combined features if they exist
    if os.path.exists("combined_features.npy"):
        # Copy to timestamped file in the directory
        import shutil
        shutil.copy("combined_features.npy", features_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Features saved to {features_path}")
    
    return model_path, features_path


def extract_ensemble_features(model_dict, dataloader, device):
    """Extract features using ensemble of pretrained models."""
    features_dict = {model_name: [] for model_name in model_dict.keys()}
    labels = []
    
    # Process data through each model
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            
            # Extract features from each model
            for model_name, model in model_dict.items():
                model.eval()
                outputs = model(inputs)
                features_dict[model_name].append(outputs.cpu().numpy())
            
            # Only store labels once
            labels.append(targets.numpy())
    
    # Concatenate features from each model
    for model_name in features_dict:
        features_dict[model_name] = np.vstack(features_dict[model_name])
    
    # Concatenate labels
    labels = np.concatenate(labels)

    # Combine all features horizontally
    combined_features = np.hstack([features_dict[model_name] for model_name in model_dict.keys()])
    # store the combined feature
    np.save('combined_features.npy', combined_features)
    logger.info(f"Combined features shape: {combined_features.shape}")
    logger.info(f"Features saved to combined_features.npy")
    return combined_features, labels


def load_data(data_dir: str, batch_size: int = 32, test_split: float = 0.2, seed: int = 42, 
              feature_selection: bool = False) -> Tuple:
    """Load and preprocess data, extract features using ensemble with optional feature selection."""
    
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
    
    # Get ensemble feature extractor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    ensemble_models = get_ensemble_feature_extractor()
    
    # Extract features using ensemble
    logger.info("Extracting ensemble features...")
    train_features, train_labels = extract_ensemble_features(ensemble_models, train_loader, device)
    valid_features, valid_labels = extract_ensemble_features(ensemble_models, valid_loader, device)
    
    logger.info(f"Extracted features - Train: {train_features.shape}, Valid: {valid_features.shape}")
    
    # Apply feature selection if enabled
    if feature_selection:
        logger.info("Applying feature selection...")
        feature_selector = FeatureSelector(method="bwoa")
        train_features = feature_selector.fit_transform(train_features, train_labels)
        valid_features = feature_selector.transform(valid_features)
        logger.info(f"After feature selection - Features: {train_features.shape[1]}")
    
    # Create DMatrix objects for XGBoost
    train_dmatrix = xgb.DMatrix(train_features, label=train_labels)
    valid_dmatrix = xgb.DMatrix(valid_features, label=valid_labels)
    
    return train_dmatrix, valid_dmatrix, train_size, test_size


class AsyncXGBoostClient(fl.client.Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        async_interval=None,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.async_interval = async_interval
        self.local_model = None
        self.global_model = None
        self.current_round = 0
        self.training_thread = None
        self.training_ready = False
        self.lock = threading.Lock()
        self.training_time = 0.0  # Track training time per client
        
    def start_background_training(self):
        """Start background training thread if async_interval is set."""
        if self.async_interval is not None and self.training_thread is None:
            self.training_thread = threading.Thread(target=self._background_training)
            self.training_thread.daemon = True  # Thread will exit when main thread exits
            self.training_thread.start()
            logger.info("Started background training thread")
    
    def _background_training(self):
        """Background thread for continuous local model updates."""
        while True:
            if self.global_model is not None:
                logger.info("Background training: updating local model...")
                
                with self.lock:
                    # Create a local booster with the global model's parameters
                    bst = xgb.Booster(params=self.params)
                    bst.load_model(self.global_model)
                    
                    # Local training
                    start_time = time.time()
                    for i in range(self.num_local_round):
                        bst.update(self.train_dmatrix, bst.num_boosted_rounds())
                    
                    # Bagging: extract the last N trees
                    local_bst = bst[
                        bst.num_boosted_rounds() - self.num_local_round : bst.num_boosted_rounds()
                    ]
                    
                    # Save updated model
                    self.local_model = local_bst.save_raw("json")
                    self.training_ready = True
                    self.training_time = time.time() - start_time  # Track training time
                    
                    logger.info("Background training: local model updated")
            
            # Sleep for the specified interval
            time.sleep(self.async_interval)
    
    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        start_time = time.time()
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())
        
        # Bagging: extract the last N=num_local_round trees for server aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds() - self.num_local_round : bst_input.num_boosted_rounds()
        ]
        
        self.training_time = time.time() - start_time  # Track training time
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        """Train model locally and return updated model parameters."""
        logger.info(f"Fitting on {self.num_train} examples")
        
        # Get global round number
        self.current_round = int(ins.config["global_round"])
        
        # Start background training for asynchronous updates if not started
        self.start_background_training()
        
        # If async mode is on and we have already trained, return
                # If async mode is on and we have already trained, return immediately
        if self.async_interval is not None and self.training_ready:
            logger.info("Using asynchronously pre-trained model")
            with self.lock:
                local_model_bytes = bytes(self.local_model)
                self.training_ready = False  # Reset flag for next round
                return FitRes(
                    status=Status(
                        code=Code.OK,
                        message="OK",
                    ),
                    parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
                    num_examples=self.num_train,
                    metrics={
                        "training_time": self.training_time,  # Report training time
                        "model_size": len(local_model_bytes),  # Report model size in bytes
                    },
                )
        
        # If not in async mode or no pre-trained model available, train locally
        if self.current_round == 1:
            # First round local training
            logger.info("First round: training new model")
            start_time = time.time()
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
            self.training_time = time.time() - start_time  # Track training time
        else:
            logger.info(f"Round {self.current_round}: updating global model")
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])
            
            # Load global model into booster
            bst.load_model(global_model)
            
            # Local training
            bst = self._local_boost(bst)
        
        # Save model with timestamp
        model_path, features_path = save_model_with_timestamp(bst)
        
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
            metrics={
                "training_time": self.training_time,  # Report training time
                "model_size": len(local_model_bytes),  # Report model size in bytes
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the model on the local validation set."""
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


def main(data_dir: str, server_address: str, num_local_rounds: int, async_interval: Optional[int] = None, feature_selection: bool = False):
    """Main function to start the client."""
    
    # Load and preprocess data
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        data_dir=data_dir,
        batch_size=32,
        test_split=0.2,
        feature_selection=feature_selection,
    )
    
    # Get XGBoost parameters
    params = get_xgboost_params()
    
    # Create client
    client = AsyncXGBoostClient(
        train_dmatrix=train_dmatrix,
        valid_dmatrix=valid_dmatrix,
        num_train=num_train,
        num_val=num_val,
        num_local_round=num_local_rounds,
        params=params,
        async_interval=async_interval,
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
    parser.add_argument(
        "--async-interval",
        type=int,
        default=None,
        help="interval in seconds for asynchronous updates (None for synchronous)",
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="enable feature selection using BWOA",
    )
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        server_address=args.server_address,
        num_local_rounds=args.num_local_rounds,
        async_interval=args.async_interval,
        feature_selection=args.feature_selection,
    )