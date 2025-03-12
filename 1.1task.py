"""federated_image_classification: A Flower / Ensemble (MobileNet, DenseNet, EfficientNet) + XGBoost app with BWOA feature selection."""
# version: 1.1
import os
from logging import INFO
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import xgboost as xgb
from flwr.common import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import random
from sklearn.feature_selection import SelectKBest, f_classif


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
    
    return combined_features, labels


def train_test_split(partition, test_fraction, seed):
    """Split the data into train and validation sets given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(features, labels):
    """Transform features and labels to DMatrix format for xgboost."""
    return xgb.DMatrix(features, label=labels)


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_clients, feature_selection=False):
    """Load partition of image data and extract features using ensemble models with optional feature selection."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_clients)
        fds = FederatedDataset(
            dataset="path/to/image/dataset",  # Replace with actual dataset
            partitioners={"train": partitioner},
        )

    # Load the partition for this `partition_id`
    partition = fds.load_partition(partition_id, split="train")
    
    # Train/test splitting
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=0.2, seed=42
    )
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets and dataloaders
    train_dataset = ImageDataset(train_data, transform=transform)
    valid_dataset = ImageDataset(valid_data, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    
    # Get ensemble feature extractor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble_models = get_ensemble_feature_extractor()
    
    # Extract features using ensemble
    log(INFO, f"Extracting ensemble features for partition {partition_id}...")
    train_features, train_labels = extract_ensemble_features(ensemble_models, train_loader, device)
    valid_features, valid_labels = extract_ensemble_features(ensemble_models, valid_loader, device)
    
    # Apply feature selection if enabled
    if feature_selection:
        log(INFO, f"Applying feature selection for partition {partition_id}...")
        feature_selector = FeatureSelector(method="bwoa")
        train_features = feature_selector.fit_transform(train_features, train_labels)
        valid_features = feature_selector.transform(valid_features)
    
    # Create DMatrix objects for XGBoost
    train_dmatrix = transform_dataset_to_dmatrix(train_features, train_labels)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_features, valid_labels)
    
    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict