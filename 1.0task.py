"""federated_image_classification: A Flower / DenseNet + XGBoost app for image classification."""
# version: 1.0
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


# Create a feature extractor based on DenseNet
def get_feature_extractor():
    """Get a pre-trained DenseNet model for feature extraction."""
    model = models.densenet121(pretrained=True)
    # Replace the classifier with an identity module to get features
    model.classifier = torch.nn.Identity()
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_clients):
    """Load partition of image data and extract features using DenseNet."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_clients)
        fds = FederatedDataset(
            dataset="path/to/image/dataset",  # This should be replaced with your actual dataset
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
    
    # Get feature extractor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor = get_feature_extractor().to(device)
    
    # Extract features
    log(INFO, f"Extracting features for partition {partition_id}...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    valid_features, valid_labels = extract_features(feature_extractor, valid_loader, device)
    
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