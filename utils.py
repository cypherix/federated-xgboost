import torch
from torchvision import transforms, models
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Dataset Class
class LungCTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['Infected', 'Normal']
        self.image_paths = []
        self.labels = []

        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls)
            if os.path.exists(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(cls_dir, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Handle file loading error
            print(f"Warning: Could not load image at {img_path}. Using empty image.")
            image = np.zeros((224, 224), dtype=np.uint8)
            
        image = cv2.resize(image, (224, 224))
        # Convert grayscale to 3-channel RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to tensor - this is the correct approach
        image = torch.from_numpy(image).float()
        # Rearrange from (H,W,C) to (C,H,W) format
        image = image.permute(2, 0, 1)
        
        # Normalize the tensor
        image = image / 255.0
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])(image)
        
        label = self.labels[idx]
        return image, label

# Feature Extractor
class FeatureExtractor:
    def __init__(self, use_cache=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Use weights parameter instead of pretrained
        self.models = {
            "mobilenet": models.mobilenet_v2(weights="DEFAULT").to(self.device),
            "resnet18": models.resnet18(weights="DEFAULT").to(self.device),
        }
        
        # Set models to evaluation mode
        for model_name, model in self.models.items():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            print(f"Loaded {model_name} model")
            
        # Feature caching to improve performance
        self.use_cache = use_cache
        self.feature_cache = {}

    def extract_features(self, images):
        """Extract features from a batch of images using multiple models"""
        batch_size = images.shape[0]
        
        # Move images to device
        images = images.to(self.device)
        
        # Generate batch identifier for caching
        if self.use_cache:
            # Simple hash of the image batch for cache key
            batch_hash = hash(images.cpu().numpy().tobytes())
            if batch_hash in self.feature_cache:
                return self.feature_cache[batch_hash]
        
        features_list = []
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                # For MobileNetV2, extract features before the classifier
                if model_name == "mobilenet":
                    # Get features before the classifier layer
                    features = model.features(images)
                    # Global average pooling
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                    features = features.reshape(batch_size, -1)
                
                # For ResNet18, extract features before the final layer
                elif model_name == "resnet18":
                    # Get features before the final fully connected layer
                    x = model.conv1(images)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)
                    x = model.avgpool(x)
                    features = torch.flatten(x, 1)
                
                features_list.append(features.cpu().numpy())
        
        # Concatenate all features
        combined_features = np.concatenate(features_list, axis=1)
        
        # Cache the features if enabled
        if self.use_cache:
            self.feature_cache[batch_hash] = combined_features
            
        return combined_features

# Data Loader
def get_data_loaders(data_dir, batch_size=16, test_split=0.2):
    """Create data loaders for training and testing"""
    dataset = LungCTDataset(data_dir)
    
    # Split indices for training and testing
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    
    # Randomize indices
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=2
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=2
    )
    
    return train_loader, test_loader