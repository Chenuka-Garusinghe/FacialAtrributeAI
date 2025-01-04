import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
import torch.multiprocessing as mp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import logging
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device setup with support for CUDA, MPS, and CPU
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
logger.info(f"Using device: {device}")

class OptimizedCelebADataset(Dataset):
    def __init__(self, data, img_folder, transform=None):
        self.data = data
        self.img_folder = img_folder
        self.transform = transform
        self.image_paths = [os.path.join(img_folder, img_name) for img_name in tqdm(self.data['image_id'], desc="Loading image paths")]
        
        # Get attribute names (excluding 'image_id' and 'partition')
        self.attribute_names = self.data.columns.tolist()
        self.attribute_names.remove('image_id')
        if 'partition' in self.attribute_names:
            self.attribute_names.remove('partition')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Convert to numpy array and apply transformations
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get labels using attribute names
        labels = self.data.iloc[idx][self.attribute_names].values.astype('float32')
        return image, torch.tensor(labels)

# Original Model (Model 1)
class OptimizedFacialAttributeModel(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

# Final Model with Attention
class ImprovedAttributeModel(nn.Module):
    def __init__(self, num_attributes=40):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        backbone_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.shared_features = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_attributes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        shared = self.shared_features(x)
        return self.classifier(shared)

def load_attributes(attr_file):
    """Load attribute data with proper column names"""
    logger.info("Loading attributes...")
    with open(attr_file, 'r') as f:
        _ = f.readline()  # Skip the number of images
        attribute_names = f.readline().strip().split()
    
    # Read attributes with correct column names
    attributes = pd.read_csv(attr_file, sep=r"\s+", skiprows=2, names=['image_id'] + attribute_names)
    attributes['image_id'] = attributes['image_id'].str.strip()
    # Replace -1 with 0 for binary labels
    attributes[attribute_names] = attributes[attribute_names].replace(-1, 0)
    return attributes, attribute_names

def load_model(model_path, model_type='final'):
    logger.info(f"Loading {model_type} model from {model_path}")
    if model_type == 'final':
        model = ImprovedAttributeModel(num_attributes=40)
    else:
        model = OptimizedFacialAttributeModel(num_classes=40)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, threshold=0.5):
    all_labels = []
    all_preds = []
    all_probs = []
    
    logger.info(f"\nStarting evaluation on {len(test_loader.dataset)} images...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating batches"):
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                probabilities = torch.sigmoid(outputs)
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())
            all_preds.append((probabilities > threshold).cpu().numpy())
    
    logger.info("Calculating metrics...")
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    
    metrics = {}
    attribute_names = test_loader.dataset.attribute_names
    
    for i, attr in enumerate(tqdm(attribute_names, desc="Processing attributes")):
        true_labels = all_labels[:, i]
        pred_labels = all_preds[:, i]
        probabilities = all_probs[:, i]
        
        acc = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0)
        
        metrics[attr] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "avg_confidence": float(np.mean(probabilities))
        }
    
    # Add aggregate metrics
    metrics["overall"] = {
        "accuracy": float(np.mean([m["accuracy"] for m in metrics.values() if isinstance(m, dict)])),
        "precision": float(np.mean([m["precision"] for m in metrics.values() if isinstance(m, dict)])),
        "recall": float(np.mean([m["recall"] for m in metrics.values() if isinstance(m, dict)])),
        "f1": float(np.mean([m["f1"] for m in metrics.values() if isinstance(m, dict)]))
    }
    
    return metrics

def main():
    # Data paths
    attr_file = '../Anno/list_attr_celeba.txt'
    partition_file = '../Eval/list_eval_partition.txt'
    img_folder = '../img_align_celeba'
    models_to_evaluate = {
        'model_1': {'path': './model_1.pth', 'type': 'original'},
        'final_model': {'path': './final_model.pth', 'type': 'final'}
    }

    # Load attributes and partitions
    logger.info("Loading datasets...")
    attributes, attribute_names = load_attributes(attr_file)
    
    # Load partitions
    partitions = pd.read_csv(partition_file, sep=r"\s+", header=None, 
                           names=["image_id", "partition"])
    partitions['image_id'] = partitions['image_id'].str.strip()
    
    # Merge data
    data = attributes.merge(partitions, on="image_id")
    test_data = data[data['partition'] == 2].reset_index(drop=True)

    # Define transforms for evaluation
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Create test dataset and loader
    test_dataset = OptimizedCelebADataset(test_data, img_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)

    # Evaluate each model
    all_results = {}
    for model_name, model_info in models_to_evaluate.items():
        logger.info(f"\nEvaluating {model_name}...")
        try:
            model = load_model(model_info['path'], model_info['type'])
            metrics = evaluate_model(model, test_loader)
            
            # Save individual model results
            with open(f'{model_name}_evaluation_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            all_results[model_name] = metrics
            
            # Display summary metrics
            logger.info(f"\n{model_name} Overall Metrics:")
            overall = metrics['overall']
            logger.info(f"Accuracy: {overall['accuracy']:.4f}")
            logger.info(f"Precision: {overall['precision']:.4f}")
            logger.info(f"Recall: {overall['recall']:.4f}")
            logger.info(f"F1 Score: {overall['f1']:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
    
    # Save comparative results
    with open('comparative_evaluation_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == '__main__':
    main()