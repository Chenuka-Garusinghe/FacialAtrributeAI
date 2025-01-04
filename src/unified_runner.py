import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model Definitions
class Model1(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

class Model2(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

class Model3(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        self.attention = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_features),
            nn.Sigmoid()
        )
        
        self.classifiers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.base_model(x)
        attention = self.attention(features)
        weighted_features = features * attention
        return self.classifiers(weighted_features)

class ModelFinal(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
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
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        shared = self.shared_features(x)
        outputs = self.classifier(shared)
        return outputs

class UnifiedPredictor:
    def __init__(self, model_version, model_path, attr_names_path):
        self.model_version = model_version
        self.model_path = model_path
        self.device = device
        
        # Load attribute names
        with open(attr_names_path, 'r') as f:
            _ = f.readline()  # Skip first line
            self.attribute_names = f.readline().strip().split()
        
        # Initialize appropriate model based on version
        self.model = self._initialize_model()
        self.transform = self._get_transforms()
        
    def _initialize_model(self):
        models_map = {
            '1': Model1,
            '2': Model2,
            '3': Model3,
            'final': ModelFinal
        }
        
        if self.model_version not in models_map:
            raise ValueError(f"Invalid model version. Choose from: {list(models_map.keys())}")
        
        model_class = models_map[self.model_version]
        model = model_class(num_classes=len(self.attribute_names))
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transforms(self):
        if self.model_version == 'final':
            return A.Compose([
                A.Resize(height=218, width=178),
                A.CenterCrop(height=218, width=178),
                A.Resize(height=224, width=224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        elif self.model_version in ['1', '2']:
            return A.Compose([
                A.Resize(128, 128),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:  # Model 3
            return A.Compose([
                A.Resize(178, 218),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def preprocess_image(self, image_path):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        return image_tensor
    
    def predict(self, image_path, threshold=0.5):
        """Predict attributes for a given image"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # Format predictions
            predictions = {}
            for attr_name, prob in zip(self.attribute_names, probabilities):
                predictions[attr_name] = {
                    'present': bool(prob >= threshold),
                    'confidence': float(prob)
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Unified Model Runner for CelebA Attribute Prediction')
    parser.add_argument('model_version', choices=['1', '2', '3', 'final'], help='Model version to use')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--attr_file', type=str, default='../Anno/list_attr_celeba.txt', 
                        help='Path to attributes file')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # Determine model path based on version
    model_paths = {
        '1': 'model_1.pth',
        '2': 'model_2.pth',
        '3': 'model_3.pth',
        'final': 'final_model.pth'
    }
    
    model_path = f"./{model_paths[args.model_version]}"
    
    # Initialize predictor
    predictor = UnifiedPredictor(
        model_version=args.model_version,
        model_path=model_path,
        attr_names_path=args.attr_file
    )
    
    # Make prediction
    predictions = predictor.predict(args.image_path, args.threshold)
    
    if predictions:
        print(f"\nPredictions using Model {args.model_version}:")
        print("-" * 50)
        
        # Sort by confidence
        sorted_predictions = sorted(
            predictions.items(), 
            key=lambda x: x[1]['confidence'], 
            reverse=True
        )
        
        # Print predictions
        for attr_name, pred in sorted_predictions:
            status = "Present" if pred['present'] else "Absent"
            confidence = pred['confidence'] * 100
            print(f"{attr_name:20s}: {status} (Confidence: {confidence:.2f}%)")
    else:
        print("Failed to generate predictions.")

if __name__ == "__main__":
    main()