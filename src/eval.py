import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.multiprocessing as mp

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Custom dataset class
class OptimizedCelebADataset(Dataset):
    def __init__(self, data, img_folder, transform=None):
        self.data = data
        self.img_folder = img_folder
        self.transform = transform
        print("Preparing image paths...")
        self.image_paths = [os.path.join(img_folder, img_name) for img_name in tqdm(self.data['image_id'], desc="Loading image paths")]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        labels = self.data.iloc[idx][1:-1].values.astype('float32')
        return image, torch.tensor(labels)

# Model class
class OptimizedFacialAttributeModel(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

# Function to load the trained model
def load_model(model_path):
    print("Loading model weights...")
    model = OptimizedFacialAttributeModel(num_classes=40)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Function to evaluate the model
def evaluate_model(model, test_loader):
    all_labels = []
    all_preds = []
    
    total_batches = len(test_loader)
    print(f"\nStarting evaluation on {len(test_loader.dataset)} images...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating batches", total=total_batches):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
    
    print("\nCalculating metrics...")
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds) > 0.5
    
    metrics = {}
    for i, attr in enumerate(tqdm(test_loader.dataset.data.columns[1:-1], desc="Processing attributes")):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels[:, i], all_preds[:, i], average='binary')
        metrics[attr] = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    
    return metrics

if __name__ == '__main__':
    # Paths to data files (update as necessary)
    attr_file = '../Anno/list_attr_celeba.txt'
    partition_file = '../Eval/list_eval_partition.txt'
    img_folder = '../img_align_celeba'
    
    # Load and preprocess attributes
    print("Loading attributes...")
    attributes = pd.read_csv(attr_file, sep="\s+", skiprows=1).replace(-1, 0)
    attributes.reset_index(inplace=True)
    attributes.rename(columns={'index': 'image_id'}, inplace=True)
    attributes['image_id'] = attributes['image_id'].str.strip()
    
    # Load partitions
    print("Loading partitions...")
    partitions = pd.read_csv(partition_file, sep="\s+", header=None, names=["image_id", "partition"])
    partitions['image_id'] = partitions['image_id'].str.strip()
    
    # Merge attributes and partitions with progress bar
    print("Merging datasets...")
    data = attributes.merge(partitions, on="image_id")
    
    # Define transformations for evaluation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Prepare test dataset and loader
    print("\nPreparing test dataset...")
    test_data = data[data['partition'] == 2].reset_index(drop=True)
    test_dataset = OptimizedCelebADataset(test_data, img_folder, transform=transform)
    
    print("\nInitializing data loader...")
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=mp.cpu_count(), pin_memory=True)
    
    # Load the model
    model = load_model('optimized_model.pth')
    
    # Evaluate the model
    metrics = evaluate_model(model, test_loader)
    
    # Display metrics for each attribute
    print("\nEvaluation Metrics:")
    for attr, metric in metrics.items():
        print(f"{attr}: Accuracy={metric['accuracy']:.2f}, Precision={metric['precision']:.2f}, Recall={metric['recall']:.2f}, F1={metric['f1']:.2f}")