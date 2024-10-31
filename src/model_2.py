import pandas as pd
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.amp.autocast_mode import autocast
from functools import partial
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Enable Metal performance acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Optimize CPU thread usage
torch.set_num_threads(mp.cpu_count())
torch.set_num_interop_threads(mp.cpu_count())

# Enhanced transforms with hardware acceleration and augmentations to images
transform = A.Compose([
    A.Resize(128, 128),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.RandomShadow(p=0.2),
    A.RandomRain(p=0.2),
    A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=0.5),  # Random Erasing
    ToTensorV2()
])

# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# 
# Custom dataset class with Albumentations
class AugmentedCelebADataset(Dataset):
    def __init__(self, data, img_folder, transform=None):
        self.data = data
        self.img_folder = img_folder
        self.transform = transform
        self.image_paths = [os.path.join(img_folder, img_name) for img_name in self.data['image_id']]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]))  # Convert to numpy array for Albumentations
        labels = self.data.iloc[idx][1:-1].values.astype('float32')

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.tensor(labels)

class OptimizedCelebADataset(Dataset):
    def __init__(self, data, img_folder, transform=None):
        self.data = data
        self.img_folder = img_folder
        self.transform = transform
        # Prefetch file paths
        self.image_paths = [os.path.join(img_folder, img_name) 
                          for img_name in self.data['image_id']]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        labels = self.data.iloc[idx][1:-1].values.astype('float32')
        return image, torch.tensor(labels)

class OptimizedFacialAttributeModel(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def create_optimized_dataloaders(data, img_folder, batch_size=64, num_workers=mp.cpu_count()):
    datasets = {
        'train': OptimizedCelebADataset(data[data['partition'] == 0], img_folder, transform),
        'val': OptimizedCelebADataset(data[data['partition'] == 1], img_folder, transform),
        'test': OptimizedCelebADataset(data[data['partition'] == 2], img_folder, transform)
    }
    
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, 
                          shuffle=True, num_workers=num_workers,
                          pin_memory=True, prefetch_factor=2),
        'val': DataLoader(datasets['val'], batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True),
        'test': DataLoader(datasets['test'], batch_size=batch_size,
                          num_workers=num_workers, pin_memory=True)
    }
    return loaders

def train_optimized_model(model, train_loader, val_loader, epochs=10):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=False)
            
            with autocast(device_type='mps', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        val_loss = 0.0
        
        # validation phase, no need gradients so we can save memeory
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Summary - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # Probability of correct classification
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
# Usage in training loop
criterion = FocalLoss(alpha=1, gamma=2)

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Initialize the optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Update training loop to include scheduler step
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
    
    # Step the scheduler based on validation loss
    scheduler.step(val_loss)

if __name__ == '__main__':
    # Data paths
    attr_file = '../Anno/list_attr_celeba.txt'
    partition_file = '../Eval/list_eval_partition.txt'
    img_folder = '../img_align_celeba'
    
    # Load and preprocess attributes
    print("Loading attributes...")
    attributes = pd.read_csv(attr_file, sep="\s+", skiprows=1).replace(-1, 0)
    attributes.reset_index(inplace=True)
    attributes.rename(columns={'index': 'image_id'}, inplace=True)
    attributes['image_id'] = attributes['image_id'].str.strip()
    # Use the augmented dataset in your DataLoader
    train_data = data[data['partition'] == 0].reset_index(drop=True)
    train_dataset = AugmentedCelebADataset(train_data, img_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=mp.cpu_count(), pin_memory=True)
    # Load partitions
    print("Loading partitions...")
    partitions = pd.read_csv(partition_file, sep="\s+", header=None, 
                            names=["image_id", "partition"])
    partitions['image_id'] = partitions['image_id'].str.strip()
    
    # Merge data
    print("Merging datasets...")
    data = attributes.merge(partitions, on="image_id")
    
    # Create optimized data loaders
    print("Creating data loaders...")
    loaders = create_optimized_dataloaders(data, img_folder)
    
    # Initialize model
    print("Initializing model...")
    model = OptimizedFacialAttributeModel()
    
    # Train model
    print("Starting training...")
    train_optimized_model(model, loaders['train'], loaders['val'])
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), 'optimized_model.pth')
    print("Training complete!")