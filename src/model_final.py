import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
import cv2
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import json
import os
import json
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

##############* Model stats extractor class ############################

class TrainingLogger:
    def __init__(self, model_name, base_dir='training_logs'):
        # Create directory structure
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_name = model_name
        self.log_dir = self.base_dir / model_name / self.timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage with empty lists
        self.metrics = {
            'epoch': [],           
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'learning_rates': [],
            'batch_losses': []
        }
        
        # Training metadata
        self.metadata = {
            'model_name': model_name,
            'start_time': time.time(),
            'epochs_completed': 0,
            'total_training_time': 0,
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        # Keep track of current batch for proper logging
        self.current_epoch = 0
        self.batch_losses_temp = []  # Temporary storage for batch losses

    def log_epoch(self, epoch, train_loss, val_loss, epoch_time, learning_rate):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['epoch_times'].append(epoch_time)
        self.metrics['learning_rates'].append(learning_rate)
        
        # Calculate average batch loss for the epoch and store it
        if self.batch_losses_temp:
            avg_batch_loss = sum(self.batch_losses_temp) / len(self.batch_losses_temp)
            self.metrics['batch_losses'].append(avg_batch_loss)
        else:
            self.metrics['batch_losses'].append(train_loss)  # Fallback to train_loss if no batch data
            
        # Clear temporary batch losses for next epoch
        self.batch_losses_temp = []
        
        # Update metadata
        self.current_epoch = epoch
        self.metadata['epochs_completed'] = epoch + 1
        if val_loss < self.metadata['best_val_loss']:
            self.metadata['best_val_loss'] = val_loss
            self.metadata['best_epoch'] = epoch
        
        # Save current state
        self._save_metrics()
        
    def log_batch(self, batch_loss):
        # Store batch loss temporarily
        self.batch_losses_temp.append(batch_loss)

    def finish_training(self):
        self.metadata['total_training_time'] = time.time() - self.metadata['start_time']
        self._save_metrics()
        self._generate_training_report()
        self._create_visualizations()

    def _save_metrics(self):
        # Create a DataFrame ensuring all columns have the same length
        metrics_df = pd.DataFrame({
            'epoch': self.metrics['epoch'],
            'train_loss': self.metrics['train_loss'],
            'val_loss': self.metrics['val_loss'],
            'epoch_time': self.metrics['epoch_times'],
            'learning_rate': self.metrics['learning_rates'],
            'avg_batch_loss': self.metrics['batch_losses']
        })
        
        # Save metrics to CSV
        metrics_df.to_csv(self.log_dir / 'metrics.csv', index=False)
        
        # Save metadata to JSON
        with open(self.log_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def _generate_training_report(self):
        report = f"""Training Report for {self.model_name}
        =======================================
        Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Total epochs: {self.metadata['epochs_completed']}
        Total training time: {self.metadata['total_training_time']:.2f} seconds
        Best validation loss: {self.metadata['best_val_loss']:.4f} (Epoch {self.metadata['best_epoch'] + 1})
        Final training loss: {self.metrics['train_loss'][-1]:.4f}
        Final validation loss: {self.metrics['val_loss'][-1]:.4f}
        Average epoch time: {sum(self.metrics['epoch_times']) / len(self.metrics['epoch_times']):.2f} seconds
        """
        
        with open(self.log_dir / 'training_report.txt', 'w') as f:
            f.write(report)

    def _create_visualizations(self):
        # Set style for all plots
        plt.style.use('seaborn')
        
        # 1. Loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['epoch'], self.metrics['train_loss'], label='Training Loss')
        plt.plot(self.metrics['epoch'], self.metrics['val_loss'], label='Validation Loss')
        plt.title(f'{self.model_name} - Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.log_dir / 'loss_curves.png')
        plt.close()
        
        # 2. Learning rate over time
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['epoch'], self.metrics['learning_rates'])
        plt.title(f'{self.model_name} - Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(self.log_dir / 'learning_rate.png')
        plt.close()
        
        # 3. Epoch times
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['epoch'], self.metrics['epoch_times'])
        plt.title(f'{self.model_name} - Epoch Training Times')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.savefig(self.log_dir / 'epoch_times.png')
        plt.close()
        
        # 4. Batch loss distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.metrics['batch_losses'], bins=50)
        plt.title(f'{self.model_name} - Average Batch Loss Distribution')
        plt.xlabel('Loss')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(self.log_dir / 'batch_loss_dist.png')
        plt.close()

class ModelComparison:
    def __init__(self, base_dir='training_logs'):
        self.base_dir = Path(base_dir)
    
    def compare_models(self, model_names):
        all_metrics = {}
        for model_name in model_names:
            model_dir = self.base_dir / model_name
            if not model_dir.exists():
                continue
                
            # Get the latest run
            latest_run = max(os.listdir(model_dir))
            metrics_file = model_dir / latest_run / 'metrics.csv'
            
            if metrics_file.exists():
                metrics = pd.read_csv(metrics_file)
                all_metrics[model_name] = metrics
        
        if not all_metrics:
            return
        
        # Create comparison visualizations
        self._create_comparison_plots(all_metrics)
    
    def _create_comparison_plots(self, all_metrics):
        comparison_dir = self.base_dir / 'model_comparison'
        comparison_dir.mkdir(exist_ok=True)
        
        # 1. Training loss comparison
        plt.figure(figsize=(12, 8))
        for model_name, metrics in all_metrics.items():
            plt.plot(metrics['train_loss'], label=f'{model_name} (Train)')
            plt.plot(metrics['val_loss'], label=f'{model_name} (Val)', linestyle='--')
        plt.title('Model Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(comparison_dir / 'loss_comparison.png')
        plt.close()
        
        # 2. Training time comparison
        plt.figure(figsize=(10, 6))
        epoch_times = {name: metrics['epoch_times'].mean() 
                      for name, metrics in all_metrics.items()}
        plt.bar(epoch_times.keys(), epoch_times.values())
        plt.title('Average Epoch Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(comparison_dir / 'time_comparison.png')
        plt.close()
##############* End of Model stats extractor class ############################



class AttributeProcessor:
    """Process CelebA attributes and calculate statistics"""
    def __init__(self, attr_file):
        self.attr_file = attr_file
        self.attribute_stats = {}
        self.attribute_names = []
        
    def load_attributes(self):
        logger.info("Loading attributes...")
        try:
            with open(self.attr_file, 'r') as f:
                num_images = int(f.readline().strip())
                self.attribute_names = f.readline().strip().split()
            
            df = pd.read_csv(
                self.attr_file, 
                sep=r'\s+', 
                skiprows=2,
                names=['image_id'] + self.attribute_names
            )
            logger.info(f"Loaded {len(df)} images with {len(self.attribute_names)} attributes")
            self.attribute_stats = self.calculate_attribute_stats(df)
            df = df.replace(-1, 0).set_index('image_id')
            return df, self.attribute_stats
        except Exception as e:
            logger.error(f"Error loading attributes: {str(e)}")
            raise
    
    def calculate_attribute_stats(self, df):
        stats = {}
        for attr in self.attribute_names:
            pos_count = (df[attr] == 1).sum()
            neg_count = (df[attr] == 0).sum()
            total = pos_count + neg_count
            stats[attr] = {
                'positive_ratio': pos_count / total,
                'weight': total / (2 * pos_count * neg_count) if pos_count > 0 and neg_count > 0 else 1.0,
                'pos_weight': neg_count / pos_count if pos_count > 0 else 1.0
            }
        return stats

class ImprovedCelebADataset(Dataset):
    def __init__(self, data, img_folder, attribute_stats, training=True):
        self.data = data
        self.img_folder = img_folder
        self.attribute_stats = attribute_stats
        self.image_paths = [os.path.join(img_folder, idx) for idx in self.data.index]
        
        if training:
            self.transform = A.Compose([
                # A.RandomResizedCrop(height=218, width=178, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                # A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image = transformed['image'].to(torch.float32)  # Ensure float32 for MPS compatibility
        
        labels = torch.tensor([self.data.iloc[idx][attr] for attr in self.attribute_stats.keys()], dtype=torch.float32)
        return image, labels

class ImprovedAttributeModel(nn.Module):
    """Enhanced model for facial attribute prediction with EfficientNet backbone"""
    def __init__(self, num_attributes=40):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        backbone_features = self.backbone.classifier[1].in_features  # Should be 1280
        self.backbone.classifier = nn.Identity()  # Remove the classification layer
        
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
        x = self.backbone(x)  # Output shape: (batch_size, 1280)
        shared = self.shared_features(x)
        outputs = self.classifier(shared)
        return outputs

class FocalLossWithWeights(nn.Module):
    """Weighted focal loss for handling class imbalance"""
    def __init__(self, attribute_stats, gamma=2.0, alpha=0.25):
        super().__init__()
        weights = torch.tensor([stats['weight'] for stats in attribute_stats.values()], dtype=torch.float32, device=device)
        self.register_buffer('weights', weights)
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return (focal_loss * self.weights.unsqueeze(0)).mean()

def plot_training_progress(metrics):
    plt.figure(figsize=(12, 6))
    train_losses = [x['value'] for x in metrics['train_loss']]
    val_losses = [x['value'] for x in metrics['val_loss']]
    epochs = range(len(train_losses))
    
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig('training_progress.png', dpi=300)
    plt.close()

def train_model(model, train_loader, val_loader, attribute_stats, epochs=10, device=device):
    model.to(device)
    criterion = FocalLossWithWeights(attribute_stats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=3e-4, epochs=epochs, steps_per_epoch=len(train_loader))
    
    metrics = {'train_loss': [], 'val_loss': []}
    training_logger = TrainingLogger("model_final")
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Use mixed precision for MPS
            with torch.amp.autocast(device_type='mps', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            training_logger.log_batch(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        # Log epoch metrics
        training_logger.log_epoch(epoch, avg_train_loss, avg_val_loss, epoch_time, optimizer.param_groups[0]['lr'])
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model with early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'final_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

    training_logger.finish_training()
    return model, metrics


def main():
    # Paths to data
    attr_file = '../Anno/list_attr_celeba.txt'
    partition_file = '../Eval/list_eval_partition.txt'
    img_folder = '../img_align_celeba'
    
    try:
        # Load data as before
        processor = AttributeProcessor(attr_file)
        attributes_df, attribute_stats = processor.load_attributes()
        partitions = pd.read_csv(partition_file, sep=r'\s+', header=None, names=['image_id', 'partition'])
        data = attributes_df.merge(partitions.set_index('image_id'), left_index=True, right_index=True)
        train_data = data[data['partition'] == 0].drop('partition', axis=1)
        val_data = data[data['partition'] == 1].drop('partition', axis=1)

        # DataLoader with reduced workers and persistent workers
        batch_size = 32
        num_workers = min(4, os.cpu_count() - 1)
        train_dataset = ImprovedCelebADataset(train_data, img_folder, attribute_stats, training=True)
        val_dataset = ImprovedCelebADataset(val_data, img_folder, attribute_stats, training=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=min(4, os.cpu_count() - 1), persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=min(4, os.cpu_count() - 1), persistent_workers=True
        )

        # Initialize and train model
        model = ImprovedAttributeModel(num_attributes=len(attribute_stats))
        model, metrics = train_model(model, train_loader, val_loader, attribute_stats, epochs=10, device=device)

        # Save metrics
        with open('training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        # Generate model comparisons
        # comparator = ModelComparison()
        # comparator.compare_models(['model_1','model_final'])

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()