import os
import copy
import time
import json
import psutil
import subprocess
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.inception import InceptionOutputs
from torch.utils.data import Dataset, DataLoader
import cv2
import optuna
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR, ReduceLROnPlateau
from torchvision.transforms.functional import to_pil_image

###############################################################################
# 1) DIRECTORY SETUP
###############################################################################
base_dir = os.path.expanduser("~/experiment_results_komlos_final")
visualizations_dir = os.path.join(base_dir, "visualizations_komlos")
image_path = os.path.expanduser("~/Zoom_Images/KomlosEncoding")
data_path = os.path.expanduser("~/Datasets/Small_Data_Human_InP.json")

###############################################################################
# 2) PREPROCESSING AND DATASET
###############################################################################
def custom_preprocess_op(image, config):
    """Enhanced preprocessing with full config support"""
    if not config["enable_custom_preprocess"]:
        return image
        
    image_np = np.array(image)
    channels = cv2.split(image_np)
    
    # Normalization
    normalized_channels = [
        cv2.normalize(
            chan, None,
            alpha=config["alpha"],
            beta=config["beta"],
            norm_type=cv2.NORM_MINMAX
        ) for chan in channels
    ]
    merged = cv2.merge(normalized_channels)
    
    # CLAHE
    ycrcb_img = cv2.cvtColor(merged, cv2.COLOR_BGR2YCrCb)
    c = list(cv2.split(ycrcb_img))
    clahe = cv2.createCLAHE(
        clipLimit=config["cliplimit"],
        tileGridSize=(config["tileGrid"], config["tileGrid"])
    )
    c[0] = clahe.apply(c[0])
    clahe_img = cv2.merge(c)
    
    final = cv2.cvtColor(clahe_img, cv2.COLOR_YCrCb2BGR)
    return Image.fromarray(final)

class ImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.dataframe.iloc[idx]['image_path'])
        with Image.open(img_path) as image:
            image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
        label = self.dataframe.iloc[idx]['label']
        return image, label

def create_train_transform(config):
    """Creates training transforms from config"""
    return transforms.Compose([
        transforms.Lambda(lambda img: custom_preprocess_op(img, config)),
        transforms.RandomResizedCrop(
            size=299,
            scale=(config["scale_min"], config["scale_max"]),
            ratio=(config["ratio_min"], config["ratio_max"])
        ),
        transforms.RandomHorizontalFlip(p=config["random_hflip_p"]),
        transforms.RandomRotation(config["random_rotation_degrees"]),
        transforms.ColorJitter(
            brightness=config["brightness"],
            contrast=config["contrast"],
            saturation=config["saturation"]
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def create_val_transform(config=None):
    """Creates validation/test transforms"""
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

###############################################################################
# 3) MODEL AND TRAINING
###############################################################################
def partial_freeze_inception(model):
    """
    Freeze only the earliest layers. We keep the deeper layers (from Mixed_6 onwards) trainable.
    This helps the model adapt to images very different from ImageNet domain.
    """
    freeze = True
    for name, child in model.named_children():
        # Once we reach 'Mixed_6a', we unfreeze everything deeper.
        if name == 'Mixed_6a':
            freeze = False

        for param_name, param in child.named_parameters():
            param.requires_grad = not freeze

    return model

def initialize_model(config):
    """Initialize InceptionV3 with aux logits"""
    model = models.inception_v3(weights="DEFAULT", aux_logits=True)
    
    # Main classifier
    num_ftrs_main = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_ftrs_main),
        torch.nn.Dropout(config['dropout']),
        torch.nn.Linear(num_ftrs_main, 1)
    )
    
    # Aux classifier
    if model.AuxLogits is not None:
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_ftrs_aux),
            torch.nn.Dropout(config['dropout']),
            torch.nn.Linear(num_ftrs_aux, 1)
        )
    
    # Apply partial freezing
    model = partial_freeze_inception(model)
    
    return model

def get_optimizer(model, config):
    """Get optimizer with proper parameters"""
    if config["optimizer"] == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    elif config["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:  # sgd
        return torch.optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )

class ResourceUsageLogger:
    def __init__(self, output_dir):
        self.stats = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def log_usage(self, epoch_idx, phase="train", epoch_time=None):
        """
        Log GPU, CPU, memory usage and epoch time
        """
        p = psutil.Process()
        mem_info = p.memory_info().rss / (1024*1024)  # MB
        cpu_percent = psutil.cpu_percent(interval=None)
        gpu_percent = self.measure_gpu_usage()
        gpu_memory = self.measure_gpu_memory()
        
        self.stats.append({
            "epoch": epoch_idx,
            "phase": phase,
            "cpu": cpu_percent,
            "gpu": gpu_percent,
            "mem": mem_info,
            "gpu_mem": gpu_memory,
            "epoch_time": epoch_time  # in seconds
        })
    
    def measure_gpu_usage(self):
        """Attempts to read GPU usage via nvidia-smi"""
        if not torch.cuda.is_available():
            return 0
        try:
            cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split()).decode().strip().split('\n')
            return int(output[0])
        except:
            return 0
            
    def measure_gpu_memory(self):
        """Get GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return 0
        try:
            cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split()).decode().strip().split('\n')
            return int(output[0])
        except:
            return 0
    
    def save_stats(self, config_name):
        """Save resource usage statistics to CSV and generate plots"""
        # Save raw data to CSV
        df = pd.DataFrame(self.stats)
        csv_path = os.path.join(self.output_dir, f"{config_name}_resource_usage.csv")
        df.to_csv(csv_path, index=False)
        
        # Plot CPU Usage
        plt.figure(figsize=(16, 12))
        train_data = df[df.phase=="train"]
        val_data = df[df.phase=="val"]
        x_train = train_data.index
        x_val = val_data.index
        
        plt.plot(x_train, train_data["cpu"], 'b-', label="Train CPU")
        plt.plot(x_val, val_data["cpu"], 'r-', label="Val CPU")
        plt.xlabel("Epoch")
        plt.ylabel("CPU Usage (%)")
        plt.title(f"CPU Usage by Epoch - {config_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{config_name}_cpu_usage.png"))
        plt.close()
        
        # Plot GPU Usage
        plt.figure(figsize=(16, 12))
        plt.plot(x_train, train_data["gpu"], 'b-', label="Train GPU")
        plt.plot(x_val, val_data["gpu"], 'r-', label="Val GPU")
        plt.xlabel("Epoch")
        plt.ylabel("GPU Usage (%)")
        plt.title(f"GPU Usage by Epoch - {config_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{config_name}_gpu_usage.png"))
        plt.close()
        
        # Plot Memory Usage
        plt.figure(figsize=(16, 12))
        plt.plot(x_train, train_data["mem"], 'b-', label="Train Memory")
        plt.plot(x_val, val_data["mem"], 'r-', label="Val Memory")
        plt.xlabel("Epoch")
        plt.ylabel("Memory Usage (MB)")
        plt.title(f"Memory Usage by Epoch - {config_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{config_name}_memory_usage.png"))
        plt.close()
        
        # Plot GPU Memory Usage
        plt.figure(figsize=(16, 12))
        plt.plot(x_train, train_data["gpu_mem"], 'b-', label="Train GPU Memory")
        plt.plot(x_val, val_data["gpu_mem"], 'r-', label="Val GPU Memory")
        plt.xlabel("Epoch")
        plt.ylabel("GPU Memory Usage (MB)")
        plt.title(f"GPU Memory Usage by Epoch - {config_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{config_name}_gpu_memory_usage.png"))
        plt.close()
      
def train_model(model, dataloaders, criterion, optimizer, scheduler, config, 
                device, save_dir, prefix, resource_monitor=None):
    """Complete training loop with resource monitoring and timing"""
    num_epochs = config["epochs"]
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    history = []
    patience = config["patience"]
    no_improve = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Training phase
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            if isinstance(outputs, InceptionOutputs):
                main_out = outputs.logits
                aux_out = outputs.aux_logits
                loss_main = criterion(main_out, labels)
                loss_aux = criterion(aux_out, labels) * 0.3
                loss = loss_main + loss_aux
                preds = torch.sigmoid(main_out)
            else:
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs)
            
            loss.backward()
            optimizer.step()
            
            if scheduler and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += ((preds >= 0.5) == labels).sum().item()
        
        train_time = time.time() - epoch_start_time
        
        if resource_monitor:
            resource_monitor.log_usage(epoch, "train", train_time)
            
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects / len(dataloaders['train'].dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {train_time:.2f}s')
        
        # Validation phase
        val_start_time = time.time()
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)
                
                outputs = model(inputs)
                if isinstance(outputs, InceptionOutputs):
                    main_out = outputs.logits
                    aux_out = outputs.aux_logits
                    loss_main = criterion(main_out, labels)
                    loss_aux = criterion(aux_out, labels) * 0.3
                    loss = loss_main + loss_aux
                    preds = torch.sigmoid(main_out)
                else:
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += ((preds >= 0.5) == labels).sum().item()
        
        val_time = time.time() - val_start_time
        
        if resource_monitor:
            resource_monitor.log_usage(epoch, "val", val_time)
            
        val_loss = running_loss / len(dataloaders['val'].dataset)
        val_acc = running_corrects / len(dataloaders['val'].dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Time: {val_time:.2f}s')
        
        if scheduler and not isinstance(scheduler, OneCycleLR):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_time': train_time,
            'val_time': val_time,
            'total_epoch_time': train_time + val_time
        })
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Save final results
    model.load_state_dict(best_model_wts)
    
    df_history = pd.DataFrame(history) 
    # Training Time Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df_history['epoch'], df_history['train_time'], 'g-', label='Train Time')
    plt.plot(df_history['epoch'], df_history['val_time'], 'm-', label='Val Time')
    plt.title('Epoch Times')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_epoch_times.png"), dpi=300)
    plt.close()

    # Cumulative Time Plot
    plt.figure(figsize=(8, 6))
    cumulative_time = np.cumsum(df_history['total_epoch_time'])
    plt.plot(df_history['epoch'], cumulative_time, 'k-', label='Total Time')
    plt.title('Cumulative Training Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_cumulative_time.png"), dpi=300)
    plt.close()
        
    return model, history

def main():
    print("Loading data...")
    df = pd.read_json(data_path, lines=True)
    df['image_path'] = df.index.map(lambda x: f"Zoom_Komlos_{x}.png")
    
    # Use the same data split as the second implementation
    total_count = len(df)
    train_count = int(total_count * 0.70)
    val_count = int(total_count * 0.15)
    test_count = total_count - train_count - val_count

    train_df, val_test_df = train_test_split(df, test_size=val_count+test_count,
                                           random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(val_test_df, test_size=test_count,
                                     random_state=42, stratify=val_test_df["label"])
    
    config_five = {
        # Optimizer settings
        'lr': 0.000458717013355822,
        'optimizer': 'sgd',
        'weight_decay': 0.0000776616517657627,
        
        # Training settings
        'batch_size': 32,
        'dropout': 0.0761305705164538,
        
        # Learning rate scheduler
        'scheduler_type': 'ExponentialLR',
        'scheduler_gamma': 0.859057706777196,
        'lr_reduction_factor': 0.844318420386553,
        
        # Preprocessing settings
        'enable_custom_preprocess': False,
        
        # CLAHE parameters
        'alpha': 38,
        'beta': 218,
        'cliplimit': 0.878215086022067,
        'tileGrid': 6,
        
        # Scale parameters
        'scale_min': 0.673068273786107,
        'scale_max': 0.962445735473066,
        
        # Aspect ratio parameters
        'ratio_min': 0.865336468643324,
        'ratio_max': 1.27373321448826,
        
        # Augmentation parameters
        'random_hflip_p': 0.123579195493028,
        'random_rotation_degrees': 30,
        'brightness': 0.0711561863348833,
        'contrast': 0.488200345292253,
        'saturation': 0.158886738074651
    }
    
    config_five["epochs"] = 100
    config_five["patience"] = 15
    
    configs = {
        "config_five": config_five,
    }

    # Train both models
    for config_name, config in configs.items():
        print(f"\nTraining {config_name} model...")
        
        # Create transforms
        train_transform = create_train_transform(config)
        val_transform = create_val_transform()
        
        # Save sample transformations
        if len(train_df) > 0:
            sample_img_path = os.path.join(
                image_path,
                train_df.sample(n=1, random_state=42).iloc[0]['image_path']
            )
        
        # Create datasets with optimized DataLoader settings
        train_dataset = ImageDataset(train_df, image_path, train_transform)
        val_dataset = ImageDataset(val_df, image_path, val_transform)
        test_dataset = ImageDataset(test_df, image_path, val_transform)
        
        # Create dataloaders with the same settings as second implementation
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
            'val': DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True, drop_last=True),
            'test': DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
        }
        
        # Initialize model and training components
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = initialize_model(config)
        model = model.to(device)
        
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = get_optimizer(model, config)
        
        # Setup scheduler
        if config["scheduler_type"] == "OneCycleLR":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config['lr'],
                steps_per_epoch=len(dataloaders['train']),
                epochs=config['epochs'],
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif config["scheduler_type"] == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, gamma=config['scheduler_gamma'])
        else:
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min',
                factor=config['lr_reduction_factor'],
                patience=5
            )
        
        # Setup resource monitoring
        resource_monitor = ResourceUsageLogger(visualizations_dir)
        
        # Train model
        model, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler,
            config, device, visualizations_dir, config_name,
            resource_monitor
        )
        
        # Save resource usage plots
        resource_monitor.save_stats(config_name)

if __name__ == "__main__":
    main()