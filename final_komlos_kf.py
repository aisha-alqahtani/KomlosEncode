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
os.makedirs(base_dir, exist_ok=True)

models_dir = os.path.join(base_dir, "best_model_komlos")
os.makedirs(models_dir, exist_ok=True)

image_samples_dir = os.path.join(base_dir, "processed_images_komlos")
os.makedirs(image_samples_dir, exist_ok=True)

optuna_runs_dir = os.path.join(base_dir, "optuna_study_runs_komlos")
metrics_dir = os.path.join(base_dir, "metrics_komlos")
visualizations_dir = os.path.join(base_dir, "visualizations_komlos")
os.makedirs(visualizations_dir, exist_ok=True)

# Paths to data & images
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

def save_transformation_samples(image_path, config, out_dir, prefix):
    """Save original and transformed versions of sample images"""
    original_img = Image.open(image_path).convert('RGB')
    
    # Save original
    original_save_path = os.path.join(out_dir, f"{prefix}_original.png")
    original_img.save(original_save_path)
    
    # Custom preprocessing
    preprocessed_img = custom_preprocess_op(original_img, config)
    preproc_save_path = os.path.join(out_dir, f"{prefix}_preprocessed.png")
    preprocessed_img.save(preproc_save_path)
    
    # Full transform pipeline
    transform = create_train_transform(config)
    transformed_tensor = transform(original_img)
    transformed_img = to_pil_image(transformed_tensor)
    transform_save_path = os.path.join(out_dir, f"{prefix}_full_transform.png")
    transformed_img.save(transform_save_path)
    
    return {
        "original": original_save_path,
        "preprocessed": preproc_save_path,
        "transformed": transform_save_path
    }

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
    def __init__(self):
        self.stats = []
        
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
    
    def plot_usage(self, suffix):
        """Plot resource usage over epochs"""
        df = pd.DataFrame(self.stats)
        bar_width = 0.4
        
        # CPU Usage
        plt.figure(figsize=(10, 6))
        train_data = df[df.phase=="train"]
        val_data = df[df.phase=="val"]
        x_train = np.arange(len(train_data))
        x_val = x_train + bar_width
        
        plt.bar(x_train, train_data["cpu"], width=bar_width, label="Train CPU")
        plt.bar(x_val, val_data["cpu"], width=bar_width, label="Val CPU")
        plt.xlabel("Epoch")
        plt.ylabel("CPU Usage (%)")
        plt.title(f"CPU Usage by Epoch")
        plt.legend()
        plt.xticks(x_train + bar_width/2, train_data["epoch"].astype(int))
        plt.savefig(os.path.join(visualizations_dir, f"cpu_usage_{suffix}.png"))
        plt.close()
        
        # GPU Usage & Memory
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # GPU Utilization
        ax1.bar(x_train, train_data["gpu"], width=bar_width, label="Train GPU")
        ax1.bar(x_val, val_data["gpu"], width=bar_width, label="Val GPU")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("GPU Usage (%)")
        ax1.set_title(f"GPU Usage by Epoch")
        ax1.legend()
        ax1.set_xticks(x_train + bar_width/2)
        ax1.set_xticklabels(train_data["epoch"].astype(int))
        
        # GPU Memory
        ax2.bar(x_train, train_data["gpu_mem"], width=bar_width, label="Train GPU Mem")
        ax2.bar(x_val, val_data["gpu_mem"], width=bar_width, label="Val GPU Mem")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("GPU Memory (MB)")
        ax2.set_title(f"GPU Memory Usage by Epoch")
        ax2.legend()
        ax2.set_xticks(x_train + bar_width/2)
        ax2.set_xticklabels(train_data["epoch"].astype(int))
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, f"gpu_metrics_{suffix}.png"))
        plt.close()
        
        # Epoch Times
        plt.figure(figsize=(10, 6))
        plt.plot(train_data["epoch"], train_data["epoch_time"], 
                marker='o', label="Train Time", linewidth=2)
        plt.plot(val_data["epoch"], val_data["epoch_time"], 
                marker='o', label="Val Time", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.title(f"Epoch Training Time - {suffix}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(visualizations_dir, f"epoch_times_{suffix}.png"))
        plt.close()
        
        # Save stats to CSV
        df.to_csv(os.path.join(metrics_dir, f"resource_usage_{suffix}.csv"), index=False)
        
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
                loss = criterion(main_out, labels) + 0.3 * criterion(aux_out, labels)
                preds = torch.sigmoid(main_out)
            else:
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
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
                    loss = criterion(main_out, labels) + 0.3 * criterion(aux_out, labels)
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
    torch.save(model.state_dict(), os.path.join(save_dir, f"{prefix}_model.pth"))
    
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(save_dir, f"{prefix}_training_history.csv"), index=False)
    
    # Loss Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df_history['epoch'], df_history['train_loss'], 'b-', label='Train Loss')
    plt.plot(df_history['epoch'], df_history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_loss.png"), dpi=300)
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(df_history['epoch'], df_history['train_acc'], 'b-', label='Train Acc')
    plt.plot(df_history['epoch'], df_history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_accuracy.png"), dpi=300)
    plt.close()

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

def plot_confusion_matrix_and_roc(true_labels, pred_scores, save_prefix):
    """Enhanced confusion matrix and ROC plotting"""
    # Compute predicted labels
    pred_labels = (pred_scores >= 0.5).astype(int)
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    roc_auc = roc_auc_score(true_labels, pred_scores)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"{save_prefix}_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm, roc_auc

def compute_metrics(true_labels, pred_scores, threshold=0.5):
    """Compute comprehensive metrics"""
    pred_labels = (pred_scores >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels),
        'roc_auc': roc_auc_score(true_labels, pred_scores),
        'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist(),
        'threshold': threshold
    }
    
    return metrics

def evaluate_model(model, dataloader, device, save_dir, prefix):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
            preds = torch.sigmoid(outputs).cpu().numpy().ravel()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().ravel())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute and save metrics
    metrics = compute_metrics(all_labels, all_preds)
    with open(os.path.join(save_dir, f"{prefix}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix and ROC
    plot_confusion_matrix_and_roc(
        all_labels, all_preds,
        os.path.join(save_dir, f"{prefix}")
    )
    
    return metrics

def analyze_trial_stability(metrics_df):
    """
    Analyze trial stability based on:
    1. Correlation between train/val curves
    2. Fluctuations in validation metrics
    3. Gap between train/val performance
    4. Overall trend of improvement
    """
    # Convert columns to numeric if needed
    numeric_cols = ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
    for col in numeric_cols:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
    
    # Only analyze if we have enough epochs
    if len(metrics_df) < 5:
        return None
        
    # Calculate correlation between train/val
    loss_corr = np.corrcoef(metrics_df['train_loss'], metrics_df['val_loss'])[0,1]
    acc_corr = np.corrcoef(metrics_df['train_accuracy'], metrics_df['val_accuracy'])[0,1]
    
    # Calculate stability metrics
    val_loss_fluctuation = metrics_df['val_loss'].diff().abs().mean()
    val_acc_fluctuation = metrics_df['val_accuracy'].diff().abs().mean()
    
    # Calculate gaps between train/val
    loss_gap = np.mean(abs(metrics_df['train_loss'] - metrics_df['val_loss']))
    acc_gap = np.mean(abs(metrics_df['train_accuracy'] - metrics_df['val_accuracy']))
    
    # Calculate overall trends
    epochs = np.arange(len(metrics_df))
    val_loss_trend = np.polyfit(epochs, metrics_df['val_loss'], 1)[0]
    val_acc_trend = np.polyfit(epochs, metrics_df['val_accuracy'], 1)[0]
    
    # Early convergence check (want steady improvement but not too fast)
    early_conv_penalty = 0
    if len(metrics_df) < 10:  # Penalize very short training
        early_conv_penalty = 1.0
    
    # Compute stability score (lower is better)
    stability_score = (
        val_loss_fluctuation * 2.0 +      # Heavily penalize loss fluctuations
        val_acc_fluctuation * 1.5 +       # Penalize accuracy fluctuations
        loss_gap * 1.0 +                  # Consider train/val gaps
        acc_gap * 1.0 -                   # Consider train/val gaps
        loss_corr * 0.5 -                 # Reward good correlations
        acc_corr * 0.5 +                  # Reward good correlations
        abs(val_loss_trend) * 0.3 +       # Small penalty for steep trends
        early_conv_penalty                # Penalize too-early convergence
    )
    
    return {
        'num_epochs': len(metrics_df),
        'final_val_loss': metrics_df['val_loss'].iloc[-1],
        'final_val_acc': metrics_df['val_accuracy'].iloc[-1],
        'loss_correlation': loss_corr,
        'acc_correlation': acc_corr,
        'val_loss_fluctuation': val_loss_fluctuation,
        'val_acc_fluctuation': val_acc_fluctuation,
        'loss_gap': loss_gap,
        'acc_gap': acc_gap,
        'val_loss_trend': val_loss_trend,
        'val_acc_trend': val_acc_trend,
        'stability_score': stability_score
    }

def find_most_stable_trial(base_dir):
    """
    Analyze all trials to find the most stable one
    """
    # Load full study results
    study_results = pd.read_csv(os.path.join(base_dir, "metrics_komlos/full_optuna_study_results.csv"))
    
    # Dictionary to store trial metrics
    trial_metrics = {}
    
    # Walk through all trial directories
    trials_dir = os.path.join(base_dir, "optuna_study_runs_komlos")
    for trial_dir in os.listdir(trials_dir):
        if trial_dir.startswith("trial_"):
            trial_num = int(trial_dir.split("_")[1])
            metrics_path = os.path.join(trials_dir, trial_dir, "training_metrics.csv")
            
            if os.path.exists(metrics_path):
                # Load and analyze trial metrics
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    stability_metrics = analyze_trial_stability(metrics_df)
                    
                    if stability_metrics is not None:
                        trial_metrics[trial_num] = stability_metrics
                except Exception as e:
                    print(f"Error processing trial {trial_num}: {e}")
    
    # Convert to DataFrame
    stability_df = pd.DataFrame.from_dict(trial_metrics, orient='index')
    
    # Sort by stability score (lower is better)
    most_stable_trials = stability_df.sort_values('stability_score').head(5)
    
    print("\nTop 5 Most Stable Trials:")
    print(most_stable_trials[['num_epochs', 'final_val_acc', 'stability_score']])
    
    # Get configuration for most stable trial
    most_stable_trial = most_stable_trials.index[0]
    stable_config = study_results.iloc[most_stable_trial].to_dict()
    
    print("\nMost Stable Trial Configuration:")
    for key, value in stable_config.items():
        print(f"{key}: {value}")
        
    return stable_config, most_stable_trials

def main():
    print("Loading data...")
    df = pd.read_json(data_path, lines=True)
    df['image_path'] = df.index.map(lambda x: f"Zoom_Komlos_{x}.png")
    
    print("Loading Optuna study...")
    storage_url = f"sqlite:///{optuna_runs_dir}/optuna_study.db"
    study = optuna.load_study(
        study_name="Komlos_final_study",
        storage=storage_url
    )
    
    # Get best trial config
    best_trial = study.best_trial
    best_config = best_trial.params
    best_config["epochs"] = 100
    best_config["patience"] = 15
    
    # Create stable config
    stable_config, stability_metrics = find_most_stable_trial(base_dir)

    
    # Data split
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'])
    
    configs = {
        "best_val": best_config,
        "stable": stable_config
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
            save_transformation_samples(
                sample_img_path,
                config,
                image_samples_dir,
                f"{config_name}"
            )
        
        # Create datasets
        train_dataset = ImageDataset(train_df, image_path, train_transform)
        val_dataset = ImageDataset(val_df, image_path, val_transform)
        test_dataset = ImageDataset(test_df, image_path, val_transform)
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=4),
            'test': DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=4)
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
        resource_monitor = ResourceUsageLogger()
        
        # Train model
        model, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler,
            config, device, models_dir, config_name,
            resource_monitor
        )
        
        # Save resource usage plots
        resource_monitor.plot_usage(config_name)
        
        # Evaluate on test set
        print(f"\nEvaluating {config_name} model...")
        metrics = evaluate_model(
            model, dataloaders['test'], device,
            metrics_dir, f"{config_name}_test"
        )
        
        print(f"\n{config_name} Test Metrics:")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"{k}: {v}")

if __name__ == "__main__":
    main()