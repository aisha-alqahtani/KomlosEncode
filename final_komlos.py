import os
import copy
import time
import json
import psutil
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             roc_curve, roc_auc_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.inception import InceptionOutputs
from torch.utils.data import Dataset, DataLoader
import cv2
import optuna
from optuna.pruners import MedianPruner

# Additional Optuna visualization
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_edf,
    plot_contour
)

# For LR schedulers
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, OneCycleLR

#####################################################
# DIRECTORY SETUP FOR THE "SECOND" OPTUNA RUN
#####################################################
base_dir = os.path.expanduser("~/experiment_results_komlos_secOptuna")
os.makedirs(base_dir, exist_ok=True)

optuna_runs_dir = os.path.join(base_dir, "optuna_study_runs_secOptuna")
os.makedirs(optuna_runs_dir, exist_ok=True)

best_model_dir = os.path.join(base_dir, "best_model_secOptuna")
os.makedirs(best_model_dir, exist_ok=True)

configs_dir = os.path.join(base_dir, "configs_secOptuna")
os.makedirs(configs_dir, exist_ok=True)

metrics_dir = os.path.join(base_dir, "metrics_secOptuna")
os.makedirs(metrics_dir, exist_ok=True)

visualizations_dir = os.path.join(base_dir, "visualizations_secOptuna")
os.makedirs(visualizations_dir, exist_ok=True)

processed_images_dir = os.path.join(base_dir, "processed_images_secOptuna")
os.makedirs(processed_images_dir, exist_ok=True)

#####################################################
# PATHS TO DATA
#####################################################
image_path = os.path.expanduser("~/Zoom_Images/KomlosEncoding")
data_path = os.path.expanduser("~/Datasets/Small_Data_Human_InP.json")

#####################################################
# IMAGE PREPROCESSING
#####################################################
def normalize_image(image):
    channels = cv2.split(image)
    normalized_channels = [
        cv2.normalize(chan, None, alpha=10, beta=245, norm_type=cv2.NORM_MINMAX)
        for chan in channels
    ]
    return cv2.merge(normalized_channels)

def CLAHE(image):
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb_img))
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(2, 2))
    channels[0] = clahe.apply(channels[0])
    clahe_img = cv2.merge(channels)
    return cv2.cvtColor(clahe_img, cv2.COLOR_YCrCb2BGR)

class CustomPreprocess:
    def __call__(self, image):
        image = np.array(image)
        normalized_image = normalize_image(image)
        clahe_image = CLAHE(normalized_image)
        return Image.fromarray(clahe_image)

#####################################################
# UPDATED TRANSFORMS (Stronger Augmentations)
#####################################################
def get_transforms():
    custom_preprocess = CustomPreprocess()
    return {
        'train': transforms.Compose([
            transforms.Lambda(lambda img: custom_preprocess(img)),
            transforms.RandomResizedCrop(299, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

#####################################################
# DATA LOADING
#####################################################
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

def load_data():
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"File {data_path} does not exist")
    df = pd.read_json(data_path, lines=True)
    # Ensure the correct image filename is set
    df['image_path'] = df.index.map(lambda x: f"Zoom_Komlos_{x}.png")
    return df

#####################################################
# MODEL INITIALIZATION
# Use pretrained InceptionV3 (weights="DEFAULT"),
# then disable the AuxLogits manually.
#####################################################
def initialize_model(config):
    # This call enforces aux_logits=True internally, but we won't use them
    model = models.inception_v3(weights="DEFAULT")

    # Disable the AuxLogits branch
    model.AuxLogits = None

    # Replace final FC
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_ftrs),
        torch.nn.Dropout(config['dropout']),
        torch.nn.Linear(num_ftrs, 1),
    )
    return model

#####################################################
# FREEZE LAYERS EXCEPT FOR THE FINAL PART
#####################################################
def partial_freeze_inception(model):
    """
    Freeze all params, then unfreeze the final Inception blocks + fc.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze final blocks: Mixed_7b and Mixed_7c
    if hasattr(model, 'Mixed_7b'):
        for param in model.Mixed_7b.parameters():
            param.requires_grad = True
    if hasattr(model, 'Mixed_7c'):
        for param in model.Mixed_7c.parameters():
            param.requires_grad = True

    # Unfreeze final FC
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def get_optimizer(model, config):
    opt = config.get("optimizer", "adam")
    if opt == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    elif opt == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    elif opt == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError("Unsupported optimizer type provided!")

#####################################################
# TRAIN/VAL LOOPS
# Must handle the possibility of InceptionOutputs
#####################################################
def train_loop(model, criterion, optimizer, dataloaders, device, scheduler=None, epoch_idx=0, config=None):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device).view(-1, 1).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        # If the model still returns an InceptionOutputs object:
        if isinstance(outputs, InceptionOutputs):
            outputs = outputs.logits

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # If using OneCycleLR, step after each batch
        if config and config.get("scheduler_type", "") == "OneCycleLR" and scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs).round()
        running_corrects += torch.sum(preds == labels).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc  = running_corrects / total_samples
    return epoch_loss, epoch_acc

def validate_loop(model, criterion, dataloaders, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1).float()

            outputs = model(inputs)
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs).round()
            running_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc  = running_corrects / total_samples
    return epoch_loss, epoch_acc

#####################################################
# CREATE DATALOADERS
#####################################################
def create_dataloaders(df, batch_size, num_workers=4):
    """Creates train/val/test splits (70/15/15) w/ random_state=42."""
    transforms_ = get_transforms()
    total_count = len(df)
    train_count = int(total_count * 0.70)
    val_count   = int(total_count * 0.15)
    test_count  = total_count - train_count - val_count

    train_df, val_test_df = train_test_split(df, test_size=val_count+test_count, random_state=42)
    val_df, test_df       = train_test_split(val_test_df, test_size=test_count, random_state=42)

    train_dataset = ImageDataset(train_df, image_path, transform=transforms_['train'])
    val_dataset   = ImageDataset(val_df,   image_path, transform=transforms_['val'])
    test_dataset  = ImageDataset(test_df,  image_path, transform=transforms_['val'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True),
        'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=True),
        'test':  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    }

    return train_df, val_df, test_df, dataloaders

#####################################################
# OPTUNA OBJECTIVE FUNCTION
# => We DO NOT pass aux_logits anymore
# => We partial freeze after initialization
#####################################################
def objective(trial):
    config = {
        "epochs": 40,  # can adjust
        "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "patience": 5,
        "scheduler_type": trial.suggest_categorical("scheduler_type", ["OneCycleLR", "ExponentialLR", "ReduceLROnPlateau"]),
        "scheduler_gamma": trial.suggest_float("scheduler_gamma", 0.85, 0.95),
        "lr_reduction_factor": trial.suggest_float("lr_reduction_factor", 0.7, 0.9),
    }

    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data()
    _, _, _, dataloaders = create_dataloaders(df, config["batch_size"], 4)

    model = initialize_model(config)
    # partial freeze
    model = partial_freeze_inception(model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, config)

    # Steps per epoch
    steps_per_epoch = len(dataloaders['train'])

    if config["scheduler_type"] == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            steps_per_epoch=steps_per_epoch,
            epochs=config["epochs"],
            anneal_strategy='cos'
        )
    elif config["scheduler_type"] == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=config["scheduler_gamma"])
    else:  # ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=config["lr_reduction_factor"],
                                      patience=5, verbose=False)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improvement_count = 0

    trial_dir = os.path.join(optuna_runs_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    training_metrics_log = []

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_loop(
            model, criterion, optimizer, dataloaders, device,
            scheduler=(scheduler if config["scheduler_type"] == "OneCycleLR" else None),
            epoch_idx=epoch, config=config
        )
        val_loss, val_acc = validate_loop(model, criterion, dataloaders, device)

        # Step for ExponentialLR or ReduceLROnPlateau
        if config["scheduler_type"] == "ExponentialLR":
            scheduler.step()
        elif config["scheduler_type"] == "ReduceLROnPlateau":
            scheduler.step(val_loss)

        metrics = {
            "epoch": epoch+1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc)
        }
        training_metrics_log.append(metrics)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if no_improvement_count >= config["patience"]:
            print(f"Early stopping trial {trial.number} at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)

    df_metrics = pd.DataFrame(training_metrics_log)
    df_metrics.to_csv(os.path.join(trial_dir, "training_metrics.csv"), index=False)

    return best_loss

#####################################################
# UTILITY TO SAVE OPTUNA PLOTS
#####################################################
def save_optuna_plot(fig, filename):
    plt.tight_layout()
    if hasattr(fig, "savefig"):
        fig.savefig(filename, dpi=300)
        plt.close(fig)
    elif hasattr(fig, "get_figure"):
        fig.get_figure().savefig(filename, dpi=300)
        plt.close(fig.get_figure())
    elif isinstance(fig, np.ndarray):
        ax = fig.flatten()[0]
        ax.get_figure().savefig(filename, dpi=300)
        plt.close(ax.get_figure())

#####################################################
# SIMPLE UTILITY TO SAVE A FEW PRE-PROCESSED IMAGES
#####################################################
def save_sample_processed_images(df):
    from torchvision.transforms import ToPILImage
    transforms_ = get_transforms()
    sample_df = df.sample(3, random_state=42)

    out_dir = processed_images_dir
    os.makedirs(out_dir, exist_ok=True)

    for i, row in sample_df.iterrows():
        img_path = os.path.join(image_path, row['image_path'])
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            proc = transforms_['train'](im)
            pil_img = ToPILImage()(proc)
            pil_img.save(os.path.join(out_dir, f"sample_processed_{i}.png"))

#####################################################
# MAIN FUNCTION
#####################################################
def main_secOptuna():
    start_time = time.time()

    print("=== Starting Second Optuna Study with partial freezing and new transforms ===")
    storage_url = f"sqlite:///{optuna_runs_dir}/optuna_study_secOptuna.db"

    study = optuna.create_study(
        study_name="komlos_second_study",
        direction="minimize",
        storage=storage_url,
        load_if_exists=False,
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=0, interval_steps=1)
    )

    study.optimize(objective, n_trials=25)

    best_trial = study.best_trial
    best_config = best_trial.params
    best_config["epochs"] = 40
    print("\nBest trial found:", best_trial.number, "value:", best_trial.value)
    print("Best hyperparams:", best_config)

    # 2) SAVE STUDY RESULTS + PLOTS
    all_results = []
    for t in study.trials:
        row = t.params.copy()
        row["value"] = t.value
        row["state"] = t.state.name
        all_results.append(row)
    df_study = pd.DataFrame(all_results)
    df_study.to_csv(os.path.join(metrics_dir, "full_optuna_study_results_secOptuna.csv"), index=False)
    print(f"Saved all study trials to {metrics_dir}")

    fig = plot_optimization_history(study)
    save_optuna_plot(fig, os.path.join(visualizations_dir, "optuna_optimization_history_secOptuna.png"))

    fig = plot_param_importances(study)
    save_optuna_plot(fig, os.path.join(visualizations_dir, "optuna_param_importances_secOptuna.png"))

    fig = plot_parallel_coordinate(study)
    save_optuna_plot(fig, os.path.join(visualizations_dir, "optuna_parallel_coordinate_secOptuna.png"))

    fig = plot_slice(study)
    save_optuna_plot(fig, os.path.join(visualizations_dir, "optuna_slice_secOptuna.png"))

    fig = plot_edf(study)
    save_optuna_plot(fig, os.path.join(visualizations_dir, "optuna_edf_secOptuna.png"))

    if len(best_trial.params) > 1:
        fig = plot_contour(study)
        save_optuna_plot(fig, os.path.join(visualizations_dir, "optuna_contour_secOptuna.png"))

    final_config = {
        "epochs": 40,
        "lr": float(best_config.get("lr", 1e-4)),
        "optimizer": best_config.get("optimizer", "adam"),
        "weight_decay": float(best_config.get("weight_decay", 1e-5)),
        "batch_size": int(best_config.get("batch_size", 32)),
        "dropout": float(best_config.get("dropout", 0.0)),
        "patience": 5,
        "scheduler_type": best_config.get("scheduler_type", "OneCycleLR"),
        "scheduler_gamma": float(best_config.get("scheduler_gamma", 0.9)),
        "lr_reduction_factor": float(best_config.get("lr_reduction_factor", 0.8))
    }

    with open(os.path.join(configs_dir, "best_config_secOptuna.json"), "w") as f:
        json.dump(final_config, f, indent=2)
    print(f"Saved best config to {configs_dir}/best_config_secOptuna.json")

    # Sample images
    df = load_data()
    save_sample_processed_images(df)

    # STRATIFIED K-FOLD
    print("\n=== Starting Stratified K-Fold with the best config ===")
    total_count = len(df)
    train_count = int(total_count * 0.70)
    val_count   = int(total_count * 0.15)
    test_count  = total_count - train_count - val_count

    train_df, val_test_df = train_test_split(df, test_size=val_count + test_count, random_state=42)
    val_df, test_df = train_test_split(val_test_df, test_size=test_count, random_state=42)

    trainval_df = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")
    print(f"Combined train+val size: {len(trainval_df)}")

    transforms_ = get_transforms()
    trainval_labels = trainval_df["label"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    fold_idx = 0

    def measure_resource_usage():
        p = psutil.Process()
        mem_info = p.memory_info().rss / (1024*1024)  # MB
        cpu_percent = psutil.cpu_percent(interval=None)
        return mem_info, cpu_percent

    class ImageDatasetCV(Dataset):
        def __init__(self, df_, img_dir, transform=None):
            self.df_ = df_
            self.img_dir = img_dir
            self.transform = transform
        def __len__(self):
            return len(self.df_)
        def __getitem__(self, idx):
            row_ = self.df_.iloc[idx]
            img_path_ = os.path.join(self.img_dir, row_['image_path'])
            with Image.open(img_path_) as im_:
                im_ = im_.convert('RGB')
                if self.transform:
                    im_ = self.transform(im_)
            return im_, row_['label']

    for train_index, val_index in skf.split(trainval_df, trainval_labels):
        start_fold_time = time.time()
        mem_before, cpu_before = measure_resource_usage()

        fold_idx += 1
        print(f"\n--- Fold {fold_idx}/5 ---")

        train_fold_df = trainval_df.iloc[train_index].reset_index(drop=True)
        val_fold_df   = trainval_df.iloc[val_index].reset_index(drop=True)

        train_dataset = ImageDatasetCV(train_fold_df, image_path, transform=transforms_["train"])
        val_dataset   = ImageDatasetCV(val_fold_df,   image_path, transform=transforms_["val"])

        train_loader = DataLoader(train_dataset, batch_size=final_config["batch_size"],
                                  shuffle=True, num_workers=4, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=final_config["batch_size"],
                                  shuffle=False, num_workers=4, drop_last=True)

        model = initialize_model(final_config)
        model = partial_freeze_inception(model)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device_)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = get_optimizer(model, final_config)
        steps_per_epoch = len(train_loader)

        if final_config["scheduler_type"] == "OneCycleLR":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=final_config['lr'],
                steps_per_epoch=steps_per_epoch,
                epochs=final_config["epochs"],
                anneal_strategy='cos'
            )
        elif final_config["scheduler_type"] == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, gamma=final_config["scheduler_gamma"])
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                          factor=final_config["lr_reduction_factor"],
                                          patience=5, verbose=False)

        best_loss = float('inf')
        best_weights = copy.deepcopy(model.state_dict())
        no_improvement_count = 0
        patience = final_config["patience"]

        train_losses, val_losses = [], []
        train_accs, val_accs     = [], []

        fold_dataloaders = {'train': train_loader, 'val': val_loader}

        for epoch in range(final_config["epochs"]):
            t_loss, t_acc = train_loop(
                model, criterion, optimizer, fold_dataloaders, device_,
                scheduler=(scheduler if final_config["scheduler_type"] == "OneCycleLR" else None),
                epoch_idx=epoch, config=final_config
            )
            v_loss, v_acc = validate_loop(model, criterion, fold_dataloaders, device_)

            train_losses.append(t_loss)
            val_losses.append(v_loss)
            train_accs.append(t_acc)
            val_accs.append(v_acc)

            # Step the scheduler if not OneCycleLR
            if final_config["scheduler_type"] == "ExponentialLR":
                scheduler.step()
            elif final_config["scheduler_type"] == "ReduceLROnPlateau":
                scheduler.step(v_loss)

            if v_loss < best_loss:
                best_loss = v_loss
                best_weights = copy.deepcopy(model.state_dict())
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping on epoch {epoch+1}, fold {fold_idx}")
                break

        # Load the best weights for this fold
        model.load_state_dict(best_weights)

        # Plot training curves
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold_idx} - Loss Curves')
        plt.legend()
        plt.savefig(os.path.join(visualizations_dir, f'loss_curve_cv_fold_{fold_idx}.png'))
        plt.close()

        plt.figure()
        plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
        plt.plot(range(1, len(val_accs)+1), val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Fold {fold_idx} - Accuracy Curves')
        plt.legend()
        plt.savefig(os.path.join(visualizations_dir, f'accuracy_curve_cv_fold_{fold_idx}.png'))
        plt.close()

        # Evaluate on val fold
        model.eval()
        val_labels, val_probs, val_preds = [], [], []
        with torch.no_grad():
            for inp, lbl in val_loader:
                inp = inp.to(device_)
                lbl = lbl.to(device_).float()
                out = model(inp)

                if isinstance(out, InceptionOutputs):
                    out = out.logits

                prob = torch.sigmoid(out).cpu().numpy().flatten()
                pred = (prob >= 0.5).astype(int)
                val_probs.extend(prob)
                val_labels.extend(lbl.cpu().numpy())
                val_preds.extend(pred)

        f1_val  = f1_score(val_labels, val_preds)
        acc_val = accuracy_score(val_labels, val_preds)
        auc_val = roc_auc_score(val_labels, val_probs)
        prec_val = precision_score(val_labels, val_preds)
        rec_val  = recall_score(val_labels, val_preds)
        cf_val   = confusion_matrix(val_labels, val_preds)

        df_cf_fold = pd.DataFrame(cf_val, index=["Actual_0", "Actual_1"], columns=["Pred_0","Pred_1"])
        df_cf_fold.to_csv(os.path.join(metrics_dir, f"confusion_matrix_cv_fold_{fold_idx}.csv"), index=False)

        # Plot confusion matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(
            cf_val,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=["Pred_0","Pred_1"],
            yticklabels=["Actual_0","Actual_1"],
            cbar=False  # remove color bar if you want
        )
        plt.title(f'Fold {fold_idx} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(visualizations_dir, f'confusion_matrix_cv_fold_{fold_idx}.png'))
        plt.close()

        # Plot ROC
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Fold {fold_idx}')
        plt.plot([0,1],[0,1],'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Fold {fold_idx} - ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(visualizations_dir, f'roc_curve_cv_fold_{fold_idx}.png'))
        plt.close()

        fold_time = time.time() - start_fold_time
        mem_after, cpu_after = measure_resource_usage()

        fold_result = {
            "fold": fold_idx,
            "val_f1": f1_val,
            "val_accuracy": acc_val,
            "val_auc": auc_val,
            "val_precision": prec_val,
            "val_recall": rec_val,
            "training_time_sec": fold_time,
            "memory_usage_before_MB": mem_before,
            "memory_usage_after_MB": mem_after,
            "cpu_usage_before_%": cpu_before,
            "cpu_usage_after_%": cpu_after
        }
        fold_results.append(fold_result)

    cv_results_df = pd.DataFrame(fold_results)
    cv_results_path = os.path.join(metrics_dir, "cross_validation_metrics_secOptuna.csv")
    cv_results_df.to_csv(cv_results_path, index=False)
    print(f"\nCross-validation metrics saved to: {cv_results_path}")

    plt.figure()
    plt.bar(cv_results_df['fold'], cv_results_df['training_time_sec'])
    plt.xlabel('Fold')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time per Fold (CV)')
    plt.savefig(os.path.join(visualizations_dir, "training_time_cv_secOptuna.png"))
    plt.close()

    # FINAL RETRAIN ON WHOLE TRAIN+VAL => EVAL ON TEST
    print("\n--- Final Retraining on the entire train+val split ---")

    trainval_dataset = ImageDatasetCV(trainval_df, image_path, transform=transforms_["train"])
    trainval_loader  = DataLoader(trainval_dataset, batch_size=final_config["batch_size"],
                                  shuffle=True, num_workers=4, drop_last=True)

    test_dataset = ImageDatasetCV(test_df, image_path, transform=transforms_["val"])
    test_loader  = DataLoader(test_dataset, batch_size=final_config["batch_size"],
                              shuffle=False, num_workers=4, drop_last=True)

    model = initialize_model(final_config)
    model = partial_freeze_inception(model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device_)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, final_config)
    steps_per_epoch = len(trainval_loader)

    if final_config["scheduler_type"] == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=final_config['lr'],
            steps_per_epoch=steps_per_epoch,
            epochs=final_config["epochs"],
            anneal_strategy='cos'
        )
    elif final_config["scheduler_type"] == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=final_config["scheduler_gamma"])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=final_config["lr_reduction_factor"],
                                      patience=5, verbose=False)

    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    no_improvement_count = 0

    # (Optional) internal train/val split for early stopping
    int_tv_df, int_val_df = train_test_split(trainval_df, test_size=0.1, random_state=42)
    int_tv_dataset   = ImageDatasetCV(int_tv_df, image_path, transform=transforms_["train"])
    int_val_dataset  = ImageDatasetCV(int_val_df, image_path, transform=transforms_["val"])
    int_tv_loader    = DataLoader(int_tv_dataset, batch_size=final_config["batch_size"],
                                  shuffle=True, num_workers=4, drop_last=True)
    int_val_loader   = DataLoader(int_val_dataset, batch_size=final_config["batch_size"],
                                  shuffle=False, num_workers=4, drop_last=True)

    fold_dataloaders = {'train': int_tv_loader, 'val': int_val_loader}

    for epoch in range(final_config["epochs"]):
        t_loss, t_acc = train_loop(
            model, criterion, optimizer, fold_dataloaders, device_,
            scheduler=(scheduler if final_config["scheduler_type"] == "OneCycleLR" else None),
            epoch_idx=epoch, config=final_config
        )
        v_loss, v_acc = validate_loop(model, criterion, fold_dataloaders, device_)

        if final_config["scheduler_type"] == "ExponentialLR":
            scheduler.step()
        elif final_config["scheduler_type"] == "ReduceLROnPlateau":
            scheduler.step(v_loss)

        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= final_config["patience"]:
            print(f"Early stopping final retrain at epoch {epoch+1}")
            break

    model.load_state_dict(best_weights)

    final_ckpt_path = os.path.join(best_model_dir, "best_model_weights_secOptuna_CV.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Saved final cross-validation model to: {final_ckpt_path}")

    # Evaluate on the test set
    model.eval()
    test_labels, test_probs, test_preds = [], [], []
    with torch.no_grad():
        for inp, lbl in test_loader:
            inp = inp.to(device_)
            lbl = lbl.to(device_).float()
            out = model(inp)

            if isinstance(out, InceptionOutputs):
                out = out.logits

            prob = torch.sigmoid(out).cpu().numpy().flatten()
            pred = (prob >= 0.5).astype(int)
            test_labels.extend(lbl.cpu().numpy())
            test_probs.extend(prob)
            test_preds.extend(pred)

    test_f1   = f1_score(test_labels, test_preds)
    test_acc  = accuracy_score(test_labels, test_preds)
    test_auc  = roc_auc_score(test_labels, test_probs)
    test_prec = precision_score(test_labels, test_preds)
    test_rec  = recall_score(test_labels, test_preds)
    test_cf   = confusion_matrix(test_labels, test_preds)

    final_metrics = {
        "test_f1": test_f1,
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "test_precision": test_prec,
        "test_recall": test_rec
    }
    final_metrics_df = pd.DataFrame([final_metrics])
    final_metrics_cv_path = os.path.join(metrics_dir, "test_metrics_secOptuna_cv.csv")
    final_metrics_df.to_csv(final_metrics_cv_path, index=False)

    df_test_cf = pd.DataFrame(test_cf, index=["Actual_0","Actual_1"], columns=["Pred_0","Pred_1"])
    df_test_cf.to_csv(os.path.join(metrics_dir, "confusion_matrix_secOptuna_test.csv"), index=False)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        test_cf,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Pred_0","Pred_1"],
        yticklabels=["Actual_0","Actual_1"],
        cbar=False
    )
    plt.title('Final Test Confusion Matrix (SecOptuna)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(visualizations_dir, "confusion_matrix_secOptuna_test.png"))
    plt.close()

    plt.figure()
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    plt.plot(fpr, tpr, label='Test ROC')
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Final Test ROC (SecOptuna)')
    plt.legend()
    plt.savefig(os.path.join(visualizations_dir, "roc_curve_secOptuna_test.png"))
    plt.close()

    print("\nFinal test metrics (after CV) saved to:", final_metrics_cv_path)

    end_time = time.time()
    total_time = end_time - start_time
    readme_path = os.path.join(base_dir, "README_secOptuna.md")
    with open(readme_path, "w") as f:
        f.write("# Second Optuna Run (komlos_secOptuna)\n\n")
        f.write("## Summary:\n")
        f.write("- This run has partial freezing of Inception v3 (pretrained), no direct aux_logits.\n")
        f.write("- We disable the auxiliary head post-creation (`model.AuxLogits = None`).\n")
        f.write("- Stronger augmentation helps reduce overfitting.\n")
        f.write("- Confusion matrices, ROC curves, and other plots saved in the visualizations directory.\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n")

    print(f"\nAll done! Total execution time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main_secOptuna()
