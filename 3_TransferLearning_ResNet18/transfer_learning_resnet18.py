"""
==============================================================
Tugas Deep Learning - Implementasi 3: Transfer Learning (ResNet-18)
==============================================================
Nama   : Imam Dzaqhoir
NIM    : H071241048
Kelas  : [Kelas]
Dosen  : [Nama Dosen]

Deskripsi:
  Implementasi Transfer Learning menggunakan arsitektur ResNet-18
  pretrained pada ImageNet untuk klasifikasi jenis sampah daur ulang
  (TrashNet - 6 kelas).

  Metode Fine-tuning 2-Tahap:
    Tahap 1 - Feature Extraction: Bobot backbone dibekukan (frozen),
              hanya melatih Fully-Connected Head baru (5 epoch).
    Tahap 2 - Fine-tuning: Membuka lapisan atas (layer4) dan FC head
              dengan learning rate kecil diferensial (8 epoch).
==============================================================
"""

import os
import sys
import copy
import time
import zipfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

# ── Reproducibility ─────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ── Hyperparameters ─────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 5       # Feature extraction
EPOCHS_PHASE2 = 8       # Fine-tuning
LR_HEAD       = 1e-3
LR_FINETUNE   = 1e-4
IMG_SIZE      = 224     # Standar resolusi ImageNet untuk ResNet
NUM_CLASSES   = 6
DATA_DIR      = "./data/dataset-resized"
SHARED_DATA   = "../1_CNN_WasteClassification/data/dataset-resized"
DATA_URL      = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"


# ── Dataset Auto-Resolver ────────────────────────────────────
def ensure_dataset():
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) >= 6:
        return DATA_DIR
    if os.path.exists(SHARED_DATA) and len(os.listdir(SHARED_DATA)) >= 6:
        print(f"[INFO] Using shared dataset from {SHARED_DATA}")
        return SHARED_DATA

    os.makedirs("./data", exist_ok=True)
    zip_path = "./data/dataset-resized.zip"
    if not os.path.exists(zip_path):
        print(f"[INFO] Downloading TrashNet dataset (~41 MB)...")
        req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(zip_path, 'wb') as out_f:
            out_f.write(resp.read())

    print("[INFO] Extracting TrashNet...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall("./data")
    return DATA_DIR


active_data_dir = ensure_dataset()

# ── Transforms & Data Loaders ────────────────────────────────
# Normalisasi standar ImageNet (wajib untuk model pretrained ResNet)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

full_dataset = ImageFolder(root=active_data_dir)
CLASSES = full_dataset.classes
print(f"[INFO] Classes ({len(CLASSES)}): {CLASSES}")

total_len = len(full_dataset)
train_len = int(0.70 * total_len)
val_len   = int(0.15 * total_len)
test_len  = total_len - train_len - val_len

train_subset, val_subset, test_subset = random_split(
    full_dataset, [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(SEED)
)

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

train_loader = DataLoader(TransformSubset(train_subset, train_transform), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TransformSubset(val_subset,   val_transform),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(TransformSubset(test_subset,  val_transform),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ── Model Construction (ResNet-18) ───────────────────────────
def build_resnet_model(num_classes=NUM_CLASSES, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Ganti Fully Connected layer terakhir
    in_features = model.fc.in_features  # 512 untuk ResNet-18
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model


# ── Training & Evaluation Loops ─────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


# ── Visualizations ──────────────────────────────────────────
def plot_parameter_comparison(model_frozen, model_unfrozen):
    p_frozen_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    p_frozen_total     = sum(p.numel() for p in model_frozen.parameters())
    p_frozen_locked    = p_frozen_total - p_frozen_trainable

    p_unfrozen_trainable = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
    p_unfrozen_total     = sum(p.numel() for p in model_unfrozen.parameters())
    p_unfrozen_locked    = p_unfrozen_total - p_unfrozen_trainable

    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['Tahap 1 (Feature Extraction)', 'Tahap 2 (Fine-Tuning)']
    trainable_counts = [p_frozen_trainable, p_unfrozen_trainable]
    frozen_counts    = [p_frozen_locked, p_unfrozen_locked]

    ax.barh(categories, trainable_counts, color='seagreen', label='Trainable Parameters')
    ax.barh(categories, frozen_counts, left=trainable_counts, color='lightgray', label='Frozen Parameters')

    ax.set_xlabel('Jumlah Parameter', fontweight='bold')
    ax.set_title('ResNet-18: Perbandingan Parameter Trainable vs Frozen', fontweight='bold')
    ax.legend(loc='lower right')

    for i, (t_val, f_val) in enumerate(zip(trainable_counts, frozen_counts)):
        ax.text(t_val / 2, i, f"{t_val:,}", ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig("../assets/tl_param_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Parameter comparison plot saved to assets/tl_param_comparison.png")


def plot_combined_history(hist1, hist2):
    t_loss = hist1['train_loss'] + hist2['train_loss']
    v_loss = hist1['val_loss']   + hist2['val_loss']
    t_acc  = hist1['train_acc']  + hist2['train_acc']
    v_acc  = hist1['val_acc']    + hist2['val_acc']

    boundary = len(hist1['train_loss'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(t_loss, label='Train Loss', color='purple', lw=2)
    axes[0].plot(v_loss, label='Val Loss',   color='crimson', lw=2)
    axes[0].axvline(boundary - 0.5, color='gray', linestyle='--', label='Start Fine-Tuning')
    axes[0].set_title('Loss across 2 Phases', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t_acc, label='Train Acc', color='purple', lw=2)
    axes[1].plot(v_acc, label='Val Acc',   color='crimson', lw=2)
    axes[1].axvline(boundary - 0.5, color='gray', linestyle='--', label='Start Fine-Tuning')
    axes[1].set_title('Akurasi across 2 Phases', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('Transfer Learning (ResNet-18) — Combined 2-Phase History', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/tl_training_history.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Combined training history saved to assets/tl_training_history.png")


def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix — Transfer Learning ResNet-18', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/tl_confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Confusion matrix saved to assets/tl_confusion_matrix.png")


# ── Main ─────────────────────────────────────────────────────
def main():
    os.makedirs("../assets", exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # ── Tahap 1: Feature Extraction ─────────────────────────
    print("\n" + "="*60)
    print("TAHAP 1: FEATURE EXTRACTION (Hanya Melatih FC Head Baru)")
    print("="*60)
    model = build_resnet_model(NUM_CLASSES, freeze_backbone=True).to(device)
    frozen_model_ref = copy.deepcopy(model)

    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p      = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable: {trainable_p1:,} / {total_p:,} ({100*trainable_p1/total_p:.2f}%)\n")

    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-3
    )

    history_p1 = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc_p1, best_weights_p1 = 0.0, None

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer_p1)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)

        history_p1['train_loss'].append(t_loss); history_p1['val_loss'].append(v_loss)
        history_p1['train_acc'].append(t_acc);   history_p1['val_acc'].append(v_acc)

        if v_acc > best_acc_p1:
            best_acc_p1 = v_acc
            best_weights_p1 = copy.deepcopy(model.state_dict())

        print(f"[Tahap 1] Epoch [{epoch}/{EPOCHS_PHASE1}] "
              f"Train Loss: {t_loss:.4f} (Acc: {t_acc:.2f}%) | "
              f"Val Loss: {v_loss:.4f} (Acc: {v_acc:.2f}%)")

    model.load_state_dict(best_weights_p1)
    print(f"\n[INFO] Selesai Tahap 1! Akurasi Validasi Terbaik: {best_acc_p1:.2f}%\n")

    # ── Tahap 2: Fine-Tuning ─────────────────────────────────
    print("="*60)
    print("TAHAP 2: FINE-TUNING (Membuka Layer 4 & FC Head)")
    print("="*60)

    # Unfreeze layer4 dan fc
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    unfrozen_model_ref = copy.deepcopy(model)
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters di Tahap 2: {trainable_p2:,} / {total_p:,} ({100*trainable_p2/total_p:.2f}%)\n")

    # Differential learning rates
    backbone_layer4_params = [p for n, p in model.named_parameters() if "layer4" in n and p.requires_grad]
    fc_params = list(model.fc.parameters())

    optimizer_p2 = optim.AdamW([
        {'params': backbone_layer4_params, 'lr': LR_FINETUNE * 0.5},
        {'params': fc_params,              'lr': LR_FINETUNE}
    ], weight_decay=1e-3)

    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=EPOCHS_PHASE2)

    history_p2 = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc_p2 = 0.0

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer_p2)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler_p2.step()

        history_p2['train_loss'].append(t_loss); history_p2['val_loss'].append(v_loss)
        history_p2['train_acc'].append(t_acc);   history_p2['val_acc'].append(v_acc)

        is_best = v_acc > best_acc_p2
        if is_best:
            best_acc_p2 = v_acc
            torch.save(model.state_dict(), "tl_best_model.pth")

        print(f"[Tahap 2] Epoch [{epoch}/{EPOCHS_PHASE2}] "
              f"Train Loss: {t_loss:.4f} (Acc: {t_acc:.2f}%) | "
              f"Val Loss: {v_loss:.4f} (Acc: {v_acc:.2f}%)"
              + ("  [BEST SAVED]" if is_best else ""))

    # ── Final Test Evaluation ───────────────────────────────
    model.load_state_dict(torch.load("tl_best_model.pth"))
    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)

    print(f"\n[RESULT] Final Test Set Accuracy: {test_acc:.2f}%\n")
    print("[RESULT] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Plots
    plot_parameter_comparison(frozen_model_ref, unfrozen_model_ref)
    plot_combined_history(history_p1, history_p2)
    plot_confusion_matrix(all_labels, all_preds, CLASSES)
    print("[SUCCESS] All Transfer Learning outputs generated successfully.")


if __name__ == "__main__":
    main()
