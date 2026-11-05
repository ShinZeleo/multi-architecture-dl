"""
==============================================================
Tugas Deep Learning - Implementasi 3: Transfer Learning (ResNet18)
==============================================================
Nama   : [Nama Mahasiswa]
NIM    : [NIM]
Kelas  : [Kelas]
Dosen  : [Nama Dosen]

Deskripsi:
  Implementasi Transfer Learning menggunakan ResNet-18 pretrained
  pada ImageNet untuk klasifikasi bunga (Oxford 102 Flowers subset
  disederhanakan menjadi 5 kelas menggunakan torchvision).

  Strategi Fine-tuning:
    Phase 1 — Feature Extraction:  Hanya melatih head FC baru (5 epoch)
    Phase 2 — Fine-tuning seluruh layer (10 epoch dengan LR kecil)

Arsitektur:
  ResNet-18 (pretrained ImageNet) → Replace FC → 5 kelas
==============================================================
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset, random_split

# ── Reproducibility ─────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ── Hyperparameters ─────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS_FE    = 5    # Feature Extraction phase
EPOCHS_FT    = 10   # Fine-tuning phase
LR_HEAD      = 1e-3  # Head learning rate (phase 1)
LR_FT        = 1e-4  # Fine-tune learning rate (phase 2)
NUM_CLASSES  = 5
IMG_SIZE     = 224

# Oxford Flowers102 — 5 kelas yang dipilih (class index dalam dataset):
# 0:pink primrose, 1:hard-leaved pocket orchid, 2:canterbury bells,
# 3:sweet pea, 4:wild geranium
FLOWER_CLASSES = ['Pink Primrose', 'Orchid', 'Canterbury Bells',
                  'Sweet Pea', 'Wild Geranium']
SELECTED_LABELS = [0, 1, 2, 3, 4]  # first 5 classes


# ── Data Transforms ─────────────────────────────────────────
# ImageNet normalization (required for pretrained ResNet-18)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


# ── Load Dataset (Flowers102) ────────────────────────────────
def filter_subset(dataset, selected_labels):
    """Return indices of samples belonging to selected_labels."""
    label_to_new = {old: new for new, old in enumerate(selected_labels)}
    indices = [i for i, (_, lbl) in enumerate(dataset)
               if lbl in selected_labels]
    return indices, label_to_new


print("[INFO] Downloading Oxford 102 Flowers dataset...")
try:
    full_train = torchvision.datasets.Flowers102(
        root='./data', split='train', download=True, transform=train_transform)
    full_val   = torchvision.datasets.Flowers102(
        root='./data', split='val',   download=True, transform=val_transform)
    full_test  = torchvision.datasets.Flowers102(
        root='./data', split='test',  download=True, transform=val_transform)

    # Filter to 5 classes
    train_idx, lbl_map = filter_subset(full_train, SELECTED_LABELS)
    val_idx,   _       = filter_subset(full_val,   SELECTED_LABELS)
    test_idx,  _       = filter_subset(full_test,  SELECTED_LABELS)

    # Remap labels
    class RemappedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, label_map):
            self.dataset   = dataset
            self.indices   = indices
            self.label_map = label_map

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img, lbl = self.dataset[self.indices[idx]]
            return img, self.label_map[lbl]

    train_ds = RemappedSubset(full_train, train_idx, lbl_map)
    val_ds   = RemappedSubset(full_val,   val_idx,   lbl_map)
    test_ds  = RemappedSubset(full_test,  test_idx,  lbl_map)

except Exception as e:
    # Fallback: use CIFAR-10 first 5 classes if Flowers102 download fails
    print(f"[WARN] Flowers102 error ({e}), falling back to CIFAR-10 (5 classes)...")
    FLOWER_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer']

    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    full_test  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform)

    keep = [i for i, (_, l) in enumerate(full_train) if l < 5]
    train_keep = keep[:int(len(keep) * 0.9)]
    val_keep   = keep[int(len(keep) * 0.9):]
    test_keep  = [i for i, (_, l) in enumerate(full_test) if l < 5]

    class SubsetData(torch.utils.data.Dataset):
        def __init__(self, ds, idx): self.ds, self.idx = ds, idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i): return self.ds[self.idx[i]]

    train_ds = SubsetData(full_train, train_keep)
    val_ds   = SubsetData(full_train, val_keep)
    test_ds  = SubsetData(full_test,  test_keep)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")


# ── Build Model ─────────────────────────────────────────────
def build_model(num_classes, freeze_backbone=True):
    """
    Load pretrained ResNet-18 dan ganti FC layer untuk num_classes.
    freeze_backbone=True → hanya head yang dilatih (feature extraction).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classification head
    in_features = model.fc.in_features   # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


# ── Training & Evaluation ────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct  += preds.eq(labels).sum().item()
            total    += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


# ── Plots ────────────────────────────────────────────────────
def plot_history(history_fe, history_ft):
    """Plot combined history of both training phases."""
    tl = history_fe['train_loss'] + history_ft['train_loss']
    vl = history_fe['val_loss']   + history_ft['val_loss']
    ta = history_fe['train_acc']  + history_ft['train_acc']
    va = history_fe['val_acc']    + history_ft['val_acc']
    phase_boundary = EPOCHS_FE

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, train_v, val_v, ylabel, title in zip(
        axes,
        [tl, ta], [vl, va],
        ['Loss', 'Accuracy (%)'],
        ['Loss per Epoch', 'Accuracy per Epoch'],
    ):
        ax.plot(train_v, label='Train', color='steelblue')
        ax.plot(val_v,   label='Val',   color='tomato')
        ax.axvline(phase_boundary - 0.5, color='gray', linestyle='--',
                   label='Fine-tune starts')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
        ax.text(phase_boundary / 2 - 1, ax.get_ylim()[0],
                'Feature Extraction', fontsize=8, color='gray')
        ax.text(phase_boundary + 0.5, ax.get_ylim()[0],
                'Fine-tuning', fontsize=8, color='gray')

    plt.suptitle('Transfer Learning (ResNet-18) — Training History', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/tl_training_history.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Training history saved → assets/tl_training_history.png")


def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Transfer Learning (ResNet-18) — Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/tl_confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Confusion matrix saved → assets/tl_confusion_matrix.png")


def compare_params(model_frozen, model_unfrozen):
    """Visualize trainable vs frozen parameters."""
    frozen   = sum(p.numel() for p in model_frozen.parameters()
                   if not p.requires_grad)
    trainable_fe = sum(p.numel() for p in model_frozen.parameters()
                       if p.requires_grad)
    trainable_ft = sum(p.numel() for p in model_unfrozen.parameters()
                       if p.requires_grad)
    total = frozen + trainable_fe

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        ['Phase 1\n(Feature Extraction)', 'Phase 2\n(Fine-tuning)'],
        [trainable_fe, trainable_ft],
        color=['steelblue', 'darkorange'],
    )
    ax.barh(
        ['Phase 1\n(Feature Extraction)'], [frozen], left=[trainable_fe],
        color='lightgray', label='Frozen'
    )
    ax.set_xlabel('Number of Parameters')
    ax.set_title('Trainable vs Frozen Parameters by Phase', fontweight='bold')
    for bar in bars:
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():,}', ha='center', va='center',
                color='white', fontweight='bold')
    ax.legend(['Trainable', 'Frozen'], loc='lower right')
    plt.tight_layout()
    plt.savefig("../assets/tl_param_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Parameter comparison saved → assets/tl_param_comparison.png")


# ── Main ─────────────────────────────────────────────────────
def main():
    os.makedirs("../assets", exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Feature Extraction ─────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: Feature Extraction (Backbone Frozen)")
    print("="*60)
    model = build_model(NUM_CLASSES, freeze_backbone=True).to(device)
    frozen_model = copy.deepcopy(model)   # keep reference for plot

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable: {trainable:,} / Total: {total_p:,} ({100*trainable/total_p:.1f}%)")

    optimizer_fe = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)

    history_fe = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc_fe, best_state = 0.0, None
    start = time.time()

    for epoch in range(1, EPOCHS_FE + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer_fe)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)
        history_fe['train_loss'].append(t_loss); history_fe['val_loss'].append(v_loss)
        history_fe['train_acc'].append(t_acc);   history_fe['val_acc'].append(v_acc)
        if v_acc > best_acc_fe:
            best_acc_fe = v_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"[FE] Epoch [{epoch}/{EPOCHS_FE}] "
              f"Train: {t_acc:.2f}% | Val: {v_acc:.2f}%")

    model.load_state_dict(best_state)
    print(f"[INFO] Phase 1 Best Val Acc: {best_acc_fe:.2f}%")

    # ── Phase 2: Fine-tuning ─────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning (All Layers Unfrozen)")
    print("="*60)
    for param in model.parameters():
        param.requires_grad = True

    trainable_ft = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable: {trainable_ft:,} / Total: {total_p:,} ({100*trainable_ft/total_p:.1f}%)")

    # Differential learning rates
    backbone_params = [p for name, p in model.named_parameters()
                       if 'fc' not in name]
    head_params     = list(model.fc.parameters())
    optimizer_ft = optim.Adam([
        {'params': backbone_params, 'lr': LR_FT},
        {'params': head_params,     'lr': LR_FT * 10},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=EPOCHS_FT)

    history_ft = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc_ft = 0.0

    for epoch in range(1, EPOCHS_FT + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer_ft)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        history_ft['train_loss'].append(t_loss); history_ft['val_loss'].append(v_loss)
        history_ft['train_acc'].append(t_acc);   history_ft['val_acc'].append(v_acc)
        if v_acc > best_acc_ft:
            best_acc_ft = v_acc
            torch.save(model.state_dict(), "tl_best_model.pth")
        print(f"[FT] Epoch [{epoch}/{EPOCHS_FT}] "
              f"Train: {t_acc:.2f}% | Val: {v_acc:.2f}%"
              + (" ✓ BEST" if v_acc == best_acc_ft else ""))

    elapsed = time.time() - start
    print(f"\n[INFO] Total training time: {elapsed/60:.1f} minutes")
    print(f"[INFO] Best Val Acc — Phase 1: {best_acc_fe:.2f}% | Phase 2: {best_acc_ft:.2f}%")

    # ── Final Test Evaluation ───────────────────────────────
    model.load_state_dict(torch.load("tl_best_model.pth"))
    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)
    print(f"\n[RESULT] Final Test Accuracy: {test_acc:.2f}%")
    print("\n[RESULT] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=FLOWER_CLASSES))

    # ── Plots ───────────────────────────────────────────────
    # Unfrozen model reference (all params trainable)
    unfrozen_model = build_model(NUM_CLASSES, freeze_backbone=False)
    compare_params(frozen_model, unfrozen_model)
    plot_history(history_fe, history_ft)
    plot_confusion_matrix(all_labels, all_preds, FLOWER_CLASSES)
    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
