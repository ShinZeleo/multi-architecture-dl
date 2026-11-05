"""
==============================================================
Tugas Deep Learning - Implementasi 1: CNN (Waste Classification)
==============================================================
Nama   : Imam Dzaqhoir
NIM    : H071241048
Kelas  : [Kelas]
Dosen  : [Nama Dosen]

Deskripsi:
  Implementasi Convolutional Neural Network (CNN) dari awal (scratch)
  untuk klasifikasi jenis sampah daur ulang menggunakan dataset
  TrashNet (Stanford - Gary Thung).
  Dataset: 2.527 citra dalam 6 kelas (cardboard, glass, metal, paper, plastic, trash).
==============================================================
"""

import os
import sys
import io
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
EPOCHS        = 15
LEARNING_RATE = 1e-3
IMG_SIZE      = 128
NUM_CLASSES   = 6
DATA_DIR      = "./data/dataset-resized"
DATA_URL      = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"


# ── Auto Download & Extract Dataset ─────────────────────────
def download_trashnet(data_dir=DATA_DIR, data_url=DATA_URL):
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) >= 6:
        print(f"[INFO] Dataset already exists at: {data_dir}")
        return

    os.makedirs("./data", exist_ok=True)
    zip_path = "./data/dataset-resized.zip"

    if not os.path.exists(zip_path):
        print(f"[INFO] Downloading TrashNet dataset (~41 MB)...")
        req = urllib.request.Request(data_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded * 100 // total_size
                    print(f"\r  Downloading: {pct}% ({downloaded//1024//1024}MB / {total_size//1024//1024}MB)", end="")
        print("\n[INFO] Download completed!")

    print("[INFO] Extracting TrashNet...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall("./data")

    # Clean macOS metadata folder if present
    ds_store = os.path.join(data_dir, ".DS_Store")
    if os.path.exists(ds_store):
        try:
            os.remove(ds_store)
        except Exception:
            pass

    print(f"[INFO] Dataset extracted to {data_dir}")


download_trashnet()

# ── Data Transforms & Loaders ───────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_dataset = ImageFolder(root=DATA_DIR)
CLASSES = full_dataset.classes
print(f"[INFO] Detected Classes ({len(CLASSES)}): {CLASSES}")

# Train/Val/Test Split (70% Train, 15% Val, 15% Test)
total_len = len(full_dataset)
train_len = int(0.70 * total_len)
val_len   = int(0.15 * total_len)
test_len  = total_len - train_len - val_len

train_subset, val_subset, test_subset = random_split(
    full_dataset, [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(SEED)
)

# Apply transforms to subsets
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

train_ds = TransformSubset(train_subset, train_transform)
val_ds   = TransformSubset(val_subset,   val_transform)
test_ds  = TransformSubset(test_subset,  val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"[INFO] Samples - Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")


# ── Model Definition ────────────────────────────────────────
class WasteCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network for Waste Classification.
    Input : (B, 3, 128, 128)
    Output: (B, 6)
    """
    def __init__(self, num_classes=6):
        super(WasteCNN, self).__init__()

        # Conv Block 1: 128x128 -> 64x64
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1)
        )

        # Conv Block 2: 64x64 -> 32x32
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        # Conv Block 3: 32x32 -> 16x16
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        # Conv Block 4: 16x16 -> 8x8
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )

        # Classifier Head: 256 * 8 * 8 = 16384
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


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


# ── Visualization Helpers ───────────────────────────────────
def show_sample_images(loader, classes):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
        img = images[i].numpy().transpose((1, 2, 0))
        img = np.clip(std * img + mean, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Class: {classes[labels[i]]}", fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle("TrashNet — Sampel Citra Sampah Daur Ulang", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/cnn_sample_images.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Sample images saved to assets/cnn_sample_images.png")


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history['train_loss'], label='Train Loss', color='steelblue', lw=2)
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='tomato',    lw=2)
    axes[0].set_title('Loss per Epoch', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', color='steelblue', lw=2)
    axes[1].plot(history['val_acc'],   label='Val Acc',   color='tomato',    lw=2)
    axes[1].set_title('Akurasi per Epoch', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('Custom CNN (TrashNet) — Training & Validation History', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/cnn_training_history.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Training history saved to assets/cnn_training_history.png")


def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix — Custom CNN (TrashNet)', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/cnn_confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Confusion matrix saved to assets/cnn_confusion_matrix.png")


# ── Main Training Routine ────────────────────────────────────
def main():
    os.makedirs("../assets", exist_ok=True)
    show_sample_images(train_loader, CLASSES)

    model = WasteCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Model Architecture Initialized.")
    print(f"[INFO] Trainable Parameters: {total_params:,}\n")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        is_best = v_acc > best_acc
        if is_best:
            best_acc = v_acc
            torch.save(model.state_dict(), "cnn_best_model.pth")

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%"
              + ("  [BEST SAVED]" if is_best else ""))

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training finished in {elapsed/60:.2f} minutes.")
    print(f"[INFO] Best Validation Accuracy: {best_acc:.2f}%\n")

    # Evaluate on Test Set
    model.load_state_dict(torch.load("cnn_best_model.pth"))
    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)

    print(f"[RESULT] Final Test Set Accuracy: {test_acc:.2f}%\n")
    print("[RESULT] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    plot_history(history)
    plot_confusion_matrix(all_labels, all_preds, CLASSES)
    print("[SUCCESS] All CNN outputs generated successfully.")


if __name__ == "__main__":
    main()
