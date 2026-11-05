"""
==============================================================
Tugas Deep Learning - Implementasi 1: CNN (CIFAR-10)
==============================================================
Nama   : [Nama Mahasiswa]
NIM    : [NIM]
Kelas  : [Kelas]
Dosen  : [Nama Dosen]

Deskripsi:
  Implementasi Convolutional Neural Network (CNN) untuk klasifikasi
  gambar menggunakan dataset CIFAR-10 (10 kelas objek).

Arsitektur:
  - Conv Block 1 : Conv2d(3->32)  + BN + ReLU + MaxPool
  - Conv Block 2 : Conv2d(32->64) + BN + ReLU + MaxPool
  - Conv Block 3 : Conv2d(64->128)+ BN + ReLU + MaxPool + Dropout(0.3)
  - FC Layer     : 128*4*4 -> 256 -> 10
==============================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time

# ── Reproducibility ─────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ── Hyperparameters ─────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 20
LEARNING_RATE = 0.001
NUM_CLASSES  = 10

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ── Data Preparation ────────────────────────────────────────
print("[INFO] Loading CIFAR-10 dataset...")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform)
test_dataset  = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"[INFO] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")


# ── Model Definition ────────────────────────────────────────
class CNN(nn.Module):
    """
    Custom CNN untuk CIFAR-10.
    Input : (B, 3, 32, 32)
    Output: (B, 10)
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional Blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 32x32 → 16x16
            nn.Dropout2d(0.1),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16x16 → 8x8
            nn.Dropout2d(0.2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 8x8 → 4x4
            nn.Dropout2d(0.3),
        )

        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


# ── Training & Evaluation Helpers ───────────────────────────
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


# ── Visualize Sample Images ──────────────────────────────────
def show_sample_images(loader, classes):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2470, 0.2435, 0.2616])
    for i, ax in enumerate(axes.flat):
        img = images[i].numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(classes[labels[i]], fontsize=8)
        ax.axis('off')
    plt.suptitle("Sample CIFAR-10 Training Images", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/cnn_sample_images.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Sample images saved → assets/cnn_sample_images.png")


# ── Plot Training History ────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train Loss', color='steelblue')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='tomato')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', color='steelblue')
    axes[1].plot(history['val_acc'],   label='Val Acc',   color='tomato')
    axes[1].set_title('Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('CNN CIFAR-10 — Training History', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/cnn_training_history.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Training history saved → assets/cnn_training_history.png")


# ── Plot Confusion Matrix ────────────────────────────────────
def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('CNN CIFAR-10 — Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/cnn_confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Confusion matrix saved → assets/cnn_confusion_matrix.png")


# ── Main ─────────────────────────────────────────────────────
def main():
    os.makedirs("../assets", exist_ok=True)

    # Visualize samples
    show_sample_images(train_loader, CLASSES)

    # Build model
    model     = CNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n[INFO] Model Architecture:\n{model}\n")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable Parameters: {total_params:,}\n")

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), "cnn_best_model.pth")

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%"
              + (" ✓ BEST" if v_acc == best_acc else ""))

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training completed in {elapsed/60:.1f} minutes")
    print(f"[INFO] Best Validation Accuracy: {best_acc:.2f}%")

    # Final evaluation with best model
    model.load_state_dict(torch.load("cnn_best_model.pth"))
    _, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)

    print(f"\n[RESULT] Final Test Accuracy: {final_acc:.2f}%")
    print("\n[RESULT] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Save plots
    plot_history(history)
    plot_confusion_matrix(all_labels, all_preds, CLASSES)
    print("\n[INFO] All assets saved. Done!")


if __name__ == "__main__":
    main()
