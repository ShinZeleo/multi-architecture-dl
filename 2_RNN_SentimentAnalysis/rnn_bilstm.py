"""
==============================================================
Tugas Deep Learning - Implementasi 2: RNN/LSTM (Sentiment Analysis)
==============================================================
Nama   : Imam Dzaqhoir
NIM    : H071241048
Kelas  : [Kelas]
Dosen  : [Nama Dosen]

Deskripsi:
  Implementasi Recurrent Neural Network (RNN) dengan arsitektur
  LSTM Bidirectional untuk analisis sentimen teks menggunakan
  dataset SST-2 (Stanford Sentiment Treebank) via IMDB subset
  yang diambil dari torchtext/sklearn.

Arsitektur:
  - Embedding Layer    : vocab_size × embed_dim
  - Bi-LSTM            : 2 layer, hidden=256, dropout=0.5
  - Attention (Global Avg + Max Pool)
  - FC Classifier      : 512 → 128 → 2
==============================================================
"""

import os
import re
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── Reproducibility ─────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ── Hyperparameters ─────────────────────────────────────────
MAX_LEN    = 200    # max tokens per document
VOCAB_SIZE = 15000  # vocabulary size
EMBED_DIM  = 128    # embedding dimension
HIDDEN_DIM = 256    # LSTM hidden units
N_LAYERS   = 2      # LSTM layers
DROPOUT    = 0.5
BATCH_SIZE = 64
EPOCHS     = 15
LR         = 1e-3

# We use 4 topic categories and treat as 2-class:
#   Sport (rec.sport.*) → positive / tech (comp.*) → negative
CATEGORIES_POS = ['rec.sport.hockey', 'rec.sport.baseball']
CATEGORIES_NEG = ['comp.graphics', 'comp.os.ms-windows.misc']
CLASSES = ['Non-Sport (Teknologi)', 'Sport']


# ── Data Preparation ────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'\S+@\S+', '', text)         # remove emails
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)    # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def build_vocab(texts, vocab_size=VOCAB_SIZE):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    # Reserve 0=PAD, 1=UNK
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    # Padding
    ids += [vocab['<PAD>']] * (max_len - len(ids))
    return ids


print("[INFO] Loading 20Newsgroups dataset (Sport vs Technology)...")

news_pos = fetch_20newsgroups(subset='all', categories=CATEGORIES_POS,
                               remove=('headers', 'footers', 'quotes'))
news_neg = fetch_20newsgroups(subset='all', categories=CATEGORIES_NEG,
                               remove=('headers', 'footers', 'quotes'))

texts  = [clean_text(t) for t in news_pos.data] + \
         [clean_text(t) for t in news_neg.data]
labels = [1] * len(news_pos.data) + [0] * len(news_neg.data)

print(f"[INFO] Total samples: {len(texts)} | Positive: {sum(labels)} | Negative: {len(labels)-sum(labels)}")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train)

print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

vocab = build_vocab(X_train)
print(f"[INFO] Vocabulary size: {len(vocab)}")


# ── Custom Dataset ───────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.data   = [encode(t, vocab) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


train_ds = TextDataset(X_train, y_train, vocab)
val_ds   = TextDataset(X_val,   y_val,   vocab)
test_ds  = TextDataset(X_test,  y_test,  vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)


# ── Model Definition ────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM dengan Global Pooling untuk klasifikasi teks.

    Alur:
      Token IDs → Embedding → Bi-LSTM (2 layer) →
      [GlobalAvgPool ⊕ GlobalMaxPool] → FC → Logits
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout, num_classes=2):
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # After concat avg+max pool: 2 * hidden_dim * 2 (bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, L)
        embedded = self.dropout(self.embedding(x))    # (B, L, E)
        output, _ = self.lstm(embedded)               # (B, L, 2H)

        # Global pooling over time dimension
        avg_pool = output.mean(dim=1)                 # (B, 2H)
        max_pool, _ = output.max(dim=1)               # (B, 2H)

        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 4H)
        logits = self.classifier(self.dropout(pooled))
        return logits


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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


# ── Visualizations ──────────────────────────────────────────
def plot_text_length_distribution(texts, title="Text Length Distribution"):
    lengths = [len(t.split()) for t in texts]
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    plt.axvline(MAX_LEN, color='red', linestyle='--', label=f'MAX_LEN={MAX_LEN}')
    plt.title(title, fontweight='bold')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("../assets/rnn_text_length_dist.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Text length distribution saved → assets/rnn_text_length_dist.png")


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train Loss', color='steelblue')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='tomato')
    axes[0].set_title('Loss per Epoch'); axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', color='steelblue')
    axes[1].plot(history['val_acc'],   label='Val Acc',   color='tomato')
    axes[1].set_title('Accuracy per Epoch'); axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('BiLSTM — Training History (Sport vs Technology)', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/rnn_training_history.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Training history saved → assets/rnn_training_history.png")


def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('BiLSTM — Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/rnn_confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Confusion matrix saved → assets/rnn_confusion_matrix.png")


# ── Inference Demo ──────────────────────────────────────────
def predict_text(model, text, vocab, classes):
    model.eval()
    cleaned = clean_text(text)
    ids     = torch.tensor([encode(cleaned, vocab)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(ids)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
    print(f"  Text    : {text[:80]}...")
    print(f"  Predicted: {classes[pred]} (confidence: {probs[pred]*100:.1f}%)")
    return pred


# ── Main ─────────────────────────────────────────────────────
def main():
    os.makedirs("../assets", exist_ok=True)

    # EDA
    plot_text_length_distribution(texts)

    # Build model
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    print(f"\n[INFO] Model Architecture:\n{model}\n")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable Parameters: {total_params:,}\n")

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(v_acc)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), "rnn_best_model.pth")

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%"
              + (" ✓ BEST" if v_acc == best_acc else ""))

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training completed in {elapsed/60:.1f} minutes")
    print(f"[INFO] Best Validation Accuracy: {best_acc:.2f}%")

    # Final test evaluation
    model.load_state_dict(torch.load("rnn_best_model.pth"))
    _, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)

    print(f"\n[RESULT] Final Test Accuracy: {final_acc:.2f}%")
    print("\n[RESULT] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Plots
    plot_history(history)
    plot_confusion_matrix(all_labels, all_preds, CLASSES)

    # Demo predictions
    print("\n[DEMO] Sample Predictions:")
    demo_texts = [
        "The hockey game was incredible, the players scored amazing goals in overtime",
        "Windows operating system crashes frequently with blue screen of death errors",
        "Baseball pitcher threw a perfect game with 27 strikeouts in a row",
        "Graphics card rendering pipeline and GPU shader compilation process",
    ]
    for t in demo_texts:
        predict_text(model, t, vocab, CLASSES)
        print()

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
