"""
==============================================================
Tugas Deep Learning - Implementasi 2: BiGRU/LSTM (Indonesian Emotion)
==============================================================
Nama   : Imam Dzaqhoir
NIM    : H071241048
Kelas  : Deep Learning
Dosen  : Prof. Dr. Armin Lawi, S.Si., M.Si.

Deskripsi:
  Implementasi Recurrent Neural Network (RNN / Bidirectional GRU)
  untuk analisis multi-class emosi pada teks media sosial (Twitter/X)
  berbahasa Indonesia menggunakan Indonesian Twitter Emotion Dataset.

  5 Kategori Emosi:
    1. anger   (kemarahan)
    2. happy   (kebahagiaan)
    3. sadness (kesedihan)
    4. fear    (ketakutan)
    5. love    (kasih sayang/cinta)
==============================================================
"""

import os
import re
import io
import time
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ─────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ── Hyperparameters ─────────────────────────────────────────
MAX_LEN       = 60      # Maksimum panjang token per tweet
VOCAB_SIZE    = 8000    # Ukuran kosakata kata unik
EMBED_DIM     = 128     # Dimensi embedding
HIDDEN_DIM    = 128     # Dimensi unit GRU tersembunyi
N_LAYERS      = 2       # Jumlah layer bertumpuk
DROPOUT       = 0.3
BATCH_SIZE    = 32
EPOCHS        = 15
LEARNING_RATE = 1e-3
DATA_URL      = "https://raw.githubusercontent.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/master/Twitter_Emotion_Dataset.csv"
DATA_PATH     = "./data/Twitter_Emotion_Dataset.csv"


# ── Auto Download Dataset ───────────────────────────────────
def download_dataset():
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"[INFO] Downloading Indonesian Twitter Emotion Dataset from GitHub...")
        req = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(DATA_PATH, 'wb') as f:
            f.write(resp.read())
        print("[INFO] Dataset saved locally!")


download_dataset()

# ── Preprocessing Teks Bahasa Indonesia ─────────────────────
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Total Dataset: {len(df)} baris")

def clean_indonesian_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)       # Hapus mention Twitter @user
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Hapus URL
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)       # Hapus hashtag
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)         # Simpan hanya huruf alfabet
    text = re.sub(r'\s+', ' ', text).strip()         # Normalisasi spasi
    return text.lower()

df['clean_tweet'] = df['tweet'].apply(clean_indonesian_text)

# Label encoding
EMOTION_CLASSES = sorted(df['label'].unique().tolist())
label2id = {lbl: i for i, lbl in enumerate(EMOTION_CLASSES)}
id2label = {i: lbl for lbl, i in label2id.items()}
df['label_id'] = df['label'].map(label2id)

print(f"[INFO] Classes: {label2id}")
print(df['label'].value_counts())

# ── Split Dataset ───────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    df['clean_tweet'].values, df['label_id'].values,
    test_size=0.3, random_state=SEED, stratify=df['label_id'].values
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print(f"[INFO] Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Vocabulary Builder ──────────────────────────────────────
def build_vocab(texts, max_vocab=VOCAB_SIZE):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    # 0 = <PAD>, 1 = <UNK>
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(X_train)
print(f"[INFO] Vocabulary Size: {len(vocab)} kata")

def encode_text(text, vocab, max_len=MAX_LEN):
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    ids += [vocab['<PAD>']] * (max_len - len(ids))
    return ids

# ── PyTorch Dataset & DataLoader ────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.data = [encode_text(t, vocab) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))

train_loader = DataLoader(EmotionDataset(X_train, y_train, vocab), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(EmotionDataset(X_val,   y_val,   vocab), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(EmotionDataset(X_test,  y_test,  vocab), batch_size=BATCH_SIZE, shuffle=False)


# ── Model Architecture: Bidirectional GRU ───────────────────
class BiGRUEmotion(nn.Module):
    """
    Bidirectional GRU Network untuk klasifikasi multi-kelas emosi.
    Input : (Batch, Sequence_Length)
    Output: (Batch, Num_Classes)
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, num_classes, dropout):
        super(BiGRUEmotion, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop  = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0
        )

        # Global average & max pooling (concatenated = 4 * hidden_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embedded = self.emb_drop(self.embedding(x))     # (B, L, E)
        output, _ = self.gru(embedded)                  # (B, L, 2H)

        avg_pool = torch.mean(output, dim=1)            # (B, 2H)
        max_pool, _ = torch.max(output, dim=1)          # (B, 2H)

        pooled = torch.cat([avg_pool, max_pool], dim=1) # (B, 4H)
        logits = self.fc(pooled)
        return logits


# ── Training & Evaluation Loops ─────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


# ── Visualization Helpers ───────────────────────────────────
def plot_dataset_dist(df):
    plt.figure(figsize=(7, 4))
    order = df['label'].value_counts().index
    sns.countplot(data=df, x='label', order=order, palette='crest')
    plt.title('Distribusi Kelas Indonesian Twitter Emotion Dataset', fontweight='bold')
    plt.xlabel('Kategori Emosi', fontweight='bold')
    plt.ylabel('Jumlah Tweet', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/rnn_emotion_dist.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Dataset distribution plot saved to assets/rnn_emotion_dist.png")


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history['train_loss'], label='Train Loss', color='forestgreen', lw=2)
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='orange',      lw=2)
    axes[0].set_title('Loss per Epoch', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', color='forestgreen', lw=2)
    axes[1].plot(history['val_acc'],   label='Val Acc',   color='orange',      lw=2)
    axes[1].set_title('Akurasi per Epoch', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('Bidirectional GRU — Training & Validation History', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/rnn_training_history.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Training history saved to assets/rnn_training_history.png")


def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix — BiGRU Indonesian Emotion', fontweight='bold')
    plt.tight_layout()
    plt.savefig("../assets/rnn_confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("[INFO] Confusion matrix saved to assets/rnn_confusion_matrix.png")


# ── Interactive Predictor ───────────────────────────────────
def predict_emotion(model, text, vocab, classes):
    model.eval()
    cleaned = clean_indonesian_text(text)
    ids = torch.tensor([encode_text(cleaned, vocab)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(ids)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = np.argmax(probs)
    return classes[pred_idx], probs[pred_idx]


# ── Main ─────────────────────────────────────────────────────
def main():
    os.makedirs("../assets", exist_ok=True)
    plot_dataset_dist(df)

    model = BiGRUEmotion(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        num_classes=len(EMOTION_CLASSES),
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] BiGRU Model Initialized.")
    print(f"[INFO] Trainable Parameters: {total_params:,}\n")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(v_acc)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        is_best = v_acc > best_acc
        if is_best:
            best_acc = v_acc
            torch.save(model.state_dict(), "rnn_best_model.pth")

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%"
              + ("  [BEST SAVED]" if is_best else ""))

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training finished in {elapsed/60:.2f} minutes.")
    print(f"[INFO] Best Validation Accuracy: {best_acc:.2f}%\n")

    # Final test evaluation
    model.load_state_dict(torch.load("rnn_best_model.pth"))
    _, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)

    print(f"[RESULT] Final Test Set Accuracy: {test_acc:.2f}%\n")
    print("[RESULT] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=EMOTION_CLASSES))

    plot_history(history)
    plot_confusion_matrix(all_labels, all_preds, EMOTION_CLASSES)

    # Demo testing with sample Indonesian sentences
    print("\n" + "="*50)
    print("DEMO PREDIKSI KALIMAT BAHASA INDONESIA:")
    print("="*50)
    demo_sentences = [
        "Alhamdulillah seneng banget akhirnya tugas deep learning selesai!",
        "Gila ya bener-bener kesel dan marah banget sama sikap dia!",
        "Sedih banget dengar kabar duka ini, gak terasa air mata menetes.",
        "Merinding parah, takut banget lewat jalan sepi tadi malam.",
        "Terima kasih selalu ada untukku sayang, aku sangat mencintaimu.",
    ]
    for s in demo_sentences:
        pred, conf = predict_emotion(model, s, vocab, EMOTION_CLASSES)
        print(f" Kalimat  : \"{s}\"")
        print(f" Prediksi : [{pred.upper()}] (Keyakinan: {conf*100:.1f}%)\n")

    print("[SUCCESS] All RNN outputs generated successfully.")


if __name__ == "__main__":
    main()
