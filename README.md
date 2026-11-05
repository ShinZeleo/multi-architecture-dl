# Tugas Deep Learning — Implementasi Arsitektur CNN, RNN, dan Transfer Learning

> **Mata Kuliah:** Deep Learning  
> **Nama:** [Nama Mahasiswa]  
> **NIM:** [NIM]  
> **Deadline:** Jumat, 6 November 2026

---

## 📌 Deskripsi Proyek

Proyek ini mengimplementasikan **3 arsitektur Deep Learning** menggunakan PyTorch:

| # | Arsitektur | Dataset | Task |
|---|-----------|---------|------|
| 1 | **CNN** (Custom Deep CNN) | CIFAR-10 | Image Classification (10 kelas) |
| 2 | **RNN/BiLSTM** | 20Newsgroups | Text Classification (Sport vs Teknologi) |
| 3 | **Transfer Learning** (ResNet-18) | Oxford Flowers-102 | Flower Classification (5 kelas) |

---

## 🗂️ Struktur Repository

```
TugasDL/
├── 1_CNN_CIFAR10/
│   ├── cnn_cifar10.py          # Implementasi CNN dari scratch
│   └── data/                   # Dataset CIFAR-10 (auto-download)
│
├── 2_RNN_SentimentAnalysis/
│   ├── rnn_bilstm.py           # Implementasi BiLSTM classifier
│   └── data/                   # Dataset 20Newsgroups (auto-download)
│
├── 3_TransferLearning_FlowerClassification/
│   ├── transfer_learning_resnet18.py   # ResNet-18 fine-tuning
│   └── data/                           # Flowers-102 (auto-download)
│
├── assets/                     # Grafik hasil training (PNG)
│   ├── cnn_sample_images.png
│   ├── cnn_training_history.png
│   ├── cnn_confusion_matrix.png
│   ├── rnn_text_length_dist.png
│   ├── rnn_training_history.png
│   ├── rnn_confusion_matrix.png
│   ├── tl_param_comparison.png
│   ├── tl_training_history.png
│   └── tl_confusion_matrix.png
│
├── requirements.txt
└── README.md
```

---

## 🧠 Arsitektur Model

### 1. CNN — CIFAR-10 Image Classification

**Dataset:** CIFAR-10 — 60.000 gambar (32×32), 10 kelas

**Arsitektur Custom CNN:**
```
Input (3×32×32)
  → Conv Block 1: Conv2d(3→32) + Conv2d(32→32) + BN + ReLU + MaxPool + Dropout(0.1) → 32×16×16
  → Conv Block 2: Conv2d(32→64) + Conv2d(64→64) + BN + ReLU + MaxPool + Dropout(0.2) → 64×8×8
  → Conv Block 3: Conv2d(64→128) + Conv2d(128→128) + BN + ReLU + MaxPool + Dropout(0.3) → 128×4×4
  → Flatten → FC(2048→256) + ReLU + Dropout(0.5) → FC(256→10)
  → Output (10 kelas)
```

**Teknik Regularisasi:**
- Batch Normalization di setiap Conv layer
- Dropout (0.1 → 0.3 → 0.5)
- Data Augmentation: RandomCrop, RandomFlip, ColorJitter
- Weight Decay (L2) + CosineAnnealing Scheduler

---

### 2. RNN/BiLSTM — Text Classification (Sport vs Teknologi)

**Dataset:** 20 Newsgroups — artikel berita teks bahasa Inggris

**Arsitektur Bidirectional LSTM:**
```
Input (token IDs, max_len=200)
  → Embedding Layer (vocab_size=15.000, embed_dim=128)
  → Bi-LSTM (hidden=256, 2 layer, dropout=0.5)
  → Global Avg Pool + Global Max Pool
  → Concat (1024-dim) → FC(128) + ReLU + Dropout → FC(2)
  → Output (2 kelas: Sport / Non-Sport)
```

**Perbedaan RNN vs LSTM vs GRU:**
| | RNN Vanilla | LSTM | GRU |
|--|------------|------|-----|
| Memory | ❌ Short-term only | ✅ Long-term (cell state) | ✅ Long-term (simpler) |
| Gates | Tidak ada | Input, Forget, Output | Reset, Update |
| Parameter | Sedikit | Paling banyak | Sedang |
| Vanishing Grad | ⚠️ Rentan | ✅ Diatasi | ✅ Diatasi |

---

### 3. Transfer Learning — Flower Classification (ResNet-18)

**Dataset:** Oxford 102 Flowers (5 kelas dipilih)

**Strategi Fine-tuning (2 Phase):**

```
ResNet-18 (pretrained ImageNet, 11.2M params)
         ↓
Phase 1 — Feature Extraction (5 epoch):
  • Backbone FROZEN
  • Hanya melatih head: FC(512→256→5)
  • LR = 1e-3

Phase 2 — Fine-tuning (10 epoch):
  • Semua layer UNFROZEN
  • Backbone LR = 1e-4 (kecil)
  • Head LR = 1e-3 (lebih besar)
  • CosineAnnealing Scheduler
```

**Keunggulan Transfer Learning:**
- ✅ Tidak perlu data berlimpah
- ✅ Training lebih cepat
- ✅ Fitur pretrained (tepi, tekstur, pola) sudah kuat
- ✅ Mencegah overfitting pada dataset kecil

---

## ⚙️ Cara Menjalankan

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Jalankan CNN (CIFAR-10)
```bash
cd 1_CNN_CIFAR10
python cnn_cifar10.py
```

### 2. Jalankan RNN/BiLSTM
```bash
cd 2_RNN_SentimentAnalysis
python rnn_bilstm.py
```

### 3. Jalankan Transfer Learning
```bash
cd 3_TransferLearning_FlowerClassification
python transfer_learning_resnet18.py
```

> ⚠️ Dataset akan **didownload otomatis** saat pertama kali dijalankan.

---

## 📊 Hasil Eksperimen

### CNN — CIFAR-10
| Metrik | Nilai |
|--------|-------|
| Epochs | 20 |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealing |
| Test Accuracy | *lihat output training* |

### RNN/BiLSTM — Text Classification
| Metrik | Nilai |
|--------|-------|
| Epochs | 15 |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | ReduceLROnPlateau |
| Max Seq Length | 200 token |
| Vocab Size | 15.000 |
| Test Accuracy | *lihat output training* |

### Transfer Learning — ResNet-18
| Metrik | Nilai |
|--------|-------|
| Backbone | ResNet-18 (ImageNet pretrained) |
| Phase 1 Epochs | 5 (frozen backbone) |
| Phase 2 Epochs | 10 (all unfrozen) |
| Input Size | 224×224 |
| Test Accuracy | *lihat output training* |

---

## 🔧 Requirements

Lihat [`requirements.txt`](requirements.txt) untuk daftar dependensi lengkap.

**Library utama:**
- `torch` ≥ 2.0
- `torchvision` ≥ 0.15
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`
- `tqdm`

---

## 📚 Referensi

1. LeCun, Y., et al. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
3. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
4. Pan, S. J., & Yang, Q. (2010). *A Survey on Transfer Learning.* IEEE TKDE.
5. PyTorch Documentation: https://pytorch.org/docs/stable/index.html

---

*Dibuat untuk memenuhi tugas mata kuliah Deep Learning*
