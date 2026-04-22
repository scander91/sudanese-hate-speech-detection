#!/usr/bin/env python3
"""
Multi-Task Learning: Hate Speech + Sentiment Joint Training
=============================================================
Proposed model: Shared BERT encoder with two task-specific heads.

Architecture:
    Input → Shared BERT Encoder → [CLS] → Dropout
                                    ├── HS Head (Linear → num_hs_labels)
                                    └── Sent Head (Linear → num_sent_labels)

Joint Loss: L = α * L_hate + (1-α) * L_sentiment

References:
    - Plaza-del-Arco et al. (2021) "A Multi-Task Learning Approach to
      Hate Speech Detection Leveraging Sentiment Analysis"
    - Yuan & Rizoiu (2025) "Generalizing Hate Speech Detection Using MTL"

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 mtl_trainer.py --run_all
    CUDA_VISIBLE_DEVICES=1 python3 mtl_trainer.py --model marbertv2 --hs_task binary --alpha 0.7
    CUDA_VISIBLE_DEVICES=1 python3 mtl_trainer.py --explain --model marbertv2 --hs_task binary

Output: results/mtl_hate_sentiment/
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix as sk_confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
#  CONFIGURATION
# ============================================================

BASE_DIR = os.path.expanduser("~/sudanese_dialect_project")
RESULTS_DIR = os.path.join(BASE_DIR, "results/mtl_hate_sentiment")
SEED = 42

# Hate speech datasets
HS_DATASETS = {
    "binary": {
        "path": os.path.join(BASE_DIR, "data/labeling_corpus/dataset_binary.tsv"),
        "labels": ["HARMFUL", "NEUTRAL"],
    },
    "3class": {
        "path": os.path.join(BASE_DIR, "data/labeling_corpus/dataset_3class.tsv"),
        "labels": ["HATE", "OFFENSIVE", "NEUTRAL"],
    },
}

# Sentiment dataset (auxiliary task)
SENT_DATASET = {
    "train": os.path.join(BASE_DIR, "data/sentiment_prepared/telecom_train.json"),
    "test": os.path.join(BASE_DIR, "data/sentiment_prepared/telecom_test.json"),
    "labels": ["neg", "obj", "pos"],
}

# Models to evaluate
MODELS = {
    "marbertv2":    "UBC-NLP/MARBERTv2",
    "marbert":      "UBC-NLP/MARBERT",
    "arabertv2":    "aubmindlab/bert-base-arabertv02",
    "camelbert_da": "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "sudabert_v2":  os.path.join(BASE_DIR, "models/sudabert_v2/sudabert_v2/"),
}

# Hyperparameters
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.1
ALPHA_VALUES = [0.5, 0.7, 0.9]  # weight for hate speech loss

# ============================================================
#  REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
#  DATA LOADING
# ============================================================

def load_hs_tsv(path, label_names):
    """Load hate speech TSV → texts, labels (int)."""
    label2id = {name: i for i, name in enumerate(label_names)}
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            text, label = parts[0].strip(), parts[1].strip()
            if label not in label2id or not text:
                continue
            texts.append(text)
            labels.append(label2id[label])
    return texts, labels


def load_sent_json(path):
    """Load sentiment JSON → texts, labels_str."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    return texts, labels


def split_data_80_10_10(texts, labels, seed=SEED):
    """80/10/10 stratified split — same as hate_speech_trainer.py."""
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels
    )
    return (train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels)


# ============================================================
#  DATASET CLASS
# ============================================================

class TextDataset(Dataset):
    """Generic text dataset with pre-computed encodings."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def encode_texts(tokenizer, texts, max_len=MAX_SEQ_LEN):
    """Tokenize texts."""
    return tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


# ============================================================
#  MTL MODEL
# ============================================================

class MTLModel(nn.Module):
    """
    Multi-Task Learning model with shared BERT encoder
    and task-specific classification heads.

    Architecture:
        BERT → [CLS] → Dropout → {HS Head, Sentiment Head}
    """
    def __init__(self, encoder_name, num_hs_labels, num_sent_labels, dropout=DROPOUT):
        super().__init__()
        self.config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Task-specific heads
        self.hs_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_hs_labels),
        )

        self.sent_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_sent_labels),
        )

    def forward(self, input_ids, attention_mask, task="hs"):
        """
        Forward pass.
        Args:
            task: "hs" for hate speech, "sent" for sentiment
        Returns:
            logits for the specified task
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        if task == "hs":
            return self.hs_head(cls_output)
        elif task == "sent":
            return self.sent_head(cls_output)
        else:
            raise ValueError(f"Unknown task: {task}")


# ============================================================
#  TRAINING
# ============================================================

def train_mtl(model_name, model_path, hs_task, alpha, skip_existing=False):
    """
    Train MTL model on hate speech (primary) + sentiment (auxiliary).
    """
    exp_name = f"mtl_{model_name}_{hs_task}_a{alpha:.1f}"
    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    results_file = os.path.join(exp_dir, "results.json")

    if skip_existing and os.path.exists(results_file):
        print(f"\n  SKIP {exp_name} — results exist")
        with open(results_file) as f:
            return json.load(f)

    os.makedirs(exp_dir, exist_ok=True)
    set_seed(SEED)

    print(f"\n{'='*70}")
    print(f"  MTL: {model_name} | HS={hs_task} | α={alpha}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    t0 = time.time()

    # ---- Load hate speech data ----
    hs_cfg = HS_DATASETS[hs_task]
    hs_label_names = hs_cfg["labels"]
    num_hs_labels = len(hs_label_names)

    hs_texts, hs_labels = load_hs_tsv(hs_cfg["path"], hs_label_names)
    print(f"  HS data: {len(hs_texts):,} samples, {num_hs_labels} classes")
    hs_dist = Counter(hs_labels)
    for name in hs_label_names:
        idx = hs_label_names.index(name)
        print(f"    {name}: {hs_dist[idx]:,}")

    hs_tr_t, hs_tr_l, hs_va_t, hs_va_l, hs_te_t, hs_te_l = split_data_80_10_10(
        hs_texts, hs_labels
    )
    print(f"  HS split: train={len(hs_tr_t):,} val={len(hs_va_t):,} test={len(hs_te_t):,}")

    # ---- Load sentiment data ----
    sent_label_names = SENT_DATASET["labels"]
    num_sent_labels = len(sent_label_names)
    sent_label2id = {name: i for i, name in enumerate(sent_label_names)}

    sent_tr_t, sent_tr_l_str = load_sent_json(SENT_DATASET["train"])
    sent_te_t, sent_te_l_str = load_sent_json(SENT_DATASET["test"])
    sent_tr_l = [sent_label2id[l] for l in sent_tr_l_str]
    sent_te_l = [sent_label2id[l] for l in sent_te_l_str]

    # Carve 10% validation from sentiment train
    sent_tr_t, sent_va_t, sent_tr_l, sent_va_l = train_test_split(
        sent_tr_t, sent_tr_l, test_size=0.1, random_state=SEED, stratify=sent_tr_l
    )
    print(f"  Sent data: train={len(sent_tr_t):,} val={len(sent_va_t):,} test={len(sent_te_t):,}")

    # ---- Tokenize ----
    print(f"\n  Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    hs_tr_enc = encode_texts(tokenizer, hs_tr_t)
    hs_va_enc = encode_texts(tokenizer, hs_va_t)
    hs_te_enc = encode_texts(tokenizer, hs_te_t)
    sent_tr_enc = encode_texts(tokenizer, sent_tr_t)
    sent_va_enc = encode_texts(tokenizer, sent_va_t)
    sent_te_enc = encode_texts(tokenizer, sent_te_t)

    hs_tr_ds = TextDataset(hs_tr_enc, hs_tr_l)
    hs_va_ds = TextDataset(hs_va_enc, hs_va_l)
    hs_te_ds = TextDataset(hs_te_enc, hs_te_l)
    sent_tr_ds = TextDataset(sent_tr_enc, sent_tr_l)
    sent_va_ds = TextDataset(sent_va_enc, sent_va_l)
    sent_te_ds = TextDataset(sent_te_enc, sent_te_l)

    hs_tr_loader = DataLoader(hs_tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    hs_va_loader = DataLoader(hs_va_ds, batch_size=64, shuffle=False)
    hs_te_loader = DataLoader(hs_te_ds, batch_size=64, shuffle=False)
    sent_tr_loader = DataLoader(sent_tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    sent_va_loader = DataLoader(sent_va_ds, batch_size=64, shuffle=False)
    sent_te_loader = DataLoader(sent_te_ds, batch_size=64, shuffle=False)

    # ---- Build model ----
    print(f"  Building MTL model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTLModel(model_path, num_hs_labels, num_sent_labels).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {total_params:.1f}M")

    # ---- Optimizer + Scheduler ----
    # Steps per epoch = max of HS and sentiment batches
    hs_steps_per_epoch = len(hs_tr_loader)
    sent_steps_per_epoch = len(sent_tr_loader)
    steps_per_epoch = max(hs_steps_per_epoch, sent_steps_per_epoch)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")
    print(f"  α={alpha} (HS weight), 1-α={1-alpha:.1f} (Sent weight)")

    # ---- Training loop ----
    history = {
        "train_loss": [], "train_hs_loss": [], "train_sent_loss": [],
        "val_hs_f1": [], "val_sent_f1": [], "val_hs_acc": [], "val_sent_acc": [],
    }
    best_val_hs_f1 = 0
    best_epoch = 0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_hs_loss = 0
        epoch_sent_loss = 0
        epoch_total_loss = 0
        n_hs_batches = 0
        n_sent_batches = 0

        # Cycle the shorter loader
        if len(hs_tr_loader) >= len(sent_tr_loader):
            hs_iter = iter(hs_tr_loader)
            sent_iter = cycle(sent_tr_loader)
            n_steps = len(hs_tr_loader)
        else:
            hs_iter = cycle(hs_tr_loader)
            sent_iter = iter(sent_tr_loader)
            n_steps = len(sent_tr_loader)

        for step in range(n_steps):
            optimizer.zero_grad()

            # HS batch
            hs_batch = next(hs_iter)
            hs_input_ids = hs_batch["input_ids"].to(device)
            hs_attn_mask = hs_batch["attention_mask"].to(device)
            hs_labels_b = hs_batch["labels"].to(device)

            hs_logits = model(hs_input_ids, hs_attn_mask, task="hs")
            loss_hs = criterion(hs_logits, hs_labels_b)

            # Sentiment batch
            sent_batch = next(sent_iter)
            sent_input_ids = sent_batch["input_ids"].to(device)
            sent_attn_mask = sent_batch["attention_mask"].to(device)
            sent_labels_b = sent_batch["labels"].to(device)

            sent_logits = model(sent_input_ids, sent_attn_mask, task="sent")
            loss_sent = criterion(sent_logits, sent_labels_b)

            # Combined loss
            loss = alpha * loss_hs + (1 - alpha) * loss_sent
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_hs_loss += loss_hs.item()
            epoch_sent_loss += loss_sent.item()
            epoch_total_loss += loss.item()
            n_hs_batches += 1
            n_sent_batches += 1

        avg_hs_loss = epoch_hs_loss / n_hs_batches
        avg_sent_loss = epoch_sent_loss / n_sent_batches
        avg_total_loss = epoch_total_loss / n_steps

        history["train_loss"].append(avg_total_loss)
        history["train_hs_loss"].append(avg_hs_loss)
        history["train_sent_loss"].append(avg_sent_loss)

        # ---- Validation ----
        model.eval()

        # Validate HS
        hs_val_preds, hs_val_true = [], []
        with torch.no_grad():
            for batch in hs_va_loader:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    task="hs"
                )
                preds = logits.argmax(dim=-1).cpu().numpy()
                hs_val_preds.extend(preds)
                hs_val_true.extend(batch["labels"].numpy())

        val_hs_acc = accuracy_score(hs_val_true, hs_val_preds)
        val_hs_f1 = f1_score(hs_val_true, hs_val_preds, average="macro", zero_division=0)

        # Validate sentiment
        sent_val_preds, sent_val_true = [], []
        with torch.no_grad():
            for batch in sent_va_loader:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    task="sent"
                )
                preds = logits.argmax(dim=-1).cpu().numpy()
                sent_val_preds.extend(preds)
                sent_val_true.extend(batch["labels"].numpy())

        val_sent_acc = accuracy_score(sent_val_true, sent_val_preds)
        val_sent_f1 = f1_score(sent_val_true, sent_val_preds, average="macro", zero_division=0)

        history["val_hs_f1"].append(val_hs_f1)
        history["val_sent_f1"].append(val_sent_f1)
        history["val_hs_acc"].append(val_hs_acc)
        history["val_sent_acc"].append(val_sent_acc)

        print(f"  Epoch {epoch}/{NUM_EPOCHS} | "
              f"Loss: {avg_total_loss:.4f} (HS:{avg_hs_loss:.4f} Sent:{avg_sent_loss:.4f}) | "
              f"Val-HS: Acc={val_hs_acc:.4f} F1={val_hs_f1:.4f} | "
              f"Val-Sent: Acc={val_sent_acc:.4f} F1={val_sent_f1:.4f}")

        # Track best
        if val_hs_f1 > best_val_hs_f1:
            best_val_hs_f1 = val_hs_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- Load best model ----
    print(f"\n  Best epoch: {best_epoch} (Val HS F1={best_val_hs_f1:.4f})")
    model.load_state_dict(best_state)
    model.eval()

    # ---- Save best model ----
    best_model_dir = os.path.join(exp_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(best_model_dir, "model.pt"))
    tokenizer.save_pretrained(best_model_dir)
    # Save model config for reloading
    model_config = {
        "encoder_name": model_path,
        "num_hs_labels": num_hs_labels,
        "num_sent_labels": num_sent_labels,
        "hs_label_names": hs_label_names,
        "sent_label_names": sent_label_names,
    }
    with open(os.path.join(best_model_dir, "mtl_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # ---- Test Evaluation ----
    # Evaluate HS on test
    hs_te_preds, hs_te_true = [], []
    with torch.no_grad():
        for batch in hs_te_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                task="hs"
            )
            preds = logits.argmax(dim=-1).cpu().numpy()
            hs_te_preds.extend(preds)
            hs_te_true.extend(batch["labels"].numpy())

    hs_te_acc = accuracy_score(hs_te_true, hs_te_preds)
    hs_te_f1 = f1_score(hs_te_true, hs_te_preds, average="macro", zero_division=0)
    hs_te_f1_w = f1_score(hs_te_true, hs_te_preds, average="weighted", zero_division=0)
    hs_te_f1_per = f1_score(hs_te_true, hs_te_preds, average=None, zero_division=0)
    hs_cm = sk_confusion_matrix(hs_te_true, hs_te_preds)
    hs_report = classification_report(hs_te_true, hs_te_preds,
                                       target_names=hs_label_names, digits=4,
                                       output_dict=True, zero_division=0)

    print(f"\n  ┌─── HS TEST RESULTS ({hs_task}) ──────────────┐")
    print(f"  │  Accuracy: {hs_te_acc*100:.2f}%")
    print(f"  │  F1-macro: {hs_te_f1*100:.2f}%")
    for i, name in enumerate(hs_label_names):
        print(f"  │  F1-{name}: {hs_te_f1_per[i]*100:.2f}%")
    print(f"  └──────────────────────────────────────┘")

    # Evaluate sentiment on test
    sent_te_preds, sent_te_true = [], []
    with torch.no_grad():
        for batch in sent_te_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                task="sent"
            )
            preds = logits.argmax(dim=-1).cpu().numpy()
            sent_te_preds.extend(preds)
            sent_te_true.extend(batch["labels"].numpy())

    sent_te_acc = accuracy_score(sent_te_true, sent_te_preds)
    sent_te_f1 = f1_score(sent_te_true, sent_te_preds, average="macro", zero_division=0)
    sent_te_f1_per = f1_score(sent_te_true, sent_te_preds, average=None, zero_division=0)
    sent_cm = sk_confusion_matrix(sent_te_true, sent_te_preds)

    print(f"\n  ┌─── SENTIMENT TEST RESULTS ─────────────┐")
    print(f"  │  Accuracy: {sent_te_acc*100:.2f}%")
    print(f"  │  F1-macro: {sent_te_f1*100:.2f}%")
    for i, name in enumerate(sent_label_names):
        print(f"  │  F1-{name}: {sent_te_f1_per[i]*100:.2f}%")
    print(f"  └──────────────────────────────────────┘")

    total_time = time.time() - t0

    # ---- Save confusion matrices ----
    # HS confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(hs_cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=hs_label_names, yticklabels=hs_label_names)
    ax.set_title(f"MTL {model_name} — HS {hs_task} (α={alpha})\nF1={hs_te_f1*100:.1f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(exp_dir, "confusion_matrix_hs.png")
    plt.savefig(cm_path, dpi=150)
    plt.savefig(cm_path.replace(".png", ".pdf"), format="pdf")
    plt.close()
    print(f"  Saved: {cm_path}")

    # Sentiment confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(sent_cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                xticklabels=sent_label_names, yticklabels=sent_label_names)
    ax.set_title(f"MTL {model_name} — Sentiment (α={alpha})\nAcc={sent_te_acc*100:.1f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_sent_path = os.path.join(exp_dir, "confusion_matrix_sent.png")
    plt.savefig(cm_sent_path, dpi=150)
    plt.savefig(cm_sent_path.replace(".png", ".pdf"), format="pdf")
    plt.close()
    print(f"  Saved: {cm_sent_path}")

    # ---- Save training curves ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(history["train_loss"], "b-", label="Total")
    axes[0].plot(history["train_hs_loss"], "r--", label="HS")
    axes[0].plot(history["train_sent_loss"], "g--", label="Sent")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    # HS validation
    axes[1].plot(history["val_hs_f1"], "r-o", label="HS F1")
    axes[1].plot(history["val_hs_acc"], "r--", label="HS Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("HS Validation")
    axes[1].legend()

    # Sentiment validation
    axes[2].plot(history["val_sent_f1"], "g-o", label="Sent F1")
    axes[2].plot(history["val_sent_acc"], "g--", label="Sent Acc")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Sentiment Validation")
    axes[2].legend()

    plt.suptitle(f"MTL {model_name} | HS={hs_task} | α={alpha}")
    plt.tight_layout()
    curve_path = os.path.join(exp_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.savefig(curve_path.replace(".png", ".pdf"), format="pdf")
    plt.close()
    print(f"  Saved: {curve_path}")

    # ---- Save classification report ----
    report_str = classification_report(hs_te_true, hs_te_preds,
                                        target_names=hs_label_names, digits=4,
                                        zero_division=0)
    report_path = os.path.join(exp_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"MTL {model_name} | HS={hs_task} | α={alpha}\n")
        f.write(f"{'='*60}\n\n")
        f.write("=== HATE SPEECH ===\n")
        f.write(report_str)
        f.write(f"\n\n=== SENTIMENT ===\n")
        sent_report_str = classification_report(sent_te_true, sent_te_preds,
                                                 target_names=sent_label_names, digits=4,
                                                 zero_division=0)
        f.write(sent_report_str)
    print(f"  Saved: {report_path}")

    # ---- Save results JSON ----
    results = {
        "experiment_id": exp_name,
        "approach": "multi_task_learning",
        "model_name": model_name,
        "model_path": model_path,
        "hs_task": hs_task,
        "alpha": alpha,
        "num_hs_labels": num_hs_labels,
        "num_sent_labels": num_sent_labels,
        "hs_label_names": hs_label_names,
        "sent_label_names": sent_label_names,
        "hs_train_size": len(hs_tr_t),
        "hs_val_size": len(hs_va_t),
        "hs_test_size": len(hs_te_t),
        "sent_train_size": len(sent_tr_t),
        "sent_val_size": len(sent_va_t),
        "sent_test_size": len(sent_te_t),
        "hs_accuracy": round(hs_te_acc * 100, 2),
        "hs_f1_macro": round(hs_te_f1 * 100, 2),
        "hs_f1_weighted": round(hs_te_f1_w * 100, 2),
        "hs_per_class_f1": {
            name: round(float(hs_te_f1_per[i]) * 100, 2)
            for i, name in enumerate(hs_label_names)
        },
        "hs_confusion_matrix": hs_cm.tolist(),
        "hs_classification_report": hs_report,
        "sent_accuracy": round(sent_te_acc * 100, 2),
        "sent_f1_macro": round(sent_te_f1 * 100, 2),
        "sent_per_class_f1": {
            name: round(float(sent_te_f1_per[i]) * 100, 2)
            for i, name in enumerate(sent_label_names)
        },
        "sent_confusion_matrix": sent_cm.tolist(),
        "best_epoch": best_epoch,
        "best_val_hs_f1": round(best_val_hs_f1 * 100, 2),
        "training_history": history,
        "total_params_M": round(total_params, 1),
        "training_time_seconds": round(total_time, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_file}")

    print(f"\n  DONE — {exp_name} in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Cleanup
    del model, optimizer, scheduler, best_state
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================
#  EXPLAINABILITY (LIME + SHAP)
# ============================================================

def run_explainability(model_name, model_path, hs_task, alpha):
    """Run LIME and SHAP on the best MTL model."""
    exp_name = f"mtl_{model_name}_{hs_task}_a{alpha:.1f}"
    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    best_model_dir = os.path.join(exp_dir, "best_model")
    explain_dir = os.path.join(exp_dir, "explainability")
    os.makedirs(explain_dir, exist_ok=True)

    if not os.path.exists(os.path.join(best_model_dir, "model.pt")):
        print(f"  ERROR: No trained model at {best_model_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  EXPLAINABILITY: {exp_name}")
    print(f"{'='*70}")

    # Load config
    with open(os.path.join(best_model_dir, "mtl_config.json")) as f:
        config = json.load(f)

    hs_label_names = config["hs_label_names"]
    num_hs_labels = config["num_hs_labels"]
    num_sent_labels = config["num_sent_labels"]
    encoder_name = config["encoder_name"]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTLModel(encoder_name, num_hs_labels, num_sent_labels).to(device)
    model.load_state_dict(torch.load(os.path.join(best_model_dir, "model.pt"),
                                      map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)

    # Load test data
    hs_cfg = HS_DATASETS[hs_task]
    hs_texts, hs_labels = load_hs_tsv(hs_cfg["path"], hs_label_names)
    _, _, _, _, te_texts, te_labels = split_data_80_10_10(hs_texts, hs_labels)

    # predict_proba function for LIME/SHAP
    def predict_proba(texts_list):
        if isinstance(texts_list, np.ndarray):
            texts_list = [str(t) for t in texts_list.flatten()]
        elif isinstance(texts_list, str):
            texts_list = [texts_list]
        else:
            texts_list = [str(t) for t in texts_list]

        all_probs = []
        batch_sz = 16
        for i in range(0, len(texts_list), batch_sz):
            batch = texts_list[i:i + batch_sz]
            enc = tokenizer(batch, truncation=True, padding="max_length",
                           max_length=MAX_SEQ_LEN, return_tensors="pt")
            with torch.no_grad():
                logits = model(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                    task="hs"
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    # ---- LIME ----
    print("\n  Running LIME...")
    try:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=hs_label_names)

        # Select examples: correct + misclassified per class
        preds = np.argmax(predict_proba(te_texts[:200]), axis=-1)
        true_arr = np.array(te_labels[:200])

        lime_results = []
        examples_per_type = 1

        for true_label_idx, true_label_name in enumerate(hs_label_names):
            # Correct prediction
            correct_mask = (true_arr == true_label_idx) & (preds == true_label_idx)
            correct_indices = np.where(correct_mask)[0]
            if len(correct_indices) > 0:
                idx = correct_indices[0]
                exp = explainer.explain_instance(
                    te_texts[idx], predict_proba, num_features=15, num_samples=500
                )
                lime_results.append({
                    "text": te_texts[idx],
                    "true_label": true_label_name,
                    "predicted_label": true_label_name,
                    "type": f"correct_{true_label_name}",
                    "top_features": exp.as_list(),
                })
                # Save HTML
                html_path = os.path.join(explain_dir, f"lime_correct_{true_label_name}.html")
                exp.save_to_file(html_path)
                print(f"    Saved: {html_path}")

            # Misclassified
            for pred_label_idx, pred_label_name in enumerate(hs_label_names):
                if pred_label_idx == true_label_idx:
                    continue
                mis_mask = (true_arr == true_label_idx) & (preds == pred_label_idx)
                mis_indices = np.where(mis_mask)[0]
                if len(mis_indices) > 0:
                    idx = mis_indices[0]
                    exp = explainer.explain_instance(
                        te_texts[idx], predict_proba, num_features=15, num_samples=500
                    )
                    etype = f"misclassified_{true_label_name}_as_{pred_label_name}"
                    lime_results.append({
                        "text": te_texts[idx],
                        "true_label": true_label_name,
                        "predicted_label": pred_label_name,
                        "type": etype,
                        "top_features": exp.as_list(),
                    })
                    html_path = os.path.join(explain_dir, f"lime_{etype}.html")
                    exp.save_to_file(html_path)
                    print(f"    Saved: {html_path}")

        # Save LIME results JSON
        lime_json_path = os.path.join(explain_dir, "lime_results.json")
        with open(lime_json_path, "w", encoding="utf-8") as f:
            json.dump(lime_results, f, indent=2, ensure_ascii=False)
        print(f"    Saved: {lime_json_path}")

    except Exception as e:
        print(f"  LIME failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- SHAP ----
    print("\n  Running SHAP...")
    try:
        import shap

        masker = shap.maskers.Text(tokenizer=r"\s+")
        shap_explainer = shap.Explainer(predict_proba, masker,
                                         output_names=hs_label_names)

        # Use 5 samples
        sample_texts = te_texts[:5]
        shap_values = shap_explainer(sample_texts)

        # Save SHAP bar plot
        shap_bar_path = os.path.join(explain_dir, "shap_bar.png")
        plt.figure()
        shap.plots.bar(shap_values[:, :, 0], show=False)
        plt.tight_layout()
        plt.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
        plt.savefig(shap_bar_path.replace(".png", ".pdf"), format="pdf",
                    bbox_inches="tight")
        plt.close()
        print(f"    Saved: {shap_bar_path}")

        # Save SHAP text plot
        shap_text_path = os.path.join(explain_dir, "shap_text.html")
        html = shap.plots.text(shap_values[:3], display=False)
        if html:
            with open(shap_text_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"    Saved: {shap_text_path}")

        # Save SHAP summary JSON
        shap_summary = []
        for i, text in enumerate(sample_texts):
            entry = {"text": text, "shap_values": {}}
            for c_idx, c_name in enumerate(hs_label_names):
                word_vals = []
                for j, word in enumerate(shap_values[i].data):
                    word_vals.append([str(word), round(float(shap_values[i].values[j, c_idx]), 4)])
                entry["shap_values"][c_name] = word_vals
            shap_summary.append(entry)

        shap_json_path = os.path.join(explain_dir, "shap_summary.json")
        with open(shap_json_path, "w", encoding="utf-8") as f:
            json.dump(shap_summary, f, indent=2, ensure_ascii=False)
        print(f"    Saved: {shap_json_path}")

    except Exception as e:
        print(f"  SHAP failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n  Explainability complete for {exp_name}")


# ============================================================
#  COMPARISON TABLE
# ============================================================

def print_comparison():
    """Print comparison of all MTL experiments."""
    if not os.path.exists(RESULTS_DIR):
        print("  No results found.")
        return

    results = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        rpath = os.path.join(RESULTS_DIR, d, "results.json")
        if os.path.isfile(rpath):
            with open(rpath) as f:
                results.append(json.load(f))

    if not results:
        print("  No results found.")
        return

    print(f"\n{'='*90}")
    print(f"  MTL RESULTS COMPARISON")
    print(f"{'='*90}")

    # Group by hs_task
    for task in ["binary", "3class"]:
        task_results = [r for r in results if r.get("hs_task") == task]
        if not task_results:
            continue

        task_results.sort(key=lambda x: x.get("hs_f1_macro", 0), reverse=True)

        print(f"\n  HS Task: {task}")
        print(f"  {'Model':<15s} {'α':>5s} {'HS-Acc':>8s} {'HS-F1':>8s} "
              f"{'Sent-Acc':>9s} {'Sent-F1':>8s} {'Time':>6s}")
        print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*6}")

        for r in task_results:
            model = r.get("model_name", "?")
            a = r.get("alpha", 0)
            hs_acc = r.get("hs_accuracy", 0)
            hs_f1 = r.get("hs_f1_macro", 0)
            sent_acc = r.get("sent_accuracy", 0)
            sent_f1 = r.get("sent_f1_macro", 0)
            t = r.get("training_time_seconds", 0) / 60
            print(f"  {model:<15s} {a:>5.1f} {hs_acc:>7.2f}% {hs_f1:>7.2f}% "
                  f"{sent_acc:>8.2f}% {sent_f1:>7.2f}% {t:>5.1f}m")

    # Compare with single-task baselines
    print(f"\n  {'='*60}")
    print(f"  COMPARISON WITH SINGLE-TASK BASELINES")
    print(f"  {'='*60}")

    st_dir = os.path.join(BASE_DIR, "results/hate_speech_models")
    if os.path.exists(st_dir):
        for task in ["binary", "3class"]:
            mtl_best = max(
                [r for r in results if r.get("hs_task") == task],
                key=lambda x: x.get("hs_f1_macro", 0),
                default=None
            )
            if not mtl_best:
                continue

            print(f"\n  {task}:")
            print(f"    MTL best ({mtl_best['model_name']}, α={mtl_best['alpha']}): "
                  f"F1={mtl_best['hs_f1_macro']:.2f}%")

            # Load single-task results
            for model_dir in sorted(os.listdir(st_dir)):
                if model_dir.endswith(f"_{task}"):
                    rpath = os.path.join(st_dir, model_dir, "results.json")
                    if os.path.exists(rpath):
                        with open(rpath) as f:
                            st_r = json.load(f)
                        st_f1 = st_r.get("f1_macro", 0)
                        if st_f1 < 1:
                            st_f1 *= 100
                        mname = st_r.get("model_name", model_dir)
                        print(f"    STL {mname}: F1={st_f1:.2f}%")

    # Also compare with hybrid results
    hybrid_dir = os.path.join(BASE_DIR, "results/hate_speech_hybrid")
    if os.path.exists(hybrid_dir):
        print(f"\n  Hybrid baselines:")
        for hd in sorted(os.listdir(hybrid_dir)):
            rpath = os.path.join(hybrid_dir, hd, "results.json")
            if os.path.isfile(rpath):
                with open(rpath) as f:
                    h_r = json.load(f)
                h_f1 = h_r.get("f1_macro", 0)
                if h_f1 < 1:
                    h_f1 *= 100
                ds = h_r.get("dataset_name", "?")
                eid = h_r.get("experiment_id", hd)
                print(f"    {eid}: F1={h_f1:.2f}% ({ds})")


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MTL Hate Speech + Sentiment")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODELS.keys()),
                        help="Single model to run")
    parser.add_argument("--hs_task", type=str, default=None,
                        choices=["binary", "3class"],
                        help="Hate speech task")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Weight for HS loss (0-1)")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all model × task × alpha combinations")
    parser.add_argument("--run_best", action="store_true",
                        help="Run top-3 models with best alpha on both tasks")
    parser.add_argument("--explain", action="store_true",
                        help="Run LIME/SHAP on best model")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments with existing results")
    parser.add_argument("--summary", action="store_true",
                        help="Print comparison table only")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.summary:
        print_comparison()
        return

    # GPU info
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print("\n  WARNING: No GPU detected")

    if args.explain:
        # Run explainability on specified or best model
        model_name = args.model or "marbertv2"
        hs_task = args.hs_task or "binary"
        alpha = args.alpha or 0.7
        run_explainability(model_name, MODELS[model_name], hs_task, alpha)
        return

    # Build experiment list
    if args.run_all:
        # All models × both tasks × 3 alpha values
        experiments = [
            (m, MODELS[m], t, a)
            for m in MODELS
            for t in ["binary", "3class"]
            for a in ALPHA_VALUES
        ]
    elif args.run_best:
        # Top 3 models × both tasks × best alpha (0.7)
        top3 = ["marbertv2", "marbert", "arabertv2"]
        experiments = [
            (m, MODELS[m], t, 0.7)
            for m in top3
            for t in ["binary", "3class"]
        ]
    elif args.model and args.hs_task:
        alpha = args.alpha if args.alpha is not None else 0.7
        experiments = [(args.model, MODELS[args.model], args.hs_task, alpha)]
    else:
        parser.error("Provide --run_all, --run_best, or --model + --hs_task")

    print(f"\n  Experiments to run: {len(experiments)}")
    for m, _, t, a in experiments:
        print(f"    {m} × {t} × α={a}")

    # Run experiments
    all_results = []
    failed = []

    for i, (model_name, model_path, hs_task, alpha) in enumerate(experiments, 1):
        print(f"\n\n  ╔═══════════════════════════════════════════╗")
        print(f"  ║  {i}/{len(experiments)}: {model_name} × {hs_task} × α={alpha}")
        print(f"  ╚═══════════════════════════════════════════╝")

        try:
            result = train_mtl(model_name, model_path, hs_task, alpha,
                               skip_existing=args.skip_existing)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ✗ FAILED: {model_name} × {hs_task} × α={alpha}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((model_name, hs_task, alpha, str(e)))
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    if all_results:
        print_comparison()

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for m, t, a, e in failed:
            print(f"    {m} × {t} × α={a}: {e}")

    print(f"\n  DONE — {len(all_results)} succeeded, {len(failed)} failed")
    print(f"  Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
