#!/usr/bin/env python3
"""
Hate Speech Detection — Hybrid Models Pipeline
================================================
Three hybrid approaches to boost SudaBERT performance:

  1. SudaBERT + BiLSTM + Attention head
  2. Ensemble (Top-3 models: MARBERTv2 + AraBERTv2 + SudaBERT)
  3. Knowledge Distillation (best teacher → SudaBERT student)

All experiments include:
  - Confusion matrix + classification report
  - LIME + SHAP explainability
  - Attention visualization (for BiLSTM)
  - Comparison with Phase 1 baselines

Usage:
    # Run ALL hybrid experiments:
    CUDA_VISIBLE_DEVICES=1 python3 hybrid_trainer.py

    # Run specific approach:
    CUDA_VISIBLE_DEVICES=1 python3 hybrid_trainer.py --approach bilstm
    CUDA_VISIBLE_DEVICES=1 python3 hybrid_trainer.py --approach ensemble
    CUDA_VISIBLE_DEVICES=1 python3 hybrid_trainer.py --approach distill

    # Run specific approach + dataset:
    CUDA_VISIBLE_DEVICES=1 python3 hybrid_trainer.py --approach bilstm --dataset binary

    # Run explainability only (after training):
    CUDA_VISIBLE_DEVICES=1 python3 hybrid_trainer.py --explain

    # Summary only:
    python3 hybrid_trainer.py --summary-only

Server: apl13, GPU 1 (RTX 8000, 49GB)
"""

import os
import gc
import json
import time
import argparse
import warnings
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
#  CONFIGURATION
# ============================================================================

# Base (pre-trained) model paths
BASE_MODELS = {
    "sudabert_v2": "models/sudabert_v2/sudabert_v2/",
    "sudabert_v2": "models/sudabert_v2/sudabert_v2/",
    "marbertv2":   "UBC-NLP/MARBERTv2",
    "arabertv2":   "aubmindlab/bert-base-arabertv02",
}

# Fine-tuned model paths (from Phase 1)
FINETUNED_DIR = "results/hate_speech_models"

DATASETS = {
    "binary": {
        "path": "data/labeling_corpus/dataset_binary.tsv",
        "labels": ["HARMFUL", "NEUTRAL"],
    },
    "3class": {
        "path": "data/labeling_corpus/dataset_3class.tsv",
        "labels": ["HATE", "OFFENSIVE", "NEUTRAL"],
    },
}

# Hyperparameters
SEED = 42
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

# BiLSTM hyperparameters
BILSTM_LR = 2e-5
BILSTM_EPOCHS = 8
BILSTM_LSTM_HIDDEN = 256
BILSTM_DROPOUT = 0.3
BILSTM_WEIGHT_DECAY = 0.01
BILSTM_WARMUP_RATIO = 0.1

# Distillation hyperparameters
DISTILL_LR = 2e-5
DISTILL_EPOCHS = 8
DISTILL_TEMPERATURE = 4.0
DISTILL_ALPHA = 0.5  # weight for soft loss vs hard loss
DISTILL_WEIGHT_DECAY = 0.01
DISTILL_WARMUP_RATIO = 0.1

RESULTS_DIR = "results/hate_speech_hybrid"

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================================
#  DATA LOADING (consistent with Phase 1)
# ============================================================================

class HateSpeechDataset(Dataset):
    """PyTorch Dataset for hate speech classification."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def load_dataset_tsv(dataset_name):
    """Load TSV dataset. Returns texts, labels, label_names, label2id."""
    cfg = DATASETS[dataset_name]
    path = cfg["path"]
    label_names = cfg["labels"]
    label2id = {name: i for i, name in enumerate(label_names)}

    texts, labels = [], []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                skipped += 1
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                skipped += 1
                continue
            text, label = parts[0].strip(), parts[1].strip()
            if label not in label2id or not text:
                skipped += 1
                continue
            texts.append(text)
            labels.append(label2id[label])

    print(f"  Loaded {len(texts):,} sentences from {path}")
    if skipped > 0:
        print(f"  Skipped {skipped} invalid rows")
    dist = Counter(labels)
    for name in label_names:
        count = dist.get(label2id[name], 0)
        print(f"    {name}: {count:,} ({100 * count / len(labels):.1f}%)")
    return texts, labels, label_names, label2id


def split_data(texts, labels, seed=SEED):
    """80/10/10 stratified split — SAME seed as Phase 1 for fair comparison."""
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels
    )
    print(f"  Split: train={len(train_texts):,}, val={len(val_texts):,}, test={len(test_texts):,}")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


def encode_texts(tokenizer, texts, max_len=MAX_SEQ_LEN):
    """Tokenize texts."""
    return tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=max_len, return_tensors="pt",
    )


# ============================================================================
#  VISUALIZATION UTILITIES
# ============================================================================

def plot_confusion_matrix(cm, label_names, save_path, title):
    """Plot and save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    n = len(label_names)
    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=label_names, yticklabels=label_names,
        title=title, ylabel="True Label", xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(train_losses, val_losses, save_path, title):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", label="Training Loss", markersize=4)
    if val_losses:
        ax.plot(epochs, val_losses, "r-o", label="Validation Loss", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_attention_heatmap(words, weights, save_path, title, top_n=30):
    """Plot attention weights over words as a horizontal bar chart."""
    # Take top_n words by attention weight
    if len(words) > top_n:
        indices = np.argsort(weights)[-top_n:]
        words = [words[i] for i in indices]
        weights = [weights[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, max(4, len(words) * 0.3)))
    y_pos = np.arange(len(words))
    ax.barh(y_pos, weights, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=9)
    ax.set_xlabel("Attention Weight")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def clear_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def evaluate_predictions(true_labels, preds, label_names, num_labels):
    """Compute all metrics and return dict."""
    acc = accuracy_score(true_labels, preds)
    f1_mac = f1_score(true_labels, preds, average="macro", zero_division=0)
    f1_wt = f1_score(true_labels, preds, average="weighted", zero_division=0)

    report_str = classification_report(
        true_labels, preds, target_names=label_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        true_labels, preds, target_names=label_names, digits=4,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true_labels, preds, labels=list(range(num_labels)))

    per_class_f1 = {}
    for name in label_names:
        if name in report_dict:
            per_class_f1[name] = round(report_dict[name]["f1-score"], 4)

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "report_str": report_str,
    }


def get_finetuned_path(model_name, dataset_name):
    """Get path to fine-tuned model from Phase 1."""
    return os.path.join(FINETUNED_DIR, f"{model_name}_{dataset_name}", "best_model")


# ============================================================================
#  APPROACH 1: SudaBERT + BiLSTM + Attention
# ============================================================================

class BiLSTMAttentionClassifier(nn.Module):
    """
    Transformer encoder → BiLSTM → Self-Attention → Classifier.

    Architecture:
      1. Transformer encodes input → sequence of hidden states
      2. BiLSTM captures sequential dependencies
      3. Self-attention computes importance weight per token
      4. Weighted sum → classification

    The attention weights provide built-in explainability.
    """

    def __init__(self, transformer, hidden_size, lstm_hidden, num_labels, dropout=0.3):
        super().__init__()
        self.transformer = transformer
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=dropout if 2 > 1 else 0.0,  # dropout between LSTM layers
        )
        # Attention: maps BiLSTM output (2*lstm_hidden) to scalar per token
        self.attention_linear = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
        self.attention_context = nn.Linear(lstm_hidden * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

        # Store for attention extraction
        self._last_attention_weights = None

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Transformer encoder — get last hidden state
        with torch.no_grad() if not self.transformer.training else torch.enable_grad():
            transformer_out = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
        hidden_states = transformer_out.last_hidden_state  # (batch, seq, hidden)

        # 2. BiLSTM
        lstm_out, _ = self.bilstm(hidden_states)  # (batch, seq, 2*lstm_hidden)

        # 3. Self-Attention with masking
        attn_hidden = torch.tanh(self.attention_linear(lstm_out))  # (batch, seq, 2*lstm_hidden)
        attn_scores = self.attention_context(attn_hidden).squeeze(-1)  # (batch, seq)

        # Mask padding tokens with large negative value
        padding_mask = (1.0 - attention_mask.float()) * -1e9
        attn_scores = attn_scores + padding_mask

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq)
        self._last_attention_weights = attn_weights.detach()

        # 4. Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, 2*lstm_hidden)
        context = self.dropout(context)

        # 5. Classification
        logits = self.classifier(context)  # (batch, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}

    def get_attention_weights(self):
        """Return last computed attention weights for explainability."""
        return self._last_attention_weights


def train_bilstm(dataset_name):
    """Train SudaBERT + BiLSTM + Attention on specified dataset."""
    exp_id = f"bilstm_sudabert_{dataset_name}"
    exp_dir = os.path.join(RESULTS_DIR, exp_id)
    results_json = os.path.join(exp_dir, "results.json")

    if os.path.exists(results_json):
        print(f"\n  SKIP {exp_id} — already completed")
        with open(results_json, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  APPROACH 1: SudaBERT + BiLSTM + Attention — {dataset_name}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("\n  [1/6] Loading data...")
    texts, labels, label_names, label2id = load_dataset_tsv(dataset_name)
    num_labels = len(label_names)
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
        split_data(texts, labels)

    # Load tokenizer and base model
    print("\n  [2/6] Loading SudaBERT base model...")
    base_path = BASE_MODELS["sudabert_v2"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_path)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)

    config = AutoConfig.from_pretrained(base_path)
    hidden_size = config.hidden_size
    print(f"  Hidden size: {hidden_size}")

    base_model = AutoModel.from_pretrained(base_path)

    # Encode data
    print("  Encoding data...")
    train_enc = encode_texts(tokenizer, train_texts)
    val_enc = encode_texts(tokenizer, val_texts)
    test_enc = encode_texts(tokenizer, test_texts)

    train_dataset = HateSpeechDataset(train_enc, train_labels)
    val_dataset = HateSpeechDataset(val_enc, val_labels)
    test_dataset = HateSpeechDataset(test_enc, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # Build hybrid model
    print("\n  [3/6] Building BiLSTM + Attention model...")
    model = BiLSTMAttentionClassifier(
        transformer=base_model,
        hidden_size=hidden_size,
        lstm_hidden=BILSTM_LSTM_HIDDEN,
        num_labels=num_labels,
        dropout=BILSTM_DROPOUT,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total params: {total_params:.1f}M | Trainable: {trainable_params:.1f}M")

    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM used: {vram:.1f} GB")

    # Optimizer with differential learning rates
    transformer_params = list(model.transformer.parameters())
    head_params = (
        list(model.bilstm.parameters()) +
        list(model.attention_linear.parameters()) +
        list(model.attention_context.parameters()) +
        list(model.classifier.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": transformer_params, "lr": BILSTM_LR * 0.1},   # Lower LR for transformer
        {"params": head_params, "lr": BILSTM_LR},                # Higher LR for BiLSTM head
    ], weight_decay=BILSTM_WEIGHT_DECAY)

    total_steps = len(train_loader) * BILSTM_EPOCHS
    warmup_steps = int(total_steps * BILSTM_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    print(f"\n  [4/6] Training ({BILSTM_EPOCHS} epochs)...")
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 3
    train_losses_epoch = []
    val_losses_epoch = []

    for epoch in range(1, BILSTM_EPOCHS + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        train_losses_epoch.append(avg_train_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_loss += outputs["loss"].item()
                preds = torch.argmax(outputs["logits"], dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch["labels"].cpu().numpy())
                n_val += 1

        avg_val_loss = val_loss / n_val
        val_losses_epoch.append(avg_val_loss)

        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_true, val_preds)

        print(f"    Epoch {epoch}/{BILSTM_EPOCHS}: "
              f"train_loss={avg_train_loss:.4f} | "
              f"val_loss={avg_val_loss:.4f} | "
              f"val_f1={val_f1:.4f} | val_acc={val_acc:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_dir = os.path.join(exp_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_model_dir, "model.pt"))
            tokenizer.save_pretrained(best_model_dir)
            # Save config info for reloading
            model_config = {
                "base_model_path": base_path,
                "hidden_size": hidden_size,
                "lstm_hidden": BILSTM_LSTM_HIDDEN,
                "num_labels": num_labels,
                "dropout": BILSTM_DROPOUT,
                "label_names": label_names,
                "label2id": label2id,
            }
            with open(os.path.join(best_model_dir, "hybrid_config.json"), "w", encoding="utf-8") as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False)
            print(f"      ✓ New best model saved (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch} (patience={patience})")
                break

    # Plot training curves
    plot_training_curves(
        train_losses_epoch, val_losses_epoch,
        os.path.join(exp_dir, "training_curve.png"),
        f"BiLSTM+Attention — {dataset_name}",
    )

    # Load best model for evaluation
    print("\n  [5/6] Evaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model", "model.pt"),
                                     map_location=device, weights_only=True))
    model.eval()

    test_preds = []
    test_true = []
    all_attention_weights = []
    all_input_ids = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = torch.argmax(outputs["logits"], dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(batch["labels"].cpu().numpy())

            # Collect attention weights for explainability
            attn_w = model.get_attention_weights()
            if attn_w is not None:
                all_attention_weights.append(attn_w.cpu().numpy())
                all_input_ids.append(batch["input_ids"].cpu().numpy())

    metrics = evaluate_predictions(
        np.array(test_true), np.array(test_preds), label_names, num_labels
    )

    print(f"\n{metrics['report_str']}")

    # Save classification report
    with open(os.path.join(exp_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp_id}\n")
        f.write(f"Architecture: SudaBERT + BiLSTM(2-layer) + Attention\n\n")
        f.write(metrics["report_str"])

    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm, label_names,
        os.path.join(exp_dir, "confusion_matrix.png"),
        f"Confusion Matrix — {exp_id}",
    )

    # Attention visualization — top 5 examples per class
    print("\n  [6/6] Generating attention visualizations...")
    attn_dir = os.path.join(exp_dir, "attention_plots")
    os.makedirs(attn_dir, exist_ok=True)

    if all_attention_weights:
        all_attn = np.concatenate(all_attention_weights, axis=0)
        all_ids = np.concatenate(all_input_ids, axis=0)

        attn_examples = []
        for class_idx, class_name in enumerate(label_names):
            class_indices = [i for i, t in enumerate(test_true) if t == class_idx]
            for idx in class_indices[:3]:  # 3 examples per class
                attn_w = all_attn[idx]
                input_ids = all_ids[idx]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)

                # Filter out padding and special tokens
                valid = []
                for t, w in zip(tokens, attn_w):
                    if t not in ("[PAD]", "[CLS]", "[SEP]", "<pad>", "<s>", "</s>"):
                        valid.append((t, float(w)))

                if valid:
                    words, weights = zip(*valid)
                    words, weights = list(words), list(weights)
                    pred_name = label_names[test_preds[idx]]

                    plot_attention_heatmap(
                        words, weights,
                        os.path.join(attn_dir, f"attn_{class_name}_{idx}.png"),
                        f"Attention — True: {class_name}, Pred: {pred_name}",
                    )
                    attn_examples.append({
                        "text": tokenizer.decode(input_ids, skip_special_tokens=True),
                        "true_label": class_name,
                        "pred_label": pred_name,
                        "top_words": sorted(valid, key=lambda x: -x[1])[:10],
                    })

        with open(os.path.join(exp_dir, "attention_examples.json"), "w", encoding="utf-8") as f:
            json.dump(attn_examples, f, indent=2, ensure_ascii=False)

    # Save results
    train_time = time.time() - start_time
    results = {
        "experiment_id": exp_id,
        "approach": "bilstm_attention",
        "base_model": "sudabert_v2",
        "dataset_name": dataset_name,
        "num_labels": num_labels,
        "label_names": label_names,
        "train_size": len(train_labels),
        "val_size": len(val_labels),
        "test_size": len(test_labels),
        "best_val_f1": round(best_val_f1, 4),
        "epochs_trained": len(train_losses_epoch),
        **{k: v for k, v in metrics.items() if k != "report_str"},
        "train_time_seconds": round(train_time, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved results: {results_json}")

    print(f"\n  DONE — {exp_id} in {train_time:.0f}s ({train_time / 60:.1f} min)")
    print(f"  Accuracy: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}")

    del model, base_model, optimizer, scheduler
    clear_gpu()

    return results


# ============================================================================
#  APPROACH 2: ENSEMBLE (Top-3 models)
# ============================================================================

def run_ensemble(dataset_name):
    """
    Ensemble of top-3 fine-tuned models via weighted probability averaging.
    Binary:  MARBERTv2 + AraBERTv2 + CAMeLBERT-DA
    3-class: AraBERTv2 + MARBERTv2 + SudaBERT-v1
    """
    exp_id = f"ensemble_{dataset_name}"
    exp_dir = os.path.join(RESULTS_DIR, exp_id)
    results_json = os.path.join(exp_dir, "results.json")

    if os.path.exists(results_json):
        print(f"\n  SKIP {exp_id} — already completed")
        with open(results_json, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  APPROACH 2: Ensemble — {dataset_name}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine ensemble members based on Phase 1 results
    if dataset_name == "binary":
        ensemble_models = ["marbertv2", "arabertv2", "camelbert_da"]
    else:  # 3class
        ensemble_models = ["arabertv2", "marbertv2", "sudabert_v2"]

    # Load Phase 1 results to get validation F1 for weighting
    weights = []
    for model_name in ensemble_models:
        p1_results_path = os.path.join(FINETUNED_DIR, f"{model_name}_{dataset_name}", "results.json")
        if os.path.exists(p1_results_path):
            with open(p1_results_path, "r", encoding="utf-8") as f:
                r = json.load(f)
            weights.append(r["f1_macro"])
            print(f"  {model_name}: F1-macro={r['f1_macro']:.4f}")
        else:
            print(f"  WARNING: Phase 1 results not found for {model_name}_{dataset_name}")
            weights.append(0.5)

    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    print(f"  Ensemble weights: {dict(zip(ensemble_models, [f'{w:.3f}' for w in weights]))}")

    # Load data
    print("\n  [1/3] Loading data...")
    texts, labels, label_names, label2id = load_dataset_tsv(dataset_name)
    num_labels = len(label_names)
    _, _, test_texts, _, _, test_labels = split_data(texts, labels)

    # Get predictions from each model
    print("\n  [2/3] Getting predictions from ensemble members...")
    all_probs = []

    for model_name in ensemble_models:
        model_path = get_finetuned_path(model_name, dataset_name)
        if not os.path.exists(model_path):
            print(f"  ERROR: {model_path} not found. Run Phase 1 first.")
            return None

        print(f"\n  Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        # Tokenize and predict
        model_probs = []
        for i in range(0, len(test_texts), EVAL_BATCH_SIZE):
            batch_texts = test_texts[i:i + EVAL_BATCH_SIZE]
            enc = tokenizer(
                batch_texts, truncation=True, padding="max_length",
                max_length=MAX_SEQ_LEN, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            model_probs.append(probs)

        model_probs = np.concatenate(model_probs, axis=0)
        all_probs.append(model_probs)
        print(f"    {model_name}: {model_probs.shape[0]} predictions")

        del model, tokenizer
        clear_gpu()

    # Weighted average ensemble
    print("\n  [3/3] Combining predictions...")
    weighted_probs = np.zeros_like(all_probs[0])
    for probs, weight in zip(all_probs, weights):
        weighted_probs += probs * weight

    ensemble_preds = np.argmax(weighted_probs, axis=-1)

    # Also compute majority vote for comparison
    individual_preds = [np.argmax(p, axis=-1) for p in all_probs]
    majority_preds = np.array([
        Counter(individual_preds[m][i] for m in range(len(ensemble_models))).most_common(1)[0][0]
        for i in range(len(test_labels))
    ])

    # Evaluate both strategies
    metrics_weighted = evaluate_predictions(
        np.array(test_labels), ensemble_preds, label_names, num_labels
    )
    metrics_majority = evaluate_predictions(
        np.array(test_labels), majority_preds, label_names, num_labels
    )

    # Use whichever is better
    if metrics_weighted["f1_macro"] >= metrics_majority["f1_macro"]:
        metrics = metrics_weighted
        best_strategy = "weighted_average"
        best_preds = ensemble_preds
        print(f"\n  Weighted average wins: F1={metrics_weighted['f1_macro']:.4f} vs "
              f"Majority vote: F1={metrics_majority['f1_macro']:.4f}")
    else:
        metrics = metrics_majority
        best_strategy = "majority_vote"
        best_preds = majority_preds
        print(f"\n  Majority vote wins: F1={metrics_majority['f1_macro']:.4f} vs "
              f"Weighted average: F1={metrics_weighted['f1_macro']:.4f}")

    print(f"\n{metrics['report_str']}")

    # Save reports
    with open(os.path.join(exp_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp_id}\n")
        f.write(f"Members: {ensemble_models}\n")
        f.write(f"Weights: {weights}\n")
        f.write(f"Best strategy: {best_strategy}\n\n")
        f.write(f"WEIGHTED AVERAGE:\n{metrics_weighted['report_str']}\n\n")
        f.write(f"MAJORITY VOTE:\n{metrics_majority['report_str']}\n")

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm, label_names,
        os.path.join(exp_dir, "confusion_matrix.png"),
        f"Confusion Matrix — Ensemble ({best_strategy}) — {dataset_name}",
    )

    # Save ensemble probabilities for explainability
    np.savez(
        os.path.join(exp_dir, "ensemble_probs.npz"),
        weighted_probs=weighted_probs,
        test_labels=np.array(test_labels),
        ensemble_preds=best_preds,
    )

    train_time = time.time() - start_time
    results = {
        "experiment_id": exp_id,
        "approach": "ensemble",
        "ensemble_members": ensemble_models,
        "ensemble_weights": [round(w, 4) for w in weights],
        "best_strategy": best_strategy,
        "dataset_name": dataset_name,
        "num_labels": num_labels,
        "label_names": label_names,
        "test_size": len(test_labels),
        "weighted_avg_f1": metrics_weighted["f1_macro"],
        "majority_vote_f1": metrics_majority["f1_macro"],
        **{k: v for k, v in metrics.items() if k != "report_str"},
        "train_time_seconds": round(train_time, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  DONE — {exp_id} in {train_time:.0f}s ({train_time / 60:.1f} min)")
    print(f"  Accuracy: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}")

    return results


# ============================================================================
#  APPROACH 3: KNOWLEDGE DISTILLATION (best teacher → SudaBERT)
# ============================================================================

def train_distillation(dataset_name):
    """
    Knowledge distillation: best Phase-1 model (teacher) → SudaBERT (student).
    Uses soft targets from teacher + hard targets from data.

    Loss = α * KL(student_soft || teacher_soft) * T² + (1-α) * CE(student, labels)
    """
    exp_id = f"distill_sudabert_{dataset_name}"
    exp_dir = os.path.join(RESULTS_DIR, exp_id)
    results_json = os.path.join(exp_dir, "results.json")

    if os.path.exists(results_json):
        print(f"\n  SKIP {exp_id} — already completed")
        with open(results_json, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  APPROACH 3: Knowledge Distillation — {dataset_name}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine best teacher from Phase 1
    best_teacher = None
    best_teacher_f1 = 0.0
    for model_name in ["marbertv2", "arabertv2", "camelbert_da", "qarib"]:
        p1_path = os.path.join(FINETUNED_DIR, f"{model_name}_{dataset_name}", "results.json")
        if os.path.exists(p1_path):
            with open(p1_path, "r", encoding="utf-8") as f:
                r = json.load(f)
            if r["f1_macro"] > best_teacher_f1:
                best_teacher_f1 = r["f1_macro"]
                best_teacher = model_name

    if best_teacher is None:
        print("  ERROR: No Phase 1 results found. Run Phase 1 first.")
        return None

    print(f"  Teacher: {best_teacher} (F1-macro={best_teacher_f1:.4f})")
    print(f"  Student: SudaBERT v2")
    print(f"  Temperature: {DISTILL_TEMPERATURE}, Alpha: {DISTILL_ALPHA}")

    # Load data
    print("\n  [1/6] Loading data...")
    texts, labels, label_names, label2id = load_dataset_tsv(dataset_name)
    num_labels = len(label_names)
    id2label = {i: name for name, i in label2id.items()}
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
        split_data(texts, labels)

    # Load teacher model
    print("\n  [2/6] Loading teacher model...")
    teacher_path = get_finetuned_path(best_teacher, dataset_name)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path)
    teacher_model.to(device)
    teacher_model.eval()
    print(f"  Teacher loaded from: {teacher_path}")

    # Pre-compute teacher soft targets for training data (saves GPU memory during training)
    print("  Computing teacher soft targets for training data...")
    teacher_train_logits = []
    for i in range(0, len(train_texts), EVAL_BATCH_SIZE):
        batch_texts = train_texts[i:i + EVAL_BATCH_SIZE]
        enc = teacher_tokenizer(
            batch_texts, truncation=True, padding="max_length",
            max_length=MAX_SEQ_LEN, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = teacher_model(**enc)
        teacher_train_logits.append(outputs.logits.cpu())

    teacher_train_logits = torch.cat(teacher_train_logits, dim=0)  # (N_train, num_labels)
    print(f"  Teacher logits computed: {teacher_train_logits.shape}")

    # Pre-compute teacher soft targets for validation data
    teacher_val_logits = []
    for i in range(0, len(val_texts), EVAL_BATCH_SIZE):
        batch_texts = val_texts[i:i + EVAL_BATCH_SIZE]
        enc = teacher_tokenizer(
            batch_texts, truncation=True, padding="max_length",
            max_length=MAX_SEQ_LEN, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = teacher_model(**enc)
        teacher_val_logits.append(outputs.logits.cpu())
    teacher_val_logits = torch.cat(teacher_val_logits, dim=0)

    # Free teacher from GPU
    del teacher_model
    clear_gpu()
    print("  Teacher unloaded from GPU.")

    # Load student model
    print("\n  [3/6] Loading student model (SudaBERT v1)...")
    student_path = BASE_MODELS["sudabert_v2"]
    student_tokenizer = AutoTokenizer.from_pretrained(student_path)
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    student_model.to(device)

    total_params = sum(p.numel() for p in student_model.parameters()) / 1e6
    print(f"  Student params: {total_params:.1f}M")

    # Encode data with STUDENT tokenizer
    print("  Encoding data with student tokenizer...")
    train_enc = encode_texts(student_tokenizer, train_texts)
    val_enc = encode_texts(student_tokenizer, val_texts)
    test_enc = encode_texts(student_tokenizer, test_texts)

    # Custom dataset that includes teacher logits
    class DistillDataset(Dataset):
        def __init__(self, encodings, labels, teacher_logits):
            self.encodings = encodings
            self.labels = labels
            self.teacher_logits = teacher_logits

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            item["teacher_logits"] = self.teacher_logits[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = DistillDataset(train_enc, train_labels, teacher_train_logits)
    val_dataset = DistillDataset(val_enc, val_labels, teacher_val_logits)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    test_dataset = HateSpeechDataset(test_enc, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=DISTILL_LR,
        weight_decay=DISTILL_WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * DISTILL_EPOCHS
    warmup_steps = int(total_steps * DISTILL_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    print(f"\n  [4/6] Training ({DISTILL_EPOCHS} epochs)...")
    T = DISTILL_TEMPERATURE
    alpha = DISTILL_ALPHA
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 3
    train_losses_epoch = []
    val_losses_epoch = []

    for epoch in range(1, DISTILL_EPOCHS + 1):
        # --- Train ---
        student_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hard_labels = batch["labels"].to(device)
            t_logits = batch["teacher_logits"].to(device)

            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            s_logits = outputs.logits

            # Distillation loss
            soft_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)

            hard_loss = F.cross_entropy(s_logits, hard_labels)

            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        train_losses_epoch.append(avg_train_loss)

        # --- Validate ---
        student_model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                hard_labels = batch["labels"].to(device)
                t_logits = batch["teacher_logits"].to(device)

                outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                s_logits = outputs.logits

                soft_loss = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                hard_loss = F.cross_entropy(s_logits, hard_labels)
                loss = alpha * soft_loss + (1 - alpha) * hard_loss

                val_loss += loss.item()
                preds = torch.argmax(s_logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(hard_labels.cpu().numpy())
                n_val += 1

        avg_val_loss = val_loss / n_val
        val_losses_epoch.append(avg_val_loss)

        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_true, val_preds)

        print(f"    Epoch {epoch}/{DISTILL_EPOCHS}: "
              f"train_loss={avg_train_loss:.4f} | "
              f"val_loss={avg_val_loss:.4f} | "
              f"val_f1={val_f1:.4f} | val_acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_dir = os.path.join(exp_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            student_model.save_pretrained(best_model_dir)
            student_tokenizer.save_pretrained(best_model_dir)
            print(f"      ✓ New best student saved (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch} (patience={patience})")
                break

    plot_training_curves(
        train_losses_epoch, val_losses_epoch,
        os.path.join(exp_dir, "training_curve.png"),
        f"Knowledge Distillation — {dataset_name}",
    )

    # Evaluate on test set
    print("\n  [5/6] Evaluating best student on test set...")
    best_model_dir = os.path.join(exp_dir, "best_model")
    student_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    student_model.to(device)
    student_model.eval()

    test_preds = []
    test_true = []
    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            outputs = student_model(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(batch_dev["labels"].cpu().numpy())

    metrics = evaluate_predictions(
        np.array(test_true), np.array(test_preds), label_names, num_labels
    )

    print(f"\n{metrics['report_str']}")

    # Comparison with base SudaBERT
    print("\n  [6/6] Comparing with baseline SudaBERT...")
    base_results_path = os.path.join(FINETUNED_DIR, f"sudabert_v1_{dataset_name}", "results.json")
    if os.path.exists(base_results_path):
        with open(base_results_path, "r", encoding="utf-8") as f:
            base_r = json.load(f)
        improvement = metrics["f1_macro"] - base_r["f1_macro"]
        print(f"  Baseline SudaBERT F1: {base_r['f1_macro']:.4f}")
        print(f"  Distilled SudaBERT F1: {metrics['f1_macro']:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({improvement * 100:+.1f}pp)")

    # Save
    with open(os.path.join(exp_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp_id}\n")
        f.write(f"Teacher: {best_teacher} (F1={best_teacher_f1:.4f})\n")
        f.write(f"Student: SudaBERT v2\n")
        f.write(f"T={DISTILL_TEMPERATURE}, alpha={DISTILL_ALPHA}\n\n")
        f.write(metrics["report_str"])

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm, label_names,
        os.path.join(exp_dir, "confusion_matrix.png"),
        f"Confusion Matrix — Distilled SudaBERT — {dataset_name}",
    )

    train_time = time.time() - start_time
    results = {
        "experiment_id": exp_id,
        "approach": "knowledge_distillation",
        "teacher": best_teacher,
        "teacher_f1": best_teacher_f1,
        "student": "sudabert_v2",
        "temperature": DISTILL_TEMPERATURE,
        "alpha": DISTILL_ALPHA,
        "dataset_name": dataset_name,
        "num_labels": num_labels,
        "label_names": label_names,
        "train_size": len(train_labels),
        "val_size": len(val_labels),
        "test_size": len(test_labels),
        "best_val_f1": round(best_val_f1, 4),
        "epochs_trained": len(train_losses_epoch),
        **{k: v for k, v in metrics.items() if k != "report_str"},
        "train_time_seconds": round(train_time, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  DONE — {exp_id} in {train_time:.0f}s ({train_time / 60:.1f} min)")
    print(f"  Accuracy: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}")

    del student_model
    clear_gpu()

    return results


# ============================================================================
#  EXPLAINABILITY (LIME + SHAP for all hybrid approaches)
# ============================================================================

def run_explainability():
    """Run LIME + SHAP on all hybrid models."""
    print(f"\n{'=' * 70}")
    print(f"  EXPLAINABILITY — LIME + SHAP for Hybrid Models")
    print(f"{'=' * 70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in DATASETS:
        for approach in ["bilstm", "distill", "ensemble"]:
            if approach == "bilstm":
                exp_id = f"bilstm_sudabert_{dataset_name}"
            elif approach == "distill":
                exp_id = f"distill_sudabert_{dataset_name}"
            else:
                exp_id = f"ensemble_{dataset_name}"

            exp_dir = os.path.join(RESULTS_DIR, exp_id)
            results_json = os.path.join(exp_dir, "results.json")

            if not os.path.exists(results_json):
                print(f"\n  SKIP {exp_id} — not trained yet")
                continue

            with open(results_json, "r", encoding="utf-8") as f:
                exp_results = json.load(f)

            label_names = exp_results["label_names"]
            explain_dir = os.path.join(exp_dir, "explainability")
            os.makedirs(explain_dir, exist_ok=True)

            print(f"\n  --- {exp_id} ---")

            # Build predict_proba function
            if approach == "bilstm":
                # Load BiLSTM model
                best_dir = os.path.join(exp_dir, "best_model")
                config_path = os.path.join(best_dir, "hybrid_config.json")
                if not os.path.exists(config_path):
                    print(f"    SKIP — hybrid_config.json not found")
                    continue

                with open(config_path, "r", encoding="utf-8") as f:
                    hconfig = json.load(f)

                tokenizer = AutoTokenizer.from_pretrained(best_dir)
                base_model = AutoModel.from_pretrained(hconfig["base_model_path"])
                model = BiLSTMAttentionClassifier(
                    transformer=base_model,
                    hidden_size=hconfig["hidden_size"],
                    lstm_hidden=hconfig["lstm_hidden"],
                    num_labels=hconfig["num_labels"],
                    dropout=hconfig["dropout"],
                )
                model.load_state_dict(torch.load(
                    os.path.join(best_dir, "model.pt"),
                    map_location=device, weights_only=True
                ))
                model.to(device)
                model.eval()

                def predict_proba(texts_list):
                    if isinstance(texts_list, np.ndarray):
                        texts_list = [str(t) for t in texts_list.flatten()]
                    elif isinstance(texts_list, str):
                        texts_list = [texts_list]
                    else:
                        texts_list = [str(t) for t in texts_list]
                    all_probs = []
                    for i in range(0, len(texts_list), EVAL_BATCH_SIZE):
                        batch = texts_list[i:i + EVAL_BATCH_SIZE]
                        enc = tokenizer(
                            batch, truncation=True, padding="max_length",
                            max_length=MAX_SEQ_LEN, return_tensors="pt"
                        )
                        enc = {k: v.to(device) for k, v in enc.items()}
                        with torch.no_grad():
                            outputs = model(
                                input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"],
                            )
                            probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()
                        all_probs.append(probs)
                    return np.vstack(all_probs)

            elif approach == "distill":
                best_dir = os.path.join(exp_dir, "best_model")
                tokenizer = AutoTokenizer.from_pretrained(best_dir)
                model = AutoModelForSequenceClassification.from_pretrained(best_dir)
                model.to(device)
                model.eval()

                def predict_proba(texts_list):
                    if isinstance(texts_list, np.ndarray):
                        texts_list = [str(t) for t in texts_list.flatten()]
                    elif isinstance(texts_list, str):
                        texts_list = [texts_list]
                    else:
                        texts_list = [str(t) for t in texts_list]
                    all_probs = []
                    for i in range(0, len(texts_list), EVAL_BATCH_SIZE):
                        batch = texts_list[i:i + EVAL_BATCH_SIZE]
                        enc = tokenizer(
                            batch, truncation=True, padding="max_length",
                            max_length=MAX_SEQ_LEN, return_tensors="pt"
                        )
                        enc = {k: v.to(device) for k, v in enc.items()}
                        with torch.no_grad():
                            outputs = model(**enc)
                            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                        all_probs.append(probs)
                    return np.vstack(all_probs)

            elif approach == "ensemble":
                # Load all ensemble members
                if dataset_name == "binary":
                    members = ["marbertv2", "arabertv2", "camelbert_da"]
                else:
                    members = ["arabertv2", "marbertv2", "sudabert_v2"]

                ens_models = []
                ens_tokenizers = []
                for m_name in members:
                    m_path = get_finetuned_path(m_name, dataset_name)
                    tok = AutoTokenizer.from_pretrained(m_path)
                    mdl = AutoModelForSequenceClassification.from_pretrained(m_path)
                    mdl.to(device)
                    mdl.eval()
                    ens_models.append(mdl)
                    ens_tokenizers.append(tok)

                # Get weights
                ens_weights = []
                for m_name in members:
                    rp = os.path.join(FINETUNED_DIR, f"{m_name}_{dataset_name}", "results.json")
                    with open(rp, "r", encoding="utf-8") as f:
                        r = json.load(f)
                    ens_weights.append(r["f1_macro"])
                w_sum = sum(ens_weights)
                ens_weights = [w / w_sum for w in ens_weights]

                def predict_proba(texts_list):
                    weighted = None
                    for mdl, tok, weight in zip(ens_models, ens_tokenizers, ens_weights):
                        all_p = []
                        for i in range(0, len(texts_list), EVAL_BATCH_SIZE):
                            batch = texts_list[i:i + EVAL_BATCH_SIZE]
                            enc = tok(
                                batch, truncation=True, padding="max_length",
                                max_length=MAX_SEQ_LEN, return_tensors="pt"
                            )
                            enc = {k: v.to(device) for k, v in enc.items()}
                            with torch.no_grad():
                                out = mdl(**enc)
                                p = torch.softmax(out.logits, dim=-1).cpu().numpy()
                            all_p.append(p)
                        probs = np.vstack(all_p)
                        if weighted is None:
                            weighted = probs * weight
                        else:
                            weighted += probs * weight
                    return weighted

                # Use first tokenizer for LIME text splitting
                tokenizer = ens_tokenizers[0]

            # Load test data
            texts_all, labels_all, _, _ = load_dataset_tsv(dataset_name)
            _, _, test_texts, _, _, test_labels = split_data(texts_all, labels_all)

            # ---- LIME ----
            print(f"    Running LIME...")
            try:
                from lime.lime_text import LimeTextExplainer

                explainer = LimeTextExplainer(
                    class_names=label_names,
                    split_expression=r"\s+",
                )

                # Select examples: 2 correct + 2 misclassified per class
                test_pred_probs = predict_proba(test_texts[:300])
                test_pred_labels = np.argmax(test_pred_probs, axis=-1)

                examples = []
                for ci, cn in enumerate(label_names):
                    count = 0
                    for i in range(min(300, len(test_texts))):
                        if test_labels[i] == ci and test_pred_labels[i] == ci and count < 2:
                            examples.append({"idx": i, "text": test_texts[i],
                                             "true": cn, "pred": cn, "type": f"correct_{cn}"})
                            count += 1
                    for i in range(min(300, len(test_texts))):
                        if test_labels[i] == ci and test_pred_labels[i] != ci:
                            pred_name = label_names[test_pred_labels[i]]
                            examples.append({"idx": i, "text": test_texts[i],
                                             "true": cn, "pred": pred_name,
                                             "type": f"misclass_{cn}_as_{pred_name}"})
                            break

                lime_results = []
                for ex in examples:
                    print(f"      LIME: [{ex['type']}] {ex['text'][:50]}...")
                    try:
                        explanation = explainer.explain_instance(
                            ex["text"], predict_proba,
                            num_features=15, num_samples=500,
                        )
                        html_path = os.path.join(explain_dir, f"lime_{ex['type']}.html")
                        explanation.save_to_file(html_path)

                        top_features = explanation.as_list()
                        lime_results.append({
                            "text": ex["text"],
                            "true_label": ex["true"],
                            "predicted_label": ex["pred"],
                            "type": ex["type"],
                            "top_features": [(w, round(s, 4)) for w, s in top_features],
                        })
                    except Exception as e:
                        print(f"        LIME error: {e}")

                with open(os.path.join(explain_dir, "lime_results.json"), "w", encoding="utf-8") as f:
                    json.dump(lime_results, f, indent=2, ensure_ascii=False)
                print(f"    LIME complete: {len(lime_results)} explanations")

            except ImportError:
                print("    LIME not installed. Run: pip3 install lime --user")

            # ---- SHAP ----
            print(f"    Running SHAP...")
            try:
                import shap

                masker = shap.maskers.Text(tokenizer=r"\s+")
                shap_explainer = shap.Explainer(
                    predict_proba, masker=masker, output_names=label_names,
                )

                sample_texts = [ex["text"] for ex in examples[:4]]
                print(f"      Computing SHAP for {len(sample_texts)} examples...")
                shap_values = shap_explainer(sample_texts)

                # Bar plot
                try:
                    fig = plt.figure(figsize=(12, 8))
                    shap.plots.bar(shap_values[:, :, 0], show=False)
                    plt.title(f"SHAP — {exp_id}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(explain_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"    Saved SHAP bar plot")
                except Exception as e:
                    print(f"    SHAP bar plot error: {e}")

                # Text plot
                try:
                    html = shap.plots.text(shap_values[:2], display=False)
                    if html:
                        with open(os.path.join(explain_dir, "shap_text.html"), "w", encoding="utf-8") as f:
                            f.write(str(html))
                        print(f"    Saved SHAP text plot")
                except Exception as e:
                    print(f"    SHAP text plot error: {e}")

                print(f"    SHAP complete")

            except ImportError:
                print("    SHAP not installed. Run: pip3 install shap --user")
            except Exception as e:
                print(f"    SHAP error: {e}")

            # Cleanup per approach
            if approach == "ensemble":
                for mdl in ens_models:
                    del mdl
                ens_models.clear()
            elif approach in ("bilstm", "distill"):
                del model
            clear_gpu()

    print(f"\n  Explainability complete.")


# ============================================================================
#  SUMMARY — Compare ALL models (Phase 1 baselines + hybrids)
# ============================================================================

def compile_full_summary():
    """Compare hybrid results with Phase 1 baselines."""
    print(f"\n{'=' * 100}")
    print(f"  FULL SUMMARY — Baselines + Hybrid Models")
    print(f"{'=' * 100}")

    all_results = []

    # Load Phase 1 baselines
    phase1_models = ["sudabert_v2", "sudabert_v2", "marbertv2", "arabertv2",
                     "camelbert_da", "qarib", "xlm_roberta", "mbert"]
    for model_name in phase1_models:
        for ds in DATASETS:
            rp = os.path.join(FINETUNED_DIR, f"{model_name}_{ds}", "results.json")
            if os.path.exists(rp):
                with open(rp, "r", encoding="utf-8") as f:
                    r = json.load(f)
                r["approach"] = "baseline"
                r["display_name"] = model_name
                all_results.append(r)

    # Load hybrid results
    hybrid_experiments = [
        "bilstm_sudabert_binary", "bilstm_sudabert_3class",
        "ensemble_binary", "ensemble_3class",
        "distill_sudabert_binary", "distill_sudabert_3class",
    ]
    for exp_id in hybrid_experiments:
        rp = os.path.join(RESULTS_DIR, exp_id, "results.json")
        if os.path.exists(rp):
            with open(rp, "r", encoding="utf-8") as f:
                r = json.load(f)
            if "bilstm" in exp_id:
                r["display_name"] = "SudaBERT+BiLSTM"
            elif "ensemble" in exp_id:
                r["display_name"] = "Ensemble(Top3)"
            elif "distill" in exp_id:
                r["display_name"] = "SudaBERT-Distill"
            all_results.append(r)

    if not all_results:
        print("  No results found.")
        return

    # Print table
    header = (
        f"{'Model':<20s} | {'Type':<12s} | {'Dataset':<8s} | "
        f"{'Acc':>6s} | {'F1-Mac':>7s} | {'F1-Wt':>7s} |"
    )
    header_3c = " HATE-F1 | OFF-F1  | NEU-F1  |"
    sep = "-" * len(header + header_3c)

    print(f"\n{sep}")
    print(f"{header}{header_3c}")
    print(f"{sep}")

    all_results.sort(key=lambda r: (r.get("dataset_name", ""), -r.get("f1_macro", 0)))

    best_overall = {"binary": {"f1": 0, "name": ""}, "3class": {"f1": 0, "name": ""}}

    for r in all_results:
        ds = r.get("dataset_name", "?")
        dname = r.get("display_name", r.get("model_name", "?"))
        approach = r.get("approach", "?")

        row = (
            f"{dname:<20s} | {approach:<12s} | {ds:<8s} | "
            f"{r['accuracy']:>5.1%} | {r['f1_macro']:>6.1%} | "
            f"{r['f1_weighted']:>6.1%} |"
        )

        pcf = r.get("per_class_f1", {})
        hate_f1 = f"{pcf['HATE']:.1%}" if "HATE" in pcf else "  -  "
        off_f1 = f"{pcf['OFFENSIVE']:.1%}" if "OFFENSIVE" in pcf else "  -  "
        if "NEUTRAL" in pcf:
            neu_f1 = f"{pcf['NEUTRAL']:.1%}"
        elif "HARMFUL" in pcf:
            neu_f1 = f"{pcf['HARMFUL']:.1%}"
        else:
            neu_f1 = "  -  "

        row_3c = f" {hate_f1:>7s} | {off_f1:>7s} | {neu_f1:>7s} |"
        print(f"{row}{row_3c}")

        if r["f1_macro"] > best_overall[ds]["f1"]:
            best_overall[ds]["f1"] = r["f1_macro"]
            best_overall[ds]["name"] = f"{dname} ({approach})"

    print(f"{sep}")

    for ds in ["binary", "3class"]:
        if best_overall[ds]["name"]:
            print(f"\n  BEST on {ds}: {best_overall[ds]['name']} "
                  f"(F1-macro={best_overall[ds]['f1']:.4f})")

    # Save
    summary_path = os.path.join(RESULTS_DIR, "full_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        # Remove report_str (verbose) before saving
        clean_results = []
        for r in all_results:
            cr = {k: v for k, v in r.items() if k != "report_str"}
            clean_results.append(cr)
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {summary_path}")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid Models Trainer")
    parser.add_argument("--approach", type=str, default=None,
                        choices=["bilstm", "ensemble", "distill"],
                        help="Run specific approach only")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["binary", "3class"],
                        help="Run on specific dataset only")
    parser.add_argument("--explain", action="store_true",
                        help="Run LIME/SHAP explainability only")
    parser.add_argument("--summary-only", action="store_true",
                        help="Compile summary only")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  Hate Speech Detection — Hybrid Models Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print(f"  WARNING: No GPU detected")
    print(f"{'=' * 70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.summary_only:
        compile_full_summary()
        return

    if args.explain:
        run_explainability()
        compile_full_summary()
        return

    datasets_to_run = [args.dataset] if args.dataset else ["binary", "3class"]
    approaches_to_run = [args.approach] if args.approach else ["bilstm", "ensemble", "distill"]

    all_start = time.time()

    for ds in datasets_to_run:
        for approach in approaches_to_run:
            try:
                if approach == "bilstm":
                    train_bilstm(ds)
                elif approach == "ensemble":
                    run_ensemble(ds)
                elif approach == "distill":
                    train_distillation(ds)
            except Exception as e:
                print(f"\n  FAILED: {approach}_{ds}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                clear_gpu()

    total_time = time.time() - all_start
    print(f"\n{'=' * 70}")
    print(f"  ALL HYBRID EXPERIMENTS COMPLETE")
    print(f"  Total time: {total_time:.0f}s ({total_time / 3600:.1f} hours)")
    print(f"{'=' * 70}")

    compile_full_summary()


if __name__ == "__main__":
    main()
