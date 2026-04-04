#!/usr/bin/env python3
"""
Hate Speech Detection — Multi-Model Training Pipeline
=====================================================
Fine-tunes 8 transformer models on binary (HARMFUL/NEUTRAL) and
3-class (HATE/OFFENSIVE/NEUTRAL) Sudanese Arabic datasets.

Outputs per experiment:
  - Trained model saved to results/hate_speech_models/{model}_{dataset}/
  - Confusion matrix PNG
  - Classification report (text + JSON)
  - Training loss curve PNG

Final outputs:
  - results/hate_speech_models/summary.json — all metrics
  - results/hate_speech_models/summary_table.txt — comparison table
  - results/hate_speech_models/explainability/ — LIME + SHAP on best model

Usage:
    # Run ALL 16 experiments (sequential):
    CUDA_VISIBLE_DEVICES=1 python3 hate_speech_trainer.py

    # Run ONE specific experiment:
    CUDA_VISIBLE_DEVICES=1 python3 hate_speech_trainer.py --model marbertv2 --dataset binary

    # Resume (skip already-completed experiments):
    CUDA_VISIBLE_DEVICES=1 python3 hate_speech_trainer.py --resume

    # Compile summary from existing results:
    python3 hate_speech_trainer.py --summary-only

    # Run LIME/SHAP on best model (after training):
    CUDA_VISIBLE_DEVICES=1 python3 hate_speech_trainer.py --explain

Server: apl13, GPU 1 (RTX 8000, 49GB)
"""

import os
import sys
import gc
import json
import time
import argparse
import warnings
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

# Suppress non-critical warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")

# Must import AFTER setting CUDA_VISIBLE_DEVICES (done externally)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
#  CONFIGURATION
# ============================================================================

MODELS = {
    "sudabert_v1": "models/sudabert_enhanced/",
    "sudabert_v2": "models/sudabert_v2/sudabert_v2/",
    "marbertv2":   "UBC-NLP/MARBERTv2",
    "arabertv2":   "aubmindlab/bert-base-arabertv02",
    "camelbert_da": "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "qarib":       "qarib/bert-base-qarib",
    "xlm_roberta": "xlm-roberta-base",
    "mbert":       "bert-base-multilingual-cased",
}

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

# Hyperparameters (literature-backed)
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
MAX_SEQ_LEN = 128
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

RESULTS_DIR = "results/hate_speech_models"


# ============================================================================
#  DATASET CLASS
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


# ============================================================================
#  DATA LOADING
# ============================================================================

def load_dataset(dataset_name):
    """
    Load TSV file and return texts, labels, label_names, label2id.
    TSV format: text<TAB>label (with header row).
    """
    cfg = DATASETS[dataset_name]
    path = cfg["path"]
    label_names = cfg["labels"]
    label2id = {name: i for i, name in enumerate(label_names)}

    texts = []
    labels = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if "\t" not in line:
                skipped += 1
                continue
            # Split on FIRST tab only (text may contain tabs, though cleaned)
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                skipped += 1
                continue
            text, label = parts[0], parts[1].strip()
            if label not in label2id:
                skipped += 1
                continue
            if not text.strip():
                skipped += 1
                continue
            texts.append(text)
            labels.append(label2id[label])

    print(f"  Loaded {len(texts):,} sentences from {path}")
    if skipped > 0:
        print(f"  Skipped {skipped} invalid rows")

    # Verify distribution
    from collections import Counter
    dist = Counter(labels)
    for label_name in label_names:
        lid = label2id[label_name]
        count = dist.get(lid, 0)
        print(f"    {label_name}: {count:,} ({100 * count / len(labels):.1f}%)")

    return texts, labels, label_names, label2id


def split_data(texts, labels, seed=SEED):
    """
    80/10/10 stratified split.
    Returns (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """
    # First split: 80% train, 20% temp
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    # Second split: 50/50 of temp → 10% val, 10% test
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels
    )

    print(f"  Split: train={len(train_texts):,}, val={len(val_texts):,}, test={len(test_texts):,}")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


def encode_data(tokenizer, texts, max_len=MAX_SEQ_LEN):
    """Tokenize texts using the model's tokenizer."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    return encodings


# ============================================================================
#  METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """Compute metrics for Trainer evaluation."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro", zero_division=0)
    f1_wt = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
    }


# ============================================================================
#  VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, label_names, save_path, title):
    """Plot and save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    n_classes = len(label_names)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=label_names,
        yticklabels=label_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix: {save_path}")


def plot_training_loss(log_history, save_path, title):
    """Plot training and eval loss curves from Trainer log history."""
    train_losses = []
    train_steps = []
    eval_losses = []
    eval_steps = []  # Use steps (not epochs) for consistent x-axis

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_losses.append(entry["loss"])
            train_steps.append(entry.get("step", len(train_steps)))
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
            eval_steps.append(entry.get("step", len(eval_steps)))

    fig, ax = plt.subplots(figsize=(10, 5))

    if train_losses:
        ax.plot(train_steps, train_losses, "b-", alpha=0.6, label="Training Loss")
    if eval_losses:
        ax.plot(eval_steps, eval_losses, "r-o", label="Eval Loss", markersize=5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training curve: {save_path}")


# ============================================================================
#  MEMORY MANAGEMENT
# ============================================================================

def clear_gpu_memory():
    """Force GPU memory cleanup between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ============================================================================
#  SINGLE EXPERIMENT
# ============================================================================

def run_experiment(model_name, dataset_name, resume=False):
    """
    Fine-tune one model on one dataset.
    Returns dict with all metrics, or None if skipped/failed.
    """
    experiment_id = f"{model_name}_{dataset_name}"
    exp_dir = os.path.join(RESULTS_DIR, experiment_id)
    results_json = os.path.join(exp_dir, "results.json")

    # Check if already completed
    if resume and os.path.exists(results_json):
        print(f"\n  SKIP {experiment_id} — already completed (--resume)")
        with open(results_json, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT: {experiment_id}")
    print(f"  Model: {MODELS[model_name]}")
    print(f"  Dataset: {DATASETS[dataset_name]['path']}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    start_time = time.time()

    # ---- Load data ----
    print("\n  [1/5] Loading data...")
    texts, labels, label_names, label2id = load_dataset(dataset_name)
    num_labels = len(label_names)
    id2label = {i: name for name, i in label2id.items()}

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
        split_data(texts, labels)

    # ---- Load tokenizer ----
    print("\n  [2/5] Loading tokenizer and model...")
    model_path = MODELS[model_name]
    is_local = os.path.isdir(model_path)

    tokenizer_kwargs = {"token": True} if not is_local else {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    except Exception as e:
        print(f"  ERROR loading tokenizer: {e}")
        return None

    # Encode data
    print("  Encoding train set...")
    train_enc = encode_data(tokenizer, train_texts)
    print("  Encoding val set...")
    val_enc = encode_data(tokenizer, val_texts)
    print("  Encoding test set...")
    test_enc = encode_data(tokenizer, test_texts)

    train_dataset = HateSpeechDataset(train_enc, train_labels)
    val_dataset = HateSpeechDataset(val_enc, val_labels)
    test_dataset = HateSpeechDataset(test_enc, test_labels)

    # ---- Load model ----
    model_kwargs = {"token": True} if not is_local else {}
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            **model_kwargs,
        )
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return None

    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Model loaded: {model_params:.1f}M params ({trainable_params:.1f}M trainable)")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram_used:.1f} GB / {vram_total:.1f} GB")

    # ---- Training ----
    print("\n  [3/5] Training...")

    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        logging_first_step=True,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    train_result = trainer.train()
    train_time = time.time() - start_time

    # Save training loss curve
    plot_training_loss(
        trainer.state.log_history,
        os.path.join(exp_dir, "training_curve.png"),
        f"Training Loss — {experiment_id}",
    )

    # ---- Evaluation on TEST set ----
    print("\n  [4/5] Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = np.array(test_labels)

    # Classification report
    report_str = classification_report(
        true_labels, preds,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        true_labels, preds,
        target_names=label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{report_str}")

    # Save classification report
    report_path = os.path.join(exp_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment_id}\n")
        f.write(f"Model: {MODELS[model_name]}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Train time: {train_time:.0f}s ({train_time / 60:.1f} min)\n\n")
        f.write(report_str)
    print(f"  Saved report: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds, labels=list(range(num_labels)))
    plot_confusion_matrix(
        cm, label_names,
        os.path.join(exp_dir, "confusion_matrix.png"),
        f"Confusion Matrix — {experiment_id}",
    )

    # ---- Save model ----
    print("\n  [5/5] Saving best model...")
    best_model_dir = os.path.join(exp_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"  Saved model to: {best_model_dir}")

    # ---- Compile results ----
    acc = accuracy_score(true_labels, preds)
    f1_mac = f1_score(true_labels, preds, average="macro", zero_division=0)
    f1_wt = f1_score(true_labels, preds, average="weighted", zero_division=0)

    per_class_f1 = {}
    for label_name in label_names:
        if label_name in report_dict:
            per_class_f1[label_name] = round(report_dict[label_name]["f1-score"], 4)

    results = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "model_path": MODELS[model_name],
        "dataset_name": dataset_name,
        "num_labels": num_labels,
        "label_names": label_names,
        "train_size": len(train_labels),
        "val_size": len(val_labels),
        "test_size": len(test_labels),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "train_time_seconds": round(train_time, 1),
        "model_params_M": round(model_params, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved results: {results_json}")

    # Cleanup
    del model, trainer, tokenizer, train_dataset, val_dataset, test_dataset
    del train_enc, val_enc, test_enc
    clear_gpu_memory()

    elapsed = time.time() - start_time
    print(f"\n  DONE — {experiment_id} in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Accuracy: {acc:.4f} | F1-macro: {f1_mac:.4f} | F1-weighted: {f1_wt:.4f}")

    return results


# ============================================================================
#  SUMMARY TABLE
# ============================================================================

def compile_summary():
    """Read all results.json files and print comparison table."""
    print(f"\n{'=' * 100}")
    print(f"  SUMMARY — All Experiments")
    print(f"{'=' * 100}")

    all_results = []
    for model_name in MODELS:
        for dataset_name in DATASETS:
            exp_id = f"{model_name}_{dataset_name}"
            results_json = os.path.join(RESULTS_DIR, exp_id, "results.json")
            if os.path.exists(results_json):
                with open(results_json, "r", encoding="utf-8") as f:
                    all_results.append(json.load(f))

    if not all_results:
        print("  No results found. Run training first.")
        return

    # Print table
    # Header
    header = (
        f"{'Model':<16s} | {'Dataset':<8s} | {'Acc':>6s} | {'F1-Mac':>7s} | "
        f"{'F1-Wt':>7s} | {'Time':>6s} |"
    )
    # Add per-class columns for 3class
    header_3c = " HATE-F1 | OFF-F1  | NEU-F1  |"

    separator = "-" * len(header + header_3c)

    print(f"\n{separator}")
    print(f"{header}{header_3c}")
    print(f"{separator}")

    # Sort by dataset then by F1-macro descending
    all_results.sort(key=lambda r: (r["dataset_name"], -r["f1_macro"]))

    best_f1 = {"binary": 0, "3class": 0}
    best_model = {"binary": "", "3class": ""}

    for r in all_results:
        ds = r["dataset_name"]
        time_str = f"{r['train_time_seconds'] / 60:.0f}m"

        row = (
            f"{r['model_name']:<16s} | {ds:<8s} | "
            f"{r['accuracy']:>5.1%} | {r['f1_macro']:>6.1%} | "
            f"{r['f1_weighted']:>6.1%} | {time_str:>6s} |"
        )

        pcf = r.get("per_class_f1", {})
        hate_f1 = f"{pcf['HATE']:.1%}" if "HATE" in pcf else "  -  "
        off_f1 = f"{pcf['OFFENSIVE']:.1%}" if "OFFENSIVE" in pcf else "  -  "
        neu_f1_key = "NEUTRAL" if "NEUTRAL" in pcf else None
        harm_f1_key = "HARMFUL" if "HARMFUL" in pcf else None

        if neu_f1_key:
            neu_f1 = f"{pcf[neu_f1_key]:.1%}"
        elif harm_f1_key:
            neu_f1 = f"{pcf[harm_f1_key]:.1%}"
        else:
            neu_f1 = "  -  "

        row_3c = f" {hate_f1:>7s} | {off_f1:>7s} | {neu_f1:>7s} |"
        print(f"{row}{row_3c}")

        if r["f1_macro"] > best_f1[ds]:
            best_f1[ds] = r["f1_macro"]
            best_model[ds] = r["model_name"]

    print(f"{separator}")

    for ds in ["binary", "3class"]:
        if best_model[ds]:
            print(f"\n  BEST on {ds}: {best_model[ds]} (F1-macro={best_f1[ds]:.4f})")

    # Save summary JSON
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {summary_path}")

    # Save summary as text
    summary_txt = os.path.join(RESULTS_DIR, "summary_table.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Hate Speech Detection — Model Comparison\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"{separator}\n")
        f.write(f"{header}{header_3c}\n")
        f.write(f"{separator}\n")
        for r in all_results:
            ds = r["dataset_name"]
            time_str = f"{r['train_time_seconds'] / 60:.0f}m"
            row = (
                f"{r['model_name']:<16s} | {ds:<8s} | "
                f"{r['accuracy']:>5.1%} | {r['f1_macro']:>6.1%} | "
                f"{r['f1_weighted']:>6.1%} | {time_str:>6s} |"
            )
            pcf = r.get("per_class_f1", {})
            hate_f1 = f"{pcf.get('HATE', 0):.1%}" if "HATE" in pcf else "  -  "
            off_f1 = f"{pcf.get('OFFENSIVE', 0):.1%}" if "OFFENSIVE" in pcf else "  -  "
            neu_f1 = f"{pcf.get('NEUTRAL', 0):.1%}" if "NEUTRAL" in pcf else (
                f"{pcf.get('HARMFUL', 0):.1%}" if "HARMFUL" in pcf else "  -  "
            )
            row_3c = f" {hate_f1:>7s} | {off_f1:>7s} | {neu_f1:>7s} |"
            f.write(f"{row}{row_3c}\n")
        f.write(f"{separator}\n")
        for ds in ["binary", "3class"]:
            if best_model[ds]:
                f.write(f"\nBEST on {ds}: {best_model[ds]} (F1-macro={best_f1[ds]:.4f})\n")
    print(f"  Saved: {summary_txt}")

    return all_results


# ============================================================================
#  EXPLAINABILITY (LIME + SHAP)
# ============================================================================

def run_explainability():
    """
    Run LIME and SHAP on the best model for each dataset.
    Requires: pip3 install lime shap --user
    """
    print(f"\n{'=' * 70}")
    print(f"  EXPLAINABILITY — LIME + SHAP")
    print(f"{'=' * 70}")

    # Find best model for each dataset
    all_results = []
    for model_name in MODELS:
        for dataset_name in DATASETS:
            rp = os.path.join(RESULTS_DIR, f"{model_name}_{dataset_name}", "results.json")
            if os.path.exists(rp):
                with open(rp, "r", encoding="utf-8") as f:
                    all_results.append(json.load(f))

    if not all_results:
        print("  No trained models found. Run training first.")
        return

    for dataset_name in DATASETS:
        ds_results = [r for r in all_results if r["dataset_name"] == dataset_name]
        if not ds_results:
            continue

        best = max(ds_results, key=lambda r: r["f1_macro"])
        exp_id = best["experiment_id"]
        model_dir = os.path.join(RESULTS_DIR, exp_id, "best_model")
        label_names = best["label_names"]

        print(f"\n  Best model for {dataset_name}: {best['model_name']} "
              f"(F1-macro={best['f1_macro']:.4f})")
        print(f"  Loading from: {model_dir}")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Prediction function for LIME
        def predict_proba(texts_list):
            """Return probability array for LIME/SHAP."""
            if isinstance(texts_list, np.ndarray):
                texts_list = [str(t) for t in texts_list.flatten()]
            elif isinstance(texts_list, str):
                texts_list = [texts_list]
            else:
                texts_list = [str(t) for t in texts_list]
            all_probs = []
            # Process in batches to avoid OOM
            batch_sz = 16
            for i in range(0, len(texts_list), batch_sz):
                batch = texts_list[i:i + batch_sz]
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

        # Load test data for examples
        texts, labels, _, label2id = load_dataset(dataset_name)
        _, _, test_texts, _, _, test_labels = split_data(texts, labels)

        explain_dir = os.path.join(RESULTS_DIR, "explainability", dataset_name)
        os.makedirs(explain_dir, exist_ok=True)

        # ---- LIME ----
        print("\n  Running LIME...")
        try:
            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(
                class_names=label_names,
                split_expression=r"\s+",
            )

            # Select diverse examples (one per class + some misclassified)
            # Get predictions for test set
            test_preds = predict_proba(test_texts[:200])
            test_pred_labels = np.argmax(test_preds, axis=-1)

            examples_to_explain = []
            # One correct example per class
            for class_idx, class_name in enumerate(label_names):
                for i in range(len(test_texts[:200])):
                    if test_labels[i] == class_idx and test_pred_labels[i] == class_idx:
                        examples_to_explain.append({
                            "idx": i,
                            "text": test_texts[i],
                            "true": class_name,
                            "pred": class_name,
                            "type": f"correct_{class_name}",
                        })
                        break

            # Some misclassified examples
            for i in range(len(test_texts[:200])):
                if test_labels[i] != test_pred_labels[i]:
                    true_name = label_names[test_labels[i]]
                    pred_name = label_names[test_pred_labels[i]]
                    examples_to_explain.append({
                        "idx": i,
                        "text": test_texts[i],
                        "true": true_name,
                        "pred": pred_name,
                        "type": f"misclassified_{true_name}_as_{pred_name}",
                    })
                    if len(examples_to_explain) >= len(label_names) + 5:
                        break

            lime_results = []
            for ex in examples_to_explain:
                print(f"    LIME explaining: [{ex['type']}] {ex['text'][:60]}...")
                try:
                    explanation = explainer.explain_instance(
                        ex["text"],
                        predict_proba,
                        num_features=15,
                        num_samples=500,
                    )

                    # Save HTML
                    html_path = os.path.join(
                        explain_dir, f"lime_{ex['type']}.html"
                    )
                    explanation.save_to_file(html_path)

                    # Extract top features
                    top_features = explanation.as_list()
                    lime_results.append({
                        "text": ex["text"],
                        "true_label": ex["true"],
                        "predicted_label": ex["pred"],
                        "type": ex["type"],
                        "top_features": [(w, round(s, 4)) for w, s in top_features],
                    })
                    print(f"      Top 5 words: {top_features[:5]}")

                except Exception as e:
                    print(f"      LIME error: {e}")

            # Save LIME results JSON
            lime_json_path = os.path.join(explain_dir, "lime_results.json")
            with open(lime_json_path, "w", encoding="utf-8") as f:
                json.dump(lime_results, f, indent=2, ensure_ascii=False)
            print(f"  Saved LIME results: {lime_json_path}")

        except ImportError:
            print("  LIME not installed. Run: pip3 install lime --user")

        # ---- SHAP ----
        print("\n  Running SHAP...")
        try:
            import shap

            # Use SHAP's partition explainer (works well with text)
            shap_tokenizer = lambda s: s.split()  # noqa: E731

            # Take a small background dataset for SHAP
            background_texts = test_texts[:50]

            masker = shap.maskers.Text(tokenizer=r"\s+")
            shap_explainer = shap.Explainer(
                predict_proba,
                masker=masker,
                output_names=label_names,
            )

            # Explain a few examples
            sample_texts = [ex["text"] for ex in examples_to_explain[:5]]
            print(f"    Computing SHAP values for {len(sample_texts)} examples...")

            shap_values = shap_explainer(sample_texts)

            # Save SHAP bar plot
            shap_bar_path = os.path.join(explain_dir, "shap_bar.png")
            fig = plt.figure(figsize=(12, 8))
            shap.plots.bar(shap_values[:, :, 0], show=False)
            plt.title(f"SHAP Feature Importance — {dataset_name}")
            plt.tight_layout()
            plt.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved SHAP bar plot: {shap_bar_path}")

            # Save SHAP text plot as HTML
            shap_text_path = os.path.join(explain_dir, "shap_text.html")
            html = shap.plots.text(shap_values[:3], display=False)
            if html:
                with open(shap_text_path, "w", encoding="utf-8") as f:
                    f.write(str(html))
                print(f"  Saved SHAP text plot: {shap_text_path}")

            # Save SHAP values
            shap_json_path = os.path.join(explain_dir, "shap_summary.json")
            shap_summary = []
            for i, text in enumerate(sample_texts):
                words = text.split()
                values_per_class = {}
                for c_idx, c_name in enumerate(label_names):
                    try:
                        vals = shap_values[i, :, c_idx].values
                        word_importance = list(zip(words[:len(vals)],
                                                   [round(float(v), 4) for v in vals]))
                        values_per_class[c_name] = word_importance
                    except (IndexError, AttributeError):
                        pass
                shap_summary.append({
                    "text": text,
                    "shap_values": values_per_class,
                })
            with open(shap_json_path, "w", encoding="utf-8") as f:
                json.dump(shap_summary, f, indent=2, ensure_ascii=False)
            print(f"  Saved SHAP summary: {shap_json_path}")

        except ImportError:
            print("  SHAP not installed. Run: pip3 install shap --user")
        except Exception as e:
            print(f"  SHAP error: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
        del model, tokenizer
        clear_gpu_memory()

    print(f"\n  Explainability complete.")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hate Speech Multi-Model Trainer")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODELS.keys()),
                        help="Train specific model only")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help="Train on specific dataset only")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed experiments")
    parser.add_argument("--summary-only", action="store_true",
                        help="Just compile summary from existing results")
    parser.add_argument("--explain", action="store_true",
                        help="Run LIME/SHAP on best model")
    args = parser.parse_args()

    # Print header
    print(f"\n{'=' * 70}")
    print(f"  Hate Speech Detection — Multi-Model Training Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print(f"  WARNING: No GPU detected — training will be very slow")
    print(f"{'=' * 70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.summary_only:
        compile_summary()
        return

    if args.explain:
        run_explainability()
        return

    # Determine which experiments to run
    model_list = [args.model] if args.model else list(MODELS.keys())
    dataset_list = [args.dataset] if args.dataset else list(DATASETS.keys())

    total_experiments = len(model_list) * len(dataset_list)
    print(f"\n  Planned experiments: {total_experiments}")
    for m in model_list:
        for d in dataset_list:
            print(f"    - {m} × {d}")

    # Run experiments
    completed = 0
    failed = 0
    all_start = time.time()

    for model_name in model_list:
        for dataset_name in dataset_list:
            try:
                result = run_experiment(model_name, dataset_name, resume=args.resume)
                if result is not None:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n  EXPERIMENT FAILED: {model_name}_{dataset_name}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                clear_gpu_memory()

    total_time = time.time() - all_start

    print(f"\n{'=' * 70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Completed: {completed}/{total_experiments}")
    if failed > 0:
        print(f"  Failed: {failed}/{total_experiments}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 3600:.1f} hours)")
    print(f"{'=' * 70}")

    # Compile summary
    compile_summary()


if __name__ == "__main__":
    main()
