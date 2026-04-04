#!/usr/bin/env python3
"""
Sentiment Evaluation Trainer — Sudanese Arabic Hate Speech Project
===================================================================
Fine-tunes 7 BERT models on 3 sentiment datasets to demonstrate
cross-task generalization.

Datasets (from data/sentiment_prepared/):
  1. Telecom  (Paper 1):     3-class (neg/obj/pos), 4276 train + 1069 test
  2. SudSenti2 (Mhamed et al.): 2-class (neg/pos),  3528/271/202 train/val/test
  3. SudSenti3 (Mhamed et al.): 3-class (neg/obj/pos), 6455/201/301 train/val/test

Models (same 7 as hate speech Phase 1):
  marbertv2, arabertv2, camelbert_da, qarib, sudabert_v2, xlm_roberta, mbert

Server: apl13, GPU 1 (RTX 8000, 49GB), Python 3.10.12,
        torch 2.10.0, transformers 5.2.0

Usage:
  # Run ALL 21 experiments (7 models × 3 datasets)
  CUDA_VISIBLE_DEVICES=1 python3 sentiment_trainer.py --run_all

  # Run single experiment
  CUDA_VISIBLE_DEVICES=1 python3 sentiment_trainer.py --model marbertv2 --dataset telecom

  # Resume after interruption (skip completed)
  CUDA_VISIBLE_DEVICES=1 python3 sentiment_trainer.py --run_all --skip_existing

  # Run ensemble for a dataset (after baselines complete)
  CUDA_VISIBLE_DEVICES=1 python3 sentiment_trainer.py --ensemble --dataset telecom

Output: results/sentiment_evaluation/{model}_{dataset}/
"""

import argparse
import json
import os
import sys
import time
import gc
import warnings
import re
import traceback
import numpy as np
import shutil
from collections import Counter, OrderedDict

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
#  CONFIGURATION — all verified from handoff reports
# ============================================================

BASE_DIR = os.path.expanduser("~/sudanese_dialect_project")
DATA_DIR = os.path.join(BASE_DIR, "data/sentiment_prepared")
RESULTS_DIR = os.path.join(BASE_DIR, "results/sentiment_evaluation")

# Hyperparameters (same as hate_speech_trainer.py)
SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1   # used to compute warmup_steps
PATIENCE = 2             # early stopping epochs

# 9 models — 7 from hate speech Phase 1 + MARBERT/ARBERT originals (Mhamed et al.)
# (no sudabert_v1: directory is empty)
MODELS = OrderedDict([
    ("marbertv2",    "UBC-NLP/MARBERTv2"),
    ("marbert",      "UBC-NLP/MARBERT"),
    ("arbert",       "UBC-NLP/ARBERT"),
    ("arabertv2",    "aubmindlab/bert-base-arabertv02"),
    ("camelbert_da", "CAMeL-Lab/bert-base-arabic-camelbert-da"),
    ("qarib",        "qarib/bert-base-qarib"),
    ("sudabert_v2",  os.path.join(BASE_DIR, "models/sudabert_v2/sudabert_v2/")),
    ("xlm_roberta",  "xlm-roberta-base"),
    ("mbert",        "bert-base-multilingual-cased"),
])

# Dataset file mappings (verified from prepare_sentiment_data.py output)
DATASET_CONFIG = {
    "telecom": {
        "train": "telecom_train.json",
        "val":   None,                   # will split 10% from train
        "test":  "telecom_test.json",
        "desc":  "Telecom Customer Reviews (Paper 1), 3-class",
    },
    "sudsenti2": {
        "train": "sudsenti2_train.json",
        "val":   "sudsenti2_val.json",
        "test":  "sudsenti2_test.json",
        "desc":  "SudSenti2 — Mhamed et al. (2-class: pos/neg)",
    },
    "sudsenti3": {
        "train": "sudsenti3_train.json",
        "val":   "sudsenti3_val.json",
        "test":  "sudsenti3_test.json",
        "desc":  "SudSenti3 — Mhamed et al. (3-class: pos/neg/obj)",
    },
}

# Published results for comparison (from handoff + Mhamed et al. paper)
PUBLISHED = {
    "telecom": [
        ("MARBERT (Paper1)",      75.68, 64.21),
        ("MARBERTv2 (Paper1)",    75.02, 63.98),
        ("SudaBERT-v2 (Paper1)",  74.74, 63.29),
        ("SudaBERT (Paper1)",     73.99, 63.16),
        ("CAMeLBERT-DA (Paper1)", 73.06, 62.76),
        ("AraBERT (Paper1)",      68.85, 60.10),
    ],
    "sudsenti2": [
        ("SCM+MMA (Mhamed)",     92.25, None),
        ("MARBERT+FT (Mhamed)",  92.14, None),
        ("MARBERT (Mhamed)",     91.11, None),
        ("ARBERT (Mhamed)",      90.12, None),
        ("CNN-LSTM (Mhamed)",    89.00, None),
        ("CNN (Mhamed)",         87.75, None),
    ],
    "sudsenti3": [
        ("MARBERT+FT (Mhamed)",  88.44, None),
        ("MARBERT (Mhamed)",     86.83, None),
        ("SCM+MMA (Mhamed)",     85.23, None),
        ("ARBERT (Mhamed)",      85.09, None),
        ("CNN (Mhamed)",         83.61, None),
        ("CNN-LSTM (Mhamed)",    81.01, None),
    ],
}


# ============================================================
#  TEXT PREPROCESSING
# ============================================================

# Arabic diacritics: fathah, dammah, kasrah, sukun, shadda, tanwin
_DIACRITICS_RE = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')

_EMOJI_RE = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)


def preprocess_text_minimal(text):
    """Minimal Arabic preprocessing — consistent with Paper 1 methodology."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # URLs
    text = re.sub(r'@\w+', '', text)                      # mentions
    text = text.replace('#', '')                           # hashtag symbol
    text = re.sub(r'[إأآا]', 'ا', text)                    # Alef normalization
    text = _EMOJI_RE.sub('', text)                         # emojis
    text = ' '.join(text.split())                          # whitespace
    return text.strip()


def preprocess_text_mhamed(text):
    """
    Mhamed et al. preprocessing (Paper Section 4.1).
    Steps verified from paper:
      1. Remove URLs
      2. Remove @mentions
      3. Remove hashtag symbol (keep text)
      4. Remove diacritics
      5. Strip elongation (repeated chars → max 2)
      6. Heh normalization: ة → ه
      7. Yeh normalization: ى → ي
      8. Hamza normalization: ئ, ؤ → ء
      9. Alef normalization: آ, أ, إ → ا
     10. Remove emojis
     11. Remove numbers
     12. Remove non-Arabic characters (keeps Arabic + spaces only)
     13. Normalize whitespace
    Note: Mhamed's 269-word stopword list is not publicly available.
          Their ablation (Table 15) shows it adds only 0.51-0.82%.
    """
    if not text or not isinstance(text, str):
        return ""
    # 1. URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 2. Mentions
    text = re.sub(r'@\w+', '', text)
    # 3. Hashtag symbol
    text = text.replace('#', '')
    # 4. Diacritics
    text = _DIACRITICS_RE.sub('', text)
    # 5. Elongation: any Arabic char repeated 3+ times → 2
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 6. Heh normalization: ة → ه
    text = text.replace('ة', 'ه')
    # 7. Yeh normalization: ى → ي
    text = text.replace('ى', 'ي')
    # 8. Hamza normalization
    text = text.replace('ئ', 'ء')
    text = text.replace('ؤ', 'ء')
    # 9. Alef normalization: آ أ إ → ا
    text = re.sub(r'[إأآا]', 'ا', text)
    # 10. Emojis
    text = _EMOJI_RE.sub('', text)
    # 11. Numbers
    text = re.sub(r'[0-9٠-٩]+', '', text)
    # 12. Keep only Arabic characters and spaces
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s]', '', text)
    # 13. Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


# Global reference — set by command-line argument
_preprocess_fn = preprocess_text_minimal


# ============================================================
#  FIX LAYERNORM NAMING (gamma/beta → weight/bias)
#  MARBERTv2 and some models use old TF-style names internally.
#  This must be done BEFORE training so checkpoints save correctly.
# ============================================================

def fix_layernorm_naming(model):
    """Rename gamma→weight, beta→bias in all modules for checkpoint compat."""
    fixed = 0
    for module in model.modules():
        params = dict(module._parameters)
        if 'gamma' in params:
            module._parameters['weight'] = module._parameters.pop('gamma')
            fixed += 1
        if 'beta' in params:
            module._parameters['bias'] = module._parameters.pop('beta')
            fixed += 1
    if fixed:
        print(f"  Fixed {fixed} LayerNorm params (gamma/beta → weight/bias)")
    return model


# ============================================================
#  PYTORCH DATASET
# ============================================================

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
#  DATA LOADING
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_splits(dataset_name):
    """
    Load train/val/test splits.  Returns:
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        label_names          (sorted list, e.g. ['neg','obj','pos'])
    """
    cfg = DATASET_CONFIG[dataset_name]

    train_raw = load_json(os.path.join(DATA_DIR, cfg["train"]))
    test_raw  = load_json(os.path.join(DATA_DIR, cfg["test"]))

    if cfg["val"] is not None:
        val_raw = load_json(os.path.join(DATA_DIR, cfg["val"]))
    else:
        # Telecom: carve 10% stratified validation from train
        texts_all  = [d["text"]  for d in train_raw]
        labels_all = [d["label"] for d in train_raw]
        t_texts, v_texts, t_labels, v_labels = train_test_split(
            texts_all, labels_all,
            test_size=0.1, random_state=SEED, stratify=labels_all,
        )
        train_raw = [{"text": t, "label": l} for t, l in zip(t_texts, t_labels)]
        val_raw   = [{"text": t, "label": l} for t, l in zip(v_texts, v_labels)]

    # Determine label set (sorted for deterministic id assignment)
    all_labels = set(d["label"] for d in train_raw + val_raw + test_raw)
    label_names = sorted(all_labels)
    label2id = {l: i for i, l in enumerate(label_names)}

    def extract(data):
        texts, labels = [], []
        for d in data:
            t = _preprocess_fn(d["text"])
            if t:  # skip entries that become empty after preprocessing
                texts.append(t)
                labels.append(label2id[d["label"]])
        return texts, labels

    tr_t, tr_l = extract(train_raw)
    va_t, va_l = extract(val_raw)
    te_t, te_l = extract(test_raw)

    return tr_t, tr_l, va_t, va_l, te_t, te_l, label_names


def load_all_data(dataset_name):
    """Load ALL data (combine train+val+test) for cross-validation."""
    cfg = DATASET_CONFIG[dataset_name]

    all_data = load_json(os.path.join(DATA_DIR, cfg["train"]))
    all_data += load_json(os.path.join(DATA_DIR, cfg["test"]))
    if cfg["val"] is not None:
        all_data += load_json(os.path.join(DATA_DIR, cfg["val"]))

    texts, labels = [], []
    removed = 0
    for d in all_data:
        t = _preprocess_fn(d["text"])
        if t:  # skip entries that become empty after preprocessing
            texts.append(t)
            labels.append(d["label"])
        else:
            removed += 1
    if removed:
        print(f"  Removed {removed} empty entries after preprocessing")
    return texts, labels


# ============================================================
#  METRICS
# ============================================================

def make_compute_metrics(label_names):
    """Factory: returns a compute_metrics fn with label_names in closure."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc    = accuracy_score(labels, preds)
        f1_mac = f1_score(labels, preds, average="macro", zero_division=0)
        f1_per = f1_score(labels, preds, average=None, zero_division=0)
        out = {"accuracy": acc, "f1_macro": f1_mac}
        for i, name in enumerate(label_names):
            out[f"f1_{name}"] = float(f1_per[i])
        return out
    return compute_metrics


# ============================================================
#  PLOTTING
# ============================================================

def plot_confusion_matrix(y_true, y_pred, label_names, save_dir,
                          model_name, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"{model_name} — {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_curves(log_history, save_dir, model_name, dataset_name):
    """Extract per-epoch metrics from Trainer log_history and plot."""
    train_loss, eval_loss, eval_acc, eval_f1 = [], [], [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
        if "eval_accuracy" in entry:
            eval_acc.append(entry["eval_accuracy"])
        if "eval_f1_macro" in entry:
            eval_f1.append(entry["eval_f1_macro"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    if eval_loss:
        epochs = range(1, len(eval_loss) + 1)
        axes[0].plot(epochs, eval_loss, "o-", label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Validation Loss")
        axes[0].legend()

    # Accuracy
    if eval_acc:
        epochs = range(1, len(eval_acc) + 1)
        axes[1].plot(epochs, [a * 100 for a in eval_acc], "o-", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Validation Accuracy")

    # F1 macro
    if eval_f1:
        epochs = range(1, len(eval_f1) + 1)
        axes[2].plot(epochs, [f * 100 for f in eval_f1], "o-", color="red")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("F1-macro (%)")
        axes[2].set_title("Validation F1-macro")

    fig.suptitle(f"{model_name} — {dataset_name}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
#  SINGLE EXPERIMENT
# ============================================================

def run_experiment(model_name, dataset_name, skip_existing=False):
    """Train one model on one dataset.  Returns results dict or None."""

    exp_dir = os.path.join(RESULTS_DIR, f"{model_name}_{dataset_name}")
    results_file = os.path.join(exp_dir, "results.json")

    if skip_existing and os.path.exists(results_file):
        print(f"\n  SKIP {model_name} × {dataset_name} (results exist)")
        with open(results_file) as f:
            return json.load(f)

    os.makedirs(exp_dir, exist_ok=True)
    model_path = MODELS[model_name]

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {model_name} × {dataset_name}")
    print(f"  Model path: {model_path}")
    print(f"  Output dir: {exp_dir}")
    print(f"{'='*60}")

    t0 = time.time()

    # ---- load data ----
    (train_texts, train_labels,
     val_texts,   val_labels,
     test_texts,  test_labels,
     label_names) = load_splits(dataset_name)

    label2id = {l: i for i, l in enumerate(label_names)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label_names)

    print(f"  Classes:  {num_labels}  {label_names}")
    print(f"  Train={len(train_texts)}  Val={len(val_texts)}  Test={len(test_texts)}")
    train_dist = Counter(train_labels)
    print(f"  Train distribution: { {id2label[k]: v for k, v in sorted(train_dist.items())} }")

    # ---- tokenizer ----
    print(f"  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(texts):
        return tokenizer(
            texts, max_length=MAX_LEN,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )

    train_enc = tokenize(train_texts)
    val_enc   = tokenize(val_texts)
    test_enc  = tokenize(test_texts)

    train_ds = SentimentDataset(train_enc, train_labels)
    val_ds   = SentimentDataset(val_enc,   val_labels)
    test_ds  = SentimentDataset(test_enc,  test_labels)

    # ---- model ----
    print(f"  Loading model …")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Fix LayerNorm parameter names so checkpoints save/reload correctly
    model = fix_layernorm_naming(model)

    # ---- training args ----
    steps_per_epoch = (len(train_ds) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = int(WARMUP_FRACTION * total_steps)

    print(f"  Steps/epoch={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}")

    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",           # transformers 5.2.0 name
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=True,
        logging_dir=os.path.join(exp_dir, "logs"),
        logging_steps=50,
        seed=SEED,
        save_total_limit=NUM_EPOCHS,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(label_names),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    # ---- train ----
    print(f"  Training …")
    trainer.train()
    train_time = time.time() - t0

    # ---- reload best model robustly ----
    # With LayerNorm naming fixed above, the Trainer's own reload should work.
    # This manual reload is a safety net.
    best_ckpt = getattr(trainer.state, 'best_model_checkpoint', None)
    if best_ckpt and os.path.exists(best_ckpt):
        try:
            print(f"  Reloading best checkpoint: {best_ckpt}")
            reloaded = AutoModelForSequenceClassification.from_pretrained(
                best_ckpt,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            reloaded = fix_layernorm_naming(reloaded)
            reloaded.to(trainer.args.device)
            trainer.model = reloaded
            print(f"  Best model reloaded successfully")
        except Exception as e:
            print(f"  WARNING: Could not reload best checkpoint: {e}")
            print(f"  Using model from end of training")

    # ---- evaluate on TEST ----
    print(f"  Evaluating on test set …")
    pred_out   = trainer.predict(test_ds)
    test_preds = np.argmax(pred_out.predictions, axis=-1)
    test_true  = np.array(test_labels)

    acc       = accuracy_score(test_true, test_preds) * 100
    f1_mac    = f1_score(test_true, test_preds, average="macro",    zero_division=0) * 100
    f1_wt     = f1_score(test_true, test_preds, average="weighted", zero_division=0) * 100
    f1_per    = f1_score(test_true, test_preds, average=None,       zero_division=0) * 100

    print(f"\n  ┌─── TEST RESULTS ───────────────────┐")
    print(f"  │  Accuracy : {acc:6.2f}%               │")
    print(f"  │  F1-macro : {f1_mac:6.2f}%               │")
    for i, name in enumerate(label_names):
        print(f"  │  F1-{name:5s}: {f1_per[i]:6.2f}%               │")
    print(f"  │  Time     : {train_time:6.1f}s               │")
    print(f"  └─────────────────────────────────────┘")

    # ---- classification report ----
    report_str = classification_report(
        test_true, test_preds,
        target_names=label_names, digits=4, zero_division=0,
    )
    with open(os.path.join(exp_dir, "classification_report.txt"), "w") as f:
        f.write(f"Model:   {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Time:    {train_time:.1f}s\n\n")
        f.write(report_str)

    # ---- confusion matrix ----
    plot_confusion_matrix(test_true, test_preds, label_names,
                          exp_dir, model_name, dataset_name)

    # ---- training curves ----
    plot_training_curves(trainer.state.log_history, exp_dir,
                         model_name, dataset_name)

    # ---- find best epoch from history ----
    best_f1, best_epoch = 0.0, 1
    for entry in trainer.state.log_history:
        if "eval_f1_macro" in entry:
            if entry["eval_f1_macro"] > best_f1:
                best_f1   = entry["eval_f1_macro"]
                best_epoch = int(entry.get("epoch", best_epoch))

    # ---- save results JSON ----
    results = {
        "model":          model_name,
        "model_path":     model_path,
        "dataset":        dataset_name,
        "num_classes":    num_labels,
        "labels":         label_names,
        "train_size":     len(train_texts),
        "val_size":       len(val_texts),
        "test_size":      len(test_texts),
        "accuracy":       round(acc, 2),
        "f1_macro":       round(f1_mac, 2),
        "f1_weighted":    round(f1_wt, 2),
        "f1_per_class":   {name: round(float(f1_per[i]), 2) for i, name in enumerate(label_names)},
        "epochs_trained": int(trainer.state.epoch),
        "best_epoch":     best_epoch,
        "training_time":  round(train_time, 1),
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- save best model ----
    best_dir = os.path.join(exp_dir, "best_model")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # ---- cleanup checkpoints to save disk ----
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    # ---- free GPU ----
    del model, trainer, train_ds, val_ds, test_ds
    del train_enc, val_enc, test_enc
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================
#  10-FOLD CROSS-VALIDATION (for SudSenti — matches Mhamed et al.)
# ============================================================

def run_cv_experiment(model_name, dataset_name, n_folds=10, skip_existing=False):
    """Run n-fold stratified CV. Matches Mhamed et al. methodology."""

    exp_dir = os.path.join(RESULTS_DIR, f"{model_name}_{dataset_name}")
    results_file = os.path.join(exp_dir, "results.json")

    if skip_existing and os.path.exists(results_file):
        print(f"\n  SKIP {model_name} × {dataset_name} (results exist)")
        with open(results_file) as f:
            return json.load(f)

    os.makedirs(exp_dir, exist_ok=True)
    model_path = MODELS[model_name]

    # Load ALL data (combine train+val+test splits)
    all_texts, all_labels_str = load_all_data(dataset_name)
    label_names = sorted(set(all_labels_str))
    label2id = {l: i for i, l in enumerate(label_names)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(label_names)
    all_labels = [label2id[l] for l in all_labels_str]

    print(f"\n{'='*60}")
    print(f"  {model_name} × {dataset_name}  ({n_folds}-fold CV)")
    print(f"  Model: {model_path}")
    print(f"  Total: {len(all_texts)},  Classes: {num_labels} {label_names}")
    print(f"  Distribution: {dict(Counter(all_labels_str))}")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []
    t0 = time.time()

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(all_texts, all_labels)):
        fold_t0 = time.time()

        # Split data
        tv_texts  = [all_texts[i]  for i in train_val_idx]
        tv_labels = [all_labels[i] for i in train_val_idx]
        te_texts  = [all_texts[i]  for i in test_idx]
        te_labels = [all_labels[i] for i in test_idx]

        # Further split train_val → train (80% total) + val (10% total)
        tr_texts, va_texts, tr_labels, va_labels = train_test_split(
            tv_texts, tv_labels,
            test_size=1.0/9.0,   # 1/9 of 90% ≈ 10% of total
            random_state=SEED,
            stratify=tv_labels,
        )

        print(f"\n  Fold {fold+1}/{n_folds}:  "
              f"train={len(tr_texts)} val={len(va_texts)} test={len(te_texts)}")

        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        train_enc = tokenizer(tr_texts, max_length=MAX_LEN, padding="max_length",
                              truncation=True, return_tensors="pt")
        val_enc   = tokenizer(va_texts, max_length=MAX_LEN, padding="max_length",
                              truncation=True, return_tensors="pt")
        test_enc  = tokenizer(te_texts, max_length=MAX_LEN, padding="max_length",
                              truncation=True, return_tensors="pt")

        train_ds = SentimentDataset(train_enc, tr_labels)
        val_ds   = SentimentDataset(val_enc,   va_labels)
        test_ds  = SentimentDataset(test_enc,  te_labels)

        # Fresh model each fold
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels,
            id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        # Training — NO checkpointing, NO load_best_model to avoid reload issues
        fold_dir = os.path.join(exp_dir, f"fold_{fold}")
        steps_per_epoch = (len(train_ds) + BATCH_SIZE - 1) // BATCH_SIZE
        warmup_steps = int(WARMUP_FRACTION * steps_per_epoch * NUM_EPOCHS)

        training_args = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            fp16=True,
            seed=SEED,
            report_to="none",
            dataloader_num_workers=0,
            disable_tqdm=True,
            logging_steps=9999,       # suppress per-step logs during CV
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=make_compute_metrics(label_names),
        )

        trainer.train()

        # Evaluate on test fold using current model (last epoch)
        pred_out  = trainer.predict(test_ds)
        preds     = np.argmax(pred_out.predictions, axis=-1)
        true_arr  = np.array(te_labels)

        acc    = accuracy_score(true_arr, preds) * 100
        f1_mac = f1_score(true_arr, preds, average="macro", zero_division=0) * 100
        f1_per = f1_score(true_arr, preds, average=None,    zero_division=0) * 100

        fold_time = time.time() - fold_t0
        print(f"    → Acc={acc:.2f}%  F1={f1_mac:.2f}%  ({fold_time:.0f}s)")

        fold_metrics.append({
            "fold": fold + 1,
            "accuracy": round(acc, 2),
            "f1_macro": round(f1_mac, 2),
            "f1_per_class": {name: round(float(f1_per[i]), 2)
                             for i, name in enumerate(label_names)},
        })

        # Cleanup fold
        del model, trainer, train_ds, val_ds, test_ds
        del train_enc, val_enc, test_enc
        gc.collect()
        torch.cuda.empty_cache()
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir, ignore_errors=True)

    total_time = time.time() - t0

    # Compute averages
    accs = [f["accuracy"] for f in fold_metrics]
    f1s  = [f["f1_macro"] for f in fold_metrics]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_f1,  std_f1  = np.mean(f1s),  np.std(f1s)

    f1_per_avg = {}
    for name in label_names:
        vals = [f["f1_per_class"][name] for f in fold_metrics]
        f1_per_avg[name] = round(float(np.mean(vals)), 2)

    print(f"\n  ┌─── {n_folds}-FOLD CV RESULTS ──────────────────┐")
    print(f"  │  Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}%     │")
    print(f"  │  F1-macro : {mean_f1:.2f}% ± {std_f1:.2f}%     │")
    for name in label_names:
        print(f"  │  F1-{name:5s}: {f1_per_avg[name]:.2f}%               │")
    print(f"  │  Time     : {total_time:.0f}s                    │")
    print(f"  └─────────────────────────────────────────┘")

    results = {
        "model":         model_name,
        "model_path":    model_path,
        "dataset":       dataset_name,
        "method":        f"{n_folds}-fold CV",
        "num_classes":   num_labels,
        "labels":        label_names,
        "total_samples": len(all_texts),
        "accuracy":      round(mean_acc, 2),
        "accuracy_std":  round(std_acc, 2),
        "f1_macro":      round(mean_f1, 2),
        "f1_macro_std":  round(std_f1, 2),
        "f1_per_class":  f1_per_avg,
        "fold_results":  fold_metrics,
        "training_time": round(total_time, 1),
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save per-fold detail
    with open(os.path.join(exp_dir, "fold_details.json"), "w") as f:
        json.dump(fold_metrics, f, indent=2)

    return results


# ============================================================
#  ENSEMBLE (soft-vote over top-3 models)
# ============================================================

def run_ensemble(dataset_name):
    """Load top-3 trained models, average logits on test set, report."""

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE — {dataset_name}")
    print(f"{'='*60}")

    # Collect all baseline results for this dataset
    baseline_results = []
    for mname in MODELS:
        rpath = os.path.join(RESULTS_DIR, f"{mname}_{dataset_name}", "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                baseline_results.append(json.load(f))

    if len(baseline_results) < 3:
        print(f"  ERROR: Need at least 3 trained baselines, found {len(baseline_results)}")
        return None

    # Sort by f1_macro descending, pick top 3
    baseline_results.sort(key=lambda x: x["f1_macro"], reverse=True)
    top3 = baseline_results[:3]
    top3_names = [r["model"] for r in top3]
    print(f"  Top-3 models: {top3_names}")
    print(f"  Their F1-macro: {[r['f1_macro'] for r in top3]}")

    # Load test data
    (_, _, _, _, test_texts, test_labels, label_names) = load_splits(dataset_name)
    num_labels = len(label_names)

    # Collect logits from each model
    all_logits = []
    for r in top3:
        mname = r["model"]
        model_dir = os.path.join(RESULTS_DIR, f"{mname}_{dataset_name}", "best_model")
        print(f"  Loading {mname} from {model_dir} …")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        enc = tokenizer(
            test_texts, max_length=MAX_LEN,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            # Process in batches to avoid OOM
            logits_list = []
            bs = EVAL_BATCH_SIZE
            for start in range(0, len(test_texts), bs):
                batch = {k: v[start:start+bs] for k, v in enc.items()}
                out = model(**batch)
                logits_list.append(out.logits.cpu().numpy())
            logits = np.concatenate(logits_list, axis=0)

        all_logits.append(logits)
        del model, enc
        gc.collect()
        torch.cuda.empty_cache()

    # Average logits (soft voting)
    avg_logits = np.mean(all_logits, axis=0)
    preds = np.argmax(avg_logits, axis=-1)
    test_true = np.array(test_labels)

    acc    = accuracy_score(test_true, preds) * 100
    f1_mac = f1_score(test_true, preds, average="macro", zero_division=0) * 100
    f1_per = f1_score(test_true, preds, average=None,    zero_division=0) * 100

    print(f"\n  ┌─── ENSEMBLE RESULTS ──────────────────┐")
    print(f"  │  Models  : {top3_names}")
    print(f"  │  Accuracy: {acc:6.2f}%                    │")
    print(f"  │  F1-macro: {f1_mac:6.2f}%                    │")
    for i, name in enumerate(label_names):
        print(f"  │  F1-{name:5s}: {f1_per[i]:6.2f}%                    │")
    print(f"  └────────────────────────────────────────┘")

    # Save
    ens_dir = os.path.join(RESULTS_DIR, f"ensemble_{dataset_name}")
    os.makedirs(ens_dir, exist_ok=True)

    results = {
        "model":        "ensemble_top3",
        "top3_models":  top3_names,
        "dataset":      dataset_name,
        "num_classes":  num_labels,
        "labels":       label_names,
        "test_size":    len(test_texts),
        "accuracy":     round(acc, 2),
        "f1_macro":     round(f1_mac, 2),
        "f1_per_class": {name: round(float(f1_per[i]), 2) for i, name in enumerate(label_names)},
    }
    with open(os.path.join(ens_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    report_str = classification_report(
        test_true, preds,
        target_names=label_names, digits=4, zero_division=0,
    )
    with open(os.path.join(ens_dir, "classification_report.txt"), "w") as f:
        f.write(f"Ensemble: {top3_names}\nDataset: {dataset_name}\n\n")
        f.write(report_str)

    plot_confusion_matrix(test_true, preds, label_names,
                          ens_dir, "Ensemble(Top3)", dataset_name)

    return results


# ============================================================
#  COMPARISON TABLE
# ============================================================

def print_comparison(dataset_name, our_results):
    """Print side-by-side comparison with published results."""
    pub = PUBLISHED.get(dataset_name, [])
    if not pub:
        return

    print(f"\n{'='*60}")
    print(f"  COMPARISON TABLE — {dataset_name}")
    print(f"{'='*60}")

    # Header
    has_f1 = any(row[2] is not None for row in pub)
    if has_f1:
        print(f"  {'Model':<30s} {'Acc%':>7s} {'F1-mac%':>8s}")
        print(f"  {'-'*30} {'-'*7} {'-'*8}")
    else:
        print(f"  {'Model':<30s} {'Acc%':>7s}")
        print(f"  {'-'*30} {'-'*7}")

    # Published
    for name, acc, f1 in pub:
        if has_f1 and f1 is not None:
            print(f"  {name:<30s} {acc:7.2f} {f1:8.2f}")
        else:
            print(f"  {name:<30s} {acc:7.2f}")

    print(f"  {'— Our Results —':<30s} {'':>7s}")

    # Ours
    for r in sorted(our_results, key=lambda x: x["accuracy"], reverse=True):
        name = r["model"]
        if name == "ensemble_top3":
            name = "Ensemble(Top3) ★"
        if has_f1:
            print(f"  {name:<30s} {r['accuracy']:7.2f} {r['f1_macro']:8.2f}")
        else:
            print(f"  {name:<30s} {r['accuracy']:7.2f}")


# ============================================================
#  SUMMARY TABLE
# ============================================================

def print_summary(all_results):
    """Print a full summary of all experiments."""
    print(f"\n\n{'='*70}")
    print(f"  FULL SUMMARY — ALL EXPERIMENTS")
    print(f"{'='*70}")

    for ds in DATASET_CONFIG:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        if not ds_results:
            continue

        print(f"\n  Dataset: {DATASET_CONFIG[ds]['desc']}")
        print(f"  {'Model':<18s} {'Acc%':>7s} {'F1-mac%':>8s}", end="")

        # Get label names from first result
        label_names = ds_results[0].get("labels", [])
        for ln in label_names:
            print(f" {'F1-'+ln:>8s}", end="")
        print(f" {'Time':>7s}")

        print(f"  {'-'*18} {'-'*7} {'-'*8}", end="")
        for _ in label_names:
            print(f" {'-'*8}", end="")
        print(f" {'-'*7}")

        for r in sorted(ds_results, key=lambda x: x.get("f1_macro", 0), reverse=True):
            name = r["model"]
            if name == "ensemble_top3":
                name = "Ensemble ★"
            print(f"  {name:<18s} {r['accuracy']:7.2f} {r['f1_macro']:8.2f}", end="")
            f1pc = r.get("f1_per_class", {})
            for ln in label_names:
                val = f1pc.get(ln, 0.0)
                print(f" {val:8.2f}", end="")
            t = r.get("training_time", 0)
            print(f" {t:6.1f}s" if t else "     N/A")

    # Save summary JSON
    summary_path = os.path.join(RESULTS_DIR, "all_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {summary_path}")


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Evaluation Trainer for Sudanese Arabic")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()),
                        help="Single model to train")
    parser.add_argument("--dataset", type=str,
                        choices=list(DATASET_CONFIG.keys()),
                        help="Single dataset to use")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all 7 models × 3 datasets = 21 experiments")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments that already have results")
    parser.add_argument("--ensemble", action="store_true",
                        help="Run ensemble for --dataset (baselines must exist)")
    parser.add_argument("--preprocess", type=str, default="minimal",
                        choices=["minimal", "mhamed"],
                        help="Preprocessing: minimal (Paper 1) or mhamed (Mhamed et al.)")
    args = parser.parse_args()

    # Validate arguments
    if not args.run_all and not args.ensemble and not (args.model and args.dataset):
        parser.error("Provide --run_all, --ensemble --dataset, or --model + --dataset")

    if args.ensemble and not args.dataset:
        parser.error("--ensemble requires --dataset")

    # Setup
    set_seed(SEED)

    # Set preprocessing mode
    global _preprocess_fn, RESULTS_DIR
    if args.preprocess == "mhamed":
        _preprocess_fn = preprocess_text_mhamed
        RESULTS_DIR = os.path.join(BASE_DIR, "results/sentiment_evaluation_mhamed")
        print(f"\n  Preprocessing: MHAMED (full Arabic normalization)")
    else:
        _preprocess_fn = preprocess_text_minimal
        print(f"\n  Preprocessing: MINIMAL (Paper 1 style)")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        vram = gpu.total_memory / (1024**3)
        print(f"\n  GPU: {gpu.name}  VRAM: {vram:.1f} GB")
    else:
        print("\n  WARNING: No GPU detected. Training will be very slow.")

    # Check data exists
    for ds_name, cfg in DATASET_CONFIG.items():
        train_path = os.path.join(DATA_DIR, cfg["train"])
        if not os.path.exists(train_path):
            print(f"  ERROR: Missing {train_path}")
            print(f"  Run prepare_sentiment_data.py first from project root:")
            print(f"    cd ~/sudanese_dialect_project && python3 data/SudSenti/prepare_sentiment_data.py")
            sys.exit(1)

    # ---- Ensemble mode ----
    if args.ensemble:
        result = run_ensemble(args.dataset)
        if result:
            print_comparison(args.dataset, [result])
        return

    # ---- Build experiment list ----
    if args.run_all:
        experiments = [(m, d) for d in DATASET_CONFIG for m in MODELS]
    else:
        experiments = [(args.model, args.dataset)]

    print(f"\n  Experiments to run: {len(experiments)}")
    for m, d in experiments:
        print(f"    {m} × {d}")

    # ---- Run experiments ----
    all_results = []
    failed = []

    for i, (model_name, dataset_name) in enumerate(experiments, 1):
        print(f"\n\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  Experiment {i}/{len(experiments)}: {model_name} × {dataset_name}")
        print(f"  ╚══════════════════════════════════════════╝")

        try:
            # SudSenti datasets: 10-fold CV (matches Mhamed et al. methodology)
            # Telecom: single train/test split (matches Paper 1)
            if dataset_name in ("sudsenti2", "sudsenti3"):
                result = run_cv_experiment(model_name, dataset_name,
                                           n_folds=10,
                                           skip_existing=args.skip_existing)
            else:
                result = run_experiment(model_name, dataset_name,
                                        skip_existing=args.skip_existing)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ✗ FAILED: {model_name} × {dataset_name}")
            print(f"    Error: {e}")
            traceback.print_exc()
            failed.append((model_name, dataset_name, str(e)))
            # Free GPU and continue
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Summary ----
    if all_results:
        print_summary(all_results)

        # Comparison tables per dataset
        for ds in DATASET_CONFIG:
            ds_results = [r for r in all_results if r["dataset"] == ds]
            if ds_results:
                print_comparison(ds, ds_results)

    # ---- Report failures ----
    if failed:
        print(f"\n  FAILED EXPERIMENTS ({len(failed)}):")
        for m, d, err in failed:
            print(f"    {m} × {d}: {err}")

    print(f"\n  DONE — {len(all_results)} succeeded, {len(failed)} failed")
    print(f"  Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
