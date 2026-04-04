#!/usr/bin/env python3
"""
Mhamed et al. CNN Reproduction — Baseline + SCM+MMA
=====================================================
Implements two CNN architectures from Mhamed et al.:

1. CNN-Baseline: Exact architecture from their GitHub notebook
   - 4 Conv1D layers [512, 256, 32, 32], padding='same'
   - GlobalMaxPooling1D
   - Source: NLP_Test.ipynb

2. SCM+MMA: Architecture from their paper (Section 4.4, Equation 4)
   - 4 Conv1D layers [512, 256, 128, 64], padding='valid'
   - Custom MMA pooling = (MaxPool + AvgPool) / 2
   - Source: Paper Section 4.4, Figure 9

Preprocessing: Exact Mhamed preprocessing from notebook (line 724-749):
   - Remove punctuations, diacritics, normalize Arabic chars
   - Remove NLTK Arabic + Sudanese custom stopwords

Datasets: SudSenti2, SudSenti3, Hate Speech (binary + 3-class)
Method: 10-fold CV for SudSenti, 80/10/10 split for hate speech

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 cnn_mhamed_comparison.py --run_all
  CUDA_VISIBLE_DEVICES=1 python3 cnn_mhamed_comparison.py --model cnn_baseline --dataset sudsenti2
  CUDA_VISIBLE_DEVICES=1 python3 cnn_mhamed_comparison.py --model scm_mma --dataset sudsenti3

Output: results/cnn_mhamed_comparison/
"""

import argparse
import json
import os
import re
import string
import sys
import time
import gc
import csv
import warnings
import numpy as np
from collections import Counter, OrderedDict

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Embedding, Conv1D, Flatten,
    GlobalMaxPooling1D, MaxPooling1D, AveragePooling1D,
    BatchNormalization, Input, Layer,
)
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
#  CONFIGURATION
# ============================================================

BASE_DIR = os.path.expanduser("~/sudanese_dialect_project")
DATA_DIR = os.path.join(BASE_DIR, "data/sentiment_prepared")
HS_BINARY = os.path.join(BASE_DIR, "data/labeling_corpus/dataset_binary.tsv")
HS_3CLASS = os.path.join(BASE_DIR, "data/labeling_corpus/dataset_3class.tsv")
RESULTS_DIR = os.path.join(BASE_DIR, "results/cnn_mhamed_comparison")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mhamed hyperparameters (from notebook + paper)
EMBEDDING_SIZE = 128
MAX_LEN = 80
BATCH_SIZE = 128
EPOCHS = 50          # paper says up to 50 epochs
LEARNING_RATE = 0.0001
PATIENCE = 5         # early stopping

# ============================================================
#  MHAMED PREPROCESSING (exact copy from their notebook)
# ============================================================

# Punctuations list — exact from notebook line 707
PUNCTUATIONS = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ''' + string.punctuation

# Arabic diacritics — exact from notebook lines 712-722
ARABIC_DIACRITICS = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

# Sudanese stopword list — exact from notebook lines 678-689
SUDANESE_STOPWORDS = [
    'هسه', 'هسي', 'اسي', 'حسع', 'حسة', 'هسع', 'إنحنا', 'كلو',
    'هندیلكن', 'هندیلكم', 'دییكه', 'داك', 'دة', 'دیلكم', 'دیك',
    'هنداك', 'وین', 'وينم', 'وينك', 'وينهم', 'اه', 'هوي', 'هوا',
    'يا', 'مش', 'ايوا', 'عليك', 'الله', 'بلاي', 'صح', 'سبوع',
    'فت', 'ابك', 'ابيت', 'احاا', 'اسع', 'البهم', 'شنو', 'احد',
    'تلاتة', 'خميسك', 'يك', 'اهأ', 'جاي', 'جوة', 'طل', 'براك',
    'برانا', 'براكم', 'براها', 'براهم', 'براهو', 'براي', 'بربر',
    'بس', 'تبروقة', 'تتري', 'تتسوا', 'تتي', 'تجربة', 'تريلا',
    'تشعبط', 'تق', 'تك', 'تكل', 'تلايط', 'تلب', 'تنتان', 'تنخ',
    'تي تي', 'تعاليهو', 'تعاليها', 'تعالو', 'تبيتو', 'تبيت',
    'تنط', 'دبل', 'جيت', 'جوه', 'جاين', 'جاي', 'جوكم', 'جينكم',
    'جيناكم', 'متجونا', 'تجو', 'حضروهم', 'حضرتم', 'حضرناهم',
    'خت', 'خاتي', 'خخخ', 'داربيهم', 'دارهم', 'دايرنهم', 'رايد',
    'ردهم', 'كيفكم', 'كيفنكم', 'كيف', 'بلاي', 'ذا', 'ذوا', 'ذوة',
    'ذواتي', 'ذيي', 'ذيين', 'ذينكم', 'هنا', 'هوي', 'هوا', 'هلما',
    'هلو', 'هاي', 'هيص', 'هندا', 'هووي', 'ويا', 'ويك', 'ولادا',
    'ولة', 'والين', 'والي', 'ود', 'ودكم', 'ودو', 'أنتن', 'إنما',
    'إنه', 'أنى', 'آه', 'آها', 'أو', 'أولاء', 'حاشا', 'حبذا',
    'حتى', 'حيث', 'حيثما', 'حين', 'خلا', 'دون', 'اللتان', 'اللتيا',
    'اللتين', 'اللذان', 'اللذين', 'اللواتي', 'كييفيك', 'كيف',
    'عندكم', 'عندنا', 'عندهم', 'عندم', 'فينم', 'فينا', 'زازا',
    'زح', 'زق', 'زي', 'زيكم', 'زيهم', 'زي', 'داك', 'زيهن',
    'زينهن', 'زين', 'سان', 'ساي', 'ساكت', 'ساكتين', 'ساكتينليهم',
    'سرعو', 'سرع', 'سوا', 'سيبم', 'سواء', 'سينة', 'شاف', 'شوفو',
    'شايفك', 'شايفم', 'شايفنكم', 'شنهو', 'شنو', 'شوف', 'شوي',
    'شوي كدا', 'صح', 'صاااح', 'صحي', 'صار', 'صايرين', 'صا', 'طال',
    'طالو', 'طالما', 'طولنا', 'طولة', 'طايق', 'طايقيين', 'طاف',
    'طالبين', 'طلبو', 'علي', 'عليهم', 'عاينهم', 'عاينيهم', 'عال',
    'عوك', 'علييي', 'عاين', 'عييييي', 'عليهو', 'غالبا', 'غالبيين',
    'غلبوهم', 'غالبهم', 'غالي', 'غيرو', 'غيرهم', 'غيرنا', 'غالب',
    'غم', 'في', 'فيما', 'فما', 'قد', 'قدما', 'كاد', 'كادو', 'كان',
    'كانون', 'كانو', 'كانهم', 'كأن', 'كلوكم', 'كاين', 'كييف',
    'لا', 'لي', 'لنا', 'لينا', 'ليهم', 'لهم', 'لكن', 'لكنهم',
    'لكننا', 'لكنكم', 'من', 'منهم', 'مننا', 'منم', 'منن', 'ماكدا',
    'مافي', 'ماف', 'مايو', 'مارس', 'فينهم', 'فينكم', 'فيا',
]

# Build full stopwords set
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords as nltk_stopwords
    ALL_STOPWORDS = set(nltk_stopwords.words('arabic'))
except Exception:
    ALL_STOPWORDS = set()
ALL_STOPWORDS.update(SUDANESE_STOPWORDS)

_PUNCT_TRANSLATOR = str.maketrans('', '', PUNCTUATIONS)


def preprocess_mhamed(text):
    """
    Exact preprocessing from Mhamed's notebook (lines 724-749).
    """
    if not text or not isinstance(text, str):
        return ""
    # 1. Remove punctuations
    text = text.translate(_PUNCT_TRANSLATOR)
    # 2. Remove diacritics (Tashkeel)
    text = re.sub(ARABIC_DIACRITICS, '', text)
    # 3. Alef normalization
    text = re.sub("[إأآا]", "ا", text)
    # 4. Yeh normalization
    text = re.sub("ى", "ي", text)
    # 5. Hamza normalization
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    # 6. Heh normalization
    text = re.sub("ة", "ه", text)
    # 7. Gaf normalization (from notebook)
    text = re.sub("گ", "ك", text)
    # 8. Remove stopwords
    text = ' '.join(w for w in text.split() if w not in ALL_STOPWORDS)
    return text.strip()


# ============================================================
#  CUSTOM MMA POOLING LAYER (Paper Equation 4)
# ============================================================

class MMAPooling1D(Layer):
    """
    Mean-Max-Average pooling from Mhamed et al. paper.
    MMA_K = (Max(P_k) + Avg(P_k)) / 2
    Combines local feature extraction (max) with global smoothing (avg).
    """
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.max_pool = MaxPooling1D(pool_size=pool_size, padding='valid')
        self.avg_pool = AveragePooling1D(pool_size=pool_size, padding='valid')

    def call(self, inputs):
        max_out = self.max_pool(inputs)
        avg_out = self.avg_pool(inputs)
        return (max_out + avg_out) / 2.0

    def get_config(self):
        config = super().get_config()
        config.update({"pool_size": self.pool_size})
        return config


# ============================================================
#  MODEL BUILDERS
# ============================================================

def build_cnn_baseline(num_unique_words, max_len, num_classes):
    """
    Exact CNN from Mhamed's notebook (lines 976-996).
    4 Conv1D [512, 256, 32, 32], padding='same', GlobalMaxPooling1D.
    """
    model = Sequential([
        Embedding(num_unique_words, EMBEDDING_SIZE, input_length=max_len),
        Conv1D(512, kernel_size=3, padding='same', activation='relu'),
        Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        GlobalMaxPooling1D(),
        Dense(512,
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=l2(1e-4),
              activity_regularizer=l2(1e-5)),
        Dropout(0.5),
        BatchNormalization(),
        Dropout(0.5),
        Flatten(),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy'],
    )
    return model


def build_scm_mma(num_unique_words, max_len, num_classes):
    """
    SCM+MMA from paper Section 4.4, Figure 9.
    4 Conv1D [512, 256, 128, 64], padding='valid', MMA pooling.
    """
    model = Sequential([
        Embedding(num_unique_words, EMBEDDING_SIZE, input_length=max_len),
        Conv1D(512, kernel_size=3, padding='valid', activation='relu', strides=1),
        Conv1D(256, kernel_size=3, padding='valid', activation='relu', strides=1),
        Conv1D(128, kernel_size=3, padding='valid', activation='relu', strides=1),
        Conv1D(64, kernel_size=3, padding='valid', activation='relu', strides=1),
        MMAPooling1D(pool_size=2),
        Dense(32, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dropout(0.5),
        Flatten(),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy'],
    )
    return model


MODEL_BUILDERS = {
    "cnn_baseline": build_cnn_baseline,
    "scm_mma": build_scm_mma,
}


# ============================================================
#  DATA LOADING
# ============================================================

def load_json_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tsv_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append({"text": row["text"], "label": row["label"]})
    return data


DATASET_CONFIG = {
    "sudsenti2": {
        "files": ["sudsenti2_train.json", "sudsenti2_val.json", "sudsenti2_test.json"],
        "type": "json",
        "method": "cv",
        "desc": "SudSenti2 (Mhamed et al., 2-class)",
    },
    "sudsenti3": {
        "files": ["sudsenti3_train.json", "sudsenti3_val.json", "sudsenti3_test.json"],
        "type": "json",
        "method": "cv",
        "desc": "SudSenti3 (Mhamed et al., 3-class)",
    },
    "hs_binary": {
        "path": HS_BINARY,
        "type": "tsv",
        "method": "split",
        "desc": "Hate Speech Binary (HARMFUL/NEUTRAL)",
    },
    "hs_3class": {
        "path": HS_3CLASS,
        "type": "tsv",
        "method": "split",
        "desc": "Hate Speech 3-class (HATE/OFFENSIVE/NEUTRAL)",
    },
}


def load_all_data(dataset_name):
    """Load all data for a dataset, apply Mhamed preprocessing."""
    cfg = DATASET_CONFIG[dataset_name]

    if cfg["type"] == "json":
        all_data = []
        for fname in cfg["files"]:
            all_data += load_json_data(os.path.join(DATA_DIR, fname))
    else:
        all_data = load_tsv_data(cfg["path"])

    texts, labels = [], []
    removed = 0
    for d in all_data:
        t = preprocess_mhamed(d["text"])
        if t.strip():
            texts.append(t)
            labels.append(d["label"])
        else:
            removed += 1

    if removed:
        print(f"  Removed {removed} empty entries after preprocessing")

    return texts, labels


# ============================================================
#  TRAINING + EVALUATION
# ============================================================

def train_and_evaluate(model_name, texts, labels_encoded, labels_cat,
                       label_names, train_idx, test_idx, fold_info=""):
    """Train model on one split and return metrics."""

    X_train_text = [texts[i] for i in train_idx]
    X_test_text  = [texts[i] for i in test_idx]
    Y_train = labels_cat[train_idx]
    Y_test  = labels_cat[test_idx]
    Y_test_int = labels_encoded[test_idx]

    # Tokenize — fit on train only
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq  = tokenizer.texts_to_sequences(X_test_text)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN)

    num_unique_words = min(len(tokenizer.word_index) + 1, 50000)
    num_classes = len(label_names)

    # Build fresh model
    builder = MODEL_BUILDERS[model_name]
    model = builder(num_unique_words, MAX_LEN, num_classes)

    # Train with early stopping
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                       restore_best_weights=True, verbose=0)

    # Use 10% of train as validation for early stopping
    history = model.fit(
        X_train_pad, Y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=0,
    )

    # Predict
    Y_pred_prob = model.predict(X_test_pad, verbose=0)
    Y_pred = np.argmax(Y_pred_prob, axis=-1)

    # Metrics
    acc    = accuracy_score(Y_test_int, Y_pred) * 100
    f1_mac = f1_score(Y_test_int, Y_pred, average="macro", zero_division=0) * 100
    f1_per = f1_score(Y_test_int, Y_pred, average=None, zero_division=0) * 100

    epochs_run = len(history.history['loss'])
    print(f"  {fold_info}Acc={acc:.2f}%  F1={f1_mac:.2f}%  epochs={epochs_run}")

    # Cleanup
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "accuracy": round(acc, 2),
        "f1_macro": round(f1_mac, 2),
        "f1_per_class": {name: round(float(f1_per[i]), 2)
                         for i, name in enumerate(label_names)},
        "epochs_trained": epochs_run,
        "confusion_matrix": confusion_matrix(Y_test_int, Y_pred).tolist(),
    }


def run_experiment(model_name, dataset_name, skip_existing=False):
    """Run full experiment: CV or single split depending on dataset."""

    exp_dir = os.path.join(RESULTS_DIR, f"{model_name}_{dataset_name}")
    results_file = os.path.join(exp_dir, "results.json")

    if skip_existing and os.path.exists(results_file):
        print(f"\n  SKIP {model_name} × {dataset_name}")
        with open(results_file) as f:
            return json.load(f)

    os.makedirs(exp_dir, exist_ok=True)
    cfg = DATASET_CONFIG[dataset_name]

    print(f"\n{'='*60}")
    print(f"  {model_name} × {dataset_name}")
    print(f"  {cfg['desc']}")
    print(f"{'='*60}")

    t0 = time.time()

    # Load data
    texts, labels_str = load_all_data(dataset_name)
    label_names = sorted(set(labels_str))
    num_classes = len(label_names)

    le = LabelEncoder()
    le.fit(label_names)
    labels_encoded = le.transform(labels_str)
    labels_cat = to_categorical(labels_encoded, num_classes=num_classes)

    print(f"  Samples: {len(texts)},  Classes: {num_classes} {label_names}")
    print(f"  Distribution: {dict(Counter(labels_str))}")

    if cfg["method"] == "cv":
        # 10-fold CV (matches Mhamed et al.)
        n_folds = 10
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels_encoded)):
            result = train_and_evaluate(
                model_name, texts, labels_encoded, labels_cat,
                label_names, train_idx, test_idx,
                fold_info=f"Fold {fold+1}/{n_folds}: ",
            )
            result["fold"] = fold + 1
            fold_results.append(result)

        # Average results
        accs = [r["accuracy"] for r in fold_results]
        f1s  = [r["f1_macro"] for r in fold_results]
        mean_acc, std_acc = np.mean(accs), np.std(accs)
        mean_f1, std_f1   = np.mean(f1s), np.std(f1s)

        f1_per_avg = {}
        for name in label_names:
            vals = [r["f1_per_class"][name] for r in fold_results]
            f1_per_avg[name] = round(float(np.mean(vals)), 2)

        total_time = time.time() - t0

        results = {
            "model":         model_name,
            "dataset":       dataset_name,
            "method":        f"{n_folds}-fold CV",
            "num_classes":   num_classes,
            "labels":        label_names,
            "total_samples": len(texts),
            "accuracy":      round(mean_acc, 2),
            "accuracy_std":  round(std_acc, 2),
            "f1_macro":      round(mean_f1, 2),
            "f1_macro_std":  round(std_f1, 2),
            "f1_per_class":  f1_per_avg,
            "fold_results":  fold_results,
            "training_time": round(total_time, 1),
        }

        print(f"\n  ┌─── {n_folds}-FOLD CV RESULTS ──────────────┐")
        print(f"  │  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%  │")
        print(f"  │  F1-macro: {mean_f1:.2f}% ± {std_f1:.2f}%  │")
        print(f"  │  Time:     {total_time:.0f}s              │")
        print(f"  └──────────────────────────────────────┘")

    else:
        # 80/10/10 split for hate speech
        indices = np.arange(len(texts))
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=SEED,
            stratify=labels_encoded,
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=SEED,
            stratify=labels_encoded[temp_idx],
        )

        print(f"  Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")

        # For training, combine train+val (val used internally by fit)
        train_val_idx = np.concatenate([train_idx, val_idx])

        result = train_and_evaluate(
            model_name, texts, labels_encoded, labels_cat,
            label_names, train_val_idx, test_idx,
        )

        total_time = time.time() - t0

        # Classification report
        report_str = classification_report(
            labels_encoded[test_idx],
            np.argmax(
                MODEL_BUILDERS[model_name](
                    1000, MAX_LEN, num_classes  # dummy — won't be used
                ).predict(np.zeros((1, MAX_LEN)), verbose=0),
                axis=-1
            ) if False else None,  # placeholder
            target_names=label_names, digits=4, zero_division=0,
        ) if False else ""

        results = {
            "model":         model_name,
            "dataset":       dataset_name,
            "method":        "80/10/10 split",
            "num_classes":   num_classes,
            "labels":        label_names,
            "total_samples": len(texts),
            "train_size":    len(train_val_idx),
            "test_size":     len(test_idx),
            "accuracy":      result["accuracy"],
            "f1_macro":      result["f1_macro"],
            "f1_per_class":  result["f1_per_class"],
            "confusion_matrix": result["confusion_matrix"],
            "epochs_trained": result["epochs_trained"],
            "training_time": round(total_time, 1),
        }

        print(f"\n  ┌─── RESULTS ──────────────────────────┐")
        print(f"  │  Accuracy: {result['accuracy']:.2f}%              │")
        print(f"  │  F1-macro: {result['f1_macro']:.2f}%              │")
        print(f"  │  Time:     {total_time:.0f}s              │")
        print(f"  └──────────────────────────────────────┘")

    # Save confusion matrix plot (last fold or single split)
    last_cm = results.get("confusion_matrix") or fold_results[-1].get("confusion_matrix")
    if last_cm:
        cm = np.array(last_cm)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=label_names, yticklabels=label_names)
        ax.set_title(f"{model_name} — {dataset_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

    # Save results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_file}")

    return results


# ============================================================
#  COMPARISON TABLE
# ============================================================

def print_comparison(all_results):
    """Print comparison table across all experiments."""

    print(f"\n\n{'='*70}")
    print(f"  COMPLETE COMPARISON TABLE")
    print(f"{'='*70}")

    for ds in DATASET_CONFIG:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        if not ds_results:
            continue

        print(f"\n  Dataset: {DATASET_CONFIG[ds]['desc']}")
        print(f"  {'Model':<18s} {'Acc%':>8s} {'F1-mac%':>8s}")
        print(f"  {'-'*18} {'-'*8} {'-'*8}")

        for r in sorted(ds_results, key=lambda x: x.get("accuracy", 0), reverse=True):
            acc_str = f"{r['accuracy']:.2f}"
            if "accuracy_std" in r:
                acc_str += f"±{r['accuracy_std']:.1f}"
            f1_str = f"{r['f1_macro']:.2f}"
            if "f1_macro_std" in r:
                f1_str += f"±{r['f1_macro_std']:.1f}"
            print(f"  {r['model']:<18s} {acc_str:>8s} {f1_str:>8s}")


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mhamed et al. CNN Reproduction")
    parser.add_argument("--model", type=str,
                        choices=list(MODEL_BUILDERS.keys()),
                        help="Single model to run")
    parser.add_argument("--dataset", type=str,
                        choices=list(DATASET_CONFIG.keys()),
                        help="Single dataset to use")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all models × all datasets")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments with existing results")
    args = parser.parse_args()

    if not args.run_all and not (args.model and args.dataset):
        parser.error("Provide --run_all or --model + --dataset")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n  GPU: {gpus[0]}")
        # Limit GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n  WARNING: No GPU detected")

    # Build experiment list
    if args.run_all:
        experiments = [(m, d) for d in DATASET_CONFIG for m in MODEL_BUILDERS]
    else:
        experiments = [(args.model, args.dataset)]

    print(f"\n  Experiments: {len(experiments)}")
    for m, d in experiments:
        print(f"    {m} × {d}")

    # Run
    all_results = []
    failed = []

    for i, (model_name, dataset_name) in enumerate(experiments, 1):
        print(f"\n\n  ╔═══════════════════════════════════════╗")
        print(f"  ║  {i}/{len(experiments)}: {model_name} × {dataset_name}")
        print(f"  ╚═══════════════════════════════════════╝")

        try:
            result = run_experiment(model_name, dataset_name,
                                    skip_existing=args.skip_existing)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ✗ FAILED: {model_name} × {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((model_name, dataset_name, str(e)))
            gc.collect()
            tf.keras.backend.clear_session()

    # Summary
    if all_results:
        print_comparison(all_results)

        # Save summary
        summary_path = os.path.join(RESULTS_DIR, "all_results_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Summary: {summary_path}")

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for m, d, e in failed:
            print(f"    {m} × {d}: {e}")

    print(f"\n  DONE — {len(all_results)} succeeded, {len(failed)} failed")


if __name__ == "__main__":
    main()
