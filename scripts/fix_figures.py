#!/usr/bin/env python3
"""
Fixed Word Clouds + KDE Distribution Plots
============================================
Fixes:
  - Removes English text, hex codes (00A0, 00BD, etc.) from word clouds
  - Removes "الله" from word clouds
  - Adds KDE curve distribution plots (like Fig 3 style)

Usage:
  python3 fix_figures.py
"""

import json
import os
import re
import csv
import numpy as np
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import gaussian_kde

# ============================================================
#  CONFIGURATION
# ============================================================

BASE_DIR = os.path.expanduser("~/sudanese_dialect_project")
DATA_DIR = os.path.join(BASE_DIR, "data/sentiment_prepared")
HS_BINARY = os.path.join(BASE_DIR, "data/labeling_corpus/dataset_binary.tsv")
HS_3CLASS = os.path.join(BASE_DIR, "data/labeling_corpus/dataset_3class.tsv")
OUT_DIR  = os.path.join(BASE_DIR, "results/paper_figures/dataset_analysis")

ARABIC_FONT = os.path.expanduser("~/.fonts/Amiri-Regular.ttf")

# Words to EXCLUDE from ALL word clouds
EXCLUDE_WORDS = {
    # "الله" — should not appear in harmful/hate context
    'الله',
    # English words that leak from emoji descriptions
    'face', 'tears', 'crying', 'joy', 'heart', 'Loudly', 'loudly',
    'tear', 'joyFace', 'faceL', 'faceLoudly',
    'ed', 'https', 'http', 'www', 'com',
    'Sudan', 'RT',
    # Hex codes from Unicode escapes
    'U',
}

# Regex to detect non-Arabic tokens (English, hex codes like 00A0, numbers)
_NON_ARABIC_RE = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
_HEX_CODE_RE = re.compile(r'^[0-9A-Fa-f]{2,6}$')

COLORS = {
    "HARMFUL": "#e74c3c", "NEUTRAL": "#2ecc71",
    "HATE": "#c0392b", "OFFENSIVE": "#e67e22",
    "neg": "#e74c3c", "pos": "#2ecc71", "obj": "#3498db",
}

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 150, "font.size": 11})
sns.set_style("whitegrid")


# ============================================================
#  DATA LOADING
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_tsv(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append({"text": row["text"], "label": row["label"]})
    return data


# ============================================================
#  WORD FILTERING
# ============================================================

def is_arabic_word(word):
    """Return True if word contains Arabic characters and is not junk."""
    if not word or len(word) <= 1:
        return False
    # Reject if in exclude list
    if word in EXCLUDE_WORDS:
        return False
    # Reject pure English / hex codes / numbers
    if _NON_ARABIC_RE.match(word):
        return False
    if _HEX_CODE_RE.match(word):
        return False
    # Must contain at least one Arabic character
    has_arabic = any('\u0600' <= c <= '\u06FF' or
                     '\u0750' <= c <= '\u077F' or
                     '\uFB50' <= c <= '\uFDFF' or
                     '\uFE70' <= c <= '\uFEFF' for c in word)
    if not has_arabic:
        return False
    # Reject if majority non-Arabic (mixed tokens like "joyFace")
    arabic_count = sum(1 for c in word if '\u0600' <= c <= '\u06FF' or
                       '\u0750' <= c <= '\u077F' or
                       '\uFB50' <= c <= '\uFDFF' or
                       '\uFE70' <= c <= '\uFEFF')
    if arabic_count / len(word) < 0.5:
        return False
    return True


def clean_text_for_wordcloud(text):
    """Remove URLs, mentions, non-Arabic tokens, and excluded words."""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbol
    text = text.replace('#', '')
    # Remove numbers (Arabic and Western)
    text = re.sub(r'[0-9٠-٩]+', '', text)
    # Split and filter
    words = [w for w in text.split() if is_arabic_word(w)]
    return ' '.join(words)


# ============================================================
#  WORD CLOUD PLOTS (FIXED)
# ============================================================

def plot_wordcloud(data, name, out_dir):
    """Word cloud per class — clean Arabic only, no English/hex/الله."""
    os.makedirs(out_dir, exist_ok=True)
    labels = sorted(set(d["label"] for d in data))

    n_cols = len(labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    font_path = ARABIC_FONT if os.path.exists(ARABIC_FONT) else None

    for ax, lbl in zip(axes, labels):
        # Combine all text for this class
        raw_text = " ".join(d["text"] for d in data if d["label"] == lbl)
        # Clean: remove English, hex codes, الله, short words
        clean_text = clean_text_for_wordcloud(raw_text)

        if not clean_text.strip():
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(lbl)
            ax.axis("off")
            continue

        # Choose colormap
        if lbl in ("HARMFUL", "HATE", "neg", "OFFENSIVE"):
            cmap = "Reds"
        elif lbl in ("NEUTRAL", "pos"):
            cmap = "Greens"
        else:
            cmap = "Blues"

        wc = WordCloud(
            font_path=font_path,
            width=800, height=600,
            background_color="white",
            max_words=100,
            collocations=False,
            prefer_horizontal=0.7,
            colormap=cmap,
            min_word_length=2,
        )
        wc.generate(clean_text)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(lbl, fontsize=14, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Word Clouds — {name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}_wordclouds.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
#  KDE DISTRIBUTION PLOTS (Fig 3 style)
# ============================================================

def plot_kde_distribution(data, name, out_dir):
    """
    Word count distribution with KDE curve per class.
    Style matches Fig 3: histogram bars + smooth KDE overlay.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels_unique = sorted(set(d["label"] for d in data))

    n_cols = len(labels_unique)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    class_colors = {
        "HARMFUL": "#6464FF",   # blue-purple like Fig 3 left
        "NEUTRAL": "#B06AB3",   # purple like Fig 3 right
        "HATE":    "#e74c3c",
        "OFFENSIVE": "#e67e22",
        "neg":     "#e74c3c",
        "pos":     "#2ecc71",
        "obj":     "#3498db",
    }

    for ax, lbl in zip(axes, labels_unique):
        word_counts = [len(d["text"].split()) for d in data if d["label"] == lbl]
        word_counts = [c for c in word_counts if c > 0]

        color = class_colors.get(lbl, "#3498db")

        # Histogram
        counts_arr = np.array(word_counts)
        max_wc = min(int(np.percentile(counts_arr, 99)), 60)
        bins = np.arange(0, max_wc + 2, 2)

        ax.hist(word_counts, bins=bins, color=color, alpha=0.7,
                edgecolor="white", linewidth=0.5, density=False)

        # KDE curve overlay
        if len(word_counts) > 10:
            kde_x = np.linspace(0, max_wc, 200)
            kde = gaussian_kde(word_counts, bw_method=0.3)
            kde_y = kde(kde_x)
            # Scale KDE to match histogram height
            bin_width = bins[1] - bins[0]
            scale_factor = len(word_counts) * bin_width
            ax.plot(kde_x, kde_y * scale_factor, color=color,
                    linewidth=2.5, alpha=0.9)

        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{lbl} Word Count Distribution")
        ax.set_xlim(0, max_wc)

        # Stats annotation
        mean_wc = np.mean(word_counts)
        median_wc = np.median(word_counts)
        ax.axvline(mean_wc, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(mean_wc + 1, ax.get_ylim()[1] * 0.9,
                f"μ={mean_wc:.1f}\nn={len(word_counts)}",
                fontsize=9, color='black')

    fig.suptitle(f"Word Count Distribution — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}_kde_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
#  COMBINED DISTRIBUTION (all classes overlaid — single plot)
# ============================================================

def plot_combined_distribution(data, name, out_dir):
    """Single plot with all classes overlaid — for compact paper figures."""
    os.makedirs(out_dir, exist_ok=True)
    labels_unique = sorted(set(d["label"] for d in data))

    class_colors = {
        "HARMFUL": "#e74c3c", "NEUTRAL": "#2ecc71",
        "HATE": "#c0392b", "OFFENSIVE": "#e67e22",
        "neg": "#e74c3c", "pos": "#2ecc71", "obj": "#3498db",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for lbl in labels_unique:
        word_counts = [len(d["text"].split()) for d in data if d["label"] == lbl]
        word_counts = [c for c in word_counts if c > 0]

        color = class_colors.get(lbl, "#3498db")

        if len(word_counts) > 10:
            max_wc = min(int(np.percentile(np.array(word_counts), 99)), 60)
            kde_x = np.linspace(0, max_wc, 200)
            kde = gaussian_kde(word_counts, bw_method=0.3)
            kde_y = kde(kde_x)
            ax.fill_between(kde_x, kde_y, alpha=0.3, color=color)
            ax.plot(kde_x, kde_y, color=color, linewidth=2, label=f"{lbl} (n={len(word_counts)})")

    ax.set_xlabel("Word Count")
    ax.set_ylabel("Density")
    ax.set_title(f"Word Count Distribution — {name}")
    ax.legend()
    ax.set_xlim(0, 60)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}_combined_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
#  MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  FIXING WORD CLOUDS + KDE DISTRIBUTIONS")
    print("=" * 60)

    # Load all datasets
    datasets = {}

    # Hate speech
    datasets["HS_Binary"] = load_tsv(HS_BINARY)
    datasets["HS_3Class"] = load_tsv(HS_3CLASS)

    # Sentiment
    datasets["Telecom"] = (
        load_json(os.path.join(DATA_DIR, "telecom_train.json")) +
        load_json(os.path.join(DATA_DIR, "telecom_test.json"))
    )
    datasets["SudSenti2"] = (
        load_json(os.path.join(DATA_DIR, "sudsenti2_train.json")) +
        load_json(os.path.join(DATA_DIR, "sudsenti2_val.json")) +
        load_json(os.path.join(DATA_DIR, "sudsenti2_test.json"))
    )
    datasets["SudSenti3"] = (
        load_json(os.path.join(DATA_DIR, "sudsenti3_train.json")) +
        load_json(os.path.join(DATA_DIR, "sudsenti3_val.json")) +
        load_json(os.path.join(DATA_DIR, "sudsenti3_test.json"))
    )

    for name, data in datasets.items():
        print(f"\n  Processing: {name} ({len(data)} samples)")

        # Fixed word clouds
        plot_wordcloud(data, name, OUT_DIR)

        # KDE distribution per class (Fig 3 style)
        plot_kde_distribution(data, name, OUT_DIR)

        # Combined distribution (single plot)
        plot_combined_distribution(data, name, OUT_DIR)

    print(f"\n  {'='*60}")
    print(f"  ALL FIXED FIGURES GENERATED")
    print(f"  Output: {OUT_DIR}/")
    print(f"  {'='*60}")

    # List new files
    new_files = [f for f in os.listdir(OUT_DIR)
                 if f.endswith('.png') and ('wordcloud' in f or 'kde' in f or 'combined' in f)]
    print(f"\n  Generated {len(new_files)} figures:")
    for f in sorted(new_files):
        print(f"    {f}")


if __name__ == "__main__":
    main()
