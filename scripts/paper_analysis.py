#!/usr/bin/env python3
"""
Comprehensive Analysis & Visualization — Paper 2
==================================================
Generates all figures and tables for the hate speech paper.

Outputs → results/paper_figures/

Sections:
  1. Hate speech dataset analysis (binary + 3-class)
  2. Sentiment dataset analysis (telecom + SudSenti2 + SudSenti3)
  3. Hate speech model comparison (confusion matrices, performance bars)
  4. Sentiment model comparison
  5. Word clouds per class
  6. Cross-task comparison table

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 paper_analysis.py
"""

import json
import os
import re
import sys
import csv
import numpy as np
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from wordcloud import WordCloud

# ============================================================
#  CONFIGURATION
# ============================================================

BASE_DIR = os.path.expanduser("~/sudanese_dialect_project")
OUT_DIR  = os.path.join(BASE_DIR, "results/paper_figures")

# Data paths
HS_BINARY   = os.path.join(BASE_DIR, "data/labeling_corpus/dataset_binary.tsv")
HS_3CLASS   = os.path.join(BASE_DIR, "data/labeling_corpus/dataset_3class.tsv")
SENT_DIR    = os.path.join(BASE_DIR, "data/sentiment_prepared")

# Results paths
HS_MODELS_DIR   = os.path.join(BASE_DIR, "results/hate_speech_models")
HS_HYBRID_DIR   = os.path.join(BASE_DIR, "results/hate_speech_hybrid")
SENT_EVAL_DIR   = os.path.join(BASE_DIR, "results/sentiment_evaluation")
SENT_MHAMED_DIR = os.path.join(BASE_DIR, "results/sentiment_evaluation_mhamed")

# Arabic font
ARABIC_FONT = os.path.expanduser("~/.fonts/Amiri-Regular.ttf")
ARABIC_FONT_BOLD = os.path.expanduser("~/.fonts/Amiri-Bold.ttf")

# Colors
COLORS = {
    "HARMFUL": "#e74c3c",
    "NEUTRAL": "#2ecc71",
    "HATE":    "#c0392b",
    "OFFENSIVE": "#e67e22",
    "neg":     "#e74c3c",
    "pos":     "#2ecc71",
    "obj":     "#3498db",
}

MODEL_ORDER = [
    "marbertv2", "marbert", "arbert", "arabertv2",
    "camelbert_da", "qarib", "sudabert_v2",
    "xlm_roberta", "mbert",
]

# Plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
    "font.size": 11,
})
sns.set_style("whitegrid")


# ============================================================
#  UTILITIES
# ============================================================

def load_tsv(path):
    """Load TSV file → list of {"text": ..., "label": ...}"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append({"text": row["text"], "label": row["label"]})
    return data


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_arabic(text):
    """Simple whitespace tokenizer for Arabic text statistics."""
    return text.split()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_arabic_font_prop(size=12, bold=False):
    """Return FontProperties for Arabic text."""
    fpath = ARABIC_FONT_BOLD if bold and os.path.exists(ARABIC_FONT_BOLD) else ARABIC_FONT
    if os.path.exists(fpath):
        return fm.FontProperties(fname=fpath, size=size)
    return None


# ============================================================
#  SECTION 1: DATASET ANALYSIS
# ============================================================

def analyze_dataset(data, name, out_dir):
    """Compute and print statistics for a dataset."""
    ensure_dir(out_dir)

    labels = [d["label"] for d in data]
    texts  = [d["text"] for d in data]

    # Token stats
    token_counts = [len(tokenize_arabic(t)) for t in texts]
    char_counts  = [len(t) for t in texts]

    label_dist = Counter(labels)
    total = len(data)

    stats = {
        "name": name,
        "total_samples": total,
        "num_classes": len(label_dist),
        "class_distribution": {k: v for k, v in sorted(label_dist.items())},
        "class_percentages": {k: round(v / total * 100, 2) for k, v in sorted(label_dist.items())},
        "token_stats": {
            "total_tokens": sum(token_counts),
            "mean": round(np.mean(token_counts), 2),
            "median": round(float(np.median(token_counts)), 2),
            "std": round(np.std(token_counts), 2),
            "min": int(np.min(token_counts)),
            "max": int(np.max(token_counts)),
        },
        "char_stats": {
            "total_chars": sum(char_counts),
            "mean": round(np.mean(char_counts), 2),
            "median": round(float(np.median(char_counts)), 2),
            "min": int(np.min(char_counts)),
            "max": int(np.max(char_counts)),
        },
        "vocab_size": len(set(w for t in texts for w in tokenize_arabic(t))),
    }

    # Print
    print(f"\n  {'='*50}")
    print(f"  Dataset: {name}")
    print(f"  {'='*50}")
    print(f"  Samples: {total}")
    print(f"  Classes: {stats['num_classes']}")
    for lbl, cnt in sorted(label_dist.items()):
        pct = cnt / total * 100
        print(f"    {lbl:12s}: {cnt:6d} ({pct:.1f}%)")
    print(f"  Tokens: mean={stats['token_stats']['mean']}, "
          f"median={stats['token_stats']['median']}, "
          f"total={stats['token_stats']['total_tokens']:,}")
    print(f"  Vocab: {stats['vocab_size']:,} unique words")

    # Save
    with open(os.path.join(out_dir, f"{name}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def plot_class_distribution(stats, name, out_dir):
    """Bar chart of class distribution."""
    labels = list(stats["class_distribution"].keys())
    counts = list(stats["class_distribution"].values())
    colors = [COLORS.get(l, "#95a5a6") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, counts):
        pct = count / sum(counts) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Class Distribution — {name}")
    ax.set_ylim(0, max(counts) * 1.15)

    path = os.path.join(out_dir, f"{name}_class_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_text_length_distribution(data, name, out_dir):
    """Histogram of text lengths (in tokens)."""
    token_counts = [len(tokenize_arabic(d["text"])) for d in data]
    labels = [d["label"] for d in data]
    unique_labels = sorted(set(labels))

    fig, ax = plt.subplots(figsize=(8, 4))
    for lbl in unique_labels:
        lbl_counts = [tc for tc, l in zip(token_counts, labels) if l == lbl]
        ax.hist(lbl_counts, bins=50, alpha=0.6, label=lbl,
                color=COLORS.get(lbl, None), edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Text Length Distribution — {name}")
    ax.legend()

    path = os.path.join(out_dir, f"{name}_length_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_wordcloud(data, name, out_dir):
    """Word cloud per class — raw Arabic, no reshaper/bidi."""
    labels = sorted(set(d["label"] for d in data))

    n_cols = len(labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    font_path = ARABIC_FONT if os.path.exists(ARABIC_FONT) else None

    for ax, lbl in zip(axes, labels):
        class_texts = " ".join(d["text"] for d in data if d["label"] == lbl)

        # Remove very short words (1-2 chars) to reduce noise
        words = [w for w in class_texts.split() if len(w) > 2]
        class_texts = " ".join(words)

        if not class_texts.strip():
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(lbl)
            ax.axis("off")
            continue

        wc = WordCloud(
            font_path=font_path,
            width=800, height=600,
            background_color="white",
            max_words=100,
            collocations=False,
            prefer_horizontal=0.7,
            colormap="viridis" if lbl in ("NEUTRAL", "pos", "obj") else "Reds",
        )
        wc.generate(class_texts)

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
#  SECTION 2: MODEL RESULTS — CONFUSION MATRICES
# ============================================================

def collect_hs_results():
    """Collect all hate speech results (Phase 1 + Phase 2)."""
    results = {"binary": [], "3class": []}

    # Phase 1 baselines
    for ds in ["binary", "3class"]:
        for model_dir in sorted(os.listdir(HS_MODELS_DIR)):
            rpath = os.path.join(HS_MODELS_DIR, model_dir, "results.json")
            if os.path.isfile(rpath) and model_dir.endswith(f"_{ds}"):
                r = load_json(rpath)
                r["phase"] = "baseline"
                results[ds].append(r)

    # Phase 2 hybrids
    if os.path.exists(HS_HYBRID_DIR):
        for hybrid_dir in sorted(os.listdir(HS_HYBRID_DIR)):
            rpath = os.path.join(HS_HYBRID_DIR, hybrid_dir, "results.json")
            if os.path.isfile(rpath):
                r = load_json(rpath)
                ds_key = "binary" if "binary" in hybrid_dir else "3class"
                r["phase"] = "hybrid"
                results[ds_key].append(r)

    return results


def plot_hs_confusion_matrices(results, out_dir):
    """Grid of confusion matrices for all hate speech models."""
    ensure_dir(out_dir)

    for ds_name in ["binary", "3class"]:
        ds_results = results[ds_name]
        if not ds_results:
            continue

        # Sort by f1_macro descending
        ds_results.sort(key=lambda x: x.get("f1_macro", 0), reverse=True)

        n = len(ds_results)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for i, r in enumerate(ds_results):
            ax = axes[i]
            cm = np.array(r["confusion_matrix"])
            label_names = r.get("label_names", [])

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=label_names, yticklabels=label_names)

            model_name = r.get("model_name", r.get("experiment_id", "?"))
            f1 = r.get("f1_macro", 0)
            if isinstance(f1, float) and f1 < 1:
                f1 *= 100
            phase = r.get("phase", "")
            marker = " ★" if phase == "hybrid" else ""
            ax.set_title(f"{model_name}{marker}\nF1={f1:.1f}%", fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Hate Speech — {ds_name} Confusion Matrices", fontsize=14)
        plt.tight_layout()
        path = os.path.join(out_dir, f"hs_{ds_name}_confusion_matrices.png")
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


def plot_hs_performance_comparison(results, out_dir):
    """Bar chart comparing all hate speech models."""
    ensure_dir(out_dir)

    for ds_name in ["binary", "3class"]:
        ds_results = results[ds_name]
        if not ds_results:
            continue

        ds_results.sort(key=lambda x: x.get("f1_macro", 0), reverse=True)

        names = []
        f1_scores = []
        colors = []

        for r in ds_results:
            model_name = r.get("model_name", r.get("experiment_id", "?"))
            f1 = r.get("f1_macro", 0)
            if isinstance(f1, float) and f1 < 1:
                f1 *= 100
            phase = r.get("phase", "baseline")

            names.append(model_name + (" ★" if phase == "hybrid" else ""))
            f1_scores.append(f1)
            colors.append("#e74c3c" if phase == "hybrid" else "#3498db")

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(range(len(names)), f1_scores, color=colors,
                       edgecolor="black", linewidth=0.5)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("F1-macro (%)")
        ax.set_title(f"Hate Speech — {ds_name} Model Comparison")
        ax.invert_yaxis()

        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{score:.1f}%", va="center", fontsize=9)

        ax.set_xlim(0, max(f1_scores) * 1.08)
        plt.tight_layout()
        path = os.path.join(out_dir, f"hs_{ds_name}_model_comparison.png")
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


# ============================================================
#  SECTION 3: SENTIMENT RESULTS COMPARISON
# ============================================================

def collect_sentiment_results(eval_dir):
    """Collect all sentiment results from a directory."""
    results = {"telecom": [], "sudsenti2": [], "sudsenti3": []}

    if not os.path.exists(eval_dir):
        return results

    summary_path = os.path.join(eval_dir, "all_results_summary.json")
    if os.path.exists(summary_path):
        all_results = load_json(summary_path)
        for r in all_results:
            ds = r.get("dataset", "")
            if ds in results:
                results[ds].append(r)

    return results


def plot_sentiment_comparison(results_minimal, results_mhamed, out_dir):
    """Side-by-side comparison of sentiment results."""
    ensure_dir(out_dir)

    # Published baselines for comparison
    published = {
        "telecom": {"MARBERT(P1)": 75.68, "MARBERTv2(P1)": 75.02,
                     "SudaBERT-v2(P1)": 74.74, "AraBERT(P1)": 68.85},
        "sudsenti3": {"MARBERT+FT(M)": 88.44, "MARBERT(M)": 86.83,
                       "SCM+MMA(M)": 85.23, "ARBERT(M)": 85.09},
    }

    for ds_name in ["telecom", "sudsenti2", "sudsenti3"]:
        min_results = results_minimal.get(ds_name, [])
        mh_results  = results_mhamed.get(ds_name, [])

        if not min_results:
            continue

        # Use minimal preprocessing results as primary
        min_results.sort(key=lambda x: x.get("accuracy", 0), reverse=True)

        # Build comparison data
        model_names = []
        min_accs = []
        mh_accs = []
        mh_lookup = {r["model"]: r.get("accuracy", 0) for r in mh_results}

        for r in min_results:
            mname = r["model"]
            model_names.append(mname)
            min_accs.append(r.get("accuracy", 0))
            mh_accs.append(mh_lookup.get(mname, 0))

        # Add published baselines
        pub = published.get(ds_name, {})
        pub_names = list(pub.keys())
        pub_accs  = list(pub.values())

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, min_accs, width, label="Minimal Preprocess",
                       color="#3498db", edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width/2, mh_accs, width, label="Mhamed Preprocess",
                       color="#e67e22", edgecolor="black", linewidth=0.5)

        # Add published baseline lines
        line_colors = ["#c0392b", "#27ae60", "#8e44ad", "#2c3e50"]
        for i, (pname, pacc) in enumerate(zip(pub_names, pub_accs)):
            ax.axhline(y=pacc, linestyle="--", linewidth=1.2,
                       color=line_colors[i % len(line_colors)],
                       label=pname)

        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Sentiment Evaluation — {ds_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend(fontsize=8, loc="lower right")

        # Value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        path = os.path.join(out_dir, f"sentiment_{ds_name}_comparison.png")
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


# ============================================================
#  SECTION 4: CROSS-TASK SUMMARY TABLE
# ============================================================

def generate_cross_task_table(hs_results, sent_results, out_dir):
    """Generate LaTeX-ready comparison table across all tasks."""
    ensure_dir(out_dir)

    # Hate speech F1-macro per model
    hs_binary = {r.get("model_name", ""): r.get("f1_macro", 0)
                 for r in hs_results.get("binary", []) if r.get("phase") == "baseline"}
    hs_3class = {r.get("model_name", ""): r.get("f1_macro", 0)
                 for r in hs_results.get("3class", []) if r.get("phase") == "baseline"}

    # Sentiment accuracy per model
    sent_telecom = {r["model"]: r.get("accuracy", 0) for r in sent_results.get("telecom", [])}
    sent_ss2 = {r["model"]: r.get("accuracy", 0) for r in sent_results.get("sudsenti2", [])}
    sent_ss3 = {r["model"]: r.get("accuracy", 0) for r in sent_results.get("sudsenti3", [])}

    # Build table
    lines = []
    lines.append("=" * 90)
    lines.append(f"{'Model':<15s} {'HS-Bin F1':>9s} {'HS-3C F1':>9s} "
                 f"{'Telecom':>9s} {'SS2 Acc':>9s} {'SS3 Acc':>9s}")
    lines.append("-" * 90)

    for model in MODEL_ORDER:
        hb = hs_binary.get(model, 0)
        h3 = hs_3class.get(model, 0)
        # Normalize to percentage
        if isinstance(hb, float) and hb < 1:
            hb *= 100
        if isinstance(h3, float) and h3 < 1:
            h3 *= 100

        st = sent_telecom.get(model, 0)
        s2 = sent_ss2.get(model, 0)
        s3 = sent_ss3.get(model, 0)

        hb_s = f"{hb:.1f}" if hb else "—"
        h3_s = f"{h3:.1f}" if h3 else "—"
        st_s = f"{st:.1f}" if st else "—"
        s2_s = f"{s2:.1f}" if s2 else "—"
        s3_s = f"{s3:.1f}" if s3 else "—"

        lines.append(f"{model:<15s} {hb_s:>9s} {h3_s:>9s} {st_s:>9s} {s2_s:>9s} {s3_s:>9s}")

    lines.append("=" * 90)

    table_str = "\n".join(lines)
    print(f"\n{table_str}")

    path = os.path.join(out_dir, "cross_task_summary.txt")
    with open(path, "w") as f:
        f.write(table_str)
    print(f"\n  Saved: {path}")


# ============================================================
#  SECTION 5: ANNOTATION AGREEMENT VISUALIZATION
# ============================================================

def plot_annotation_agreement(out_dir):
    """Visualize inter-annotator agreement from agreement_analysis."""
    ensure_dir(out_dir)

    agree_path = os.path.join(BASE_DIR, "data/labeling_corpus/agreement_analysis/agreement_summary.json")
    if not os.path.exists(agree_path):
        print("  Skipping annotation agreement — file not found")
        return

    agree = load_json(agree_path)

    # Tier distribution pie chart
    tier_labels = ["Full Agreement\n(3/3)", "Partial Agreement\n(2/3)", "No Agreement\n(0/3)"]
    tier_counts = [
        agree.get("tier1_count", agree.get("full_agreement", 0)),
        agree.get("tier2_count", agree.get("partial_agreement", 0)),
        agree.get("tier3_count", agree.get("no_agreement", 0)),
    ]

    if sum(tier_counts) == 0:
        # Try alternate keys
        for key in agree:
            if "tier" in key.lower() or "agreement" in key.lower():
                print(f"    Available key: {key} = {agree[key]}")
        print("  Could not parse agreement data")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    colors_pie = ["#2ecc71", "#f1c40f", "#e74c3c"]
    wedges, texts, autotexts = ax.pie(
        tier_counts, labels=tier_labels, colors=colors_pie,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11}
    )
    ax.set_title("Inter-Annotator Agreement Distribution", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "annotation_agreement.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
#  SECTION 6: HATE SPEECH HYBRID vs BASELINE COMPARISON
# ============================================================

def plot_hybrid_comparison(hs_results, out_dir):
    """Compare hybrid models vs best baselines."""
    ensure_dir(out_dir)

    for ds_name in ["binary", "3class"]:
        ds_results = hs_results.get(ds_name, [])
        if not ds_results:
            continue

        baselines = [r for r in ds_results if r.get("phase") == "baseline"]
        hybrids   = [r for r in ds_results if r.get("phase") == "hybrid"]

        if not hybrids:
            continue

        # Sort both by f1
        baselines.sort(key=lambda x: x.get("f1_macro", 0), reverse=True)
        hybrids.sort(key=lambda x: x.get("f1_macro", 0), reverse=True)

        # Top 3 baselines + all hybrids
        show = baselines[:3] + hybrids

        names = []
        f1s = []
        colors = []
        for r in show:
            mname = r.get("model_name", r.get("experiment_id", "?"))
            f1 = r.get("f1_macro", 0)
            if isinstance(f1, float) and f1 < 1:
                f1 *= 100
            phase = r.get("phase", "baseline")
            names.append(mname)
            f1s.append(f1)
            colors.append("#e74c3c" if phase == "hybrid" else "#3498db")

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(range(len(names)), f1s, color=colors,
                      edgecolor="black", linewidth=0.5)

        for bar, score in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{score:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("F1-macro (%)")
        ax.set_title(f"Hate Speech {ds_name} — Hybrid vs Baseline")
        ax.set_ylim(min(f1s) - 3, max(f1s) + 3)

        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#3498db", label="Baseline"),
            Patch(facecolor="#e74c3c", label="Hybrid ★"),
        ], loc="lower right")

        plt.tight_layout()
        path = os.path.join(out_dir, f"hs_{ds_name}_hybrid_vs_baseline.png")
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")


# ============================================================
#  SECTION 7: ALL EXISTING FIGURES COLLECTION
# ============================================================

def collect_existing_figures(out_dir):
    """Copy existing confusion matrices and training curves to paper_figures."""
    import shutil
    ensure_dir(out_dir)

    existing_dir = os.path.join(out_dir, "existing_from_training")
    ensure_dir(existing_dir)

    count = 0

    # Hate speech model figures
    for model_dir in sorted(os.listdir(HS_MODELS_DIR)):
        full_dir = os.path.join(HS_MODELS_DIR, model_dir)
        if not os.path.isdir(full_dir):
            continue
        for fname in ["confusion_matrix.png", "training_curve.png"]:
            src = os.path.join(full_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(existing_dir, f"hs_{model_dir}_{fname}")
                shutil.copy2(src, dst)
                count += 1

    # Hate speech hybrid figures
    if os.path.exists(HS_HYBRID_DIR):
        for hybrid_dir in sorted(os.listdir(HS_HYBRID_DIR)):
            full_dir = os.path.join(HS_HYBRID_DIR, hybrid_dir)
            if not os.path.isdir(full_dir):
                continue
            for fname in os.listdir(full_dir):
                if fname.endswith(".png"):
                    src = os.path.join(full_dir, fname)
                    dst = os.path.join(existing_dir, f"hybrid_{hybrid_dir}_{fname}")
                    shutil.copy2(src, dst)
                    count += 1

    # Sentiment figures
    for eval_dir in [SENT_EVAL_DIR, SENT_MHAMED_DIR]:
        if not os.path.exists(eval_dir):
            continue
        prefix = "sent_mhamed_" if "mhamed" in eval_dir else "sent_"
        for model_dir in sorted(os.listdir(eval_dir)):
            full_dir = os.path.join(eval_dir, model_dir)
            if not os.path.isdir(full_dir):
                continue
            for fname in ["confusion_matrix.png", "training_curves.png"]:
                src = os.path.join(full_dir, fname)
                if os.path.exists(src):
                    dst = os.path.join(existing_dir, f"{prefix}{model_dir}_{fname}")
                    shutil.copy2(src, dst)
                    count += 1

    # Explainability
    explain_dir = os.path.join(HS_MODELS_DIR, "explainability")
    if os.path.exists(explain_dir):
        for root, dirs, files in os.walk(explain_dir):
            for fname in files:
                if fname.endswith(".png") or fname.endswith(".html"):
                    src = os.path.join(root, fname)
                    rel = os.path.relpath(src, explain_dir)
                    dst = os.path.join(existing_dir, f"explain_{rel.replace('/', '_')}")
                    shutil.copy2(src, dst)
                    count += 1

    print(f"  Collected {count} existing figures → {existing_dir}/")


# ============================================================
#  MAIN
# ============================================================

def main():
    ensure_dir(OUT_DIR)

    print("\n" + "=" * 60)
    print("  PAPER 2 — COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    # ---- 1. Hate speech dataset analysis ----
    print("\n\n  SECTION 1: Hate Speech Dataset Analysis")
    print("  " + "-" * 45)

    hs_binary_data = load_tsv(HS_BINARY)
    hs_3class_data = load_tsv(HS_3CLASS)

    ds_dir = os.path.join(OUT_DIR, "dataset_analysis")
    stats_bin = analyze_dataset(hs_binary_data, "HS_Binary", ds_dir)
    stats_3c  = analyze_dataset(hs_3class_data, "HS_3Class", ds_dir)

    plot_class_distribution(stats_bin, "HS_Binary", ds_dir)
    plot_class_distribution(stats_3c,  "HS_3Class", ds_dir)
    plot_text_length_distribution(hs_binary_data, "HS_Binary", ds_dir)
    plot_text_length_distribution(hs_3class_data, "HS_3Class", ds_dir)
    plot_wordcloud(hs_binary_data, "HS_Binary", ds_dir)
    plot_wordcloud(hs_3class_data, "HS_3Class", ds_dir)

    # ---- 2. Sentiment dataset analysis ----
    print("\n\n  SECTION 2: Sentiment Dataset Analysis")
    print("  " + "-" * 45)

    sent_datasets = {
        "Telecom": [
            load_json(os.path.join(SENT_DIR, "telecom_train.json")) +
            load_json(os.path.join(SENT_DIR, "telecom_test.json"))
        ][0],
        "SudSenti2": [
            load_json(os.path.join(SENT_DIR, "sudsenti2_train.json")) +
            load_json(os.path.join(SENT_DIR, "sudsenti2_val.json")) +
            load_json(os.path.join(SENT_DIR, "sudsenti2_test.json"))
        ][0],
        "SudSenti3": [
            load_json(os.path.join(SENT_DIR, "sudsenti3_train.json")) +
            load_json(os.path.join(SENT_DIR, "sudsenti3_val.json")) +
            load_json(os.path.join(SENT_DIR, "sudsenti3_test.json"))
        ][0],
    }

    for name, data in sent_datasets.items():
        stats = analyze_dataset(data, name, ds_dir)
        plot_class_distribution(stats, name, ds_dir)
        plot_text_length_distribution(data, name, ds_dir)
        plot_wordcloud(data, name, ds_dir)

    # ---- 3. Hate speech model results ----
    print("\n\n  SECTION 3: Hate Speech Model Results")
    print("  " + "-" * 45)

    hs_results = collect_hs_results()
    model_dir = os.path.join(OUT_DIR, "model_results")
    plot_hs_confusion_matrices(hs_results, model_dir)
    plot_hs_performance_comparison(hs_results, model_dir)
    plot_hybrid_comparison(hs_results, model_dir)

    # ---- 4. Sentiment model results ----
    print("\n\n  SECTION 4: Sentiment Model Comparison")
    print("  " + "-" * 45)

    sent_minimal = collect_sentiment_results(SENT_EVAL_DIR)
    sent_mhamed  = collect_sentiment_results(SENT_MHAMED_DIR)
    plot_sentiment_comparison(sent_minimal, sent_mhamed, model_dir)

    # ---- 5. Cross-task table ----
    print("\n\n  SECTION 5: Cross-Task Summary")
    print("  " + "-" * 45)

    generate_cross_task_table(hs_results, sent_minimal, OUT_DIR)

    # ---- 6. Annotation agreement ----
    print("\n\n  SECTION 6: Annotation Agreement")
    print("  " + "-" * 45)

    plot_annotation_agreement(OUT_DIR)

    # ---- 7. Collect existing figures ----
    print("\n\n  SECTION 7: Collecting Existing Figures")
    print("  " + "-" * 45)

    collect_existing_figures(OUT_DIR)

    # ---- Done ----
    print(f"\n\n  {'='*60}")
    print(f"  ALL ANALYSIS COMPLETE")
    print(f"  Output: {OUT_DIR}/")
    print(f"  {'='*60}")

    # List generated files
    total = 0
    for root, dirs, files in os.walk(OUT_DIR):
        for f in files:
            total += 1
    print(f"  Total files: {total}")


if __name__ == "__main__":
    main()
