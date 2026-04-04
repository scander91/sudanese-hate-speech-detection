#!/usr/bin/env python3
"""
Inter-Annotator Agreement Analysis
====================================
Compares 3 annotators for Sudanese Arabic hate speech:
  1. Weak Supervision v3 (keyword-based labeling functions)
  2. GPT-4o-mini (OpenAI API)
  3. Llama-3.1-70B-Instruct (local GPU)

Metrics implemented from scratch (no sklearn/nltk needed):
  - Cohen's Kappa (pairwise, 3 pairs)
  - Krippendorff's Alpha (all 3 annotators, nominal)
  - Fleiss' Kappa (all 3 annotators)
  - Confusion matrices
  - Per-category agreement
  - Human review tiers

Literature basis:
  - Krippendorff (2019): alpha ≥ 0.80 = reliable, 0.67-0.79 = tentative
  - Landis & Koch (1977): kappa > 0.80 = almost perfect, 0.61-0.80 = substantial

Usage:
    cd ~/sudanese_dialect_project
    python3 agreement_analysis.py

Output:
    data/labeling_corpus/agreement_analysis/
        merged_labels.tsv       — all labels side by side
        human_review.tsv        — sentences needing human review (priority-sorted)
        agreement_summary.json  — all statistics
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime

# ─── FILE PATHS (edit if different on your server) ───
CORPUS_PATH = "data/labeling_corpus/labeling_corpus_25k.json"
WS_TSV_PATH = "data/labeling_corpus/weak_supervision_v3/labeled_corpus_weak.tsv"
GPT_JSON_PATH = "data/labeling_corpus/llm_annotations/labels_gpt_4o_mini.json"
LLAMA_JSON_PATH = "data/labeling_corpus/llm_annotations/labels_llama31_70b.json"
OUTPUT_DIR = "data/labeling_corpus/agreement_analysis"

CATEGORIES = ["HATE", "OFFENSIVE", "NEUTRAL"]
VALID_LABELS = set(CATEGORIES)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_corpus(path):
    """Load the 40K corpus JSON. Returns list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Corpus:       {len(data):,} sentences loaded from {path}")
    return data


def load_ws_labels(path):
    """
    Load weak supervision TSV.
    Expected columns: id, text, source, keyword_cat, weak_label, label_name, confidence, human_label
    Returns dict: {id_str: {"label": str, "confidence": float}}
    """
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")

        # Find column indices dynamically (robust to column order changes)
        col_indices = {name: i for i, name in enumerate(header)}

        # Determine which column has the label
        if "label_name" in col_indices:
            label_col = col_indices["label_name"]
        elif "weak_label" in col_indices:
            label_col = col_indices["weak_label"]
        else:
            print(f"  ❌ Cannot find label column in WS TSV. Columns: {header}")
            sys.exit(1)

        id_col = col_indices.get("id", 0)
        conf_col = col_indices.get("confidence", None)

        # Map numeric labels if needed
        num_to_str = {"0": "HATE", "1": "OFFENSIVE", "2": "NEUTRAL"}

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(id_col, label_col):
                continue

            sid = parts[id_col].strip()
            raw_label = parts[label_col].strip()

            # Convert numeric to string if needed
            label = num_to_str.get(raw_label, raw_label)

            confidence = 0.0
            if conf_col is not None and conf_col < len(parts):
                try:
                    confidence = float(parts[conf_col])
                except (ValueError, IndexError):
                    confidence = 0.0

            labels[sid] = {"label": label, "confidence": confidence}

    valid_count = sum(1 for v in labels.values() if v["label"] in VALID_LABELS)
    print(f"  WS labels:    {len(labels):,} loaded, {valid_count:,} valid ({path})")
    return labels


def load_llm_labels(path, name):
    """
    Load LLM annotation JSON. Handles two formats:
    Format A (list):  [{"id": 1, "llm_label": "HATE", ...}, ...]
    Format B (dict):  {"1": {"label": "HATE", "raw_output": "..."}, ...}
    Returns dict: {id_str: {"label": str}}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    labels = {}
    if isinstance(raw, list):
        # Format A: list of dicts (GPT-4o-mini)
        for item in raw:
            sid = str(item.get("id", ""))
            label = item.get("llm_label", item.get("label", "MISSING"))
            labels[sid] = {"label": label}
    elif isinstance(raw, dict):
        # Format B: dict keyed by ID (Llama)
        for sid, val in raw.items():
            label = val.get("label", "MISSING")
            labels[str(sid)] = {"label": label}
    else:
        print(f"  ERROR: Unknown format in {path}: {type(raw)}")
        return {}

    valid_count = sum(1 for v in labels.values()
                      if v.get("label", "") in VALID_LABELS)
    print(f"  {name:12s}  {len(labels):,} loaded, {valid_count:,} valid ({path})")
    return labels


def align_all_labels(corpus, ws_labels, gpt_labels, llama_labels):
    """
    Align all annotations by sentence ID.
    Returns list of dicts with all labels, and counts of missing/invalid.
    """
    aligned = []
    missing_counts = {"ws": 0, "gpt": 0, "llama": 0}
    invalid_counts = {"ws": 0, "gpt": 0, "llama": 0}

    for i, item in enumerate(corpus):
        sid = str(i + 1)
        text = item.get("text", "")
        source = item.get("source", "")
        keyword_cat = item.get("keyword_category", "")

        # Get labels from each annotator
        ws_entry = ws_labels.get(sid, {})
        gpt_entry = gpt_labels.get(sid, {})
        llama_entry = llama_labels.get(sid, {})

        ws_label = ws_entry.get("label", "MISSING")
        gpt_label = gpt_entry.get("label", "MISSING")
        llama_label = llama_entry.get("label", "MISSING")
        ws_conf = ws_entry.get("confidence", 0.0)

        # Track missing
        if ws_label == "MISSING" or ws_label not in VALID_LABELS:
            missing_counts["ws"] += 1
            if ws_label != "MISSING":
                invalid_counts["ws"] += 1
        if gpt_label == "MISSING" or gpt_label not in VALID_LABELS:
            missing_counts["gpt"] += 1
            if gpt_label != "MISSING":
                invalid_counts["gpt"] += 1
        if llama_label == "MISSING" or llama_label not in VALID_LABELS:
            missing_counts["llama"] += 1
            if llama_label != "MISSING":
                invalid_counts["llama"] += 1

        aligned.append({
            "id": sid,
            "text": text,
            "source": source,
            "keyword_cat": keyword_cat,
            "ws": ws_label,
            "gpt": gpt_label,
            "llama": llama_label,
            "ws_conf": ws_conf,
        })

    print(f"\n  Aligned: {len(aligned):,} sentences")
    print(f"  Missing — WS: {missing_counts['ws']:,}, "
          f"GPT: {missing_counts['gpt']:,}, "
          f"Llama: {missing_counts['llama']:,}")
    if any(v > 0 for v in invalid_counts.values()):
        print(f"  Invalid — WS: {invalid_counts['ws']:,}, "
              f"GPT: {invalid_counts['gpt']:,}, "
              f"Llama: {invalid_counts['llama']:,}")

    return aligned, missing_counts


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: AGREEMENT METRICS (implemented from scratch)
# ═══════════════════════════════════════════════════════════════════════

def cohens_kappa(labels1, labels2):
    """
    Calculate Cohen's Kappa for two annotators.
    Only includes items where both have valid labels.
    Returns: (kappa, observed_agreement, expected_agreement, n_items)
    """
    # Filter to items where both have valid labels
    pairs = [(l1, l2) for l1, l2 in zip(labels1, labels2)
             if l1 in VALID_LABELS and l2 in VALID_LABELS]

    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0

    # Count agreements
    agree = sum(1 for l1, l2 in pairs if l1 == l2)
    p_o = agree / n

    # Count marginals
    count1 = Counter(l1 for l1, _ in pairs)
    count2 = Counter(l2 for _, l2 in pairs)

    # Expected agreement by chance
    p_e = sum((count1.get(c, 0) / n) * (count2.get(c, 0) / n)
              for c in CATEGORIES)

    if p_e >= 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    return kappa, p_o, p_e, n


def fleiss_kappa(annotations):
    """
    Calculate Fleiss' Kappa for 3+ annotators.
    annotations: list of lists, each inner list = [ann1_label, ann2_label, ann3_label]
    Only includes items where ALL annotators have valid labels.
    Returns: (kappa, n_items)
    """
    # Filter to items where all annotators have valid labels
    valid = [row for row in annotations
             if all(label in VALID_LABELS for label in row)]

    n = len(valid)
    if n == 0:
        return 0.0, 0

    k = len(CATEGORIES)  # number of categories
    m = len(valid[0])     # number of annotators per item

    # For each item, count how many annotators chose each category
    # n_ij = count of annotators who assigned category j to item i
    cat_to_idx = {c: j for j, c in enumerate(CATEGORIES)}

    # Compute P_i for each item
    p_items = []
    col_totals = [0] * k  # total assignments per category across all items

    for row in valid:
        counts = [0] * k
        for label in row:
            counts[cat_to_idx[label]] += 1
        for j in range(k):
            col_totals[j] += counts[j]
        # P_i = (1 / (m*(m-1))) * (sum(n_ij^2) - m)
        p_i = (sum(c * c for c in counts) - m) / (m * (m - 1))
        p_items.append(p_i)

    # P_bar = mean of P_i
    p_bar = sum(p_items) / n

    # P_e = sum of (p_j)^2 where p_j = proportion of all assignments to category j
    total_assignments = n * m
    p_e = sum((col_totals[j] / total_assignments) ** 2 for j in range(k))

    if p_e >= 1.0:
        kappa = 1.0
    else:
        kappa = (p_bar - p_e) / (1 - p_e)

    return kappa, n


def krippendorff_alpha_nominal(annotations):
    """
    Calculate Krippendorff's Alpha for nominal data with 3+ annotators.
    Uses the coincidence matrix approach (Krippendorff, 2019).

    annotations: list of lists, each inner list = [ann1_label, ann2_label, ...]
                 Values can be in VALID_LABELS or "MISSING" (treated as missing)
    Returns: (alpha, n_items_used)

    Formula: alpha = 1 - D_o / D_e
    D_o = observed disagreement (from coincidence matrix)
    D_e = expected disagreement (from marginal frequencies)
    """
    # Build coincidence matrix
    # For each unit with m_u ≥ 2 valid values, add 1/(m_u - 1) to each pair
    cat_list = list(CATEGORIES)
    cat_idx = {c: i for i, c in enumerate(cat_list)}
    n_cat = len(cat_list)

    # Coincidence matrix (symmetric)
    coincidence = [[0.0] * n_cat for _ in range(n_cat)]
    n_pairable = 0  # total number of pairable values
    n_units_used = 0

    for row in annotations:
        valid_values = [v for v in row if v in VALID_LABELS]
        m_u = len(valid_values)
        if m_u < 2:
            continue

        n_units_used += 1
        n_pairable += m_u

        # For each pair of values in this unit, add to coincidence matrix
        weight = 1.0 / (m_u - 1)
        for i in range(m_u):
            for j in range(m_u):
                if i != j:
                    ci = cat_idx[valid_values[i]]
                    cj = cat_idx[valid_values[j]]
                    coincidence[ci][cj] += weight

    if n_pairable < 2 or n_units_used == 0:
        return 0.0, 0

    # Marginal frequencies from coincidence matrix
    # n_c = sum of row c in coincidence matrix
    n_c = [sum(coincidence[c]) for c in range(n_cat)]
    n_total = sum(n_c)  # should equal n_pairable

    if n_total <= 1:
        return 0.0, 0

    # D_o: observed disagreement
    # = 1 - (sum of diagonal) / n_total
    diag_sum = sum(coincidence[c][c] for c in range(n_cat))
    d_o = 1.0 - diag_sum / n_total

    # D_e: expected disagreement for nominal metric
    # = 1 - sum(n_c * (n_c - 1)) / (n_total * (n_total - 1))
    d_e = 1.0 - sum(n_c[c] * (n_c[c] - 1) for c in range(n_cat)) / (n_total * (n_total - 1))

    if d_e == 0:
        return 1.0, n_units_used  # perfect agreement

    alpha = 1.0 - d_o / d_e
    return alpha, n_units_used


def confusion_matrix(labels1, labels2, name1, name2):
    """
    Build and format a confusion matrix between two annotators.
    Returns: (matrix_dict, formatted_string)
    """
    # Filter to valid pairs
    pairs = [(l1, l2) for l1, l2 in zip(labels1, labels2)
             if l1 in VALID_LABELS and l2 in VALID_LABELS]

    matrix = {}
    for c1 in CATEGORIES:
        for c2 in CATEGORIES:
            matrix[(c1, c2)] = 0
    for l1, l2 in pairs:
        matrix[(l1, l2)] += 1

    # Format as string
    lines = []
    # Header
    short = {"HATE": "HATE", "OFFENSIVE": "OFF", "NEUTRAL": "NEU"}
    header = f"{'':>15s}"
    for c2 in CATEGORIES:
        header += f"  {short[c2]:>6s}"
    header += f"  {'Total':>6s}"
    lines.append(f"  {name1} vs {name2}:")
    lines.append(f"  {'':>15s}{name2:>{len(CATEGORIES)*8}s}")
    lines.append(f"  {header}")
    lines.append(f"  {'─' * (15 + len(CATEGORIES) * 8 + 8)}")

    for c1 in CATEGORIES:
        row = f"  {name1:>5s} {short[c1]:>8s}"
        row_total = 0
        for c2 in CATEGORIES:
            val = matrix[(c1, c2)]
            row_total += val
            row += f"  {val:>6,}"
        row += f"  {row_total:>6,}"
        lines.append(row)

    # Column totals
    total_row = f"  {'Total':>14s}"
    grand_total = 0
    for c2 in CATEGORIES:
        col_sum = sum(matrix[(c1, c2)] for c1 in CATEGORIES)
        total_row += f"  {col_sum:>6,}"
        grand_total += col_sum
    total_row += f"  {grand_total:>6,}"
    lines.append(f"  {'─' * (15 + len(CATEGORIES) * 8 + 8)}")
    lines.append(total_row)

    return matrix, "\n".join(lines)


def interpret_kappa(kappa):
    """Interpret kappa value (Landis & Koch, 1977)."""
    if kappa > 0.80:
        return "Almost Perfect"
    elif kappa > 0.60:
        return "Substantial"
    elif kappa > 0.40:
        return "Moderate"
    elif kappa > 0.20:
        return "Fair"
    else:
        return "Slight/Poor"


def interpret_alpha(alpha):
    """Interpret Krippendorff's alpha (Krippendorff, 2019)."""
    if alpha >= 0.80:
        return "Reliable"
    elif alpha >= 0.67:
        return "Tentative"
    else:
        return "Unreliable"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: AGREEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_agreement(aligned):
    """
    Comprehensive agreement analysis.
    Returns dict with all statistics.
    """
    stats = {}

    # Extract label vectors
    ws_labels = [item["ws"] for item in aligned]
    gpt_labels = [item["gpt"] for item in aligned]
    llama_labels = [item["llama"] for item in aligned]

    # ── Per-annotator distribution ──
    stats["distributions"] = {}
    for name, labels in [("WS", ws_labels), ("GPT-4o-mini", gpt_labels),
                         ("Llama-70B", llama_labels)]:
        dist = Counter(labels)
        stats["distributions"][name] = {c: dist.get(c, 0) for c in CATEGORIES + ["MISSING"]}

    # ── Pairwise Cohen's Kappa ──
    pairs = [
        ("WS", "GPT-4o-mini", ws_labels, gpt_labels),
        ("WS", "Llama-70B", ws_labels, llama_labels),
        ("GPT-4o-mini", "Llama-70B", gpt_labels, llama_labels),
    ]

    stats["cohens_kappa"] = {}
    stats["confusion_matrices"] = {}
    for name1, name2, l1, l2 in pairs:
        kappa, p_o, p_e, n = cohens_kappa(l1, l2)
        pair_key = f"{name1} vs {name2}"
        stats["cohens_kappa"][pair_key] = {
            "kappa": round(kappa, 4),
            "observed_agreement": round(p_o, 4),
            "expected_agreement": round(p_e, 4),
            "n_items": n,
            "interpretation": interpret_kappa(kappa),
        }
        _, cm_str = confusion_matrix(l1, l2, name1, name2)
        stats["confusion_matrices"][pair_key] = cm_str

    # ── Fleiss' Kappa (all 3) ──
    all_annotations = list(zip(ws_labels, gpt_labels, llama_labels))
    fleiss_k, fleiss_n = fleiss_kappa(all_annotations)
    stats["fleiss_kappa"] = {
        "kappa": round(fleiss_k, 4),
        "n_items": fleiss_n,
        "interpretation": interpret_kappa(fleiss_k),
    }

    # ── Krippendorff's Alpha (all 3) ──
    alpha, alpha_n = krippendorff_alpha_nominal(all_annotations)
    stats["krippendorff_alpha"] = {
        "alpha": round(alpha, 4),
        "n_items": alpha_n,
        "interpretation": interpret_alpha(alpha),
    }

    # ── Agreement levels ──
    full_agree = 0
    two_agree = 0
    no_agree = 0
    majority_labels = {}  # for each pattern, count

    for item in aligned:
        labels = [item["ws"], item["gpt"], item["llama"]]
        valid = [l for l in labels if l in VALID_LABELS]

        if len(valid) < 2:
            no_agree += 1
            continue

        if len(set(valid)) == 1:
            full_agree += 1
        elif len(valid) == 3 and len(set(valid)) == 2:
            two_agree += 1
        else:
            no_agree += 1

        # Track agreement patterns
        pattern = f"{item['ws']}/{item['gpt']}/{item['llama']}"
        majority_labels[pattern] = majority_labels.get(pattern, 0) + 1

    stats["agreement_levels"] = {
        "full_agreement_3of3": full_agree,
        "partial_agreement_2of3": two_agree,
        "no_agreement_0of3": no_agree,
        "total": len(aligned),
        "full_pct": round(100 * full_agree / len(aligned), 1),
        "partial_pct": round(100 * two_agree / len(aligned), 1),
        "no_pct": round(100 * no_agree / len(aligned), 1),
    }

    # Top 20 most common label patterns
    stats["top_patterns"] = sorted(majority_labels.items(),
                                   key=lambda x: -x[1])[:20]

    # ── Per-category agreement ──
    # For each category, what % of times an annotator labels it,
    # do the other annotators agree?
    stats["per_category"] = {}
    for cat in CATEGORIES:
        # How many times each annotator used this category
        ws_count = sum(1 for l in ws_labels if l == cat)
        gpt_count = sum(1 for l in gpt_labels if l == cat)
        llama_count = sum(1 for l in llama_labels if l == cat)

        # When all 3 agree on this category
        all_agree_cat = sum(1 for item in aligned
                           if item["ws"] == cat and item["gpt"] == cat
                           and item["llama"] == cat)

        # When at least 2 of 3 agree on this category
        at_least_2 = sum(1 for item in aligned
                         if sum(1 for k in ["ws", "gpt", "llama"]
                                if item[k] == cat) >= 2)

        stats["per_category"][cat] = {
            "ws_count": ws_count,
            "gpt_count": gpt_count,
            "llama_count": llama_count,
            "all_3_agree": all_agree_cat,
            "at_least_2_agree": at_least_2,
        }

    return stats


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: HUMAN REVIEW TIERS
# ═══════════════════════════════════════════════════════════════════════

def create_review_tiers(aligned):
    """
    Categorize sentences into human review tiers.
    Returns list of items with added 'tier' and 'majority_label' fields.
    """
    for item in aligned:
        labels = [item["ws"], item["gpt"], item["llama"]]
        valid = [l for l in labels if l in VALID_LABELS]

        if len(valid) < 2:
            item["tier"] = 3  # no consensus possible
            item["majority_label"] = "UNKNOWN"
            item["agree_count"] = 0
            continue

        # Count votes per label
        votes = Counter(valid)
        most_common_label, most_common_count = votes.most_common(1)[0]

        if len(valid) == 3 and most_common_count == 3:
            item["tier"] = 1  # all 3 agree
            item["majority_label"] = most_common_label
            item["agree_count"] = 3
        elif most_common_count >= 2:
            item["tier"] = 2  # 2 of 3 agree
            item["majority_label"] = most_common_label
            item["agree_count"] = 2
        else:
            item["tier"] = 3  # all 3 disagree
            item["majority_label"] = "NO_MAJORITY"
            item["agree_count"] = 0

    # Count tiers
    tier_counts = Counter(item["tier"] for item in aligned)
    tier_label_dist = {}
    for tier in [1, 2, 3]:
        tier_items = [item for item in aligned if item["tier"] == tier]
        tier_label_dist[tier] = Counter(item["majority_label"]
                                        for item in tier_items)

    return aligned, tier_counts, tier_label_dist


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: OUTPUT
# ═══════════════════════════════════════════════════════════════════════

def save_merged_tsv(aligned, path):
    """Save all labels side by side as TSV."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("id\ttext\tsource\tkeyword_cat\t"
                "ws_label\tgpt_label\tllama_label\t"
                "tier\tmajority_label\tagree_count\tws_confidence\n")
        for item in aligned:
            text = item["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{item['id']}\t{text}\t{item['source']}\t"
                    f"{item['keyword_cat']}\t"
                    f"{item['ws']}\t{item['gpt']}\t{item['llama']}\t"
                    f"{item['tier']}\t{item['majority_label']}\t"
                    f"{item['agree_count']}\t{item['ws_conf']}\n")


def save_human_review(aligned, path):
    """
    Save sentences that need human review, sorted by priority:
    Tier 3 first (complete disagreement), then Tier 2 with low confidence.
    """
    review_items = [item for item in aligned if item["tier"] >= 2]

    # Sort: Tier 3 first (most urgent), then within same tier by ws_conf ascending
    review_items.sort(key=lambda x: (-x["tier"], x["ws_conf"]))

    with open(path, "w", encoding="utf-8") as f:
        f.write("priority\tid\ttext\tsource\tkeyword_cat\t"
                "ws_label\tgpt_label\tllama_label\t"
                "tier\tmajority_label\thuman_label\n")
        for i, item in enumerate(review_items, 1):
            text = item["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{i}\t{item['id']}\t{text}\t{item['source']}\t"
                    f"{item['keyword_cat']}\t"
                    f"{item['ws']}\t{item['gpt']}\t{item['llama']}\t"
                    f"{item['tier']}\t{item['majority_label']}\t\n")

    return len(review_items)


def save_summary_json(stats, tier_counts, tier_label_dist, path):
    """Save all statistics as JSON."""
    # Convert non-serializable items
    summary = {
        "timestamp": datetime.now().isoformat(),
        "distributions": stats["distributions"],
        "cohens_kappa": stats["cohens_kappa"],
        "fleiss_kappa": stats["fleiss_kappa"],
        "krippendorff_alpha": stats["krippendorff_alpha"],
        "agreement_levels": stats["agreement_levels"],
        "per_category": stats["per_category"],
        "top_patterns": stats["top_patterns"],
        "tier_counts": {str(k): v for k, v in tier_counts.items()},
        "tier_label_dist": {
            str(k): dict(v) for k, v in tier_label_dist.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_report(stats, tier_counts, tier_label_dist):
    """Print comprehensive report to console."""
    print(f"\n{'='*75}")
    print(f" INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*75}")

    # ── Distributions ──
    print(f"\n{'─'*75}")
    print(f" LABEL DISTRIBUTIONS")
    print(f"{'─'*75}")
    total_n = stats["agreement_levels"]["total"]
    print(f"  {'Label':<12s}  {'WS':>8s}  {'GPT-4o-mini':>12s}  {'Llama-70B':>10s}")
    print(f"  {'─'*50}")
    for cat in CATEGORIES:
        ws_n = stats["distributions"]["WS"].get(cat, 0)
        gpt_n = stats["distributions"]["GPT-4o-mini"].get(cat, 0)
        llama_n = stats["distributions"]["Llama-70B"].get(cat, 0)
        ws_pct = 100 * ws_n / total_n if total_n > 0 else 0
        gpt_pct = 100 * gpt_n / total_n if total_n > 0 else 0
        llama_pct = 100 * llama_n / total_n if total_n > 0 else 0
        print(f"  {cat:<12s}  {ws_n:>6,} ({ws_pct:4.1f}%)  "
              f"{gpt_n:>6,} ({gpt_pct:4.1f}%)  "
              f"{llama_n:>6,} ({llama_pct:4.1f}%)")

    # ── Pairwise Kappa ──
    print(f"\n{'─'*75}")
    print(f" PAIRWISE COHEN'S KAPPA")
    print(f"{'─'*75}")
    for pair_key, kdata in stats["cohens_kappa"].items():
        print(f"  {pair_key}:")
        print(f"    κ = {kdata['kappa']:.4f}  ({kdata['interpretation']})")
        print(f"    Observed agreement: {kdata['observed_agreement']:.4f}  "
              f"({kdata['observed_agreement']*100:.1f}%)")
        print(f"    Expected by chance: {kdata['expected_agreement']:.4f}  "
              f"({kdata['expected_agreement']*100:.1f}%)")
        print(f"    Items compared: {kdata['n_items']:,}")
        print()

    # ── Fleiss' Kappa ──
    print(f"{'─'*75}")
    print(f" MULTI-ANNOTATOR METRICS")
    print(f"{'─'*75}")
    fk = stats["fleiss_kappa"]
    print(f"  Fleiss' Kappa (all 3):       κ = {fk['kappa']:.4f}  "
          f"({fk['interpretation']})  [n={fk['n_items']:,}]")

    ka = stats["krippendorff_alpha"]
    print(f"  Krippendorff's Alpha (all 3): α = {ka['alpha']:.4f}  "
          f"({ka['interpretation']})  [n={ka['n_items']:,}]")

    # ── Confusion Matrices ──
    print(f"\n{'─'*75}")
    print(f" CONFUSION MATRICES")
    print(f"{'─'*75}")
    for pair_key, cm_str in stats["confusion_matrices"].items():
        print(f"\n{cm_str}")
        print()

    # ── Agreement Levels ──
    print(f"{'─'*75}")
    print(f" AGREEMENT LEVELS")
    print(f"{'─'*75}")
    al = stats["agreement_levels"]
    print(f"  Full agreement  (3/3): {al['full_agreement_3of3']:>7,}  "
          f"({al['full_pct']:5.1f}%)")
    print(f"  Partial agree   (2/3): {al['partial_agreement_2of3']:>7,}  "
          f"({al['partial_pct']:5.1f}%)")
    print(f"  No agreement    (0/3): {al['no_agreement_0of3']:>7,}  "
          f"({al['no_pct']:5.1f}%)")
    print(f"  Total:                 {al['total']:>7,}")

    # ── Per-Category Agreement ──
    print(f"\n{'─'*75}")
    print(f" PER-CATEGORY AGREEMENT")
    print(f"{'─'*75}")
    for cat in CATEGORIES:
        pc = stats["per_category"][cat]
        print(f"  {cat}:")
        print(f"    Used by — WS: {pc['ws_count']:,}, "
              f"GPT: {pc['gpt_count']:,}, Llama: {pc['llama_count']:,}")
        print(f"    All 3 agree:    {pc['all_3_agree']:>6,}")
        print(f"    At least 2/3:   {pc['at_least_2_agree']:>6,}")

    # ── Human Review Tiers ──
    print(f"\n{'─'*75}")
    print(f" HUMAN REVIEW TIERS")
    print(f"{'─'*75}")
    for tier in [1, 2, 3]:
        count = tier_counts.get(tier, 0)
        pct = 100 * count / total_n if total_n > 0 else 0
        if tier == 1:
            desc = "Full agreement   — spot-check 5-10%"
        elif tier == 2:
            desc = "Partial agreement — review 20-30%"
        else:
            desc = "No agreement     — human labels ALL"
        print(f"  Tier {tier}: {count:>7,} ({pct:5.1f}%)  {desc}")

        # Label distribution within tier
        dist = tier_label_dist.get(tier, {})
        if dist:
            dist_str = ", ".join(f"{k}: {v:,}" for k, v in
                                 sorted(dist.items(), key=lambda x: -x[1])
                                 if v > 0)
            print(f"         Labels: {dist_str}")

    # ── Top Label Patterns ──
    print(f"\n{'─'*75}")
    print(f" TOP 15 LABEL PATTERNS (WS / GPT / Llama)")
    print(f"{'─'*75}")
    for pattern, count in stats["top_patterns"][:15]:
        pct = 100 * count / total_n if total_n > 0 else 0
        print(f"  {pattern:<35s}  {count:>6,}  ({pct:5.1f}%)")

    # ── Human Effort Estimate ──
    print(f"\n{'─'*75}")
    print(f" ESTIMATED HUMAN REVIEW EFFORT")
    print(f"{'─'*75}")
    tier1 = tier_counts.get(1, 0)
    tier2 = tier_counts.get(2, 0)
    tier3 = tier_counts.get(3, 0)
    spot_check = int(tier1 * 0.05)  # 5% of tier 1
    partial_review = int(tier2 * 0.25)  # 25% of tier 2
    full_review = tier3  # 100% of tier 3
    total_human = spot_check + partial_review + full_review
    print(f"  Tier 1 spot-check (5%):   {spot_check:>6,}")
    print(f"  Tier 2 review (25%):      {partial_review:>6,}")
    print(f"  Tier 3 full review (100%): {full_review:>6,}")
    print(f"  ────────────────────────────────")
    print(f"  Total human effort:       {total_human:>6,} sentences")
    print(f"  (vs labeling all {total_n:,} from scratch)")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*75}")
    print(f" Loading annotation data...")
    print(f"{'='*75}")

    # Check all files exist before loading
    for path, name in [(CORPUS_PATH, "Corpus"),
                       (WS_TSV_PATH, "Weak Supervision TSV"),
                       (GPT_JSON_PATH, "GPT-4o-mini JSON"),
                       (LLAMA_JSON_PATH, "Llama-70B JSON")]:
        if not os.path.exists(path):
            print(f"\n  ❌ File not found: {path}")
            print(f"     Expected: {name}")
            print(f"\n  Check the paths at the top of this script.")
            sys.exit(1)

    # Load data
    corpus = load_corpus(CORPUS_PATH)
    ws_labels = load_ws_labels(WS_TSV_PATH)
    gpt_labels = load_llm_labels(GPT_JSON_PATH, "GPT-4o-mini")
    llama_labels = load_llm_labels(LLAMA_JSON_PATH, "Llama-70B")

    # Align
    aligned, missing_counts = align_all_labels(corpus, ws_labels,
                                                gpt_labels, llama_labels)

    # Analyze
    print(f"\n  Computing agreement metrics...")
    stats = analyze_agreement(aligned)

    # Create tiers
    aligned, tier_counts, tier_label_dist = create_review_tiers(aligned)

    # Print report
    print_report(stats, tier_counts, tier_label_dist)

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_path = os.path.join(OUTPUT_DIR, "merged_labels.tsv")
    review_path = os.path.join(OUTPUT_DIR, "human_review.tsv")
    summary_path = os.path.join(OUTPUT_DIR, "agreement_summary.json")

    save_merged_tsv(aligned, merged_path)
    n_review = save_human_review(aligned, review_path)
    save_summary_json(stats, tier_counts, tier_label_dist, summary_path)

    print(f"\n{'─'*75}")
    print(f" OUTPUT FILES")
    print(f"{'─'*75}")
    print(f"  📁 {merged_path}   ({len(aligned):,} rows)")
    print(f"  📁 {review_path}    ({n_review:,} rows)")
    print(f"  📁 {summary_path}")
    print(f"\n{'='*75}")
    print(f" DONE")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
