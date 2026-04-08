#!/usr/bin/env python3
"""
Prepare Sentiment Datasets for Model Evaluation
=================================================
Converts 3 sentiment datasets into consistent JSON format:

1. Telecom (Paper 1): data/raw/sudanese_sentiment_*.json → already JSON
2. SudSenti2 (Mhamed et al., 2-class): data/SudSenti/SudSenti2-Tweets.txt
3. SudSenti3 (Mhamed et al., 3-class): data/SudSenti/SudSenti3-Tweets.txt

Output: data/sentiment_prepared/{dataset}_{split}.json
Each file: [{"text": "...", "label": "..."}, ...]

Usage:
    python3 prepare_sentiment_data.py
"""

import json
import os
import re
from collections import Counter

BASE_DIR = os.path.expanduser("~/sudanese_dialect_project")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/sentiment_prepared")


def strip_bom(text):
    """Remove BOM and clean whitespace."""
    return text.lstrip("\ufeff").strip()


def parse_sudsenti_tweets(filepath):
    """
    Parse SudSenti tweet files. Format: text<TAB>label
    SudSenti3 has MULTILINE tweets — must handle carefully.
    Returns list of (text, label) tuples, 1-indexed by original line position.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Remove BOM
    raw = raw.lstrip("\ufeff")
    # Normalize line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    entries = []
    current_text_lines = []
    line_num = 0  # 1-indexed position in original file

    for line in raw.split("\n"):
        line_num += 1

        # Check if this line ends with a tab + label
        # Labels are: pos, neg, obj (possibly with trailing whitespace)
        match = re.search(r'\t(pos|neg|obj|neutral|neural)\s*$', line)

        if match:
            label = match.group(1).strip()
            if label in ("neutral", "neural"):
                label = "obj"
            # Text is everything before the last tab+label
            text_part = line[:match.start()]

            if current_text_lines:
                # This is a multiline tweet — combine previous lines with this one
                current_text_lines.append(text_part)
                full_text = "\n".join(current_text_lines)
            else:
                full_text = text_part

            full_text = full_text.strip()
            if full_text:
                entries.append({
                    "text": full_text,
                    "label": label,
                    "original_line": line_num,
                })
            current_text_lines = []
        else:
            # This line is part of a multiline tweet (no label at end)
            if line.strip():  # Skip empty lines at start
                current_text_lines.append(line)
            elif current_text_lines:
                current_text_lines.append(line)

    return entries


def parse_split_indices(filepath):
    """
    Parse split index files. Each line is a 1-indexed line number.
    Files have BOM, Windows line endings, and possibly empty first line.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    indices = set()
    for line in raw.split("\n"):
        line = line.strip()
        if line and line.isdigit():
            indices.add(int(line))
    return indices


def prepare_telecom():
    """Prepare telecom sentiment dataset (Paper 1)."""
    print("=" * 60)
    print("  DATASET 1: Telecom Sentiment (Paper 1)")
    print("=" * 60)

    train_path = "data/raw/sudanese_sentiment_train.json"
    test_path = "data/raw/sudanese_sentiment_test.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Standardize labels
    label_map = {"POSITIVE": "pos", "NEGATIVE": "neg", "OBJECTIVE": "obj"}

    for item in train_data:
        item["label"] = label_map.get(item["label"], item["label"])
    for item in test_data:
        item["label"] = label_map.get(item["label"], item["label"])

    # Print stats
    train_dist = Counter(d["label"] for d in train_data)
    test_dist = Counter(d["label"] for d in test_data)
    print(f"  Train: {len(train_data)} items — {dict(train_dist)}")
    print(f"  Test:  {len(test_data)} items — {dict(test_dist)}")

    # Save
    out_train = os.path.join(OUTPUT_DIR, "telecom_train.json")
    out_test = os.path.join(OUTPUT_DIR, "telecom_test.json")
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_train}")
    print(f"  Saved: {out_test}")

    return {"name": "telecom", "classes": 3, "labels": ["neg", "obj", "pos"],
            "train": len(train_data), "test": len(test_data)}


def prepare_sudsenti2():
    """Prepare SudSenti2 (2-class: pos/neg)."""
    print()
    print("=" * 60)
    print("  DATASET 2: SudSenti2 (Mhamed et al., 2-class)")
    print("=" * 60)

    data_path = "data/SudSenti/SudSenti2-Tweets.txt"
    train_idx_path = "data/SudSenti/2class_SudSenti2_train.txt"
    val_idx_path = "data/SudSenti/2class-SudSenti2-validation .txt"
    test_idx_path = "data/SudSenti/2class-SudSenti2-test.txt"

    # Parse all tweets
    all_entries = parse_sudsenti_tweets(data_path)
    print(f"  Total entries parsed: {len(all_entries)}")

    # Build index map: original_line -> entry
    # SudSenti2 has simple single-line tweets, so line number = entry index
    # But we need to map the split indices to entries

    # The split files contain LINE NUMBERS from the original file
    # We need to figure out the mapping
    # In SudSenti2, each tweet is on one line, so line N = tweet N

    train_indices = parse_split_indices(train_idx_path)
    val_indices = parse_split_indices(val_idx_path)
    test_indices = parse_split_indices(test_idx_path)

    print(f"  Split indices: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # Map entries by their original line number
    entry_by_line = {e["original_line"]: e for e in all_entries}

    # Build splits
    train_data = [{"text": entry_by_line[i]["text"], "label": entry_by_line[i]["label"]}
                  for i in sorted(train_indices) if i in entry_by_line]
    val_data = [{"text": entry_by_line[i]["text"], "label": entry_by_line[i]["label"]}
                for i in sorted(val_indices) if i in entry_by_line]
    test_data = [{"text": entry_by_line[i]["text"], "label": entry_by_line[i]["label"]}
                 for i in sorted(test_indices) if i in entry_by_line]

    # Check for unmatched indices
    all_lines = set(entry_by_line.keys())
    unmatched_train = len(train_indices - all_lines)
    unmatched_val = len(val_indices - all_lines)
    unmatched_test = len(test_indices - all_lines)

    if unmatched_train or unmatched_val or unmatched_test:
        print(f"  WARNING: Unmatched indices — train:{unmatched_train}, val:{unmatched_val}, test:{unmatched_test}")
        # Try 0-indexed mapping as fallback
        print("  Trying 0-indexed mapping...")
        entry_by_idx = {i: e for i, e in enumerate(all_entries)}

        train_data_0 = [{"text": entry_by_idx[i]["text"], "label": entry_by_idx[i]["label"]}
                        for i in sorted(train_indices) if i in entry_by_idx]
        if len(train_data_0) > len(train_data):
            print("  0-indexed works better, using that")
            train_data = train_data_0
            val_data = [{"text": entry_by_idx[i]["text"], "label": entry_by_idx[i]["label"]}
                        for i in sorted(val_indices) if i in entry_by_idx]
            test_data = [{"text": entry_by_idx[i]["text"], "label": entry_by_idx[i]["label"]}
                         for i in sorted(test_indices) if i in entry_by_idx]

    train_dist = Counter(d["label"] for d in train_data)
    val_dist = Counter(d["label"] for d in val_data)
    test_dist = Counter(d["label"] for d in test_data)
    print(f"  Train: {len(train_data)} items — {dict(train_dist)}")
    print(f"  Val:   {len(val_data)} items — {dict(val_dist)}")
    print(f"  Test:  {len(test_data)} items — {dict(test_dist)}")

    # Save
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out_path = os.path.join(OUTPUT_DIR, f"sudsenti2_{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}")

    return {"name": "sudsenti2", "classes": 2, "labels": ["neg", "pos"],
            "train": len(train_data), "val": len(val_data), "test": len(test_data)}


def prepare_sudsenti3():
    """Prepare SudSenti3 (3-class: pos/neg/obj)."""
    print()
    print("=" * 60)
    print("  DATASET 3: SudSenti3 (Mhamed et al., 3-class)")
    print("=" * 60)

    data_path = "data/SudSenti/SudSenti3-Tweets.txt"
    train_idx_path = "data/SudSenti/3class_SudSenti3_train.txt"
    val_idx_path = "data/SudSenti/3class-SudSenti3-validation .txt"
    test_idx_path = "data/SudSenti/3class-SudSenti3-test.txt"

    # Parse all tweets (handles multiline)
    all_entries = parse_sudsenti_tweets(data_path)
    print(f"  Total entries parsed: {len(all_entries)}")

    dist = Counter(e["label"] for e in all_entries)
    print(f"  Overall distribution: {dict(dist)}")

    # Parse split indices
    train_indices = parse_split_indices(train_idx_path)
    val_indices = parse_split_indices(val_idx_path)
    test_indices = parse_split_indices(test_idx_path)

    print(f"  Split indices: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # Map entries by original line number
    entry_by_line = {e["original_line"]: e for e in all_entries}

    # Build splits
    train_data = [{"text": entry_by_line[i]["text"], "label": entry_by_line[i]["label"]}
                  for i in sorted(train_indices) if i in entry_by_line]
    val_data = [{"text": entry_by_line[i]["text"], "label": entry_by_line[i]["label"]}
                for i in sorted(val_indices) if i in entry_by_line]
    test_data = [{"text": entry_by_line[i]["text"], "label": entry_by_line[i]["label"]}
                 for i in sorted(test_indices) if i in entry_by_line]

    # Check for unmatched
    all_lines = set(entry_by_line.keys())
    unmatched_train = len(train_indices - all_lines)
    unmatched_val = len(val_indices - all_lines)
    unmatched_test = len(test_indices - all_lines)

    if unmatched_train or unmatched_val or unmatched_test:
        print(f"  WARNING: Unmatched indices — train:{unmatched_train}, val:{unmatched_val}, test:{unmatched_test}")
        # The indices might refer to entry index (not line number) for multiline
        print("  Trying entry-index mapping (for multiline tweets)...")
        # Map by sequential entry index (1-indexed)
        entry_by_idx = {i + 1: e for i, e in enumerate(all_entries)}

        train_data_idx = [{"text": entry_by_idx[i]["text"], "label": entry_by_idx[i]["label"]}
                          for i in sorted(train_indices) if i in entry_by_idx]
        if len(train_data_idx) > len(train_data):
            print(f"  Entry-index mapping: {len(train_data_idx)} train (vs {len(train_data)} line-based)")
            train_data = train_data_idx
            val_data = [{"text": entry_by_idx[i]["text"], "label": entry_by_idx[i]["label"]}
                        for i in sorted(val_indices) if i in entry_by_idx]
            test_data = [{"text": entry_by_idx[i]["text"], "label": entry_by_idx[i]["label"]}
                         for i in sorted(test_indices) if i in entry_by_idx]

    # Still unmatched? Try 0-indexed
    if len(train_data) < len(train_indices) * 0.9:
        print("  Trying 0-indexed mapping...")
        entry_by_idx0 = {i: e for i, e in enumerate(all_entries)}
        train_data_0 = [{"text": entry_by_idx0[i]["text"], "label": entry_by_idx0[i]["label"]}
                        for i in sorted(train_indices) if i in entry_by_idx0]
        if len(train_data_0) > len(train_data):
            print(f"  0-indexed: {len(train_data_0)} train")
            train_data = train_data_0
            val_data = [{"text": entry_by_idx0[i]["text"], "label": entry_by_idx0[i]["label"]}
                        for i in sorted(val_indices) if i in entry_by_idx0]
            test_data = [{"text": entry_by_idx0[i]["text"], "label": entry_by_idx0[i]["label"]}
                         for i in sorted(test_indices) if i in entry_by_idx0]

    train_dist = Counter(d["label"] for d in train_data)
    val_dist = Counter(d["label"] for d in val_data)
    test_dist = Counter(d["label"] for d in test_data)
    print(f"  Train: {len(train_data)} items — {dict(train_dist)}")
    print(f"  Val:   {len(val_data)} items — {dict(val_dist)}")
    print(f"  Test:  {len(test_data)} items — {dict(test_dist)}")

    # Save
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out_path = os.path.join(OUTPUT_DIR, f"sudsenti3_{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}")

    return {"name": "sudsenti3", "classes": 3, "labels": ["neg", "obj", "pos"],
            "train": len(train_data), "val": len(val_data), "test": len(test_data)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Preparing All Sentiment Datasets")
    print("=" * 60)

    stats = []
    stats.append(prepare_telecom())
    stats.append(prepare_sudsenti2())
    stats.append(prepare_sudsenti3())

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("  ALL DATASETS PREPARED")
    print("=" * 60)
    for s in stats:
        print(f"  {s['name']}: {s['classes']}-class, labels={s['labels']}")
        print(f"    train={s.get('train', 'N/A')}, val={s.get('val', 'N/A')}, test={s.get('test', 'N/A')}")
    print(f"\n  Output directory: {OUTPUT_DIR}/")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
