#!/usr/bin/env python3
"""
Llama-3.1-70B-Instruct Annotation Script for Sudanese Hate Speech
=================================================================
Annotator 3 (after GPT-4o-mini as Annotator 1)

Literature basis:
- Abbass & Faili (2025): LLaMA-3 for Arabic hate speech detection
- Das et al. (2024): LLM annotation with multiple models
- LLaMA-70B achieves F1=0.836 for hate speech detection

Hardware: GPU 1 (RTX 8000, 49GB) with 4-bit NF4 quantization (~35-40GB VRAM)
Estimated runtime: ~24-36 hours for 40K sentences (1 sentence/inference)

Usage:
    # Test with 5 sentences first
    CUDA_VISIBLE_DEVICES=1 python3 llama_annotate.py --test

    # Full run (use nohup!)
    nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 -u llama_annotate.py > llama_run.log 2>&1 &

    # Resume if interrupted
    nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 -u llama_annotate.py --resume > llama_run.log 2>&1 &

    # Use 8B model for faster processing (~4-6 hours)
    nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 -u llama_annotate.py --model-size 8b > llama_run.log 2>&1 &
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# ─── MUST be before torch/transformers imports ───
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("⚠️  CUDA_VISIBLE_DEVICES not set. Defaulting to GPU 1.")
    print("   Run with: CUDA_VISIBLE_DEVICES=1 python3 llama_annotate.py")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ─── CONFIGURATION ───
MODEL_IDS = {
    "70b": "meta-llama/Llama-3.1-70B-Instruct",
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
}

CORPUS_PATH = "data/labeling_corpus/labeling_corpus_25k.json"
OUTPUT_DIR = "data/labeling_corpus/llm_annotations"

LABEL_MAP = {"HATE": "HATE", "OFFENSIVE": "OFFENSIVE", "NEUTRAL": "NEUTRAL",
             "0": "HATE", "1": "OFFENSIVE", "2": "NEUTRAL"}

VALID_LABELS = {"HATE", "OFFENSIVE", "NEUTRAL"}

# ─── SAME PROMPT AS GPT-4o-mini (for consistency) ───
SYSTEM_PROMPT = """أنت خبير في تصنيف خطاب الكراهية في اللهجة العربية السودانية.

صنّف كل جملة إلى واحدة من ثلاث فئات:

HATE = خطاب كراهية يستهدف مجموعة بسبب هويتها (قبيلة، عرق، دين، انتماء سياسي). يتضمن التجريد من الإنسانية أو التحريض على العنف ضد مجموعة.
OFFENSIVE = لغة مسيئة أو بذيئة لكنها لا تستهدف مجموعة بسبب هويتها. شتائم شخصية، ألفاظ نابية، إهانات فردية.
NEUTRAL = محايد. أخبار، تقارير، محادثات عادية، محتوى إنساني — حتى لو ذكر الحرب أو العنف كتقرير.

الفرق الحاسم: HATE يستهدف مجموعة (قبيلة/عرق/دين). OFFENSIVE يستهدف فرد أو عام بدون استهداف هوية.

أمثلة:
1. "الجنجويد ديل حيوانات لازم نخلص منهم" → HATE (تجريد مجموعة من الإنسانية)
2. "كسمك يا غبي" → OFFENSIVE (شتيمة شخصية بدون استهداف مجموعة)
3. "الجيش يسيطر على مناطق جديدة في الخرطوم" → NEUTRAL (تقرير إخباري)
4. "الشايقية ديل كلهم خونة وعملاء" → HATE (استهداف قبيلة كاملة)
5. "هسع الكهرباء قطعت تاني يا ناس" → NEUTRAL (محادثة يومية)

أجب بكلمة واحدة فقط: HATE أو OFFENSIVE أو NEUTRAL
لا تكتب أي شيء آخر. كلمة واحدة فقط."""


def build_user_prompt(sentence):
    """Build user prompt for a single sentence."""
    return f"صنّف هذه الجملة:\n\n{sentence}\n\nالتصنيف:"


def load_corpus(path):
    """Load the 40K corpus."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} sentences")
    return data


def load_model(model_size="70b"):
    """Load Llama model with 4-bit quantization."""
    model_id = MODEL_IDS[model_size]
    print(f"\n  Loading {model_id} with 4-bit NF4 quantization...")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    start = time.time()

    # 4-bit NF4 quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # saves ~0.4 bits/param
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        token=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    elapsed = time.time() - start
    vram_used = torch.cuda.memory_allocated(0) / 1e9
    print(f"  ✅ Model loaded in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  VRAM used: {vram_used:.1f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return model, tokenizer


def classify_sentence(model, tokenizer, sentence):
    """
    Classify a single sentence. Returns (label, raw_output).
    Uses chat template for proper Llama 3.1 formatting.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sentence)},
    ]

    # One-step tokenization (avoids double BOS token)
    # This is the approach used in HuggingFace's official Llama 3.1 examples
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,       # deterministic for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the NEW tokens (exclude the input)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse label from output
    label = parse_label(raw_output)

    return label, raw_output


def parse_label(raw_output):
    """Extract label from model output. Handles various formats."""
    text = raw_output.strip().upper()

    # Direct match
    if text in VALID_LABELS:
        return text

    # Check if label appears anywhere in output
    for label in ["HATE", "OFFENSIVE", "NEUTRAL"]:
        if label in text:
            return label

    # Check numeric
    for char in text:
        if char in LABEL_MAP:
            mapped = LABEL_MAP[char]
            if mapped in VALID_LABELS:
                return mapped

    return "MISSING"


def save_progress(results, progress_path, current_idx, total, start_time,
                  model_size, errors):
    """Save progress for resume capability."""
    progress = {
        "current_idx": current_idx,
        "total": total,
        "model_size": model_size,
        "timestamp": datetime.now().isoformat(),
        "errors": errors,
        "labeled_count": len(results),
        "elapsed_seconds": time.time() - start_time,
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f)

    # Also save current results
    results_path = progress_path.replace("progress_", "partial_labels_")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_final_results(results, corpus, model_size, elapsed, errors):
    """Save final results in same format as GPT-4o-mini output."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_tag = f"llama31_{model_size}"

    # JSON
    json_path = os.path.join(OUTPUT_DIR, f"labels_{model_tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # TSV (same format as GPT-4o-mini)
    tsv_path = os.path.join(OUTPUT_DIR, f"labels_{model_tag}.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("id\ttext\tsource\tkeyword_cat\tlabel\traw_output\n")
        for item in corpus:
            sid = str(item.get("id", ""))
            text = item.get("text", "").replace("\t", " ").replace("\n", " ")
            source = item.get("source", "")
            kcat = item.get("keyword_category", "")
            label = results.get(sid, {}).get("label", "MISSING")
            raw = results.get(sid, {}).get("raw_output", "").replace("\t", " ").replace("\n", " ")
            f.write(f"{sid}\t{text}\t{source}\t{kcat}\t{label}\t{raw}\n")

    # Summary
    dist = {}
    for v in results.values():
        lbl = v.get("label", "MISSING")
        dist[lbl] = dist.get(lbl, 0) + 1

    print(f"\n{'='*70}")
    print(f" DONE — Llama-3.1-{model_size.upper()}-Instruct")
    print(f"{'='*70}")
    print(f"  Sentences:     {len(corpus):,}")
    print(f"  Labeled:       {len(results):,}")
    print(f"  Missing:       {dist.get('MISSING', 0)}")
    print(f"  Errors:        {errors}")
    print(f"  Time:          {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"\n  Distribution:")
    for label in ["HATE", "OFFENSIVE", "NEUTRAL", "MISSING"]:
        if label in dist:
            pct = 100 * dist[label] / len(results) if results else 0
            print(f"    {label:15s} {dist[label]:>6,} ({pct:5.1f}%)")
    print(f"\n  📁 {json_path}")
    print(f"  📁 {tsv_path}")

    return json_path, tsv_path


def run_test(model, tokenizer):
    """Test with 5 sentences. Verify labels before full run."""
    test_sentences = [
        ("الجنجويد ديل حيوانات لازم نخلص منهم", "HATE"),
        ("كسمك يا غبي", "OFFENSIVE"),
        ("عاجل: الجيش يسيطر على مناطق جديدة في الخرطوم", "NEUTRAL"),
        ("الشايقية ديل كلهم خونة وعملاء", "HATE"),
        ("هسع الكهرباء قطعت تاني يا ناس", "NEUTRAL"),
    ]

    print(f"\n{'='*70}")
    print(f" TEST MODE — 5 sentences")
    print(f"{'='*70}")

    correct = 0
    total_time = 0

    for i, (sentence, expected) in enumerate(test_sentences, 1):
        start = time.time()
        label, raw_output = classify_sentence(model, tokenizer, sentence)
        elapsed = time.time() - start
        total_time += elapsed

        match = "✅" if label == expected else "❌"
        if label == expected:
            correct += 1

        print(f"  {i}  {match}  Expected={expected:10s}  Got={label:10s}  "
              f"Raw='{raw_output}'  ({elapsed:.1f}s)")
        # Truncate sentence for display
        display_sent = sentence[:60] + "..." if len(sentence) > 60 else sentence
        print(f"       {display_sent}")

    avg_time = total_time / len(test_sentences)
    est_total = avg_time * 40000
    print(f"\n  Accuracy: {correct}/{len(test_sentences)}")
    print(f"  Avg time per sentence: {avg_time:.1f}s")
    print(f"  Estimated full corpus time: {est_total/3600:.1f} hours")

    if correct >= 4:
        print(f"\n  ✅ Test PASSED ({correct}/5). Safe to run full corpus.")
        print(f"  Run: nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "
              f"python3 -u llama_annotate.py > llama_run.log 2>&1 &")
    else:
        print(f"\n  ❌ Test FAILED ({correct}/5). Do NOT run full corpus.")
        print(f"  Check the raw outputs above for issues.")

    return correct >= 4


def run_full(model, tokenizer, model_size, resume=False):
    """Annotate all 40K sentences."""
    corpus = load_corpus(CORPUS_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_tag = f"llama31_{model_size}"
    progress_path = os.path.join(OUTPUT_DIR, f"progress_{model_tag}.json")
    partial_path = os.path.join(OUTPUT_DIR, f"partial_labels_{model_tag}.json")

    # Resume handling
    results = {}
    start_idx = 0
    if resume and os.path.exists(progress_path) and os.path.exists(partial_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        with open(partial_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = progress["current_idx"]
        print(f"  Resuming: sentence {start_idx:,}, {len(results):,} labels done")

    total = len(corpus)
    errors = 0
    oom_count = 0
    session_labeled = 0  # Track labels in THIS session for accurate rate
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f" FULL ANNOTATION — Llama-3.1-{model_size.upper()}-Instruct")
    print(f" Sentences: {total:,} | Starting from: {start_idx:,}")
    print(f" Previously labeled: {len(results):,}")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    for i in range(start_idx, total):
        item = corpus[i]
        sid = str(item.get("id", i + 1))
        sentence = item.get("text", "")

        if sid in results:
            continue

        # Skip empty sentences
        if not sentence or not sentence.strip():
            results[sid] = {"label": "NEUTRAL", "raw_output": "EMPTY_SENTENCE"}
            session_labeled += 1
            continue

        try:
            label, raw_output = classify_sentence(model, tokenizer, sentence)
            results[sid] = {"label": label, "raw_output": raw_output}
            session_labeled += 1
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            torch.cuda.empty_cache()
            if oom_count >= 3:
                print(f"\n  ❌ GPU OUT OF MEMORY — 3 consecutive OOM errors.")
                print(f"  Saving progress and exiting. Resume with --resume")
                save_progress(results, progress_path, i, total,
                              start_time, model_size, errors)
                sys.exit(1)
            results[sid] = {"label": "MISSING", "raw_output": "OOM_ERROR"}
            errors += 1
        except Exception as e:
            errors += 1
            oom_count = 0  # Reset OOM counter on non-OOM error
            results[sid] = {"label": "MISSING", "raw_output": f"ERROR: {str(e)}"}
            if errors <= 5:
                print(f"    ⚠️  Error at sentence {i}: {str(e)[:100]}")

        # Progress update every 500 sentences
        if (i + 1) % 500 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = session_labeled / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0

            print(f"  [{i+1:,}/{total:,}] {len(results):,} labeled | "
                  f"errors:{errors} | "
                  f"{rate:.2f} sent/sec | "
                  f"ETA: {remaining/3600:.1f}h")

        # Save progress every 1000 sentences
        if (i + 1) % 1000 == 0:
            save_progress(results, progress_path, i + 1, total,
                          start_time, model_size, errors)

        # Clear GPU cache periodically to prevent memory buildup
        if (i + 1) % 5000 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    save_final_results(results, corpus, model_size, elapsed, errors)

    # Clean up progress files
    for p in [progress_path, partial_path]:
        if os.path.exists(p):
            os.remove(p)


def main():
    parser = argparse.ArgumentParser(
        description="Llama-3.1 Annotation for Sudanese Hate Speech"
    )
    parser.add_argument("--test", action="store_true",
                        help="Test with 5 sentences only")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--model-size", choices=["70b", "8b"], default="70b",
                        help="Model size: 70b (~35GB VRAM, ~24-36h) or "
                             "8b (~6GB VRAM, ~4-6h)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f" Llama-3.1-{args.model_size.upper()}-Instruct Annotation Pipeline")
    print(f" GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ No GPU available! Check CUDA_VISIBLE_DEVICES.")
        sys.exit(1)

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model, tokenizer = load_model(args.model_size)

    if args.test:
        run_test(model, tokenizer)
    else:
        run_full(model, tokenizer, args.model_size, resume=args.resume)


if __name__ == "__main__":
    main()
