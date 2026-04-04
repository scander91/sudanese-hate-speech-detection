#!/usr/bin/env python3
"""
LLM-Assisted Annotation for Sudanese Arabic Hate Speech (v2)
==============================================================
Design based on literature:
- Ghorbanpour et al. (2025, WOAH): language-customized prompting
- Bosley (2025): few-shot > zero-shot (+7%)
- Masud et al. (2024): geographical priming
- Huang et al. (2023): temperature=0

Usage:
  python3 llm_annotate.py --test --model gpt-4o-mini   # Test first!
  python3 llm_annotate.py --model gpt-4o-mini           # Full run
  python3 llm_annotate.py --model gpt-4o                # Full run
  python3 llm_annotate.py --model gpt-4o-mini --resume  # Resume

Requirements: pip3 install openai --user
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from collections import Counter
import re

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip3 install openai --user")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
CORPUS_PATH = "data/labeling_corpus/labeling_corpus_25k.json"
OUTPUT_DIR = "data/labeling_corpus/llm_annotations"
BATCH_SIZE = 20
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_DELAY = 5

# Model pricing (per 1M tokens)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT (Arabic, with geographical context)
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """أنت خبير لغوي متخصص في تصنيف خطاب الكراهية في اللهجة العربية السودانية. لديك معرفة عميقة بالسياق السوداني: الحرب الأهلية (2023-الآن)، القبائل السودانية، الفصائل السياسية، واللهجة العامية السودانية.

صنّف كل جملة إلى فئة واحدة فقط:

HATE = خطاب يستهدف مجموعة كاملة بناءً على هويتها (عرق، قبيلة، دين، انتماء سياسي). يشمل: التجريد من الإنسانية لمجموعة، التحريض على العنف ضد مجموعة، الدعوة لإقصاء مجموعة.

OFFENSIVE = لغة بذيئة أو مسيئة لكنها لا تستهدف مجموعة بسبب هويتها. تشمل: شتائم شخصية، ألفاظ نابية، إهانة أفراد.

NEUTRAL = تقارير إخبارية، محادثات يومية، محتوى إنساني، نقاش موضوعي.

القاعدة الأساسية: HATE يستهدف مجموعة كاملة. OFFENSIVE يستهدف فرد. NEUTRAL لا يحتوي على إساءة.

تنبيه مهم: أجب دائماً بالإنجليزية فقط: HATE أو OFFENSIVE أو NEUTRAL. لا تستخدم كلمات عربية في التصنيف."""


def build_batch_prompt(sentences):
    """Build user prompt with few-shot examples + numbered sentences."""

    prompt = """أمثلة للتصنيف:

الجملة: الجنجويد ديل كلاب ما بستاهلوا يعيشوا وسطنا
التصنيف: HATE
السبب: تجريد مجموعة عرقية من الإنسانية ودعوة لإقصائهم

الجملة: الكيزان حشرات لازم نبيدهم كلهم عن بكرة ابيهم
التصنيف: HATE
السبب: تجريد مجموعة سياسية من الإنسانية وتحريض على العنف ضدهم

الجملة: كسمك يا اهبل انت وكلامك الفاضي
التصنيف: OFFENSIVE
السبب: شتيمة شخصية بألفاظ نابية لكن لا تستهدف مجموعة

الجملة: البرهان ده كلب وخائن ما بيستاهل يحكم
التصنيف: OFFENSIVE
السبب: إهانة لشخص بعينه وليس لمجموعة عرقية أو قبلية

الجملة: مصادر ميدانية: الجيش السوداني يدمر عربات قتالية تابعة لقوات الدعم السريع في الخرطوم
التصنيف: NEUTRAL
السبب: تقرير إخباري موضوعي

الآن صنّف الجمل التالية. أجب بصيغة JSON فقط بدون أي نص أو شرح إضافي.
الشكل المطلوب بالضبط:
[{"id": 1, "label": "HATE"}, {"id": 2, "label": "NEUTRAL"}, {"id": 3, "label": "OFFENSIVE"}]

استخدم فقط: HATE أو OFFENSIVE أو NEUTRAL (بالإنجليزية).

الجمل:
"""
    for i, sent in enumerate(sentences):
        prompt += f"{i + 1}. {sent}\n"

    return prompt


def parse_response(response_text, expected_count, start_idx):
    """Parse model response into labels. Multiple fallback strategies."""
    text = response_text.strip()

    # Remove markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Strategy 1: JSON array parsing
    try:
        results = json.loads(text)
        if isinstance(results, list):
            labels = {}
            for item in results:
                if isinstance(item, dict) and "id" in item and "label" in item:
                    label = str(item["label"]).strip().upper()
                    if label in ("HATE", "OFFENSIVE", "NEUTRAL"):
                        labels[int(item["id"])] = label
            if labels:
                return labels
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Try to find JSON array anywhere in the text
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            results = json.loads(json_match.group())
            if isinstance(results, list):
                labels = {}
                for item in results:
                    if isinstance(item, dict) and "id" in item and "label" in item:
                        label = str(item["label"]).strip().upper()
                        if label in ("HATE", "OFFENSIVE", "NEUTRAL"):
                            labels[int(item["id"])] = label
                if labels:
                    return labels
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Line-by-line parsing
    labels = {}
    valid_labels = {"HATE", "OFFENSIVE", "NEUTRAL"}
    for line in text.split("\n"):
        line = line.strip()
        for vid in range(start_idx + 1, start_idx + expected_count + 1):
            if str(vid) in line:
                for vl in valid_labels:
                    if vl in line.upper():
                        labels[vid] = vl
                        break
                break

    return labels


def calc_cost(model, input_tokens, output_tokens):
    """Calculate exact cost based on model pricing."""
    p = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens / 1e6 * p["input"]) + (output_tokens / 1e6 * p["output"])


def annotate_batch(client, model, sentences, start_idx, batch_num, total_batches):
    """Send one batch to the API. Returns labels dict + token counts."""
    user_prompt = build_batch_prompt(sentences)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=1200,
            )

            raw_text = response.choices[0].message.content
            usage = response.usage
            parsed = parse_response(raw_text, len(sentences), 0)
            labels = {start_idx + mid: lbl for mid, lbl in parsed.items()}

            # Check completeness
            expected_ids = set(range(start_idx + 1, start_idx + len(sentences) + 1))
            got_ids = set(labels.keys())
            missing = expected_ids - got_ids

            if len(missing) > 3 and attempt < MAX_RETRIES - 1:
                print(f"    ⚠️  Batch {batch_num}: got {len(got_ids)}/{len(sentences)}, retrying...")
                time.sleep(RETRY_DELAY)
                continue

            return labels, usage.prompt_tokens, usage.completion_tokens, raw_text

        except Exception as e:
            print(f"    ❌ Batch {batch_num} attempt {attempt+1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {}, 0, 0, f"ERROR: {e}"

    return {}, 0, 0, "ERROR: max retries"


def run_test(client, model):
    """Test on 5 sentences before spending money."""
    print(f"\n{'='*70}")
    print(f" TEST MODE — 5 sentences (cost: ~$0.001)")
    print(f" Model: {model} | Temperature: {TEMPERATURE}")
    print(f"{'='*70}")

    test_sentences = [
        "الجنجويد ديل حيوانات لازم نخلص منهم",
        "كسمك يا غبي",
        "عاجل: الجيش يسيطر على مناطق جديدة في الخرطوم",
        "الشايقية ديل كلهم خونة وعملاء",
        "هسع الكهرباء قطعت تاني يا ناس",
    ]
    expected = ["HATE", "OFFENSIVE", "NEUTRAL", "HATE", "NEUTRAL"]

    labels, inp_tok, out_tok, raw_text = annotate_batch(
        client, model, test_sentences, 0, 1, 1
    )

    cost = calc_cost(model, inp_tok, out_tok)

    print(f"\n  Raw API response:")
    print(f"  {raw_text}")
    print(f"\n  Tokens: input={inp_tok}, output={out_tok}")
    print(f"  Cost: ${cost:.6f}")

    print(f"\n  {'#':<4} {'Expected':<12} {'Got':<12} {'Match':<6} Sentence")
    print(f"  {'─'*75}")
    correct = 0
    for i, (sent, exp) in enumerate(zip(test_sentences, expected)):
        got = labels.get(i + 1, "MISSING")
        match = "✅" if got == exp else "❌"
        if got == exp:
            correct += 1
        print(f"  {i+1:<4} {exp:<12} {got:<12} {match:<6} {sent[:50]}")

    print(f"\n  Accuracy: {correct}/{len(test_sentences)} ({100*correct/len(test_sentences):.0f}%)")

    # Extrapolate cost for full corpus
    # Test: 5 sentences in 1 batch. Full: 40,000 sentences in 2,000 batches of 20
    # prompt_overhead repeats per batch (2,000 times in full run)
    # sentence tokens scale by 40,000/5 = 8,000x
    avg_tok_per_sent = inp_tok // len(test_sentences)  # rough per-sentence
    prompt_overhead = inp_tok - (avg_tok_per_sent * len(test_sentences) // 2)
    prompt_overhead = max(prompt_overhead, inp_tok // 2)  # at least half is prompt
    sentence_tokens = inp_tok - prompt_overhead
    full_input = (prompt_overhead * 2000) + (sentence_tokens * 8000)
    full_output = out_tok * 8000
    full_cost = calc_cost(model, full_input, full_output)
    print(f"  Estimated full corpus cost: ~${full_cost:.2f} (rough estimate)")

    if correct >= 4:
        print(f"\n  ✅ Test PASSED. Safe to run full corpus.")
        print(f"  Run: python3 llm_annotate.py --model {model}")
    else:
        print(f"\n  ⚠️  Test accuracy low. Review the labels before running full corpus.")

    return correct


def run_full(client, model, resume=False):
    """Annotate full 40K corpus."""
    print(f"\n{'='*70}")
    print(f" FULL ANNOTATION — {model}")
    print(f" Batch: {BATCH_SIZE} | Temp: {TEMPERATURE} | Retries: {MAX_RETRIES}")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Load corpus
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    print(f"  Loaded {len(texts):,} sentences")

    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_safe = model.replace("-", "_").replace(".", "_")
    output_path = os.path.join(OUTPUT_DIR, f"labels_{model_safe}.json")
    tsv_path = os.path.join(OUTPUT_DIR, f"labels_{model_safe}.tsv")
    progress_path = os.path.join(OUTPUT_DIR, f"progress_{model_safe}.json")
    raw_path = os.path.join(OUTPUT_DIR, f"raw_responses_{model_safe}.jsonl")

    # Resume or fresh
    all_labels = {}
    total_inp = 0
    total_out = 0
    start_batch = 0

    if resume and os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            prog = json.load(f)
        all_labels = {int(k): v for k, v in prog.get("labels", {}).items()}
        total_inp = prog.get("input_tokens", 0)
        total_out = prog.get("output_tokens", 0)
        start_batch = prog.get("last_batch", 0) + 1
        print(f"  Resuming: batch {start_batch}, {len(all_labels)} labels done")

    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Batches: {n_batches} (starting from {start_batch})")

    # Open raw response file
    raw_mode = "a" if resume else "w"
    raw_file = open(raw_path, raw_mode, encoding="utf-8")

    start_time = time.time()
    errors = 0

    for bi in range(start_batch, n_batches):
        bs = bi * BATCH_SIZE
        be = min(bs + BATCH_SIZE, len(texts))
        batch_texts = texts[bs:be]

        labels, inp, out, raw = annotate_batch(client, model, batch_texts, bs, bi+1, n_batches)

        all_labels.update(labels)
        total_inp += inp
        total_out += out

        # Save raw response
        raw_file.write(json.dumps({
            "batch": bi + 1, "start_idx": bs,
            "response": raw, "labels_parsed": len(labels),
        }, ensure_ascii=False) + "\n")
        raw_file.flush()

        if not labels:
            errors += 1

        # Progress update every 100 batches
        if (bi + 1) % 100 == 0 or bi == n_batches - 1:
            elapsed = time.time() - start_time
            done_batches = bi - start_batch + 1
            eta = (elapsed / done_batches) * (n_batches - bi - 1) if done_batches > 0 else 0
            cost = calc_cost(model, total_inp, total_out)

            print(f"  [{bi+1}/{n_batches}] {len(all_labels):,} labeled | "
                  f"${cost:.2f} | errors:{errors} | ETA:{eta/60:.0f}min")

            # Save progress
            with open(progress_path, "w") as f:
                json.dump({
                    "model": model, "last_batch": bi,
                    "labels": {str(k): v for k, v in all_labels.items()},
                    "input_tokens": total_inp, "output_tokens": total_out,
                    "errors": errors,
                    "timestamp": datetime.now().isoformat(),
                }, f, ensure_ascii=False)

        time.sleep(0.3)

    raw_file.close()

    # Final cost
    total_cost = calc_cost(model, total_inp, total_out)
    elapsed = time.time() - start_time

    # Build output
    results = []
    missing = 0
    for i, item in enumerate(data):
        label = all_labels.get(i + 1, "UNKNOWN")
        if label == "UNKNOWN":
            missing += 1
        results.append({
            "id": i + 1,
            "text": item["text"],
            "source": item["source"],
            "keyword_category": item["keyword_category"],
            "llm_label": label,
            "model": model,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("id\ttext\tsource\tkeyword_cat\tllm_label\tmodel\n")
        for r in results:
            t = r["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{r['id']}\t{t}\t{r['source']}\t{r['keyword_category']}\t{r['llm_label']}\t{model}\n")

    # Stats
    label_counts = Counter(all_labels.values())

    print(f"\n{'='*70}")
    print(f" DONE — {model}")
    print(f"{'='*70}")
    print(f"  Sentences:     {len(texts):,}")
    print(f"  Labeled:       {len(all_labels):,}")
    print(f"  Missing:       {missing:,}")
    print(f"  Errors:        {errors}")
    print(f"  Time:          {elapsed/60:.1f} min")
    print(f"  Input tokens:  {total_inp:,}")
    print(f"  Output tokens: {total_out:,}")
    print(f"  TOTAL COST:    ${total_cost:.2f}")
    print(f"\n  Distribution:")
    for lab in ["HATE", "OFFENSIVE", "NEUTRAL", "UNKNOWN"]:
        c = label_counts.get(lab, 0) if lab != "UNKNOWN" else missing
        if c > 0:
            print(f"    {lab:<12} {c:>8,} ({100*c/len(texts):>5.1f}%)")
    print(f"\n  📁 {output_path}")
    print(f"  📁 {tsv_path}")
    print(f"  📁 {raw_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o"])
    parser.add_argument("--test", action="store_true", help="Test 5 sentences first")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    client = OpenAI()

    if args.test:
        run_test(client, args.model)
    else:
        run_full(client, args.model, args.resume)


if __name__ == "__main__":
    main()
