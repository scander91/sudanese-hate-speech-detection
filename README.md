# Sudanese Arabic Hate Speech Detection

Evaluating BERT-Based Models and Hybrid Approaches on a 40K Sudanese Arabic Hate Speech Corpus

## Overview

This repository contains the code, experimental results, and datasets for our paper on Sudanese Arabic hate speech detection.

**Key contributions:**
- A 40,000-sentence Sudanese Arabic hate speech corpus annotated using a multi-annotator pipeline (Weak Supervision + GPT-4o-mini + Llama-3.1-70B)
- 7 BERT baseline models evaluated on binary and 3-class classification
- 3 hybrid approaches: Ensemble (Top-3), BiLSTM+Attention, Knowledge Distillation
- SHAP and LIME explainability analysis
- Cross-task sentiment evaluation on 3 Sudanese Arabic datasets
- CNN comparison reproducing Mhamed et al. architecture

## Key Results

### Hate Speech Detection (F1-macro)

| Model | Binary | 3-Class |
|-------|--------|---------|
| **Ensemble (Top-3)** | **86.2%** | **80.2%** |
| SudaBERT-Distill | 84.5% | 78.7% |
| SudaBERT+BiLSTM | 84.2% | 78.0% |
| MARBERTv2 | 85.2% | 78.0% |
| AraBERTv2 | 84.7% | 78.4% |

### Cross-Task Sentiment (Accuracy / F1-macro)

| Model | Telecom Acc | Telecom F1 | SS2 Acc | SS3 Acc |
|-------|------------|------------|---------|---------|
| Ensemble (Top-3) | **80.4%** | **67.0%** | --- | --- |
| CAMeLBERT-DA | 79.0% | 62.8% | 72.4% | 86.1% |
| MARBERTv2 | 78.5% | 64.5% | 74.3% | 85.5% |
| MARBERT | 78.0% | 64.7% | **74.8%** | 86.9% |
| SudaBERT-v2 | 77.5% | 61.3% | 74.6% | **87.1%** |
### BERT vs CNN Comparison

| Architecture | HS Binary F1 | HS 3-Class F1 | SS3 Acc |
|-------------|-------------|---------------|---------|
| **Ensemble (BERT)** | **86.2%** | **80.2%** | --- |
| MARBERTv2 (best HS) | 85.2% | 78.0% | 85.5% |
| SudaBERT-v2 (best SS3) | 83.6% | 77.3% | **87.1%** |
| CNN Baseline | 73.3% | 64.2% | 83.0% |
| SCM+MMA (CNN) | 67.9% | 60.8% | 81.4% |
## Repository Structure

    sudanese-hate-speech-detection/
    │
    ├── scripts/                              # All Python scripts
    │   ├── hate_speech_trainer.py             # Phase 1: 7 BERT baselines x 2 datasets
    │   ├── hybrid_trainer.py                  # Phase 2: Ensemble + BiLSTM + KD
    │   ├── sentiment_trainer.py               # 9 BERT models x 3 sentiment datasets
    │   ├── cnn_mhamed_comparison.py           # Mhamed et al. CNN reproduction
    │   ├── prepare_sentiment_data.py          # SudSenti + Telecom data preparation
    │   ├── agreement_analysis.py              # Inter-annotator agreement
    │   ├── snorkel_pipeline_v3.py             # Weak supervision (42 labeling functions)
    │   ├── llm_annotate.py                    # GPT-4o-mini annotation
    │   ├── llama_annotate.py                  # Llama-3.1-70B annotation
    │   ├── updated-build_labeling_corpus_40k.py
    │   ├── paper_analysis.py                  # Dataset analysis + figures
    │   └── fix_figures.py                     # Word clouds + KDE distributions
    │
    ├── data/
    │   ├── dataset_binary.tsv                 # 40K hate speech (HARMFUL/NEUTRAL)
    │   ├── dataset_3class.tsv                 # 40K hate speech (HATE/OFFENSIVE/NEUTRAL)
    │   ├── sentiment_prepared/                # Preprocessed sentiment datasets
    │   ├── SudSenti/                          # Raw SudSenti files + split indices
    │   ├── agreement_analysis/                # Merged annotations + agreement stats
    │   └── annotations/                       # LLM + WS annotation outputs
    │
    ├── results/
    │   ├── hate_speech/                       # Phase 1 + Phase 2 results (JSON)
    │   ├── sentiment/                         # Sentiment evaluation results
    │   ├── cnn_comparison/                    # CNN reproduction results
    │   └── explainability/                    # LIME HTML + SHAP results
    │
    └── figures/
        ├── dataset_analysis/                  # Word clouds, distributions, stats
        └── model_results/                     # Confusion matrices, comparisons, SHAP
## Preprocessing

Two preprocessing modes are supported:

**Minimal (Paper 1 style):**
- URL removal, mention removal, hashtag symbol removal
- Alef normalization (إأآا → ا)
- Emoji removal, whitespace normalization

**Mhamed et al. (full Arabic normalization):**
- All minimal steps plus:
- Diacritics removal (Tashkeel)
- Elongation stripping (عااااجل → عااجل)
- Heh normalization (ة → ه)
- Yeh normalization (ى → ي)
- Hamza normalization (ئ,ؤ → ء)
- Gaf normalization (گ → ك)
- Number removal, non-Arabic character removal
- NLTK Arabic + 269 Sudanese custom stopwords

## Datasets

| Dataset | Samples | Classes | Source |
|---------|---------|---------|--------|
| Hate Speech Binary | 40,000 | HARMFUL / NEUTRAL | This work |
| Hate Speech 3-Class | 40,000 | HATE / OFFENSIVE / NEUTRAL | This work |
| Telecom Sentiment | 5,345 | neg / obj / pos | Paper 1 |
| SudSenti2 | 4,001 | neg / pos | Mhamed et al. |
| SudSenti3 | 6,957 | neg / obj / pos | Mhamed et al. |

## Usage

### Requirements
Python 3.10+
torch >= 2.0
transformers >= 5.0
tensorflow >= 2.13 (for CNN comparison only)
scikit-learn, lime, shap
matplotlib, seaborn, wordcloud
arabic_reshaper, python-bidi

### Hate Speech Training
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/hate_speech_trainer.py --run_all
CUDA_VISIBLE_DEVICES=0 python3 scripts/hybrid_trainer.py --run_all
CUDA_VISIBLE_DEVICES=0 python3 scripts/hate_speech_trainer.py --explain
```

### Sentiment Evaluation
```bash
python3 scripts/prepare_sentiment_data.py
CUDA_VISIBLE_DEVICES=0 python3 scripts/sentiment_trainer.py --run_all
CUDA_VISIBLE_DEVICES=0 python3 scripts/sentiment_trainer.py --run_all --preprocess mhamed
```

### CNN Comparison
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/cnn_mhamed_comparison.py --run_all
```

### Generate Figures
```bash
python3 scripts/paper_analysis.py
python3 scripts/fix_figures.py
```

## Data Availability

- **Sudanese Arabic Corpus** (6.7M sentences): [Google Drive](https://drive.google.com/file/d/1MgAsn284J4V1uF6v3L9DtkS_nLFnrNyR/view?usp=drive_link)
- **SudSenti Datasets**: [Mhamed et al. GitHub](https://github.com/mustafa20999/Sudanese-Arabic-Sentiment-Datasets)


## License

This code and data are available for academic research purposes only.


