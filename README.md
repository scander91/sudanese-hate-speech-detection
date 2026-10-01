# Sudanese Arabic hate speech detection: hybrid weak supervision with LLM annotation

Code for the paper "Hybrid Weak Supervision with Large Language Model Annotation Framework for
Low-Resource Sudanese Arabic Hate Speech Detection".

The repository contains the data pipeline (de-duplication, splits, weak-supervision label
aggregation), the training code for all model families (transformer baselines, BiLSTM hybrid,
knowledge distillation, multi-task learning with sentiment, CNN baselines, ensembles), the
validation-only model selection, the statistical evaluation, and the SHAP/LIME explainability jobs.

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended for the transformer jobs. The CPU is sufficient for the post-hoc jobs
(selection, evaluation) and for the smoke test.

## Data

The corpus is released as record identifiers and labels in `data/ids/`. The texts are not
redistributed.

| File | Columns |
|---|---|
| `hs_ids.tsv` | `orig_id`, `uid`, `split`, `label3`, `label_bin` |
| `telecom_ids.tsv` | `uid`, `pool_id`, `split`, `label` |
| `sudsenti2_ids.tsv`, `sudsenti3_ids.tsv` | `uid`, `pool_id`, `fold`, `label` |
| `sudsenti2_folds.json`, `sudsenti3_folds.json` | per fold: `train` / `val` / `test` lists of `uid` |

`uid` is the SHA-256 of the canonical form of a text (`rv2.textnorm.text_id`: NFKC, Algorithm 3
normalisation, Persian Yeh/Kaf folding). `orig_id` is the row identifier in the original
hate-speech corpus and `pool_id` the row number in the pooled source files of each sentiment
dataset. With access to the source texts, `rv2.textnorm.text_id(text)` maps every text to its `uid`,
so the published splits, folds and labels can be re-attached exactly.

To run the pipeline, place the source files at the paths listed under `data:` in
`configs/default.yaml` (relative to `project_root`), together with the labelling-function module
given by `data.snorkel_module` (`snorkel_pipeline_v3.py`, included). Phase 0 then writes the frozen manifests to `data_frozen/`.

## Configuration

`configs/default.yaml` holds every path, label set, seed, model, model revision and hyper-parameter.
The `hosts` section is an example (`host1`, `host2`, `host3`); replace it with your own machines,
GPU indices and routing (`kinds_first`, `exclude_kinds`). The ssh command used to start remote
workers is `launcher.ssh`.

## How to run

```bash
# unit tests of the text normalisation (no data needed)
python tests/test_textnorm.py

# Phase 0: de-duplicate, split, apply the labelling functions, write data_frozen/
python -m rv2.phase0_build

# optional checks
python tests/smoke_test.py                                   # CPU end-to-end smoke test on small subsets
CUDA_VISIBLE_DEVICES=0 python tests/gpu_gate.py --tag host1_gpu0   # short GPU check per host

# campaign
python -m rv2.launcher init                  # write runs/CAMPAIGN.json (code/config/data fingerprint)
python -m rv2.launcher plan                  # job counts and time estimates
python -m rv2.launcher worker --gpu 0 --host host1     # run jobs on this machine
python -m rv2.launcher start --host host2              # one remote worker per GPU over ssh
python -m rv2.launcher worker --cpu          # optional CPU worker for the post-hoc jobs
python -m rv2.launcher status                # progress, failures, stale locks

# single jobs
python -m rv2.run --list --kind base
python -m rv2.run --job base__marbertv2__binary__s42

# report (also run automatically as job post__report)
python -m rv2.evaluate
```

Workers run the whole job graph defined in `rv2/jobs.py`, in dependency order:

1. training jobs (`base`, `sent`, `bilstm`, `distill`, `distill4`, `mtl`, `cnn`) write training and
   validation predictions and a best-on-validation checkpoint;
2. selection jobs (`post__teacher__*`, `post__ens__*`, `post__mtlsel__*`) choose the distillation
   teacher, the ensemble members and rule, the MTL loss weight and the best single model, using
   validation results only; `post__freeze` then freezes all selections;
3. test jobs (`test__*`) reload each checkpoint and predict the test partition once;
   `post__enstest__*` materialises the ensembles;
4. explainability jobs (`xai__*`, `xai_mtlcmp__*`) run SHAP and LIME on the selected checkpoints;
5. `post__report` computes mean ± SD over seeds, confidence intervals, per-class scores, the
   Nadeau-Bengio corrected resampled t-test with Holm adjustment, and McNemar tests.

Outputs go to `runs/` (per-job directories, `runs/_selection/`, `runs/_report/`).

## Repository layout

| Path | Content |
|---|---|
| `rv2/textnorm.py` | canonical key, record identifiers, preprocessing modes (`none`, `minimal`, `alg3`, `mhamed`) |
| `rv2/phase0_build.py` | Phase 0: label rules, labelling functions, de-duplication, splits, folds, manifests |
| `rv2/manifest.py` | manifest loading, checksums, cross-partition leakage assertions |
| `rv2/models.py` | BiLSTM+attention, MTL model, CNN and SCM+MMA, Keras-style tokenizer |
| `rv2/train.py` | training and test phases of every model family |
| `rv2/select.py` | validation-only selections and the selection freeze |
| `rv2/evaluate.py` | statistics and report |
| `rv2/xai.py` | SHAP and LIME |
| `rv2/jobs.py`, `rv2/run.py`, `rv2/launcher.py` | job registry, job execution, queue and workers |
| `resources/stopwords_cnn.json` | stop-word list of the CNN preprocessing |
| `data/ids/` | record identifiers, splits, folds and labels |

## License

MIT (see `LICENSE`).
