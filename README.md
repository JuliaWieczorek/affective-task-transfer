# Affective Task Transfer

## Dissertation relationship

This repository supports Study II of Chapter 5, *Task-Level Knowledge Transfer for Joint Emotion, Intensity, and Sentiment Analysis*, of the PhD dissertation *Knowledge Transfer for Emotion Intensity Prediction in Mental Health Support Dialogues*. It extends the code for *Multi-Task Aware Learning for Joint Emotion, Intensity, and Sentiment Analysis*; the paper's historical comparison is Study I.

**A refactored and extended research implementation** of the system described in *Multi-Task Aware Learning for Joint Emotion, Intensity, and Sentiment Analysis*. This repository builds on the paper's [original `mtl-emotion-intensity-sentiment` codebase](https://github.com/JuliaWieczorek/mtl-emotion-intensity-sentiment). It carries forward the existing models, backbones, tasks, and main comparisons while improving data preparation, evaluation, and reproducibility and adding analyses of task interaction.

The model classes were migrated from source commit `b5d8aaae5266229970b641f0b3b2e6e6da8607ea`; they were not reimplemented from scratch. Their source locations and hashes are recorded in [`docs/provenance.json`](docs/provenance.json). Protocol changes are implemented in separate modules and described in [`docs/IMPLEMENTATION_AUDIT.md`](docs/IMPLEMENTATION_AUDIT.md).

## Research scope and dissertation use

The dissertation presents the historical architecture comparison as Study I and the controlled extension supported here as Study II. Study I retains its original scope, with verified metric and method descriptions corrected. Study II examines the same three affective tasks using improved data preparation, repeated training, interaction diagnostics, and an external corpus. Scores from the two protocols must be interpreted within their own evaluation conditions.

Study II uses the **start half of each historical MEISD dialogue**, consistent with the earlier study of affect available before the later conversation unfolds. The primary MEISD matrix compares three single-task models, all three two-task soft-sharing pairs, and four three-task architectures: hard sharing, soft sharing, adapters, and MMoE. The fixed primary backbone is BERT-base-uncased, with five seeds: 42, 52, 62, 72, and 82. The inherited BERT-LSTM and cross-stitch classes and the other three historical backbones remain available as optional extensions. See [`docs/STUDY_II_PROTOCOL.md`](docs/STUDY_II_PROTOCOL.md) for the prespecified comparisons and interpretation limits.

An external study uses [BRIGHTER English Track B](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-intensities) to examine emotion and intensity on a separate corpus. BRIGHTER has no sentiment labels. For this track, emotion presence is derived from an intensity greater than zero; the two labels are therefore related by construction. The BRIGHTER experiment retrains models on that corpus and does not claim zero-shot transfer from MEISD.

Outputs include accuracy, precision, recall, F1, confusion matrices, and results by task and class, including joint emotion-intensity evaluation. Training logs also record task losses, gradient norms and cosine similarities on a fixed, label-coverage training probe, and MMoE routing. Positive or negative transfer is assessed against a matched STL run with the same dataset, backbone, split, and seed. Pair-to-triple comparisons test the incremental effect of adding a task in soft sharing. These diagnostics help examine shared optimization; they do not by themselves establish a causal flow of knowledge between tasks. The architecture comparison and its interpretation limits are specified in [`docs/ARCHITECTURE_COMPARISON.md`](docs/ARCHITECTURE_COMPARISON.md).

Forecasting, training-set-size curves, and new augmentation methods are outside this project's scope. The full experiment matrix has not been run; those runs are reserved for the target computer after implementation review and runtime estimation. The earlier repository and its results remain intact.

Study II now uses a **192-token maximum**. A local token audit found that the former 128-token setting truncated 282/1,378 MEISD training texts, including 245/828 augmented texts; all current MEISD texts fit within 192 tokens. New GPU timing and memory pilots are required because the earlier BRIGHTER preflight used 128-token padding. The executable pre-run gates are in [`docs/PRE_RUN_CHECKLIST_2026-10-02.md`](docs/PRE_RUN_CHECKLIST_2026-10-02.md).

## Data audit and layout

Local source files are under `data/source/meisd/`; the supplied archive and an identical duplicate CSV are under `data/archive/`. The `data/` directory is excluded from Git. File hashes and transfer instructions are in [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md).

The supplied `multilabel_augmented_onehot_11222025.csv` contains 4,219 rows. All 2,608 rows marked `mode=llm` have empty `Utterances`. The matching `MEISD_balanced_expanded.csv` contains their actual generated texts in `augmented`. Preparation checks row alignment and label conversion before joining the texts to the one-hot labels. It also masks 216 ambiguous intensity labels where repeated slots assign different intensities to the same emotion. The existing augmentations are reused; none are generated again.

MEISD examples are historical dialogue halves; Study II selects the start halves. The split groups related examples by source dialogue and joins groups that share exact normalized text. Augmentations can enter training only when their parent belongs to training; development and test contain originals only. Conflicting duplicate labels are excluded. The prepared Study II split has 1,378 training, 118 development, and 118 test examples. These are labels inherited from the historical aggregation and generation process; the documented slot-aggregation and synthetic-label validity limits still apply.

BRIGHTER is downloaded at a recorded revision. Study II retains every record and label in the official English Track B test split, removes train/dev text overlaps before fitting, and deduplicates the training split. This produces 2,753/115/2,765 examples; the seven internally conflicting training records and two redundant training copies are documented in the prepared manifest. The [SemEval task description](https://github.com/emotion-analysis-project/semeval2025-task11) defines five English emotions and intensity levels 0 (absent) through 3 (high).

## Install and prepare

Python 3.11 or newer is required. The Windows CPU verification used PyTorch 2.6 and Transformers 4.49; install a compatible PyTorch build for the target GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\att.exe audit-data --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output docs/data_audit.json
.\.venv\Scripts\att.exe prepare-meisd --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output data/prepared/meisd-study2-v1
.\.venv\Scripts\att.exe download-brighter --output data/source/brighter
.\.venv\Scripts\att.exe prepare-brighter --input data/source/brighter --output data/prepared/brighter-study2-v1
.\.venv\Scripts\att.exe sample-label-audit --dataset data/prepared/meisd-study2-v1 --output data/audit/study2-label-review.csv --seed 2026 --per-stratum 20
```

The `--expanded` argument may be omitted when the matching file is beside the one-hot CSV. Preparation fails if augmented rows lack text and the expanded file is unavailable. Dataset manifests and run directories are never overwritten silently. The audit command writes a local, train-only worksheet with 80 examples across original/augmented and short/long strata. A human reviewer must assess the blank label-validity fields; the sample does not certify the inherited labels automatically.

## Verify and plan experiments

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/smoke.py --dataset data/prepared/meisd-study2-v1 --output outputs/smoke-meisd-study2
.\.venv\Scripts\python.exe scripts/smoke.py --dataset data/prepared/brighter-study2-v1 --output outputs/smoke-brighter-study2 --brighter
.\.venv\Scripts\att.exe make-matrix --dataset data/prepared/meisd-study2-v1 --dataset-name meisd --output outputs/matrix-meisd-study2-2026-10-02
.\.venv\Scripts\att.exe make-matrix --dataset data/prepared/brighter-study2-v1 --dataset-name brighter --output outputs/matrix-brighter-study2-2026-10-02
```

Creating the matrices does not train models. The primary protocol contains **50 MEISD and 30 BRIGHTER runs**. Generate optional extension matrices explicitly with `--backbones ...`, `--legacy-architectures`, or `--ablations`; use separate output directories. After timing representative **192-token** runs on the target computer, `att run-matrix --matrix outputs/matrix-meisd-study2-2026-10-02/matrix.json` starts the MEISD runs; `--limit N` restricts the command to the first N configurations. The same command applies to the BRIGHTER matrix. Each epoch saves history and a temporary resumable checkpoint containing model, optimiser, scheduler and random-number state. Repeating `run-matrix` resumes a configuration-matching interrupted run from its last completed epoch; it verifies completed checkpoint hashes. The temporary checkpoint is removed after successful completion to limit disk use.

Checkpoint selection and the emotion threshold use development data. Once the runs are reviewed, use `att evaluate-matrix --matrix outputs/matrix-meisd-study2-2026-10-02/matrix.json --split test` and the corresponding BRIGHTER command. Generate tables and plots only after both matrices have complete verified evaluations:

```powershell
.\.venv\Scripts\att.exe report --matrices outputs/matrix-meisd-study2-2026-10-02/matrix.json outputs/matrix-brighter-study2-2026-10-02/matrix.json --expected-runs 80 --output outputs/report-study2 --split test
```

The report rejects missing or changed runs instead of silently summarising a partial matrix. The smoke script uses a small randomly initialized BERT to check the pipeline; its scores are not research results.

Each run records configuration, library versions, GPU/HIP details, dataset hashes, seed, training time and steps, selected checkpoint, predictions, and metrics. Reports include per-seed results, paired differences from STL, direct paired architecture contrasts, pair-to-triple soft-sharing differences, class-level results, confusion matrices, learning curves, gradient diagnostics, routing, and resource use. See [`docs/RUN_HANDOFF.md`](docs/RUN_HANDOFF.md) for the target-computer workflow.
