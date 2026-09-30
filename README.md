# Affective Task Transfer

**A refactored and extended research implementation** of the system described in *Multi-Task Aware Learning for Joint Emotion, Intensity, and Sentiment Analysis*. This repository builds on the paper's [original `mtl-emotion-intensity-sentiment` codebase](https://github.com/JuliaWieczorek/mtl-emotion-intensity-sentiment). It carries forward the existing models, backbones, tasks, and main comparisons while improving data preparation, evaluation, and reproducibility and adding analyses of task interaction.

The model classes were migrated from source commit `b5d8aaae5266229970b641f0b3b2e6e6da8607ea`; they were not reimplemented from scratch. Their source locations and hashes are recorded in [`docs/provenance.json`](docs/provenance.json). Protocol changes are implemented in separate modules and described in [`docs/IMPLEMENTATION_AUDIT.md`](docs/IMPLEMENTATION_AUDIT.md).

## Research scope

The primary study uses MEISD to compare single-task learning (STL), all three two-task combinations, and joint emotion, emotion-conditioned intensity, and sentiment learning. The experiment matrix retains six model families: hard sharing, soft sharing, adapters, MMoE, BERT-LSTM, and cross-stitch. It supports the four backbones used in the earlier project: BERT uncased, BERT cased, RoBERTa, and XLM-R. Each planned condition uses five seeds: 42, 52, 62, 72, and 82.

An external study uses [BRIGHTER English Track B](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-intensities) to examine emotion and intensity on a separate corpus. BRIGHTER has no sentiment labels. For this track, emotion presence is derived from an intensity greater than zero; the two labels are therefore related by construction. The BRIGHTER experiment retrains models on that corpus and does not claim zero-shot transfer from MEISD.

Outputs include accuracy, precision, recall, F1, confusion matrices, and results by task and class. Training logs also record task losses, gradient norms and cosine similarities on a fixed training probe, and MMoE routing. Positive or negative transfer is assessed against a matched STL run with the same dataset, backbone, split, and seed. These diagnostics help examine shared optimization; they do not by themselves establish a causal flow of knowledge between tasks.

Forecasting, training-set-size curves, and new augmentation methods are outside this project's scope. The full experiment matrix has not been run; those runs are reserved for the target computer after implementation review and runtime estimation. The earlier repository and its results remain intact.

## Data audit and layout

Local source files are under `data/source/meisd/`; the supplied archive and an identical duplicate CSV are under `data/archive/`. The `data/` directory is excluded from Git. File hashes and transfer instructions are in [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md).

The supplied `multilabel_augmented_onehot_11222025.csv` contains 4,219 rows. All 2,608 rows marked `mode=llm` have empty `Utterances`. The matching `MEISD_balanced_expanded.csv` contains their actual generated texts in `augmented`. Preparation checks row alignment and label conversion before joining the texts to the one-hot labels. It also masks 216 ambiguous intensity labels where repeated slots assign different intensities to the same emotion. The existing augmentations are reused; none are generated again.

MEISD examples are historical dialogue halves. The split groups related halves by source dialogue and joins groups that share exact normalized text. Augmentations can enter training only when their parent belongs to training; development and test contain originals only. Conflicting duplicate labels are excluded. The prepared split has 2,877 training, 231 development, and 247 test examples.

BRIGHTER is downloaded at a recorded revision using its official train/development/test membership. The audit excludes 14 records with identical text but conflicting labels and two redundant training copies, leaving 2,753/115/2,759 examples. These exclusions must be reported when comparing against the unfiltered official benchmark. The [SemEval task description](https://github.com/emotion-analysis-project/semeval2025-task11) defines five English emotions and intensity levels 0 (absent) through 3 (high).

## Install and prepare

Python 3.11 or newer is required. The Windows CPU verification used PyTorch 2.6 and Transformers 4.49; install a compatible PyTorch build for the target GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\att.exe audit-data --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output docs/data_audit.json
.\.venv\Scripts\att.exe prepare-meisd --csv data/source/meisd/multilabel_augmented_onehot_11222025.csv --expanded data/source/meisd/MEISD_balanced_expanded.csv --raw-meisd data/source/meisd/MEISD_text.csv --output data/prepared/meisd-v2
.\.venv\Scripts\att.exe download-brighter --output data/source/brighter
.\.venv\Scripts\att.exe prepare-brighter --input data/source/brighter --output data/prepared/brighter-v1
```

The `--expanded` argument may be omitted when the matching file is beside the one-hot CSV. Preparation fails if augmented rows lack text and the expanded file is unavailable. Dataset manifests and run directories are never overwritten silently.

## Verify and plan experiments

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/smoke.py --dataset data/prepared/meisd-v2 --output outputs/smoke-meisd
.\.venv\Scripts\python.exe scripts/smoke.py --dataset data/prepared/brighter-v1 --output outputs/smoke-brighter --brighter
.\.venv\Scripts\att.exe make-matrix --dataset data/prepared/meisd-v2 --dataset-name meisd --output outputs/matrix-meisd-v2
.\.venv\Scripts\att.exe make-matrix --dataset data/prepared/brighter-v1 --dataset-name brighter --output outputs/matrix-brighter
```

Creating the matrices does not train models. They contain 240 MEISD and 160 BRIGHTER configurations. After estimating cost on the target computer, `att run-matrix --matrix outputs/matrix-meisd-v2/matrix.json` starts the MEISD runs; `--limit N` restricts the command to the first N configurations. The same command applies to the BRIGHTER matrix.

Checkpoint selection and the emotion threshold use development data. Once the runs are reviewed, use `att evaluate-matrix --matrix outputs/matrix-meisd-v2/matrix.json --split test` and the corresponding BRIGHTER command. Generate tables and plots with `att report --runs outputs --output outputs/report --split test`. The smoke script uses a small randomly initialized BERT to check the pipeline; its scores are not research results.

Each run records configuration, library versions, dataset hashes, seed, training time and steps, selected checkpoint, predictions, and metrics. Reports include per-seed results, paired differences from STL, class-level results, confusion matrices, learning curves, gradient diagnostics, routing, and resource use. See [`docs/RUN_HANDOFF.md`](docs/RUN_HANDOFF.md) for the target-computer workflow.
