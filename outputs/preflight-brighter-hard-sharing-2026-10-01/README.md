# BRIGHTER GPU preflight — 1 October 2026

This is a one-epoch validation run, not a Study II result. It used the full
BRIGHTER training split (2,753 records), the full development split (115
records), BERT-base-uncased, hard sharing for emotion and intensity, batch
size 16, seed 42, and enabled diagnostics. The held-out test split was not
evaluated. The run is marked `smoke=true` so normal reports exclude it.

The run completed 172 optimizer steps in 89.2 seconds of recorded training
time, with 1.7 seconds of diagnostics and 3.86 GiB peak allocated GPU memory.
Development emotion macro-F1 was 0.5796; intensity macro-F1 across emotions
was 0.2392. These scores are preliminary after one epoch.

`manifest.json` contains the exact configuration, data hashes, package
versions, checkpoint SHA-256, timing, and resource use. `history.jsonl` and
`diagnostics.jsonl` contain the epoch logs. `dev_predictions.json`,
`dev_results.json`, and `best.json` contain validation outputs. The tokenizer
and `best_model.pt` allow checkpoint reconstruction.

`best_model.pt` is stored with Git LFS. On another computer, install Git LFS,
check out the `gpu-rocm-check` branch, and run `git lfs pull` before using the
checkpoint. The prepared BRIGHTER dataset is not tracked by Git and must be
prepared separately to rerun evaluation; its exact revision and hashes are
in `manifest.json`.
