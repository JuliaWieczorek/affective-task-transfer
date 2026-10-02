# BRIGHTER learning preflight — 2 October 2026

These are two diagnostic runs, not Study II results. Both used the full
BRIGHTER training and development splits (2,753 and 115 records),
`bert-base-uncased`, 192 tokens, batch size 16, seed 42, AMD Radeon RX 9070,
and up to six epochs. The held-out test split was not evaluated. Both runs are
marked `smoke=true` and are excluded from the official report.

| Run | Best epoch | Elapsed | Peak GPU memory | Dev intensity macro-F1 across emotions | Predicted low / medium / high |
| --- | ---: | ---: | ---: | ---: | ---: |
| `brighter-hard` (emotion + intensity) | 6 | 774.4 s | 4.42 GiB | 0.3674 | 140 / 31 / 3 |
| `brighter-stl-intensity` | 6 | 756.3 s | 3.35 GiB | 0.4367 | 124 / 39 / 11 |

The development majority baseline for intensity macro-F1 across emotions is
0.2392. Both models predicted only `low` after one epoch, then began predicting
`medium` and `high` during subsequent epochs. Both reached their best recorded
score in epoch six, so these runs do not establish that learning had plateaued.

Each run directory includes the configuration, manifest, checkpoint, epoch
history, development predictions and metrics, tokenizer, and encoder config.
The hard-sharing run also includes gradient diagnostics. The manifests record
the source revision, dataset and checkpoint hashes, package versions, GPU/HIP
details, and timing. The prepared BRIGHTER dataset is not included; recreate
it separately to rerun evaluation.

The two `best_model.pt` files use Git LFS. On another computer, install Git
LFS and run `git lfs pull` after checking out this commit.
