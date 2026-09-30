# Local data and transfer to the target computer

The Git repository contains code and audits. `data/` is local and excluded from Git. Copy these MEISD files into `data/source/meisd/`:

| File | SHA-256 | Purpose |
|---|---|---|
| `multilabel_augmented_onehot_11222025.csv` | `458add484602056689a64079385b47a01f3bdba1a9531a33ba23880b1934c4ad` | Historical one-hot labels. |
| `MEISD_balanced_expanded.csv` | `20974b3232d2dd1e903d5b333e6267a826eb4845cb922d3a744e84f045e49019` | Existing generated texts, original slots and segment labels. |
| `MEISD_text.csv` | `30e5e212a2bf390af6d5fa01e8ac211072073a93b32c209917e276ca3957c749` | Raw dialogue IDs and turn order. |

The identical duplicate one-hot CSV and historical ZIP remain in the local `data/archive/`. They are not used in Study II. The `data/source/brighter/` files can be downloaded at the recorded dataset revision with the README command.

Prepare Study II data in new directories. `data/prepared/meisd-study2-v1/` selects the start half of each dialogue, groups by source dialogue and exact duplicate text, and contains 1,378 train, 118 dev, and 118 test examples. The earlier `meisd-v2` directory contains both halves and remains available for provenance. `data/prepared/brighter-study2-v1/` contains 2,753/115/2,765 examples and retains the official English Track B test unchanged. The earlier `brighter-v1` directory uses a different test-cleaning rule and must not be mixed into Study II reports.

Preparation refuses to overwrite an existing manifest. Training checks the hashes of every prepared split; evaluation checks the dataset manifest and checkpoint hash. Copy `outputs/` separately from the Git repository when transferring research results.
