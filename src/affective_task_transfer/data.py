"""Audited adapters around the historical one-hot data representation.

No augmentation is generated here. Dialogue-half reconstruction follows
meisd_project/data/MEISD_csv_to_csv.py:convert_csv_to_da_halves.
"""
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import random
import re

from .io import read_csv, sha256, write_json, write_jsonl, read_json, read_jsonl

SENTIMENTS = ["negative", "neutral", "positive"]
BRIGHTER_EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def text_id(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()

def recover_halves(raw_path):
    groups = defaultdict(list)
    for row in read_csv(raw_path):
        groups[row["dialog_ids"]].append(row)
    lookup = defaultdict(set)
    for dialog, rows in groups.items():
        # Intentionally reproduce historical grouping/order, not a new aggregation.
        rows = sorted(rows, key=lambda r: int(r["uttr_ids"]))
        mid = math.ceil(len(rows) / 2)
        for segment in (rows[:mid], rows[mid:]):
            text = " ".join(r["Utterances"].strip() for r in segment if r["Utterances"].strip())
            if text:
                lookup[normalize(text)].add("meisd:" + dialog)
    return lookup

def meisd_records(csv_path, raw_path, expanded_path=None):
    source = read_csv(csv_path)
    if expanded_path is None:
        candidate = Path(csv_path).with_name("MEISD_balanced_expanded.csv")
        expanded_path = candidate if candidate.exists() else None
    expanded = read_csv(expanded_path) if expanded_path else None
    if expanded is not None and len(expanded) != len(source):
        raise ValueError("Expanded source and one-hot CSV have different row counts")
    emotions = [k.removeprefix("emotion__") for k in source[0] if k.startswith("emotion__")]
    if not emotions:
        raise ValueError("No emotion columns")
    halves = recover_halves(raw_path)
    originals = {normalize(r["Utterances"]): r for r in source if not r["mode"].strip()}
    records, issues = [], Counter()
    for i, row in enumerate(source):
        augmented = bool(row["mode"].strip())
        parent = row["original"] if augmented else row["Utterances"]
        parent_norm = normalize(parent)
        if augmented and parent_norm not in originals:
            raise ValueError(f"Unresolved augmentation parent at record {i}")
        dialogs = halves.get(parent_norm, set())
        if len(dialogs) != 1:
            raise ValueError(f"Expected one raw dialogue for record {i}, found {len(dialogs)}")
        text = row["Utterances"].strip()
        if expanded is not None:
            previous = expanded[i]
            if any(previous[k] != row[k] for k in ("Utterances", "original", "mode")):
                raise ValueError(f"Expanded source and one-hot CSV are misaligned at record {i}")
            if previous["sentiment"].replace("positve", "positive").lower() != row["sentiment"].lower():
                raise ValueError(f"Sentiment conversion differs at record {i}")
            if previous["sentiment"] == "positve":
                issues["corrected_positve_labels"] += 1
            if augmented:
                generated = previous["augmented"].strip()
                if not generated:
                    raise ValueError(f"Missing generated text in expanded source at record {i}")
                if text and normalize(text) != normalize(generated):
                    raise ValueError(f"One-hot text and expanded augmentation differ at record {i}")
                text = generated
        if not text:
            raise ValueError(f"Empty text at record {i}; supply the matching MEISD_balanced_expanded.csv")
        sentiment = row["sentiment"].strip().lower()
        if sentiment not in SENTIMENTS:
            raise ValueError(f"Invalid sentiment at record {i}")
        e, intensities = [], []
        slot_values = defaultdict(set)
        slot_last = {}
        if expanded is not None:
            for slot in (1, 2, 3):
                label = expanded[i].get(f"emotion{slot}", "").strip()
                value = expanded[i].get(f"intensity{slot}", "").strip()
                if label and value:
                    slot_values[label].add(float(value))
                    slot_last[label] = float(value)
        for name in emotions:
            presence = float(row[f"emotion__{name}"])
            if presence not in (0, 1):
                raise ValueError(f"Invalid presence at record {i}")
            value = row[f"intensity__{name}"].strip()
            intensity = float(value) if value else None
            if expanded is not None:
                if bool(presence) != (name in slot_last):
                    raise ValueError(f"One-hot presence differs from expanded slots at record {i}, {name}")
                if name in slot_last and intensity != slot_last[name]:
                    raise ValueError(f"One-hot intensity differs from expanded slot conversion at record {i}, {name}")
            if presence and len(slot_values[name]) > 1:
                target = -100
                issues["ambiguous_repeated_emotion_intensity_positions"] += 1
            elif presence and intensity in (1, 2, 3):
                target = int(intensity) - 1
            else:
                target = -100
                if presence:
                    issues["active_missing_intensity"] += 1
                elif intensity not in (None, 0):
                    issues["inactive_nonzero_intensity"] += 1
            if intensity is not None and intensity not in (0, 1, 2, 3):
                raise ValueError(f"Out-of-range intensity at record {i}")
            e.append(int(presence))
            intensities.append(target)
        if augmented:
            original = originals[parent_norm]
            cols = ["sentiment"] + [f"{kind}__{n}" for n in emotions for kind in ("emotion", "intensity")]
            if any(row[c] != original[c] for c in cols):
                raise ValueError(f"Augmentation labels differ from parent at record {i}")
        records.append({"id": f"meisd-{i:05d}", "text": text, "source_dialogue_id": next(iter(dialogs)), "parent_id": text_id(parent), "is_augmented": augmented, "augmentation_method": row["mode"], "emotion": e, "intensity": intensities, "sentiment": SENTIMENTS.index(sentiment)})
    # Exact copies with contradictory targets cannot be silently split/deduplicated.
    by_text = defaultdict(set)
    for r in records:
        by_text[normalize(r["text"])].add((tuple(r["emotion"]), tuple(r["intensity"]), r["sentiment"]))
    conflicts = sum(len(v) > 1 for v in by_text.values())
    issues["conflicting_text_labels"] = conflicts
    audit = {"source_sha256": sha256(csv_path), "expanded_sha256": sha256(expanded_path) if expanded_path else None, "raw_sha256": sha256(raw_path), "records": len(records), "originals": sum(not r["is_augmented"] for r in records), "augmentations": sum(r["is_augmented"] for r in records), "augmentation_text_source": "expanded.augmented" if expanded is not None else "onehot.Utterances", "dialogues": len({r["source_dialogue_id"] for r in records}), "all_parents_resolved": True, "unit": "historical dialogue half", "issues": dict(issues), "limitations": ["Historical filtering and target-frequency balancing preceded this re-split; this evaluates the retained historical population.", "Segment labels are inherited, not independently reannotated. The historical converter averaged intensities by annotation slot, which need not preserve emotion identity.", "Neutral intensity and fallback labels require semantic caution; masks fix absent emotions, not annotation validity."]}
    return records, emotions, audit

def assign_groups(records, seed, ratios=(0.7, 0.15, 0.15)):
    """Union dialogue IDs across exact duplicates, then split connected components.

    Split is deterministic and not selected by downstream validation/test scores.
    """
    parents = {r["source_dialogue_id"]: r["source_dialogue_id"] for r in records}
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    texts = {}
    for r in records:
        key, dialog = normalize(r["text"]), r["source_dialogue_id"]
        if key in texts:
            a, b = find(dialog), find(texts[key])
            parents[max(a, b)] = min(a, b)
        texts[key] = dialog
    groups = sorted({find(x) for x in parents})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    first, second = int(n*ratios[0]), int(n*(ratios[0]+ratios[1]))
    if min(first, second-first, n-second) < 1:
        raise ValueError("Not enough independent dialogue groups for three partitions")
    assignments = {g: ("train" if i < first else "dev" if i < second else "test") for i,g in enumerate(groups)}
    for r in records:
        r["group_id"] = find(r["source_dialogue_id"])
        r["split"] = assignments[r["group_id"]]
    return records

def validate_splits(splits):
    for a,b in (("train","dev"),("train","test"),("dev","test")):
        for key in ("group_id", "parent_id"):
            overlap = {r[key] for r in splits[a]} & {r[key] for r in splits[b]}
            if overlap:
                raise ValueError(f"{key} overlap between {a} and {b}")
        if {normalize(r["text"]) for r in splits[a]} & {normalize(r["text"]) for r in splits[b]}:
            raise ValueError(f"Text overlap between {a} and {b}")
    if any(r["is_augmented"] for s in ("dev", "test") for r in splits[s]):
        raise ValueError("Augmentation in evaluation set")

def split_counts(rows, emotions):
    return {"records": len(rows), "groups": len({r["group_id"] for r in rows}), "augmented": sum(r["is_augmented"] for r in rows), "sentiment": dict(Counter(r["sentiment"] for r in rows)), "emotion_support": {e:sum(r["emotion"][j] for r in rows) for j,e in enumerate(emotions)}, "intensity_support": {e:dict(Counter(r["intensity"][j]+1 for r in rows if r["intensity"][j]>=0)) for j,e in enumerate(emotions)}}

def save_dataset(output, splits, emotions, audit, sentiment=True):
    output = Path(output)
    if (output/"manifest.json").exists():
        raise FileExistsError(f"Dataset exists: {output}. Use a new directory to preserve the split.")
    validate_splits(splits)
    files = {}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty {name}")
        path = output / f"{name}.jsonl"
        write_jsonl(path, rows)
        files[name] = {"file": path.name, "sha256": sha256(path), **split_counts(rows, emotions)}
    manifest = {"schema_version": 1, "emotions": emotions, "tasks": ["sentiment", "emotion", "intensity"] if sentiment else ["emotion", "intensity"], "files": files, "audit": audit}
    write_json(output/"manifest.json", manifest)
    return manifest

def prepare_meisd(csv_path, raw_path, output, seed=2026, expanded_path=None):
    records, emotions, audit = meisd_records(csv_path, raw_path, expanded_path)
    records = assign_groups(records, seed)
    targets_by_text = defaultdict(set)
    for r in records:
        targets_by_text[normalize(r["text"])].add((tuple(r["emotion"]), tuple(r["intensity"]), r["sentiment"]))
    conflicts = {text for text, targets in targets_by_text.items() if len(targets) > 1}
    excluded_conflicts = sum(normalize(r["text"]) in conflicts for r in records)
    records = [r for r in records if normalize(r["text"]) not in conflicts]
    splits = {s: [] for s in ("train", "dev", "test")}
    seen = {s:set() for s in splits}
    excluded_aug = duplicates = 0
    for r in records:
        s = r["split"]
        if s != "train" and r["is_augmented"]:
            excluded_aug += 1
            continue
        key = (normalize(r["text"]), tuple(r["emotion"]), tuple(r["intensity"]), r["sentiment"])
        if key in seen[s]:
            duplicates += 1
            continue
        seen[s].add(key)
        splits[s].append(r)
    audit.update(split_seed=seed, split_method="shuffled dialogue/duplicate connected components 70/15/15", excluded_evaluation_augmentations=excluded_aug, excluded_conflicting_label_records=excluded_conflicts, removed_exact_duplicate_records=duplicates)
    return save_dataset(output, splits, emotions, audit)

def prepare_brighter(input_dir, output, revision=None):
    splits = {}
    hashes = {}
    download_manifest_path = Path(input_dir)/"download_manifest.json"
    download_manifest = read_json(download_manifest_path) if download_manifest_path.exists() else None
    if download_manifest is not None and revision != download_manifest["revision"]:
        raise ValueError("BRIGHTER revision differs from download manifest")
    for split in ("train", "dev", "test"):
        path = Path(input_dir)/f"{split}.csv"
        rows = read_csv(path)
        hashes[split] = sha256(path)
        if download_manifest is not None and hashes[split] != download_manifest["splits"][split]["sha256"]:
            raise ValueError(f"Downloaded BRIGHTER {split} changed after conversion")
        parsed = []
        for i,r in enumerate(rows):
            values = [int(r[e]) for e in BRIGHTER_EMOTIONS]
            if not r["text"].strip() or any(v not in range(4) for v in values):
                raise ValueError(f"Invalid BRIGHTER row in {split}: {i}")
            h = text_id(r["text"])
            parsed.append({"id":r.get("id", f"{split}-{i}"), "text":r["text"], "group_id":h, "parent_id":h, "source_dialogue_id":None, "is_augmented":False, "emotion":[int(v>0) for v in values], "intensity":[v-1 if v else -100 for v in values], "sentiment":-100})
        splits[split] = parsed
    # Preserve official test rows except ambiguous identical text with conflicting
    # labels. Remove train/dev copies of evaluation text before model fitting.
    labels_by_text = defaultdict(set)
    for rows in splits.values():
        for r in rows:
            labels_by_text[normalize(r["text"])].add(tuple(r["intensity"]))
    conflicts = {text for text, labels in labels_by_text.items() if len(labels) > 1}
    removed_conflicts = {}
    for split in splits:
        before = len(splits[split])
        splits[split] = [r for r in splits[split] if normalize(r["text"]) not in conflicts]
        removed_conflicts[split] = before-len(splits[split])
    excluded_overlap = {}
    seen_later = set()
    for split in ("test", "dev", "train"):
        before = len(splits[split])
        splits[split] = [r for r in splits[split] if normalize(r["text"]) not in seen_later]
        excluded_overlap[split] = before-len(splits[split])
        seen_later.update(normalize(r["text"]) for r in splits[split])
    removed_duplicates = {}
    for split in splits:
        seen, kept = set(), []
        for r in splits[split]:
            key=(normalize(r["text"]),tuple(r["intensity"]))
            if key not in seen:
                kept.append(r);seen.add(key)
        removed_duplicates[split] = len(splits[split])-len(kept)
        splits[split] = kept
    audit = {"dataset":"brighter-dataset/BRIGHTER-emotion-intensities", "revision":revision, "input_hashes":hashes, "unit":"text snippet", "emotion_presence":"derived from Track B intensity > 0; not independent annotations", "split":"official membership with audited exact-text exclusions", "excluded_conflicting_label_records":removed_conflicts, "excluded_train_dev_overlap_with_later_splits":excluded_overlap, "removed_same_split_duplicates":removed_duplicates}
    return save_dataset(output, splits, BRIGHTER_EMOTIONS, audit, sentiment=False)

def load_dataset(path):
    path = Path(path)
    manifest = read_json(path/"manifest.json")
    splits = {}
    for split, info in manifest["files"].items():
        file = path/info["file"]
        if sha256(file) != info["sha256"]:
            raise ValueError(f"Dataset changed: {file}")
        splits[split] = read_jsonl(file)
    validate_splits(splits)
    return splits, manifest
