"""Create a reproducible, train-only worksheet for human label review."""
import csv
import random
from pathlib import Path
from transformers import AutoTokenizer
from .config import DEFAULTS
from .data import load_dataset
from .io import sha256, write_json


def sample_label_audit(dataset, output, backbone="bert-base-uncased", seed=2026,
                       per_stratum=20, historical_cutoff=128):
    if per_stratum<1 or historical_cutoff<1:raise ValueError("Sampling arguments must be positive")
    output=Path(output)
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(output)
    splits,manifest=load_dataset(dataset)
    rows=splits["train"]
    tokenizer=AutoTokenizer.from_pretrained(backbone,local_files_only=True)
    originals={r["parent_id"]:r for r in rows if not r["is_augmented"]}
    strata={f"{source}_{length}":[] for source in ("original","augmented")
            for length in ("short","long")}
    for row in rows:
        tokens=len(tokenizer(row["text"],truncation=False,add_special_tokens=True)["input_ids"])
        source="augmented" if row["is_augmented"] else "original"
        length="long" if tokens>historical_cutoff else "short"
        strata[f"{source}_{length}"].append((row,tokens))
    rng=random.Random(seed)
    selected=[]
    for stratum,candidates in strata.items():
        if len(candidates)<per_stratum:
            raise ValueError(f"Only {len(candidates)} train records in {stratum}; requested {per_stratum}")
        selected.extend((stratum,row,tokens) for row,tokens in rng.sample(candidates,per_stratum))
    if any(row["parent_id"] not in originals for _,row,_ in selected):
        raise ValueError("A selected example has no source text in training")
    fields=["id","stratum","is_augmented","tokens","source_text","text_to_review",
            "emotion_labels","intensity_labels","sentiment_label","emotion_plausible",
            "intensity_plausible","sentiment_plausible","augmentation_preserves_labels",
            "notes"]
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        for stratum,row,tokens in selected:
            parent=originals[row["parent_id"]]
            writer.writerow({"id":row["id"],"stratum":stratum,"is_augmented":row["is_augmented"],
                "tokens":tokens,"source_text":parent["text"],"text_to_review":row["text"],
                "emotion_labels":"; ".join(e for j,e in enumerate(manifest["emotions"]) if row["emotion"][j]),
                "intensity_labels":"; ".join(f"{e}:{row['intensity'][j]+1 if row['intensity'][j]>=0 else 'unknown'}"
                    for j,e in enumerate(manifest["emotions"]) if row["emotion"][j]),
                "sentiment_label":("negative","neutral","positive")[row["sentiment"]],
                "emotion_plausible":"","intensity_plausible":"","sentiment_plausible":"",
                "augmentation_preserves_labels":"","notes":""})
    audit={"dataset_manifest_sha256":sha256(Path(dataset)/"manifest.json"),"sample_sha256":sha256(output),
           "source_split":"train only","seed":seed,"per_stratum":per_stratum,
           "historical_cutoff":historical_cutoff,"current_max_length":DEFAULTS["max_length"],
           "strata_counts":{k:len(v) for k,v in strata.items()},
           "reviewer_instructions":"Complete the blank columns by reading the source and target text; generated labels were inherited from the source. Keep the worksheet local and record reviewer/date and disagreements separately."}
    write_json(output.with_suffix(".manifest.json"),audit)
    return audit
