"""Extract actual class source, without importing/executing the old training script.

Run once during migration. The resulting module is standalone and versioned.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "mtl-emotion-intensity-sentiment/EMOTIA-ML/multi_emotion_sentiment_intensity_classifier.py"
NAMES = {"MultiTaskBERT", "AdapterModule", "MultiTaskBERTWithAdapters", "MMOE_Core", "MultiTaskMMOE", "SoftSharingModel", "SingleTaskModel", "SingleTaskSentiment", "SingleTaskEmotion", "SingleTaskIntensity", "MultiTaskBERTLSTM", "CrossStitchModel", "FocalLoss", "MultiTaskDataset"}

def main():
    source = SOURCE.read_text(encoding="utf-8-sig")
    lines = source.splitlines(keepends=True)
    # Python's later definition shadows the earlier SoftSharingModel.
    selected = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in NAMES}
    destination = ROOT / "src/affective_task_transfer/models/inherited.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pieces = ['"""Classes extracted verbatim from Julia Wieczorek\'s Chapter 5 code.\nSee docs/provenance.json. Corrections live in wrapper modules.\n"""\nimport torch\nfrom torch import nn\nfrom torch.utils.data import Dataset\nfrom transformers import AutoModel\n\n']
    provenance = []
    for name, node in selected.items():
        segment = "".join(lines[node.lineno-1:node.end_lineno])
        pieces.append(segment + "\n\n")
        provenance.append({"symbol": name, "source_start": node.lineno, "source_end": node.end_lineno, "sha256": hashlib.sha256(segment.encode()).hexdigest()})
    destination.write_text("".join(pieces), encoding="utf-8")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/provenance.json").write_text(json.dumps({"source_repository": "https://github.com/JuliaWieczorek/mtl-emotion-intensity-sentiment", "source_commit": "b5d8aaae5266229970b641f0b3b2e6e6da8607ea", "source_path": "EMOTIA-ML/multi_emotion_sentiment_intensity_classifier.py", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "destination": str(destination.relative_to(ROOT)), "classes": provenance}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
