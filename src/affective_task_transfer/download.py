"""Public BRIGHTER English download; pin revision and retain official splits."""
import json
from pathlib import Path
import urllib.request
from .io import write_json, sha256

REPO = "brighter-dataset/BRIGHTER-emotion-intensities"

def download_brighter(output):
    output=Path(output)
    if (output/"download_manifest.json").exists():
        raise FileExistsError("BRIGHTER download already exists; use its manifest/revision")
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}",timeout=60) as response:
        info=json.load(response)
    revision=info["sha"]
    files=[x["rfilename"] for x in info["siblings"]]
    output.mkdir(parents=True,exist_ok=True)
    sources={}
    import pandas as pd
    for split in ("train","dev","test"):
        aliases=[split] if split!="dev" else ["dev","validation"]
        matches=[f for f in files if (f.startswith("eng/") or "/eng/" in f or "eng_" in f) and any(a in Path(f).stem for a in aliases) and f.endswith((".parquet",".csv"))]
        if not matches:
            raise ValueError(f"Cannot identify English {split} in upstream file list: {files}")
        frames=[];upstream_hashes={}
        for name in sorted(matches):
            target=output/Path(name).name
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}",target)
            upstream_hashes[name]=sha256(target)
            frames.append(pd.read_parquet(target) if name.endswith(".parquet") else pd.read_csv(target))
        frame=pd.concat(frames,ignore_index=True)
        if len(frame)==0:raise ValueError("Empty upstream split")
        frame.to_csv(output/f"{split}.csv",index=False)
        sources[split]={"upstream_files":sorted(matches),"upstream_sha256":upstream_hashes,"records":len(frame),"sha256":sha256(output/f"{split}.csv")}
    result={"repository":REPO,"revision":revision,"splits":sources}
    write_json(output/"download_manifest.json",result)
    return result
