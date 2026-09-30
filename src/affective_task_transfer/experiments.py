from itertools import combinations
from pathlib import Path
from .config import load_config
from .io import write_json, read_json

BACKBONES=["bert-base-uncased","bert-base-cased","roberta-base","xlm-roberta-base"]
SEEDS=[42,52,62,72,82]
PRIMARY_BACKBONES=["bert-base-uncased"]
PRIMARY_ARCHITECTURES=["hard_sharing","soft_sharing","adapters","mmoe"]

def matrix(dataset,output,dataset_name="meisd",include_ablation=False,include_historical_stl=False,backbones=None,include_legacy_architectures=False):
    if dataset_name not in ("meisd","brighter"):
        raise ValueError("Unknown dataset name")
    backbones=list(PRIMARY_BACKBONES if backbones is None else backbones)
    if not backbones or len(set(backbones))!=len(backbones) or set(backbones)-set(BACKBONES):
        raise ValueError("Select distinct supported backbones")
    architectures=PRIMARY_ARCHITECTURES+(["bert_lstm","cross_stitch"] if include_legacy_architectures else [])
    directory=Path(output)
    if directory.exists() and any(directory.iterdir()):raise FileExistsError(directory)
    tasks=["sentiment","emotion","intensity"] if dataset_name=="meisd" else ["emotion","intensity"]
    conditions=[]
    for task in tasks:
        conditions.append((f"stl_{task}",{"architecture":"matched_stl","tasks":[task],"diagnostics":False}))
        if include_historical_stl:
            conditions.append((f"historical_stl_{task}",{"architecture":"single_task","tasks":[task],"diagnostics":False}))
    if len(tasks)==3:
        for pair in combinations(tasks,2):
            conditions.append(("soft_"+"_".join(pair),{"architecture":"soft_sharing","tasks":list(pair)}))
    for architecture in architectures:
        conditions.append((architecture+"_"+"_".join(tasks),{"architecture":architecture,"tasks":tasks}))
    if include_ablation:
        for shared,coefficient in [(False,0),(True,0),(False,1e-4)]:
            conditions.append((f"ablation_projection{int(shared)}_l2{int(coefficient>0)}",{"architecture":"soft_sharing","tasks":tasks,"shared_projection":shared,"soft_lambda":coefficient}))
    configs=[]
    for backbone in backbones:
        for name,overrides in conditions:
            for seed in SEEDS:
                run_name=f"study2_{dataset_name}_{backbone}_{name}_seed{seed}"
                cfg=load_config(overrides=overrides|{"backbone":backbone,"seed":seed,"dataset":str(Path(dataset).resolve()),"output":str(Path("outputs")/run_name),"matrix_condition":name,"name":run_name})
                path=directory/f"{run_name}.json"
                write_json(path,cfg); configs.append(path.name)
    write_json(directory/"matrix.json",{"study":"II","protocol":"study-ii-2026-09-30","dataset":dataset_name,"backbones":backbones,"architectures":architectures,"seeds":SEEDS,"runs":len(configs),"configs":configs,"historical_stl":include_historical_stl,"ablation":include_ablation,"legacy_architectures":include_legacy_architectures,"primary_scope":backbones==PRIMARY_BACKBONES and not any((include_ablation,include_historical_stl,include_legacy_architectures))})
    return {"runs":len(configs),"manifest":str(directory/"matrix.json")}

def run_matrix(path,limit=None):
    from .training import train
    path=Path(path); manifest=read_json(path)
    selected=manifest["configs"][:limit] if limit else manifest["configs"]
    results=[]
    for file in selected:
        config=load_config(path.parent/file)
        run=Path(config["output"])
        if (run/"manifest.json").exists():
            previous=read_json(run/"manifest.json")
            if previous["status"]=="trained" and previous["config"]==config:
                results.append({"run":str(run),"status":"already_trained"}); continue
            raise ValueError(f"Incomplete or mismatched run: {run}. Preserve it and choose a new directory.")
        train(config["dataset"],run,config)
        results.append({"run":str(run),"status":"trained"})
    return results

def evaluate_matrix(path,split="test",limit=None):
    from .training import evaluate_run
    path=Path(path);manifest=read_json(path)
    selected=manifest["configs"][:limit] if limit else manifest["configs"]
    results=[]
    for file in selected:
        config=load_config(path.parent/file)
        run=Path(config["output"])
        run_manifest=read_json(run/"manifest.json")
        if run_manifest["config"]!=config or run_manifest["status"]!="trained":
            raise ValueError(f"Missing or mismatched trained run: {run}")
        result_file=run/f"{split}_results.json"
        if result_file.exists():
            if read_json(result_file)["split"]!=split:raise ValueError(f"Mismatched evaluation: {run}")
            results.append({"run":str(run),"status":"already_evaluated"})
            continue
        evaluate_run(config["dataset"],run,split)
        results.append({"run":str(run),"status":"evaluated"})
    return results
