from itertools import combinations
from pathlib import Path
from .config import load_config
from .io import write_json, read_json

BACKBONES=["bert-base-uncased","bert-base-cased","roberta-base","xlm-roberta-base"]
SEEDS=[42,52,62,72,82]

def matrix(dataset,output,dataset_name="meisd",include_ablation=False,include_historical_stl=False):
    directory=Path(output)
    if directory.exists() and any(directory.iterdir()):raise FileExistsError(directory)
    tasks=["sentiment","emotion","intensity"] if dataset_name=="meisd" else ["emotion","intensity"]
    conditions=[]
    for task in tasks:
        conditions.append((f"stl_{task}",{"architecture":"matched_stl","tasks":[task]}))
        if include_historical_stl:
            conditions.append((f"historical_stl_{task}",{"architecture":"single_task","tasks":[task]}))
    if len(tasks)==3:
        for pair in combinations(tasks,2):
            conditions.append(("soft_"+"_".join(pair),{"architecture":"soft_sharing","tasks":list(pair)}))
    for architecture in ["hard_sharing","soft_sharing","adapters","mmoe","bert_lstm","cross_stitch"]:
        conditions.append((architecture+"_"+"_".join(tasks),{"architecture":architecture,"tasks":tasks}))
    if include_ablation:
        for shared,coefficient in [(False,0),(True,0),(False,1e-4)]:
            conditions.append((f"ablation_projection{int(shared)}_l2{int(coefficient>0)}",{"architecture":"soft_sharing","tasks":tasks,"shared_projection":shared,"soft_lambda":coefficient}))
    configs=[]
    for backbone in BACKBONES:
        for name,overrides in conditions:
            for seed in SEEDS:
                run_name=f"{dataset_name}_{backbone}_{name}_seed{seed}"
                cfg=load_config(overrides=overrides|{"backbone":backbone,"seed":seed,"dataset":str(Path(dataset).resolve()),"output":str(Path("outputs")/run_name),"matrix_condition":name,"name":run_name})
                path=directory/f"{run_name}.json"
                write_json(path,cfg); configs.append(path.name)
    write_json(directory/"matrix.json",{"dataset":dataset_name,"backbones":BACKBONES,"seeds":SEEDS,"runs":len(configs),"configs":configs,"historical_stl":include_historical_stl,"ablation":include_ablation})
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
