from itertools import combinations
from datetime import datetime, timezone
from pathlib import Path
import time
from .config import load_config
from .io import write_json, read_json, sha256

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
                run_name=f"study2_192_{dataset_name}_{backbone}_{name}_seed{seed}"
                cfg=load_config(overrides=overrides|{"backbone":backbone,"seed":seed,"dataset":str(Path(dataset).resolve()),"output":str(Path("outputs")/run_name),"matrix_condition":name,"name":run_name})
                path=directory/f"{run_name}.json"
                write_json(path,cfg); configs.append(path.name)
    write_json(directory/"matrix.json",{"study":"II","protocol":"study-ii-192-2026-10-01","dataset":dataset_name,"backbones":backbones,"architectures":architectures,"seeds":SEEDS,"runs":len(configs),"configs":configs,"historical_stl":include_historical_stl,"ablation":include_ablation,"legacy_architectures":include_legacy_architectures,"primary_scope":backbones==PRIMARY_BACKBONES and not any((include_ablation,include_historical_stl,include_legacy_architectures))})
    return {"runs":len(configs),"manifest":str(directory/"matrix.json")}

def run_matrix(path,limit=None):
    from .training import train
    path=Path(path); manifest=read_json(path)
    selected=manifest["configs"][:limit] if limit else manifest["configs"]
    total=len(selected)
    progress_file=path.parent/"progress"/"status.json"
    started=time.perf_counter()
    timed_runs=[]
    results=[]
    def show_progress(completed,phase,run=None):
        # Durations are from this invocation. Different architectures can take
        # very different times, so this is a rough estimate, not a deadline.
        eta=(sum(timed_runs)/len(timed_runs))*(total-completed) if timed_runs else None
        state={"phase":phase,"completed":completed,"total":total,"current_run":run,
               "invocation_elapsed_seconds":time.perf_counter()-started,"eta_seconds":eta,
               "timed_runs":len(timed_runs),"estimate_method":"mean duration of timed runs in this invocation",
               "updated_at_utc":datetime.now(timezone.utc).isoformat()}
        write_json(progress_file,state)
        estimate="pending first completed training run" if eta is None else f"about {int(eta//3600)} h {int((eta%3600)//60):02d} min"
        print(f"matrix {completed}/{total} | {phase}"+
              (f" | {run}" if run else "")+f" | remaining: {estimate}",flush=True)
    show_progress(0,"starting")
    for index,file in enumerate(selected):
        config=load_config(path.parent/file)
        run=Path(config["output"])
        show_progress(index,"running",run.name)
        run_started=time.perf_counter()
        try:
            if (run/"manifest.json").exists():
                previous=read_json(run/"manifest.json")
                if previous["status"]=="trained" and previous["config"]==config:
                    if not (run/"best_model.pt").exists() or sha256(run/"best_model.pt")!=previous.get("checkpoint_sha256"):
                        raise ValueError(f"Missing or changed checkpoint: {run}")
                    if not (run/"dev_predictions.json").exists():
                        raise ValueError(f"Missing development predictions: {run}")
                    status="already_trained"
                elif previous["status"]=="running" and previous["config"]==config:
                    train(config["dataset"],run,config)
                    status="resumed"
                else:
                    raise ValueError(f"Mismatched run: {run}. Preserve it and choose a new directory.")
            else:
                if run.exists() and any(run.iterdir()):
                    raise ValueError(f"Run directory has no manifest: {run}")
                train(config["dataset"],run,config)
                status="trained"
        except Exception:
            show_progress(index,"failed",run.name)
            raise
        if status!="already_trained":
            timed_runs.append(time.perf_counter()-run_started)
        results.append({"run":str(run),"status":status})
        show_progress(index+1,"complete" if index+1==total else status,run.name)
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
        if sha256(run/"best_model.pt")!=run_manifest.get("checkpoint_sha256"):
            raise ValueError(f"Missing or changed checkpoint: {run}")
        if sha256(Path(config["dataset"])/"manifest.json")!=run_manifest["dataset_manifest_sha256"]:
            raise ValueError(f"Dataset changed: {run}")
        result_file=run/f"{split}_results.json"
        if result_file.exists():
            if read_json(result_file)["split"]!=split or not (run/f"{split}_predictions.json").exists():
                raise ValueError(f"Mismatched evaluation: {run}")
            hashes=run_manifest.get("evaluation_sha256",{}).get(split)
            if not hashes or any(sha256(run/name)!=digest for name,digest in hashes.items()):
                raise ValueError(f"Unverified evaluation: {run}")
            results.append({"run":str(run),"status":"already_evaluated"})
            continue
        evaluate_run(config["dataset"],run,split)
        results.append({"run":str(run),"status":"evaluated"})
    return results
