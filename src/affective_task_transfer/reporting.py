"""Report separate task results and paired seed deltas, retaining raw evidence."""
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from .io import read_json, read_jsonl, write_json

def write_csv(path,rows):
    if not rows:return
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)

def conditional_task_deltas(rows):
    """Measured effect of adding one task to a soft-sharing pair on the same seed."""
    pairs={(r["dataset_hash"],r["backbone"],r["seed"],r["task"],r["metric"],frozenset(r["tasks"].split("+"))):r
           for r in rows if r["architecture"]=="soft_sharing" and len(r["tasks"].split("+"))==2}
    result=[]
    for triple in rows:
        if triple["architecture"]!="soft_sharing" or len(triple["tasks"].split("+"))!=3 or triple["task"]=="joint":continue
        tasks=set(triple["tasks"].split("+"))
        for added in tasks-{triple["task"]}:
            pair_tasks=frozenset(tasks-{added})
            pair=pairs.get((triple["dataset_hash"],triple["backbone"],triple["seed"],triple["task"],triple["metric"],pair_tasks))
            if pair is None:continue
            delta=triple["value"]-pair["value"]
            result.append({k:triple[k] for k in ("dataset_hash","backbone","seed","task","metric")}|{"pair_condition":pair["condition"],"triple_condition":triple["condition"],"added_task":added,"delta_triple_minus_pair":delta,"benefit":(-delta if triple["metric"]=="mae" else delta)})
    return result

def plot_run(run,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history=read_jsonl(run/"history.jsonl")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for task in history[0]["train_losses"]:
        axes[0].plot([r["epoch"] for r in history],[r["train_losses"][task] for r in history],label=task+" train")
        axes[0].plot([r["epoch"] for r in history],[r["dev_losses"][task] for r in history],linestyle="--",label=task+" dev")
        axes[1].plot([r["epoch"] for r in history],[r["dev_metrics"][task]["emotion_macro_f1" if task=="intensity" else "macro_f1"] for r in history],label=task)
    axes[0].set_title("Task losses");axes[1].set_title("Development macro-F1")
    for ax in axes:ax.set_xlabel("Epoch");ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/f"{run.name}_learning.png",dpi=150);plt.close(fig)
    if (run/"diagnostics.jsonl").exists():
        ds=read_jsonl(run/"diagnostics.jsonl")
        if ds and ds[0]["cosines"]:
            fig,ax=plt.subplots(figsize=(6,4))
            for pair in ds[0]["cosines"]:ax.plot([d["epoch"] for d in ds],[d["cosines"][pair] for d in ds],label=pair)
            ax.axhline(0,color="gray",linewidth=0.7);ax.set(xlabel="Epoch",ylabel="Gradient cosine (fixed train probe)");ax.legend();fig.tight_layout();fig.savefig(output/f"{run.name}_gradients.png",dpi=150);plt.close(fig)
        if ds and "routing" in ds[-1]:
            routing=ds[-1]["routing"];fig,ax=plt.subplots(figsize=(6,4))
            im=ax.imshow([v["mean_weights"] for v in routing.values()],vmin=0,vmax=1)
            ax.set_yticks(range(len(routing)),list(routing));ax.set_xlabel("Expert");fig.colorbar(im,ax=ax,label="Mean gate weight");fig.tight_layout();fig.savefig(output/f"{run.name}_routing.png",dpi=150);plt.close(fig)

def report(root,output,split="test",allow_smoke=False):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]; runs={}; per_class=[]; confusion={}; resources=[]
    for file in sorted(root.rglob(f"{split}_results.json")):
        run=file.parent; manifest=read_json(run/"manifest.json");result=read_json(file);cfg=manifest["config"]
        if manifest["smoke"] and not allow_smoke:continue
        condition=cfg.get("matrix_condition") or cfg["architecture"]+"_"+"_".join(cfg["tasks"])
        dataset_hash=manifest["dataset_manifest_sha256"]
        runs[str(run)]={"manifest":manifest,"result":result,"condition":condition}
        resources.append({"run":str(run),"dataset_hash":dataset_hash,"backbone":cfg["backbone"],"condition":condition,"seed":cfg["seed"],"parameters":manifest["parameters"],"trainable_parameters":manifest["trainable_parameters"],"elapsed_seconds":manifest["elapsed_seconds"],"diagnostic_seconds":manifest["diagnostic_seconds"],"optimizer_steps":manifest["optimizer_steps"],"peak_cuda_bytes":manifest["peak_cuda_bytes"],"train_truncated_fraction":manifest["token_audit"]["train"]["truncated_fraction"]})
        for task in [*cfg["tasks"],*(["joint"] if "joint" in result["metrics"] else [])]:
            values=result["metrics"][task]
            detail={"overall":values}
            if task in ("emotion","intensity"):
                detail=values["per_emotion"]
            for emotion,scope in detail.items():
                confusion[f"{run}/{task}/{emotion}"]=scope["confusion_matrix"]
                for label,scores in scope["per_class"].items():
                    per_class.append({"run":str(run),"dataset_hash":dataset_hash,"backbone":cfg["backbone"],"condition":condition,"seed":cfg["seed"],"task":task,"emotion":emotion,"class":label,**scores})
            for metric in ("accuracy","subset_accuracy","label_accuracy","macro_precision","macro_recall","macro_f1","weighted_f1","micro_f1","mae","emotion_macro_f1","emotion_macro_accuracy","pair_precision","pair_recall","pair_micro_f1"):
                value=values.get(metric)
                if value is None:continue
                rows.append({"run":str(run),"dataset_hash":dataset_hash,"backbone":cfg["backbone"],"condition":condition,"architecture":cfg["architecture"],"tasks":"+".join(cfg["tasks"]),"seed":cfg["seed"],"task":task,"metric":metric,"value":value,"smoke":manifest["smoke"]})
        plot_run(run,output)
    write_csv(output/"per_seed.csv",rows)
    baseline_rows=[]
    for run,item in runs.items():
        cfg=item["manifest"]["config"]
        for task,values in item["result"].get("majority_baseline",{}).items():
            for metric in ("accuracy","macro_f1","micro_f1","emotion_macro_f1","pair_micro_f1","mae"):
                value=values.get(metric)
                if value is not None:
                    baseline_rows.append({"run":run,"dataset_hash":item["manifest"]["dataset_manifest_sha256"],"backbone":cfg["backbone"],"seed":cfg["seed"],"task":task,"metric":metric,"value":value})
    write_csv(output/"majority_baselines.csv",baseline_rows)
    write_csv(output/"resources.csv",resources)
    write_csv(output/"per_class.csv",per_class)
    write_json(output/"confusion_matrices.json",confusion)
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in ("dataset_hash","backbone","condition","task","metric"))].append(r)
    summary=[]
    for key,values in grouped.items():
        if len({r["seed"] for r in values})!=len(values):raise ValueError(f"Duplicate seeds for {key}")
        nums=[r["value"] for r in values]
        summary.append(dict(zip(("dataset_hash","backbone","condition","task","metric"),key))|{"n_seeds":len(nums),"mean":float(np.mean(nums)),"sd":float(np.std(nums,ddof=1)) if len(nums)>1 else None})
    write_csv(output/"summary.csv",summary)
    deltas=[]
    for r in rows:
        if "+" not in r["tasks"]:continue
        base=[b for b in rows if b["architecture"]=="matched_stl" and b["tasks"]==r["task"] and all(b[k]==r[k] for k in ("dataset_hash","backbone","seed","task","metric"))]
        if len(base)==1:
            delta=r["value"]-base[0]["value"]
            direction=-1 if r["metric"]=="mae" else 1
            deltas.append({k:r[k] for k in ("dataset_hash","backbone","condition","seed","task","metric")}|{"delta_mtl_minus_stl":delta,"improvement_direction":direction,"benefit":delta*direction,"transfer":"positive" if delta*direction>0 else "negative" if delta*direction<0 else "tie"})
    write_csv(output/"paired_deltas.csv",deltas)
    conditional=conditional_task_deltas(rows)
    write_csv(output/"conditional_deltas.csv",conditional)
    class_index={(r["dataset_hash"],r["backbone"],r["seed"],r["task"],r["emotion"],r["class"]):r for r in per_class if r["condition"]=="stl_"+r["task"]}
    class_deltas=[]
    for r in per_class:
        if r["condition"]=="stl_"+r["task"]:continue
        key=(r["dataset_hash"],r["backbone"],r["seed"],r["task"],r["emotion"],r["class"])
        base=class_index.get(key)
        if base is None:continue
        for metric in ("precision","recall","f1-score"):
            class_deltas.append({k:r[k] for k in ("dataset_hash","backbone","condition","seed","task","emotion","class")}|{"metric":metric,"delta_mtl_minus_stl":r[metric]-base[metric],"support":r["support"]})
    write_csv(output/"class_paired_deltas.csv",class_deltas)
    paired=defaultdict(list)
    for row in deltas:
        paired[tuple(row[k] for k in ("dataset_hash","backbone","condition","task","metric"))].append(row)
    paired_summary=[]
    for key,values in paired.items():
        if len({r["seed"] for r in values})!=len(values):raise ValueError(f"Duplicate paired seeds for {key}")
        benefits=np.array([r["benefit"] for r in values],dtype=float)
        mean=float(benefits.mean());sd=float(benefits.std(ddof=1)) if len(benefits)>1 else None
        if len(benefits)>1:
            from scipy.stats import t
            margin=float(t.ppf(0.975,len(benefits)-1)*sd/np.sqrt(len(benefits)))
        else:margin=None
        paired_summary.append(dict(zip(("dataset_hash","backbone","condition","task","metric"),key))|{"n_paired_seeds":len(benefits),"mean_benefit":mean,"sd_benefit":sd,"ci95_low":mean-margin if margin is not None else None,"ci95_high":mean+margin if margin is not None else None,"positive_seeds":sum(r["transfer"]=="positive" for r in values),"negative_seeds":sum(r["transfer"]=="negative" for r in values),"ties":sum(r["transfer"]=="tie" for r in values)})
    write_csv(output/"paired_summary.csv",paired_summary)
    write_json(output/"report_manifest.json",{"split":split,"allow_smoke":allow_smoke,"runs":list(runs),"summary_rows":len(summary),"paired_rows":len(deltas),"paired_summary_rows":len(paired_summary),"conditional_rows":len(conditional),"class_paired_rows":len(class_deltas),"interpretation":"Paired seed differences on one split use the same backbone and seed. Positive benefit means MTL outperformed matched STL; MAE is inverted. Conditional differences compare the soft-sharing pair with its three-task model. The t interval is descriptive with five seeds and does not cover dataset sampling or label uncertainty. Gradient agreement and routing do not prove causal knowledge flow. No cross-dataset pooling."})
    text=["# Experiment report", "", f"Split: {split}. Runs: {len(runs)}. Smoke inclusion: {allow_smoke}.","", "See per_seed.csv, per_class.csv, class_paired_deltas.csv, majority_baselines.csv, conditional_deltas.csv, confusion_matrices.json, summary.csv, paired_deltas.csv, paired_summary.csv and resources.csv.","", "Gradient agreement and routing are diagnostics, not proof of knowledge flow. Positive/negative transfer is evaluated against the paired STL baseline; inspect task trade-offs and intervals across seeds."]
    (output/"README.md").write_text("\n".join(text),encoding="utf-8")
    return {"runs":len(runs),"output":str(output)}
