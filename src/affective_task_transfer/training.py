"""Refactored historical AdamW/warmup/early-stopping pipeline.

Changes: masked losses, train-only weights, accumulation-aware scheduler,
development-only selection, explicit test stage, isolated diagnostics.
"""
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import importlib.metadata
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from .config import load_config
from .data import load_dataset
from .io import write_json, read_json, write_jsonl, sha256
from .models import build_model, SoftSharing
from .models.inherited import MultiTaskDataset
from .losses import TaskLosses
from .metrics import evaluate_predictions, selection_score, tune_threshold, majority_baseline
from .diagnostics import capture_batches, select_probe

class RecordsDataset(MultiTaskDataset):
    def __init__(self, records, tokenizer, max_length):
        super().__init__([r["text"] for r in records],[r["sentiment"] for r in records],[r["emotion"] for r in records],[r["intensity"] for r in records],tokenizer,max_length)

    def __getitem__(self, index):
        batch=super().__getitem__(index)
        batch["emotion"]=batch.pop("emotions")
        batch["intensity"]=batch.pop("intensities")
        return batch

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def move(batch,device):
    return {k:v.to(device) for k,v in batch.items()}

class MergeSingletonTail(BatchSampler):
    """Keep every example and avoid a size-one training batch for legacy BatchNorm."""
    def __iter__(self):
        batches=list(super().__iter__())
        if len(batches)>1 and len(batches[-1])==1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        total=super().__len__()
        return total-1 if total>1 and len(self.sampler)%self.batch_size==1 else total

def make_loader(rows,tokenizer,config,shuffle=False):
    generator=torch.Generator().manual_seed(config["seed"])
    dataset=RecordsDataset(rows,tokenizer,config["max_length"])
    if shuffle and len(dataset)>1 and config["batch_size"]>1:
        sampler=RandomSampler(dataset,generator=generator)
        batches=MergeSingletonTail(sampler,batch_size=config["batch_size"],drop_last=False)
        return DataLoader(dataset,batch_sampler=batches,num_workers=config["num_workers"])
    return DataLoader(dataset,batch_size=config["batch_size"],shuffle=shuffle,generator=generator,num_workers=config["num_workers"])

def infer(model,rows,tokenizer,config,loss_fns=None):
    model.eval(); device=next(model.parameters()).device
    predictions={}; sums={}; supports={}
    with torch.no_grad():
        for batch in make_loader(rows,tokenizer,config):
            batch=move(batch,device)
            outputs=model(batch["input_ids"],batch["attention_mask"])
            if loss_fns:
                losses=loss_fns(outputs,batch)
                for task,value in losses.items():
                    count=int(((batch["emotion"]==1)&(batch["intensity"]>=0)).sum()) if task=="intensity" else len(batch["sentiment"])
                    sums[task]=sums.get(task,0)+float(value)*count
                    supports[task]=supports.get(task,0)+count
            for task,value in outputs.items():
                if value is None:continue
                if task=="emotion": key="emotion_probabilities"; pred=value.sigmoid()
                elif task=="intensity": key=task; pred=value.reshape(value.shape[0],-1,3).argmax(-1)
                else:key=task;pred=value.argmax(-1)
                predictions.setdefault(key,[]).extend(pred.cpu().tolist())
    return predictions,{t:sums[t]/supports[t] if supports[t] else None for t in sums}

def clip_gradients(model,config):
    # Independent clipping when the no-sharing control has independent parameter blocks.
    core=model.core
    if isinstance(core,SoftSharing) and not core.shared_projection and config["soft_lambda"]==0:
        for t in core.tasks:
            torch.nn.utils.clip_grad_norm_(list(core.encoders[t].parameters())+list(core.projections[t].parameters())+list(core.heads[t].parameters()),config["clip_norm"])
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(),config["clip_norm"])

def token_audit(rows,tokenizer,max_length):
    lengths=[len(tokenizer(r["text"],truncation=False,add_special_tokens=True)["input_ids"]) for r in rows]
    return {"records":len(rows),"max_tokens":max(lengths),"mean_tokens":float(np.mean(lengths)),"truncated_records":sum(n>max_length for n in lengths),"truncated_fraction":sum(n>max_length for n in lengths)/len(rows)}

def environment():
    packages={n:importlib.metadata.version(n) for n in ("torch","transformers","numpy","scikit-learn")}
    try:commit=subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):commit=None
    return {"python":platform.python_version(),"platform":platform.platform(),"packages":packages,"git_commit":commit,"cuda":torch.version.cuda}

def train(dataset, output, config):
    config=load_config(overrides=config)
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(f"Run directory must be empty: {output}")
    output.mkdir(parents=True,exist_ok=True)
    splits,manifest=load_dataset(dataset)
    if set(config["tasks"])-set(manifest["tasks"]):raise ValueError("Dataset lacks a requested task")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(config["threads"])
    seed_everything(config["seed"])
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if config["device"]=="auto" else config["device"])
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
    rows=splits["train"][:config["train_limit"]] if config["train_limit"] else splits["train"]
    dev=splits["dev"][:config["eval_limit"]] if config["eval_limit"] else splits["dev"]
    tokenizer=AutoTokenizer.from_pretrained(config["backbone"])
    model=build_model(config,len(manifest["emotions"])).to(device)
    tokenizer.save_pretrained(output/"tokenizer")
    # Save encoder config, needed for model reconstruction on another machine.
    core=model.core
    encoder=next(iter(core.encoders.values())) if hasattr(core,"encoders") else next(getattr(core,n) for n in ("encoder","transformer","bert","encoder_a") if hasattr(core,n))
    encoder.config.save_pretrained(output/"encoder_config")
    loss_fns=TaskLosses(rows,manifest["emotions"],config["tasks"],device,config["focal_gamma"])
    loader=make_loader(rows,tokenizer,config,shuffle=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    updates_per_epoch=math.ceil(len(loader)/config["accumulation_steps"])
    total_steps=updates_per_epoch*config["epochs"]
    if config["max_steps"]:total_steps=min(total_steps,config["max_steps"])
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total_steps*config["warmup_ratio"]),total_steps)
    run_manifest={"config":config,"environment":environment(),"dataset_manifest_sha256":sha256(Path(dataset)/"manifest.json"),"dataset_manifest":manifest,"status":"running","smoke":config["smoke"],"deterministic_algorithms":True,"parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"train_records_used":len(rows),"dev_records_used":len(dev),"token_audit":{"train":token_audit(rows,tokenizer,config["max_length"]),"dev":token_audit(dev,tokenizer,config["max_length"])},"class_weights":{"emotion":loss_fns.em_weights.cpu().tolist(),"sentiment":loss_fns.sent_weights.cpu().tolist() if loss_fns.sent_weights is not None else None},"encoder_revision":getattr(encoder.config,"_commit_hash",None),"selection":"mean dev task macro-F1; intensity macro over emotions; emotion threshold selected on dev"}
    write_json(output/"manifest.json",run_manifest)
    probe_rows,probe_manifest=select_probe(rows,config["diagnostic_batch_size"])
    write_json(output/"probe_manifest.json",probe_manifest|{"ids":[r["id"] for r in probe_rows]})
    best=-float("inf"); best_epoch=0; steps=0; history=[]; diagnostics=[]
    started=time.perf_counter(); diagnostic_seconds=0
    for epoch in range(1,config["epochs"]+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        raw_sums={}; counts={}; reg_sum=0; batches=0
        for index,batch in enumerate(loader):
            batch=move(batch,device)
            out=model(batch["input_ids"],batch["attention_mask"])
            losses=loss_fns(out,batch)
            reg=model.regularization(config)
            weighted=sum(config["weights"][t]*v for t,v in losses.items())+reg
            start=(index//config["accumulation_steps"])*config["accumulation_steps"]
            window=min(config["accumulation_steps"],len(loader)-start)
            (weighted/window).backward()
            for task,value in losses.items():
                n=int(((batch["emotion"]==1)&(batch["intensity"]>=0)).sum()) if task=="intensity" else len(batch["sentiment"])
                raw_sums[task]=raw_sums.get(task,0)+float(value.detach())*n; counts[task]=counts.get(task,0)+n
            reg_sum+=float(reg.detach());batches+=1
            if (index+1)%config["accumulation_steps"]==0 or index+1==len(loader):
                clip_gradients(model,config); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); steps+=1
                if steps>=total_steps:break
        if config["diagnostics"]:
            t=time.perf_counter()
            probe_batches=(move(b,device) for b in make_loader(probe_rows,tokenizer,config))
            d=capture_batches(model,probe_batches,loss_fns,config,[r["id"] for r in probe_rows])
            diagnostic_seconds+=time.perf_counter()-t
            d.update(epoch=epoch,optimizer_steps=steps,probe_ids=[r["id"] for r in probe_rows]);diagnostics.append(d)
            write_jsonl(output/"diagnostics.jsonl",diagnostics)
        predictions,dev_losses=infer(model,dev,tokenizer,config,loss_fns)
        threshold=tune_threshold(dev,predictions,manifest["emotions"],config["tasks"],config["threshold_candidates"])
        metrics=evaluate_predictions(dev,predictions,manifest["emotions"],config["tasks"],threshold)
        score=selection_score(metrics,config["tasks"])
        losses_logged={t:raw_sums[t]/counts[t] if counts[t] else None for t in raw_sums}
        entry={"epoch":epoch,"optimizer_steps":steps,"train_losses":losses_logged,"weighted_train_losses":{t:v*config["weights"][t] if v is not None else None for t,v in losses_logged.items()},"regularization_loss":reg_sum/batches,"dev_losses":dev_losses,"dev_metrics":metrics,"threshold":threshold,"selection_score":score}
        history.append(entry);write_jsonl(output/"history.jsonl",history)
        print(f"epoch={epoch} steps={steps} dev_score={score:.4f}",flush=True)
        if score>best:
            best=score;best_epoch=epoch
            torch.save(model.state_dict(),output/"best_model.pt")
            write_json(output/"best.json",{"epoch":epoch,"threshold":threshold,"score":score,"dev_metrics":metrics})
        if epoch-best_epoch>=config["patience"] or steps>=total_steps:break
    run_manifest.update(status="trained",best_epoch=best_epoch,optimizer_steps=steps,elapsed_seconds=time.perf_counter()-started,diagnostic_seconds=diagnostic_seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else None,checkpoint_sha256=sha256(output/"best_model.pt"),seconds_per_optimizer_step=(time.perf_counter()-started)/max(steps,1))
    write_json(output/"manifest.json",run_manifest)
    # Development predictions only. Final test evaluation is an explicit command.
    model.load_state_dict(torch.load(output/"best_model.pt",map_location=device,weights_only=True))
    predictions,_=infer(model,dev,tokenizer,config)
    write_json(output/"dev_predictions.json",{"ids":[r["id"] for r in dev],"predictions":predictions})
    return run_manifest

def evaluate_run(dataset,run,split="test"):
    run=Path(run); manifest=read_json(run/"manifest.json");config=manifest["config"]
    if manifest["status"]!="trained":raise ValueError("Run has not completed training")
    if sha256(run/"best_model.pt")!=manifest["checkpoint_sha256"]:raise ValueError("Checkpoint changed after training")
    if sha256(Path(dataset)/"manifest.json")!=manifest["dataset_manifest_sha256"]:raise ValueError("Evaluation dataset differs from training manifest")
    if split not in ("dev","test"):raise ValueError(split)
    if (run/f"{split}_results.json").exists():raise FileExistsError("Evaluation already saved; preserve the existing result")
    splits,data_manifest=load_dataset(dataset)
    rows=splits[split][:config["eval_limit"]] if config["eval_limit"] else splits[split]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    seed_everything(config["seed"]);torch.set_num_threads(config["threads"])
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if config["device"]=="auto" else config["device"])
    eval_config=dict(config)
    if manifest.get("encoder_revision") and not Path(config["backbone"]).is_dir():
        # The checkpoint contains every weight. Pin the architecture source to
        # the exact pretrained revision used at training, using cache offline.
        from huggingface_hub import snapshot_download
        options=dict(repo_id=config["backbone"],revision=manifest["encoder_revision"],allow_patterns=["config.json","*.safetensors","*.bin","*.index.json"])
        try:
            eval_config["backbone"]=snapshot_download(**options,local_files_only=True)
        except FileNotFoundError:
            eval_config["backbone"]=snapshot_download(**options)
    model=build_model(eval_config,len(data_manifest["emotions"])).to(device)
    model.load_state_dict(torch.load(run/"best_model.pt",map_location=device,weights_only=True))
    tokenizer=AutoTokenizer.from_pretrained(run/"tokenizer")
    predictions,_=infer(model,rows,tokenizer,config)
    threshold=read_json(run/"best.json")["threshold"]
    metrics=evaluate_predictions(rows,predictions,data_manifest["emotions"],config["tasks"],threshold)
    write_json(run/f"{split}_predictions.json",{"ids":[r["id"] for r in rows],"group_ids":[r["group_id"] for r in rows],"predictions":predictions,"truth":{t:[r[t] for r in rows] for t in config["tasks"]}})
    result={"split":split,"smoke":config["smoke"],"threshold_from_dev":threshold,"metrics":metrics,"majority_baseline":majority_baseline(splits["train"],rows,data_manifest["emotions"],config["tasks"])}
    write_json(run/f"{split}_results.json",result)
    return result
