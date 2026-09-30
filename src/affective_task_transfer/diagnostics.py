from itertools import combinations
from collections import Counter
import random
import torch
from .models import SoftSharing

def select_probe(rows, size, seed=2026):
    """Cover available train label strata, then fill without replacement.

    This is a coverage probe, not an estimate of population prevalence.
    The fixed selection seed keeps it identical across architectures/run seeds.
    """
    def features(row):
        values={("source",bool(row.get("is_augmented",False)))}
        if row["sentiment"]>=0: values.add(("sentiment",row["sentiment"]))
        for j,present in enumerate(row["emotion"]):
            if present:
                values.add(("emotion",j))
                if row["intensity"][j]>=0:values.add(("intensity",j,row["intensity"][j]))
        return values
    strata=[features(r) for r in rows]
    support=Counter(f for values in strata for f in values)
    remaining=list(range(len(rows)));random.Random(seed).shuffle(remaining)
    selected=[];covered=set()
    while remaining and len(selected)<size:
        best=max(remaining,key=lambda i:sum(1/support[f] for f in strata[i]-covered))
        if not strata[best]-covered:break
        selected.append(best);covered.update(strata[best]);remaining.remove(best)
    selected.extend(remaining[:max(0,size-len(selected))])
    missing=set(support)-set().union(*(strata[i] for i in selected))
    return [rows[i] for i in selected],{"selection":"train label coverage followed by seeded sampling without replacement","selection_seed":seed,"requested_records":size,"selected_records":len(selected),"available_strata":len(support),"uncovered_strata":[list(f) for f in sorted(missing)],"interpretation":"Enriched diagnostic probe; not a prevalence-weighted population estimate."}

def capture_batches(model, batches, loss_fns, config, ids):
    """Retain every probe batch; plot averages without calling them pooled gradients."""
    captured=[];offset=0
    for batch in batches:
        n=len(batch["emotion"])
        d=capture(model,batch,loss_fns,config)
        d.update(probe_ids=ids[offset:offset+n],records=n)
        captured.append(d);offset+=n
    if not captured or offset!=len(ids):raise ValueError("Probe IDs and consumed batches differ")
    def average(values,weights):
        first=values[0]
        if isinstance(first,dict):return {k:average([v[k] for v in values],weights) for k in first}
        if isinstance(first,list):return [average([v[i] for v in values],weights) for i in range(len(first))]
        valid=[(v,w) for v,w in zip(values,weights) if v is not None]
        return sum(v*w for v,w in valid)/sum(w for _,w in valid) if valid else None
    weights=[d["records"] for d in captured]
    result={"probe_mode":"eval on fixed training coverage probe","aggregation":"record-weighted mean of batch diagnostics; cosines and norms are not pooled-gradient statistics","shared_parameter_count":captured[0]["shared_parameter_count"],"valid_intensity_labels":sum(d["valid_intensity_labels"] for d in captured),"probe_ids":ids,"batches":captured}
    for key in ("losses","gradient_norms","cosines","regularization_loss","encoder_task_vs_regularization","routing","routing_cosines"):
        if key in captured[0]:result[key]=average([d[key] for d in captured],weights)
    return result

def gradients(loss, parameters):
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True) if parameters else ()

def norm(gs):
    return float(sum((g.detach().double().square().sum() for g in gs if g is not None), torch.tensor(0.0)).sqrt())

def similarity(a,b):
    na,nb=norm(a),norm(b)
    if na==0 or nb==0: return None
    return float(sum((x.detach().double().mul(y.detach().double()).sum() for x,y in zip(a,b) if x is not None and y is not None),torch.tensor(0.0))/(na*nb))

def capture(model, batch, loss_fns, config):
    """Fixed train probe, eval-mode dropout, no .grad writes, RNG restored."""
    training = model.training
    device=next(model.parameters()).device
    devices=[device.index or 0] if device.type=="cuda" else []
    routing=[]; hooks=[]
    try:
        model.eval()
        if hasattr(model.core,"mmoe"):
            for gate in model.core.mmoe.gates:
                hooks.append(gate.register_forward_hook(lambda m,i,o:routing.append(o.detach())))
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            out=model(batch["input_ids"],batch["attention_mask"])
            losses=loss_fns(out,batch)
            parameters=model.diagnostic_parameters()
            grads={t:gradients(loss,parameters) for t,loss in losses.items()}
            result={"probe_mode":"eval on fixed training examples", "shared_parameter_count":sum(p.numel() for p in parameters),"losses":{t:float(v.detach()) for t,v in losses.items()},"gradient_norms":{t:{"raw":norm(g),"weighted":norm(g)*config["weights"][t]} for t,g in grads.items()},"cosines":{a+"/"+b:similarity(grads[a],grads[b]) for a,b in combinations(grads,2)},"valid_intensity_labels":int(((batch["emotion"]==1)&(batch["intensity"]>=0)).sum())}
            if isinstance(model.core,SoftSharing):
                reg=model.regularization(config)
                result["regularization_loss"]=float(reg.detach())
                result["encoder_task_vs_regularization"]={}
                for task,encoder in model.core.encoders.items():
                    params=list(encoder.parameters())
                    g=gradients(losses[task],params); r=gradients(reg,params)
                    result["encoder_task_vs_regularization"][task]={"task_norm":norm(g),"regularization_norm":norm(r),"cosine":similarity(g,r)}
            if routing:
                from .models import TASKS
                result["routing"]={t:{"mean_weights":w.mean(0).cpu().tolist(),"entropy":float(-(w*w.clamp_min(1e-12).log()).sum(1).mean())} for t,w in zip(TASKS,routing) if t in model.tasks}
                result["routing_cosines"]={a+"/"+b:float(torch.nn.functional.cosine_similarity(routing[TASKS.index(a)].mean(0),routing[TASKS.index(b)].mean(0),dim=0)) for a,b in combinations(model.tasks,2)}
            return result
    finally:
        for hook in hooks: hook.remove()
        model.train(training)
