from itertools import combinations
import torch
from .models import SoftSharing

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
