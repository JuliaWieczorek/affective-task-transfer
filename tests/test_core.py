import ast
import hashlib
from pathlib import Path
import pytest
import torch
from affective_task_transfer.config import load_config
from affective_task_transfer.models import build_model
from affective_task_transfer.losses import TaskLosses
from affective_task_transfer.metrics import evaluate_predictions
from affective_task_transfer.diagnostics import capture
from affective_task_transfer.diagnostics import capture_batches, select_probe
from affective_task_transfer.training import make_loader
from transformers import AutoTokenizer
from affective_task_transfer.io import read_json

def batch():
    return {"input_ids":torch.tensor([[2,5,6,3],[2,5,7,3]]),"attention_mask":torch.ones(2,4,dtype=torch.long),"emotion":torch.tensor([[1,0],[1,1]]),"intensity":torch.tensor([[1,-100],[2,0]]),"sentiment":torch.tensor([0,1])}

@pytest.mark.parametrize("arch",["hard_sharing","soft_sharing","adapters","mmoe","matched_stl","single_task","single_emotion","bert_lstm","cross_stitch"])
def test_architectures_forward_backward(backbone,rows,arch):
    tasks=["emotion"] if arch in ("matched_stl","single_task","single_emotion") else ["sentiment","emotion","intensity"]
    cfg=load_config(overrides={"backbone":backbone,"architecture":arch,"tasks":tasks,"expert_hidden":8,"lstm_hidden":8})
    model=build_model(cfg,2)
    out=model(batch()["input_ids"],batch()["attention_mask"])
    loss=sum(TaskLosses(rows,["a","b"],tasks,"cpu")(out,batch()).values())+model.regularization(cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())

def test_mask_invariance_and_empty(rows):
    loss=TaskLosses(rows,["a","b"],["intensity"],"cpu")
    logits=torch.randn(2,6,requires_grad=True)
    b=batch();out={"intensity":logits}
    expected=loss(out,b)["intensity"]
    expected.backward()
    assert torch.equal(logits.grad[0,3:],torch.zeros(3))
    changed=logits.detach().clone();changed[0,3:]=1000
    assert torch.allclose(expected,loss({"intensity":changed},b)["intensity"])
    b["intensity"][:]=-100
    assert loss(out,b)["intensity"]==0

def test_metrics_do_not_reward_absence():
    rows=[{"emotion":[1,0,0],"intensity":[2,-100,-100],"sentiment":0}]
    m=evaluate_predictions(rows,{"emotion_probabilities":[[0,0,0]],"intensity":[[0,0,0]]},["a","b","c"],["emotion","intensity"])
    assert m["emotion"]["micro_f1"]==0
    assert m["emotion"]["label_accuracy"]==pytest.approx(2/3)
    assert m["intensity"]["accuracy"]==0
    assert m["intensity"]["support"]==1
    assert m["joint"]["pair_micro_f1"]==0

@pytest.mark.parametrize("arch",["soft_sharing","mmoe"])
def test_diagnostics_do_not_change_training(backbone,rows,arch):
    cfg=load_config(overrides={"backbone":backbone,"architecture":arch,"dropout":0.2,"expert_hidden":8})
    model=build_model(cfg,2);model.train()
    loss=TaskLosses(rows,["a","b"],cfg["tasks"],"cpu")
    state=torch.random.get_rng_state().clone()
    before={k:v.clone() for k,v in model.state_dict().items()}
    result=capture(model,batch(),loss,cfg)
    assert torch.equal(torch.random.get_rng_state(),state)
    assert model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert len(result["gradient_norms"])==3

def test_probe_covers_labels_and_consumes_multiple_batches(backbone,rows):
    selected,meta=select_probe(rows,6)
    assert len(selected)==6 and meta["uncovered_strata"]==[]
    cfg=load_config(overrides={"backbone":backbone,"architecture":"hard_sharing","tasks":["emotion","intensity"],"batch_size":2})
    model=build_model(cfg,2)
    tokenizer=AutoTokenizer.from_pretrained(backbone)
    batches=list(make_loader(selected,tokenizer,cfg))
    result=capture_batches(model,batches,TaskLosses(rows,["a","b"],cfg["tasks"],"cpu"),cfg,[r["id"] for r in selected])
    assert len(result["batches"])==3
    assert result["probe_ids"]==[r["id"] for r in selected]
    assert result["valid_intensity_labels"]==sum(sum(v>=0 for v in r["intensity"]) for r in selected)
    assert all(p.grad is None for p in model.parameters())

def test_training_loader_keeps_singleton_tail_with_previous_batch(backbone,rows):
    cfg=load_config(overrides={"backbone":backbone,"batch_size":5})
    batches=list(make_loader(rows,AutoTokenizer.from_pretrained(backbone),cfg,shuffle=True))
    assert len(batches)==1
    assert len(batches[0]["sentiment"])==6

def test_inherited_classes_match_recorded_source_hashes():
    root=Path(__file__).resolve().parents[1]
    source=(root/"src/affective_task_transfer/models/inherited.py").read_text(encoding="utf-8")
    lines=source.splitlines(keepends=True)
    nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)}
    for record in read_json(root/"docs/provenance.json")["classes"]:
        node=nodes[record["symbol"]]
        segment="".join(lines[node.lineno-1:node.end_lineno])
        assert hashlib.sha256(segment.encode()).hexdigest()==record["sha256"]
