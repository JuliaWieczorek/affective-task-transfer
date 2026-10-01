import csv
import pytest
import torch
from affective_task_transfer.config import load_config
from affective_task_transfer.data import save_dataset
from affective_task_transfer.io import read_json, read_jsonl, sha256, write_json
from affective_task_transfer.reporting import matrix_result_files, report
from affective_task_transfer.label_audit import sample_label_audit
from affective_task_transfer import training


def test_interrupted_epoch_resumes_identically(tmp_path, backbone, rows, monkeypatch):
    splits={}
    for split, numbers in (("train", range(6)), ("dev", range(6, 8)), ("test", range(8, 10))):
        splits[split]=[]
        for i in numbers:
            item=dict(rows[i % len(rows)])
            item.update(id=str(i),text=f"i feel sad {i}",group_id=str(i),parent_id=str(i))
            splits[split].append(item)
    dataset=tmp_path/"data"
    save_dataset(dataset,splits,["sad","other"],{"test":True})
    cfg=load_config(overrides={"backbone":backbone,"architecture":"matched_stl","tasks":["intensity"],
        "epochs":3,"batch_size":2,"max_length":16,"diagnostics":False,"device":"cpu",
        "learning_rate":1e-3,"dropout":0.1})
    complete=tmp_path/"complete"
    training.train(dataset,complete,cfg)

    interrupted=tmp_path/"interrupted"
    original_save=training.save_torch_atomic
    raised=False
    def fail_after_epoch(path,value):
        nonlocal raised
        if path.name=="best_model.pt" and not raised:
            raised=True
            raise RuntimeError("simulated interruption after epoch snapshot")
        return original_save(path,value)
    monkeypatch.setattr(training,"save_torch_atomic",fail_after_epoch)
    with pytest.raises(RuntimeError,match="simulated interruption"):
        training.train(dataset,interrupted,cfg)
    assert (interrupted/"last_epoch.pt").exists()
    monkeypatch.setattr(training,"save_torch_atomic",original_save)
    resumed=training.train(dataset,interrupted,cfg)
    assert resumed["resume_count"]==1
    assert resumed["status"]=="trained"
    assert not (interrupted/"last_epoch.pt").exists()
    assert read_jsonl(complete/"history.jsonl")==read_jsonl(interrupted/"history.jsonl")
    left=torch.load(complete/"best_model.pt",weights_only=True)
    right=torch.load(interrupted/"best_model.pt",weights_only=True)
    assert all(torch.equal(left[key],right[key]) for key in left)


def test_matrix_report_requires_every_verified_evaluation(tmp_path):
    dataset=tmp_path/"data";dataset.mkdir()
    write_json(dataset/"manifest.json",{"schema_version":1})
    run=tmp_path/"run";run.mkdir()
    (run/"best_model.pt").write_bytes(b"checkpoint")
    cfg=load_config(overrides={"dataset":str(dataset),"output":str(run),"tasks":["emotion"]})
    write_json(run/"manifest.json",{"status":"trained","config":cfg,"smoke":False,
        "checkpoint_sha256":sha256(run/"best_model.pt"),
        "dataset_manifest_sha256":sha256(dataset/"manifest.json")})
    matrix=tmp_path/"matrix";matrix.mkdir()
    write_json(matrix/"case.json",cfg)
    write_json(matrix/"matrix.json",{"runs":1,"configs":["case.json"]})
    with pytest.raises(ValueError,match="evaluation"):
        matrix_result_files([matrix/"matrix.json"],"test")
    write_json(run/"test_results.json",{"split":"test"})
    write_json(run/"test_predictions.json",{"ids":[]})
    manifest=read_json(run/"manifest.json")
    manifest["evaluation_sha256"]={"test":{name:sha256(run/name) for name in
        ("test_results.json","test_predictions.json")}}
    write_json(run/"manifest.json",manifest)
    assert matrix_result_files([matrix/"matrix.json"],"test")==[run/"test_results.json"]
    with pytest.raises(ValueError,match="Expected 80"):
        report(tmp_path,tmp_path/"partial_report",matrices=[matrix/"matrix.json"],expected_runs=80)
    assert not (tmp_path/"partial_report").exists()
    (run/"test_results.json").write_text("changed",encoding="utf-8")
    with pytest.raises(ValueError,match="evaluation"):
        matrix_result_files([matrix/"matrix.json"],"test")


def test_label_audit_samples_train_only_with_parent_text(tmp_path, backbone):
    def row(identifier,text,parent,aug,group):
        return {"id":identifier,"text":text,"parent_id":parent,"group_id":group,
                "is_augmented":aug,"emotion":[1],"intensity":[1],"sentiment":0}
    train=[row("short-parent","i","short",False,"a"),
           row("long-parent","i feel sad","long",False,"b"),
           row("short-aug","feel","short",True,"a"),
           row("long-aug","i feel sad sad","long",True,"b")]
    dev=[row("dev","sad","dev",False,"c")]
    test=[row("test","other","test",False,"d")]
    dataset=tmp_path/"data"
    save_dataset(dataset,{"train":train,"dev":dev,"test":test},["sad"],{})
    output=tmp_path/"audit.csv"
    result=sample_label_audit(dataset,output,backbone,per_stratum=1,historical_cutoff=4)
    with output.open(encoding="utf-8-sig",newline="") as file:
        sampled=list(csv.DictReader(file))
    assert len(sampled)==4
    assert {r["id"] for r in sampled}=={r["id"] for r in train}
    assert next(r for r in sampled if r["id"]=="short-aug")["source_text"]=="i"
    assert all(not r["emotion_plausible"] for r in sampled)
    assert result["strata_counts"]=={s:1 for s in
        ("original_short","original_long","augmented_short","augmented_long")}
