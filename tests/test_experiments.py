import pytest
from affective_task_transfer.experiments import matrix, SEEDS, PRIMARY_ARCHITECTURES
from affective_task_transfer.io import read_json

@pytest.mark.parametrize("dataset,n,stl,pairs",[("meisd",50,3,3),("brighter",30,2,0)])
def test_primary_matrix(tmp_path,dataset,n,stl,pairs):
    directory=tmp_path/"matrix"
    result=matrix(tmp_path/"data",directory,dataset)
    manifest=read_json(result["manifest"])
    configs=[read_json(directory/p) for p in manifest["configs"]]
    assert manifest["runs"]==len(configs)==n
    assert manifest["study"]=="II" and manifest["primary_scope"]
    assert manifest["backbones"]==["bert-base-uncased"]
    assert all(c["max_length"]==192 for c in configs)
    assert all("study2_192_" in c["name"] for c in configs)
    assert manifest["architectures"]==PRIMARY_ARCHITECTURES
    assert {c["seed"] for c in configs}==set(SEEDS)
    assert len({c["output"] for c in configs})==n
    assert sum(c["architecture"]=="matched_stl" for c in configs)==stl*5
    assert all(not c["diagnostics"] for c in configs if c["architecture"]=="matched_stl")
    assert all(c["diagnostics"] for c in configs if c["architecture"] in PRIMARY_ARCHITECTURES)
    if dataset=="meisd":
        assert sum(len(c["tasks"])==2 for c in configs)==pairs*5
        assert all(c["architecture"]=="soft_sharing" for c in configs if len(c["tasks"])==2)
    else:
        assert all("sentiment" not in c["tasks"] for c in configs)
    with pytest.raises(FileExistsError):matrix(tmp_path/"data",directory,dataset)

def test_optional_matrix_is_explicit(tmp_path):
    result=matrix(tmp_path/"data",tmp_path/"matrix",backbones=["bert-base-uncased","roberta-base"],include_legacy_architectures=True)
    manifest=read_json(result["manifest"])
    assert manifest["runs"]==120
    assert not manifest["primary_scope"]
    assert manifest["legacy_architectures"]

def test_invalid_backbones_rejected(tmp_path):
    with pytest.raises(ValueError):matrix(tmp_path/"data",tmp_path/"matrix",backbones=[])
