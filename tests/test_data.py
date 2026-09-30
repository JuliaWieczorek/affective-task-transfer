import csv
import pytest
from affective_task_transfer.data import assign_groups,validate_splits,meisd_records,prepare_brighter
from affective_task_transfer.io import read_jsonl,write_jsonl

def test_duplicate_dialogue_union():
    rows=[{"id":str(i),"source_dialogue_id":str(i),"parent_id":str(i),"text":f"text {i}","is_augmented":False} for i in range(20)]
    rows[1]["text"]=rows[0]["text"]
    augmented=dict(rows[0],id="aug",text="paraphrase",is_augmented=True)
    rows.append(augmented)
    result=assign_groups(rows,42)
    assert result[0]["split"]==result[1]["split"]==result[-1]["split"]
    splits={s:[r for r in result if r["split"]==s and (s=="train" or not r["is_augmented"])] for s in ["train","dev","test"]}
    validate_splits(splits)

def test_leakage_rejected():
    record={"text":"same", "group_id":"x","parent_id":"x","is_augmented":False}
    with pytest.raises(ValueError):validate_splits({"train":[record],"dev":[record],"test":[]})

def test_unicode_text_round_trips_jsonl(tmp_path):
    file=tmp_path/"records.jsonl"
    rows=[{"text":"first\u2028second\u0085third"}]
    write_jsonl(file,rows)
    assert read_jsonl(file)==rows

def test_onehot_uses_aligned_expanded_text(tmp_path):
    def csv_file(name, columns, rows):
        path=tmp_path/name
        with path.open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        return path
    raw=csv_file("raw.csv",["dialog_ids","uttr_ids","Utterances"],[{"dialog_ids":"1","uttr_ids":"1","Utterances":"Hello"},{"dialog_ids":"1","uttr_ids":"2","Utterances":"World"}])
    originals=[{"Utterances":"Hello","original":"","mode":"","sentiment":"positive","emotion__joy":"1","intensity__joy":"2"},{"Utterances":"","original":"Hello","mode":"llm","sentiment":"positive","emotion__joy":"1","intensity__joy":"2"}]
    onehot=csv_file("onehot.csv",list(originals[0]),originals)
    expanded=[{"Utterances":"Hello","original":"","mode":"","sentiment":"positve","augmented":"","emotion1":"joy","intensity1":"2","emotion2":"","intensity2":""},{"Utterances":"","original":"Hello","mode":"llm","sentiment":"positve","augmented":"I'm joyful","emotion1":"joy","intensity1":"1","emotion2":"joy","intensity2":"2"}]
    source=csv_file("expanded.csv",list(expanded[0]),expanded)
    records,emotions,audit=meisd_records(onehot,raw,source)
    assert records[1]["text"]=="I'm joyful"
    assert records[1]["parent_id"]==records[0]["parent_id"]
    assert records[0]["intensity"]==[1]
    assert records[1]["intensity"]==[-100]
    assert audit["issues"]["corrected_positve_labels"]==2
    assert audit["issues"]["ambiguous_repeated_emotion_intensity_positions"]==1
    assert emotions==["joy"]
    with pytest.raises(ValueError,match="supply the matching"):
        meisd_records(onehot,raw)

def test_brighter_quarantines_conflicting_cross_split_text(tmp_path):
    source=tmp_path/"source";source.mkdir()
    labels={"anger":0,"fear":0,"joy":0,"sadness":0,"surprise":0}
    splits={"train":[("stay",0),("collision",1),("stay",0)],"dev":[("development",1)],"test":[("collision",2),("held out",3)]}
    for split,items in splits.items():
        with (source/f"{split}.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["id","text",*labels]);writer.writeheader()
            for i,(text,intensity) in enumerate(items):
                writer.writerow({"id":f"{split}-{i}","text":text,**(labels|{"joy":intensity})})
    manifest=prepare_brighter(source,tmp_path/"prepared")
    assert [manifest["files"][s]["records"] for s in ("train","dev","test")]==[1,1,1]
    assert manifest["audit"]["excluded_conflicting_label_records"]=={"train":1,"dev":0,"test":1}
    assert manifest["audit"]["removed_same_split_duplicates"]["train"]==1
