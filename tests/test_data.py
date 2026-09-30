import csv
import pytest
from affective_task_transfer.data import assign_groups,validate_splits,meisd_records,prepare_brighter,prepare_meisd
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

def test_brighter_preserves_test_labels_and_removes_training_overlap(tmp_path):
    source=tmp_path/"source";source.mkdir()
    labels={"anger":0,"fear":0,"joy":0,"sadness":0,"surprise":0}
    splits={"train":[("stay",0),("collision",1),("stay",0)],"dev":[("development",1)],"test":[("collision",2),("held out",3)]}
    for split,items in splits.items():
        with (source/f"{split}.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["id","text",*labels]);writer.writeheader()
            for i,(text,intensity) in enumerate(items):
                writer.writerow({"id":f"{split}-{i}","text":text,**(labels|{"joy":intensity})})
    manifest=prepare_brighter(source,tmp_path/"prepared")
    assert [manifest["files"][s]["records"] for s in ("train","dev","test")]==[1,1,2]
    assert manifest["audit"]["excluded_train_dev_overlap_with_later_splits"]["train"]==1
    assert manifest["audit"]["removed_same_split_duplicates"]["train"]==1
    assert manifest["audit"]["test_labels_used_for_cleaning"] is False
    assert [r["id"] for r in read_jsonl(tmp_path/"prepared/test.jsonl")]==["test-0","test-1"]

def test_study_two_selects_start_half_with_dialogue_groups(tmp_path):
    raw=tmp_path/"raw.csv";onehot=tmp_path/"onehot.csv";expanded=tmp_path/"expanded.csv"
    with raw.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["dialog_ids","uttr_ids","Utterances"]);writer.writeheader()
        for dialog in range(12):
            writer.writerow({"dialog_ids":dialog,"uttr_ids":0,"Utterances":f"opening {dialog}"})
            writer.writerow({"dialog_ids":dialog,"uttr_ids":1,"Utterances":f"ending {dialog}"})
    with onehot.open("w",encoding="utf-8",newline="") as a,expanded.open("w",encoding="utf-8",newline="") as b:
        w1=csv.DictWriter(a,fieldnames=["Utterances","original","mode","sentiment","emotion__joy","intensity__joy"])
        w2=csv.DictWriter(b,fieldnames=["Utterances","original","mode","sentiment","augmented","segment","emotion1","intensity1"])
        w1.writeheader();w2.writeheader()
        for dialog in range(12):
            for segment,word in (("start","opening"),("end","ending")):
                text=f"{word} {dialog}"
                w1.writerow({"Utterances":text,"original":"","mode":"","sentiment":"positive","emotion__joy":1,"intensity__joy":2})
                w2.writerow({"Utterances":text,"original":"","mode":"","sentiment":"positive","augmented":"","segment":segment,"emotion1":"joy","intensity1":2})
    manifest=prepare_meisd(onehot,raw,tmp_path/"prepared",expanded_path=expanded)
    assert manifest["audit"]["prepared_segment"]=="start"
    assert manifest["audit"]["source_half_counts"]=={"start":12,"end":12}
    assert sum(manifest["files"][s]["records"] for s in ("train","dev","test"))==12
    assert all(r["segment"]=="start" for s in ("train","dev","test") for r in read_jsonl(tmp_path/f"prepared/{s}.jsonl"))
