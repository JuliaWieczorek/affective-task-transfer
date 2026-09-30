"""Real forward/backward/checkpoint/evaluation using a tiny local BERT.

Random initialization is intentional: this verifies the pipeline, not performance.
No historical/full-size training is launched. Input samples are real prepared data.
"""
import argparse
from pathlib import Path
from transformers import BertConfig,BertModel,BertTokenizerFast
from affective_task_transfer.config import load_config
from affective_task_transfer.training import train,evaluate_run
from affective_task_transfer.reporting import report

def tiny_backbone(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    vocab=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","i","you","we","it","is","a","the","feel","sad","happy","angry","help","not","to","and","."]
    (path/"vocab.txt").write_text("\n".join(vocab),encoding="utf-8")
    BertTokenizerFast(vocab_file=str(path/"vocab.txt"),do_lower_case=True).save_pretrained(path)
    BertModel(BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=128)).save_pretrained(path)
    return str(path.resolve())

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--output",default="outputs/smoke");p.add_argument("--brighter",action="store_true");args=p.parse_args()
    root=Path(args.output);backbone=tiny_backbone(root/"tiny_backbone")
    tasks=["emotion","intensity"] if args.brighter else ["sentiment","emotion","intensity"]
    conditions=[(a,tasks,{}) for a in ["hard_sharing","soft_sharing","adapters","mmoe","bert_lstm","cross_stitch"]]
    conditions += [("matched_stl",[t],{}) for t in tasks]
    conditions += [("single_task",[t],{}) for t in tasks]
    conditions += [("single_"+t,[t],{}) for t in tasks]
    conditions += [("soft_sharing",tasks,{"shared_projection":False,"soft_lambda":0})]
    for i,(arch,active,extra) in enumerate(conditions):
        config=load_config(overrides={"architecture":arch,"backbone":backbone,"tasks":active,"smoke":True,"epochs":2,"train_limit":12,"eval_limit":8,"batch_size":4,"max_length":32,"max_steps":4,"accumulation_steps":2,"diagnostic_batch_size":4,"dropout":0.1,"learning_rate":1e-3,"device":"cpu","threads":2,"expert_hidden":16,"adapter_bottleneck":4,"lstm_hidden":8,"matrix_condition":f"smoke_{i:02d}_{arch}_{'_'.join(active)}"}|extra)
        run=root/f"{i:02d}_{arch}_{'_'.join(active)}"
        train(args.dataset,run,config)
        evaluate_run(args.dataset,run,"dev")
    report(root,root/"report",split="dev",allow_smoke=True)
    print("Smoke complete; no test-set evaluation performed.")

if __name__=="__main__":main()
