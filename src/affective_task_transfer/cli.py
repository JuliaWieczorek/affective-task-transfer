import argparse
import json
from pathlib import Path
from .config import load_config
from .io import read_json, write_json

def main():
    parser=argparse.ArgumentParser(description="Chapter 5 task-transfer experiments")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("audit-data");p.add_argument("--csv",required=True);p.add_argument("--raw-meisd",required=True);p.add_argument("--expanded");p.add_argument("--output",required=True)
    p=sub.add_parser("prepare-meisd");p.add_argument("--csv",required=True);p.add_argument("--raw-meisd",required=True);p.add_argument("--expanded");p.add_argument("--output",required=True);p.add_argument("--split-seed",type=int,default=2026)
    p=sub.add_parser("download-brighter");p.add_argument("--output",required=True)
    p=sub.add_parser("prepare-brighter");p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("train");p.add_argument("--config",required=True);p.add_argument("--dataset");p.add_argument("--output")
    p=sub.add_parser("evaluate");p.add_argument("--dataset",required=True);p.add_argument("--run",required=True);p.add_argument("--split",choices=["dev","test"],default="test")
    p=sub.add_parser("make-matrix");p.add_argument("--dataset",required=True);p.add_argument("--dataset-name",choices=["meisd","brighter"],required=True);p.add_argument("--output",required=True);p.add_argument("--ablations",action="store_true");p.add_argument("--historical-stl",action="store_true")
    p=sub.add_parser("run-matrix");p.add_argument("--matrix",required=True);p.add_argument("--limit",type=int)
    p=sub.add_parser("evaluate-matrix");p.add_argument("--matrix",required=True);p.add_argument("--split",choices=["dev","test"],default="test");p.add_argument("--limit",type=int)
    p=sub.add_parser("report");p.add_argument("--runs",required=True);p.add_argument("--output",required=True);p.add_argument("--split",choices=["dev","test"],default="test");p.add_argument("--allow-smoke",action="store_true")
    args=parser.parse_args()
    if args.command in ("audit-data","prepare-meisd"):
        from .data import meisd_records,prepare_meisd
        if args.command=="audit-data":
            _,emotions,result=meisd_records(args.csv,args.raw_meisd,args.expanded);result["emotions"]=emotions;write_json(args.output,result)
        else:result=prepare_meisd(args.csv,args.raw_meisd,args.output,args.split_seed,args.expanded)
    elif args.command=="download-brighter":
        from .download import download_brighter
        result=download_brighter(args.output)
    elif args.command=="prepare-brighter":
        from .data import prepare_brighter
        provenance=Path(args.input)/"download_manifest.json"
        revision=read_json(provenance)["revision"] if provenance.exists() else None
        result=prepare_brighter(args.input,args.output,revision)
    elif args.command=="train":
        from .training import train
        config=load_config(args.config)
        result=train(args.dataset or config["dataset"],args.output or config["output"],config)
    elif args.command=="evaluate":
        from .training import evaluate_run
        result=evaluate_run(args.dataset,args.run,args.split)
    elif args.command=="make-matrix":
        from .experiments import matrix
        result=matrix(args.dataset,args.output,args.dataset_name,args.ablations,args.historical_stl)
    elif args.command=="run-matrix":
        from .experiments import run_matrix
        result=run_matrix(args.matrix,args.limit)
    elif args.command=="evaluate-matrix":
        from .experiments import evaluate_matrix
        result=evaluate_matrix(args.matrix,args.split,args.limit)
    elif args.command=="report":
        from .reporting import report
        result=report(args.runs,args.output,args.split,args.allow_smoke)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))

if __name__=="__main__":main()
