"""Conventional metrics with explicit denominators; no absent-as-low scoring."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, mean_absolute_error

def classification(y, p, labels, names):
    y,p = np.asarray(y), np.asarray(p)
    if len(y)==0:
        return {"support":0,"accuracy":None,"macro_f1":None,"weighted_f1":None,"macro_precision":None,"macro_recall":None,"confusion_matrix":np.zeros((len(labels),len(labels)),dtype=int).tolist(),"per_class":{}}
    report = classification_report(y,p,labels=labels,target_names=names,output_dict=True,zero_division=0)
    return {"support":len(y), "accuracy":float(accuracy_score(y,p)), "macro_f1":float(f1_score(y,p,labels=labels,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y,p,labels=labels,average="weighted",zero_division=0)),"macro_precision":float(precision_score(y,p,labels=labels,average="macro",zero_division=0)),"macro_recall":float(recall_score(y,p,labels=labels,average="macro",zero_division=0)),"confusion_matrix":confusion_matrix(y,p,labels=labels).tolist(),"per_class":{n:report[n] for n in names}}

def evaluate_predictions(rows, predictions, emotions, tasks, threshold=0.5):
    truth_e=np.asarray([r["emotion"] for r in rows]); truth_i=np.asarray([r["intensity"] for r in rows])
    result={}
    if "sentiment" in tasks:
        y=np.asarray([r["sentiment"] for r in rows]); p=np.asarray(predictions["sentiment"]); valid=y>=0
        result["sentiment"]=classification(y[valid],p[valid],[0,1,2],["negative","neutral","positive"])
    if "emotion" in tasks:
        ep=(np.asarray(predictions["emotion_probabilities"])>=threshold).astype(int)
        result["emotion"]={"support":len(rows),"subset_accuracy":float(accuracy_score(truth_e,ep)),"label_accuracy":float((truth_e==ep).mean()),"macro_f1":float(f1_score(truth_e,ep,average="macro",zero_division=0)),"micro_f1":float(f1_score(truth_e,ep,average="micro",zero_division=0)),"weighted_f1":float(f1_score(truth_e,ep,average="weighted",zero_division=0)),"macro_precision":float(precision_score(truth_e,ep,average="macro",zero_division=0)),"macro_recall":float(recall_score(truth_e,ep,average="macro",zero_division=0)),"per_emotion":{e:classification(truth_e[:,j],ep[:,j],[0,1],["absent","present"]) for j,e in enumerate(emotions)}}
    if "intensity" in tasks:
        ip=np.asarray(predictions["intensity"])
        mask=(truth_e==1)&(truth_i>=0)
        result["intensity"]=classification(truth_i[mask],ip[mask],[0,1,2],["low","medium","high"])
        result["intensity"]["mae"]=float(mean_absolute_error(truth_i[mask],ip[mask])) if mask.any() else None
        per={}
        for j,e in enumerate(emotions):
            per[e]=classification(truth_i[mask[:,j],j],ip[mask[:,j],j],[0,1,2],["low","medium","high"])
        result["intensity"]["per_emotion"]=per
        vals=[v["macro_f1"] for v in per.values() if v["support"]]
        result["intensity"]["emotion_macro_f1"]=float(np.mean(vals)) if vals else None
        vals=[v["accuracy"] for v in per.values() if v["support"]]
        result["intensity"]["emotion_macro_accuracy"]=float(np.mean(vals)) if vals else None
    if "emotion" in tasks and "intensity" in tasks:
        # Joint labels 0=absent, 1..3=intensity. Exclude active unknown intensities.
        valid=(truth_e==0)|(truth_i>=0)
        y=np.where(truth_e,truth_i+1,0); p=np.where(ep,ip+1,0)
        result["joint"]=classification(y[valid],p[valid],[0,1,2,3],["absent","low","medium","high"])
        tp=int(((p==y)&(y>0)&valid).sum()); fp=int(((p>0)&(p!=y)&valid).sum()); fn=int(((y>0)&(p!=y)&valid).sum())
        result["joint"].update(pair_micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0, pair_precision=tp/(tp+fp) if tp+fp else 0,pair_recall=tp/(tp+fn) if tp+fn else 0)
    return result

def selection_score(metrics, tasks):
    values=[metrics[t]["emotion_macro_f1" if t=="intensity" else "macro_f1"] for t in tasks]
    if any(v is None for v in values):
        raise ValueError("An active task has no evaluable development labels")
    return float(np.mean(values))

def tune_threshold(rows, predictions, emotions, tasks, candidates):
    if "emotion" not in tasks: return 0.5
    # Tune only emotion F1, using development predictions, deterministic tie-break.
    y=np.asarray([r["emotion"] for r in rows]); probs=np.asarray(predictions["emotion_probabilities"])
    return max(candidates,key=lambda t:(f1_score(y,probs>=t,average="macro",zero_division=0),-abs(t-0.5),-t))

def majority_baseline(train, evaluation, emotions, tasks):
    pred={}
    if "sentiment" in tasks:
        vals=[r["sentiment"] for r in train if r["sentiment"]>=0]
        pred["sentiment"]=[int(np.bincount(vals,minlength=3).argmax())]*len(evaluation)
    if "emotion" in tasks:
        pred["emotion_probabilities"]=np.tile(np.asarray([r["emotion"] for r in train]).mean(0)>=0.5,(len(evaluation),1)).astype(float).tolist()
    if "intensity" in tasks:
        levels=[]
        for j in range(len(emotions)):
            vals=[r["intensity"][j] for r in train if r["emotion"][j] and r["intensity"][j]>=0]
            levels.append(int(np.bincount(vals,minlength=3).argmax()))
        pred["intensity"]=np.tile(levels,(len(evaluation),1)).tolist()
    return evaluate_predictions(evaluation,pred,emotions,tasks)
