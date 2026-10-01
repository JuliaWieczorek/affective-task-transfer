from pathlib import Path
from .io import read_json

DEFAULTS = dict(architecture="soft_sharing", backbone="bert-base-uncased", tasks=["sentiment","emotion","intensity"], seed=42, max_length=192, batch_size=16, epochs=6, learning_rate=2e-5, weight_decay=0.01, dropout=0.4, warmup_ratio=0.2, accumulation_steps=1, patience=2, clip_norm=1.0, weights={"sentiment":1.0,"emotion":2.0,"intensity":0.7}, focal_gamma=2.0, soft_lambda=1e-4, soft_normalization="mean_pairs", shared_projection=True, adapter_bottleneck=64, num_experts=4, expert_hidden=256, lstm_hidden=128, threshold_candidates=[0.3,0.4,0.5,0.6,0.7], device="auto", num_workers=0, diagnostic_batch_size=32, diagnostics=True, max_steps=None, train_limit=None, eval_limit=None, threads=2, smoke=False, stl_baseline="matched_stl")

def load_config(path=None, overrides=None):
    config = DEFAULTS | (read_json(path) if path else {}) | (overrides or {})
    unknown = set(config)-set(DEFAULTS)-{"dataset","output","name","matrix_condition"}
    if unknown: raise ValueError(f"Unknown configuration keys: {unknown}")
    if len(config["tasks"]) != len(set(config["tasks"])): raise ValueError("Duplicate task")
    for key in ("epochs","batch_size","accumulation_steps","max_length","patience","diagnostic_batch_size"):
        if config[key]<1: raise ValueError(f"Invalid {key}")
    for key in ("max_steps","train_limit","eval_limit"):
        if config[key] is not None and (not config["smoke"] or config[key]<1):
            raise ValueError(f"{key} is allowed only in explicitly marked smoke runs")
    if any(config["weights"].get(t,0)<=0 for t in config["tasks"]): raise ValueError("Active task weight must be positive")
    if config["soft_normalization"] not in ("sum_pairs","mean_pairs"): raise ValueError("Invalid L2 normalization")
    if not config["threshold_candidates"] or any(not 0<t<1 for t in config["threshold_candidates"]): raise ValueError("Invalid emotion thresholds")
    return config
