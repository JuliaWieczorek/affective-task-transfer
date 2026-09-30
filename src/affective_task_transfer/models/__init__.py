"""Adapters around verbatim historical models; no parallel replacement models."""
import torch
from torch import nn
from transformers import AutoModel
from . import inherited as old

TASKS = ("sentiment", "emotion", "intensity")
ARCHITECTURES = ("hard_sharing", "soft_sharing", "adapters", "mmoe", "single_task", "matched_stl", "single_sentiment", "single_emotion", "single_intensity", "bert_lstm", "cross_stitch")

class SoftSharing(old.SoftSharingModel):
    def __init__(self, transformer_name, num_emotions, tasks, dropout, shared_projection=True):
        super().__init__(transformer_name, num_emotions, tasks, dropout)
        self.shared_projection = shared_projection
        if not shared_projection:
            import copy
            self.projections = nn.ModuleDict({t:copy.deepcopy(self.shared_fc) for t in tasks})
            del self.shared_fc

    def forward(self, input_ids, attention_mask):
        if self.shared_projection:
            return super().forward(input_ids, attention_mask)
        result = {t:None for t in TASKS}
        for task,encoder in self.encoders.items():
            h = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
            result[task] = self.heads[task](torch.relu(self.projections[task](self.dropout(h))))
        return result

    def regularization(self, coefficient, normalization="mean_pairs"):
        n = len(self.encoders)
        if n<2 or coefficient==0:
            return next(self.parameters()).sum()*0
        loss = super().get_soft_sharing_loss(coefficient)
        if normalization == "mean_pairs":
            return loss / (n*(n-1)/2)
        if normalization != "sum_pairs":
            raise ValueError(normalization)
        return loss

class TaskModel(nn.Module):
    """Normalize the old tuple/dict/tensor interfaces and disable unused outputs."""
    def __init__(self, core, tasks, architecture):
        super().__init__()
        self.core, self.tasks, self.architecture = core, list(tasks), architecture

    def forward(self, input_ids, attention_mask):
        result = self.core(input_ids, attention_mask)
        if isinstance(result, tuple):
            result = dict(zip(TASKS, result))
        elif isinstance(result, torch.Tensor):
            result = {self.tasks[0]:result}
        return {task:result.get(task) if task in self.tasks else None for task in TASKS}

    def regularization(self, config):
        if isinstance(self.core, SoftSharing):
            return self.core.regularization(config["soft_lambda"], config["soft_normalization"])
        return next(self.parameters()).sum()*0

    def diagnostic_parameters(self):
        # Actual shared projection in soft sharing; shared backbone elsewhere.
        c = self.core
        if isinstance(c, SoftSharing):
            return list(c.shared_fc.parameters()) if c.shared_projection else []
        for name in ("encoder", "transformer", "bert"):
            if hasattr(c, name):
                return list(getattr(c,name).parameters())
        return list(c.shared_fc.parameters()) if hasattr(c, "shared_fc") else []

def build_model(config, num_emotions):
    arch, backbone, tasks = config["architecture"], config["backbone"], config["tasks"]
    dropout = config["dropout"]
    if not tasks or set(tasks)-set(TASKS):
        raise ValueError("Invalid tasks")
    if arch == "hard_sharing":
        core = old.MultiTaskBERT(AutoModel.from_pretrained(backbone), num_emotions, dropout)
    elif arch in ("soft_sharing", "matched_stl"):
        if arch == "matched_stl" and len(tasks)!=1:
            raise ValueError("matched_stl requires one task")
        core = SoftSharing(backbone, num_emotions, tasks, dropout, config["shared_projection"])
    elif arch == "adapters":
        core = old.MultiTaskBERTWithAdapters(backbone, num_emotions, config["adapter_bottleneck"], dropout)
    elif arch == "mmoe":
        core = old.MultiTaskMMOE(backbone, num_emotions, config["num_experts"], config["expert_hidden"], dropout)
    elif arch == "single_task":
        if len(tasks)!=1: raise ValueError("single_task requires one task")
        core = old.SingleTaskModel(backbone, tasks[0], num_emotions, dropout)
    elif arch.startswith("single_"):
        task = arch.removeprefix("single_")
        if tasks!=[task]: raise ValueError("Single head/task mismatch")
        cls = {"emotion":old.SingleTaskEmotion,"intensity":old.SingleTaskIntensity,"sentiment":old.SingleTaskSentiment}[task]
        core = cls(backbone, dropout=dropout, **({"num_emotions":num_emotions} if task!="sentiment" else {}))
    elif arch == "bert_lstm":
        core = old.MultiTaskBERTLSTM(AutoModel.from_pretrained(backbone), num_emotions, config["lstm_hidden"], dropout=dropout)
    elif arch == "cross_stitch":
        core = old.CrossStitchModel(backbone, num_emotions, dropout)
    else:
        raise ValueError(f"Unknown architecture {arch}")
    return TaskModel(core, tasks, arch)
