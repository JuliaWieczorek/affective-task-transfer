"""Historical focal/CE objectives with corrected masking and train-only weights."""
import torch
from torch import nn
from .models.inherited import FocalLoss

class TaskLosses:
    def __init__(self, rows, emotions, tasks, device, gamma=2.0):
        self.tasks = tasks
        self.sent_weights = None
        if "sentiment" in tasks:
            labels = torch.tensor([r["sentiment"] for r in rows])
            counts = torch.bincount(labels[labels>=0], minlength=3).float()
            self.sent_weights = torch.where(counts>0, counts.sum()/(counts.clamp_min(1)*max(int((counts>0).sum()),1)), 0).to(device)
        presence = torch.tensor([r["emotion"] for r in rows], dtype=torch.float)
        freq = presence.mean(0)
        self.em_weights = torch.where(freq>0, 1/freq.clamp_min(1e-6), 0).to(device)
        self.focal = FocalLoss(self.em_weights, gamma)

    def __call__(self, outputs, batch):
        losses = {}
        if outputs.get("sentiment") is not None:
            logits = outputs["sentiment"]
            mask = batch["sentiment"]>=0
            losses["sentiment"] = nn.functional.cross_entropy(logits[mask],batch["sentiment"][mask],weight=self.sent_weights) if mask.any() else logits.sum()*0
        if outputs.get("emotion") is not None:
            losses["emotion"] = self.focal(outputs["emotion"], batch["emotion"].float())
        if outputs.get("intensity") is not None:
            logits = outputs["intensity"].reshape(*batch["intensity"].shape,3)
            mask = (batch["emotion"]==1) & (batch["intensity"]>=0)
            losses["intensity"] = nn.functional.cross_entropy(logits[mask],batch["intensity"][mask]) if mask.any() else logits.sum()*0
        return losses
