"""Classes extracted verbatim from Julia Wieczorek's Chapter 5 code.
See docs/provenance.json. Corrections live in wrapper modules.
"""
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel

class MultiTaskDataset(Dataset):
    def __init__(self, texts, sentiment_labels, emotion_labels, intensity_labels, tokenizer, max_len):
        self.texts = texts
        self.sentiment_labels = sentiment_labels
        self.emotion_labels = emotion_labels
        self.intensity_labels = intensity_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['sentiment'] = torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        item['emotions'] = torch.tensor(self.emotion_labels[idx], dtype=torch.float)
        item['intensities'] = torch.tensor(self.intensity_labels[idx], dtype=torch.long)
        return item


class MultiTaskBERTLSTM(nn.Module):
    def __init__(self, bert_model, num_emotions, lstm_hidden=128, lstm_layers=1, dropout=0.3, bidirectional=True):
        super().__init__()
        self.bert = bert_model
        hidden_size = self.bert.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.shared_bn = nn.BatchNorm1d(lstm_output_dim // 2)

        # Heads
        self.sentiment_head = nn.Linear(lstm_output_dim // 2, 3)
        self.emotion_head = nn.Linear(lstm_output_dim // 2, num_emotions)
        self.intensity_head = nn.Linear(lstm_output_dim // 2, num_emotions * 3)

        nn.init.xavier_uniform_(self.shared_fc.weight)
        nn.init.constant_(self.shared_fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state

        lstm_out, (h_n, _) = self.lstm(sequence_output)

        if self.lstm.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat((h_forward, h_backward), dim=1)
        else:
            h = h_n[-1]

        x = self.dropout(h)
        x = self.shared_fc(x)
        x = self.shared_bn(x)
        x = torch.relu(x)
        x = self.dropout(x)

        sentiment_logits = self.sentiment_head(x)
        emotion_logits = self.emotion_head(x)
        intensity_logits = self.intensity_head(x)

        return sentiment_logits, emotion_logits, intensity_logits


class MultiTaskBERT(nn.Module):
    def __init__(self, encoder, num_emotions, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(hidden, hidden // 2)

        self.sentiment_head = nn.Linear(hidden // 2, 3)
        self.emotion_head = nn.Linear(hidden // 2, num_emotions)
        self.intensity_head = nn.Linear(hidden // 2, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0]     # [CLS]

        x = torch.relu(self.shared_fc(self.dropout(h)))

        return {
            "sentiment": self.sentiment_head(x),
            "emotion": self.emotion_head(x),
            "intensity": self.intensity_head(x)
        }


class AdapterModule(nn.Module):
    """
    Lightweight adapter: down-proj -> nonlinearity -> up-proj + residual
    Applied to pooled representation (CLS) or to token outputs if desired.
    """
    def __init__(self, hidden_size:int, bottleneck:int=64, dropout:float=0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # init
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.constant_(self.down.bias, 0.)
        nn.init.constant_(self.up.bias, 0.)

    def forward(self, x):
        z = self.down(x)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.up(z)
        return x + z  # residual


class MultiTaskBERTWithAdapters(nn.Module):
    """
    Hard-shared encoder (one transformer) + per-task adapters applied to pooled output.
    This is a practical 'adapter' implementation without external libs.
    """
    def __init__(self, transformer_name:str, num_emotions:int, adapter_bottleneck:int=64, dropout:float=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hid = self.transformer.config.hidden_size

        # task-specific adapters
        self.adapter_sentiment = AdapterModule(hid, adapter_bottleneck, dropout)
        self.adapter_emotion = AdapterModule(hid, adapter_bottleneck, dropout)
        self.adapter_intensity = AdapterModule(hid, adapter_bottleneck, dropout)

        # shared projection after adapter
        proj_dim = hid // 2
        self.shared_fc = nn.Linear(hid, proj_dim)
        self.dropout = nn.Dropout(dropout)

        self.sentiment_head = nn.Linear(proj_dim, 3)
        self.emotion_head = nn.Linear(proj_dim, num_emotions)
        self.intensity_head = nn.Linear(proj_dim, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # CLS pooled

        s = self.adapter_sentiment(pooled)
        e = self.adapter_emotion(pooled)
        i = self.adapter_intensity(pooled)

        # pass through shared fc
        s_x = torch.relu(self.shared_fc(self.dropout(s)))
        e_x = torch.relu(self.shared_fc(self.dropout(e)))
        i_x = torch.relu(self.shared_fc(self.dropout(i)))

        return {
            "sentiment": self.sentiment_head(s_x),
            "emotion": self.emotion_head(e_x),
            "intensity": self.intensity_head(i_x)
        }


class MMOE_Core(nn.Module):
    """
    Implementation of Mixture-of-Experts layer.
    experts: list of feed-forward networks
    gates: per-task gate producing mixture weights
    """
    def __init__(self, input_dim:int, expert_hidden:int, num_experts:int, num_tasks:int, dropout:float=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, expert_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, input_dim),
            nn.ReLU()
        ) for _ in range(num_experts)])

        # per-task gating networks
        self.gates = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        ) for _ in range(num_tasks)])

    def forward(self, x):
        # x: (B, D)
        expert_outs = []  # list (num_experts) of (B, D)
        for e in self.experts:
            expert_outs.append(e(x))  # B x D

        # stack experts -> (B, num_experts, D)
        expert_stack = torch.stack(expert_outs, dim=1)

        # For each task, compute gated sum
        task_outputs = []
        for g in self.gates:
            weights = g(x)  # (B, num_experts)
            weights = weights.unsqueeze(-1)  # (B, num_experts, 1)
            # weighted sum over experts
            task_out = (weights * expert_stack).sum(dim=1)  # (B, D)
            task_outputs.append(task_out)
        # returns list len=num_tasks of tensors (B,D)
        return task_outputs


class MultiTaskMMOE(nn.Module):
    """
    Encoder -> MMOE -> per-task head
    tasks: sentiment, emotion, intensity
    """
    def __init__(self, transformer_name:str, num_emotions:int, num_experts:int=4, expert_hidden:int=256, dropout:float=0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hid = self.transformer.config.hidden_size

        self.mmoe = MMOE_Core(input_dim=hid, expert_hidden=expert_hidden, num_experts=num_experts, num_tasks=3, dropout=dropout)

        # heads after task-specific mixture outputs
        self.sent_fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.em_fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.int_fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))

        self.sentiment_head = nn.Linear(hid//2, 3)
        self.emotion_head = nn.Linear(hid//2, num_emotions)
        self.intensity_head = nn.Linear(hid//2, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # (B, hid)

        task_feats = self.mmoe(pooled)  # list len 3 of (B, hid)
        s_feat, e_feat, i_feat = task_feats

        s_x = self.sent_fc(s_feat)
        e_x = self.em_fc(e_feat)
        i_x = self.int_fc(i_feat)

        return {
            "sentiment": self.sentiment_head(s_x),
            "emotion": self.emotion_head(e_x),
            "intensity": self.intensity_head(i_x)
        }


class SingleTaskModel(nn.Module):
    def __init__(self, transformer_name: str, task: str, num_emotions: int, dropout: float = 0.3):
        super().__init__()
        self.task = task

        self.encoder = AutoModel.from_pretrained(transformer_name)
        hid = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        if task == "sentiment":
            self.head = nn.Linear(hid, 3)

        elif task == "emotion":
            self.head = nn.Linear(hid, num_emotions)

        elif task == "intensity":
            self.head = nn.Linear(hid, num_emotions * 3)

        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, input_ids, attention_mask):
        h = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        h = self.dropout(h)
        logits = self.head(h)

        return {
            "sentiment": logits if self.task == "sentiment" else None,
            "emotion": logits if self.task == "emotion" else None,
            "intensity": logits if self.task == "intensity" else None
        }


class SoftSharingModel(nn.Module):
    """
    Soft-sharing encoder-only model.
    """
    def __init__(self, transformer_name, num_emotions, tasks, dropout=0.3):
        super().__init__()
        self.tasks = tasks

        self.encoders = nn.ModuleDict({
            task: AutoModel.from_pretrained(transformer_name)
            for task in tasks
        })

        hid = next(iter(self.encoders.values())).config.hidden_size
        proj = hid // 2

        self.shared_fc = nn.Linear(hid, proj)
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict()
        if "sentiment" in tasks:
            self.heads["sentiment"] = nn.Linear(proj, 3)
        if "emotion" in tasks:
            self.heads["emotion"] = nn.Linear(proj, num_emotions)
        if "intensity" in tasks:
            self.heads["intensity"] = nn.Linear(proj, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        outputs = {"sentiment": None, "emotion": None, "intensity": None}

        for task, encoder in self.encoders.items():
            h = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0]

            x = torch.relu(self.shared_fc(self.dropout(h)))
            outputs[task] = self.heads[task](x)

        return outputs

    def get_soft_sharing_loss(self, l2_lambda=1e-4):
        loss = 0.0
        encs = list(self.encoders.values())
        for i in range(len(encs)):
            for j in range(i + 1, len(encs)):
                for p1, p2 in zip(encs[i].parameters(), encs[j].parameters()):
                    if p1.shape == p2.shape:
                        loss += (p1 - p2).pow(2).sum()
        return l2_lambda * loss


class CrossStitchModel(nn.Module):
    """
    Cross-stitch networks: learn linear combination of feature maps from different tasks.
    We'll implement a simplified cross-stitch between {sent, emotion+intensity} streams.
    """
    def __init__(self, transformer_name:str, num_emotions:int, dropout:float=0.3):
        super().__init__()
        self.encoder_a = AutoModel.from_pretrained(transformer_name)  # e.g. sentiment
        self.encoder_b = AutoModel.from_pretrained(transformer_name)  # e.g. emotion/intensity

        hid = self.encoder_a.config.hidden_size
        self.cross_coeff = nn.Parameter(torch.eye(2))  # 2x2 cross-stitch matrix (learnable)
        self.shared_fc = nn.Linear(hid, hid//2)
        self.dropout = nn.Dropout(dropout)

        self.sentiment_head = nn.Linear(hid//2, 3)
        self.emotion_head = nn.Linear(hid//2, num_emotions)
        self.intensity_head = nn.Linear(hid//2, num_emotions * 3)

    def forward(self, input_ids, attention_mask):
        a = self.encoder_a(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]  # BxH
        b = self.encoder_b(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]  # BxH

        # combine features via learned 2x2 matrix per batch (applied same for all dims)
        # [a'; b'] = cross_coeff @ [a; b]
        stacked = torch.stack([a, b], dim=1)  # (B, 2, H)
        # apply mixing: for each batch, for each feature dimension mixing is same matrix -> do matmul
        # Result: (B, 2, H) -> reshape -> apply
        coeff = self.cross_coeff.unsqueeze(0)  # (1,2,2)
        mixed = torch.matmul(coeff, stacked)  # (1,2,2) x (B,2,H) -> broadcasting -> (B,2,H)
        a_m = mixed[:,0,:]  # (B,H)
        b_m = mixed[:,1,:]

        # use a_m for sentiment, b_m for emotion+intensity (shared)
        s = torch.relu(self.shared_fc(self.dropout(a_m)))
        e = torch.relu(self.shared_fc(self.dropout(b_m)))
        i = torch.relu(self.shared_fc(self.dropout(b_m)))

        return self.sentiment_head(s), self.emotion_head(e), self.intensity_head(i)


class SingleTaskSentiment(nn.Module):
    def __init__(self, transformer_name:str, dropout:float=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hid = self.encoder.config.hidden_size
        self.fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hid//2, 3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        x = self.fc(out)
        return self.head(x)


class SingleTaskEmotion(nn.Module):
    def __init__(self, transformer_name:str, num_emotions:int, dropout:float=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hid = self.encoder.config.hidden_size
        self.fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hid//2, num_emotions)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        x = self.fc(out)
        return self.head(x)


class SingleTaskIntensity(nn.Module):
    def __init__(self, transformer_name:str, num_emotions:int, dropout:float=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hid = self.encoder.config.hidden_size
        self.fc = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hid//2, num_emotions*3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        x = self.fc(out)
        return self.head(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
