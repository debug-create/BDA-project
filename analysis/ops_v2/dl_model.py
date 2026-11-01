"""Residual-on-logistic deep model used by the operations-v2 evaluation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


EMBED_DIMS = {
    "Reporting_Airline": 8,
    "Origin": 24,
    "Dest": 24,
    "RouteHash65536": 32,
    "ScheduledDepHour": 8,
    "DayOfWeekCat": 4,
    "HolidayPeriod": 4,
}


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, width),
        )

    def forward(self, value):
        return value + self.net(value)


class StateEncoder(nn.Module):
    def __init__(self, channels: int, kind: str = "conv"):
        super().__init__()
        self.kind = kind
        if kind == "conv":
            self.encoder = nn.Sequential(
                nn.Conv1d(channels, 16, kernel_size=3, padding=1), nn.GELU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.GELU(),
            )
            self.project = nn.Linear(32 + channels - 1, 32)
        elif kind == "gru":
            self.encoder = nn.GRU(channels, 32, batch_first=True)
            self.project = nn.Linear(32 + channels - 1, 32)
        else:
            raise ValueError(f"unknown state encoder: {kind}")

    def forward(self, sequence):
        mask = sequence[:, :, -1:].clamp(0, 1)
        if self.kind == "conv":
            encoded = self.encoder(sequence.transpose(1, 2)).transpose(1, 2)
        else:
            encoded, _ = self.encoder(sequence)
        pooled = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.project(torch.cat([pooled, sequence[:, 0, :-1]], dim=1))


class FlightDelayResidualNet(nn.Module):
    """The forward pass consumes features only; auxiliary labels are never inputs."""

    def __init__(self, dense_dim, vocab_sizes, state_channels, *, width=256,
                 dropout=0.25, embedding_scale=1.0, state_kind="conv",
                 use_states=True, use_embeddings=True, use_offset=True,
                 simple_onehot=False):
        super().__init__()
        self.category_names = list(EMBED_DIMS)
        self.use_states = use_states
        self.use_embeddings = use_embeddings
        self.use_offset = use_offset
        self.simple_onehot = simple_onehot
        self.vocab_sizes = vocab_sizes
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_sizes[name], max(1, round(EMBED_DIMS[name] * embedding_scale)), padding_idx=0)
            for name in self.category_names
        })
        if use_embeddings:
            embedding_dim = sum(layer.embedding_dim for layer in self.embeddings.values())
        elif simple_onehot:
            embedding_dim = vocab_sizes["Reporting_Airline"] + vocab_sizes["ScheduledDepHour"]
        else:
            embedding_dim = 0
        self.dense = nn.Sequential(nn.LayerNorm(dense_dim), nn.Linear(dense_dim, 128), nn.GELU(), nn.Dropout(dropout))
        self.states = nn.ModuleDict({name: StateEncoder(channels, state_kind) for name, channels in state_channels.items()})
        state_dim = 32 * len(state_channels) if use_states else 0
        fusion_in = embedding_dim + 128 + state_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, width), nn.GELU(),
            ResidualBlock(width, dropout), ResidualBlock(width, dropout),
            nn.Linear(width, 128), nn.GELU(),
        )
        self.main = nn.Linear(128, 1)
        nn.init.zeros_(self.main.weight)
        nn.init.zeros_(self.main.bias)
        self.aux_minutes = nn.Linear(128, 1)
        self.aux_dep = nn.Linear(128, 1)
        self.aux_bucket = nn.Linear(128, 5)

    def forward(self, dense, categories, states, lr_offset, *, apply_id_dropout=False):
        pieces = [self.dense(dense)]
        if self.use_embeddings:
            embedded = []
            for column, name in enumerate(self.category_names):
                ids = categories[:, column]
                if self.training and apply_id_dropout:
                    probability = 0.10 if name in {"Origin", "Dest", "RouteHash65536"} else 0.05
                    ids = ids.masked_fill(torch.rand_like(ids, dtype=torch.float32) < probability, 0)
                embedded.append(self.embeddings[name](ids))
            pieces.append(torch.cat(embedded, dim=1))
        elif self.simple_onehot:
            carrier = F.one_hot(categories[:, self.category_names.index("Reporting_Airline")],
                                self.vocab_sizes["Reporting_Airline"]).to(dense.dtype)
            hour = F.one_hot(categories[:, self.category_names.index("ScheduledDepHour")],
                             self.vocab_sizes["ScheduledDepHour"]).to(dense.dtype)
            pieces.append(torch.cat([carrier, hour], dim=1))
        if self.use_states:
            pieces.extend(self.states[name](states[name]) for name in self.states)
        hidden = self.fusion(torch.cat(pieces, dim=1))
        residual = self.main(hidden).squeeze(1)
        main = residual + lr_offset if self.use_offset else residual
        return {
            "main": main,
            "minutes": self.aux_minutes(hidden).squeeze(1),
            "dep": self.aux_dep(hidden).squeeze(1),
            "bucket": self.aux_bucket(hidden),
        }


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
