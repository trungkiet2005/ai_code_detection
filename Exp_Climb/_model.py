"""
Model — shared SpectralCode backbone.

The backbone is the paper-aligned combo: ModernBERT tokens + BiLSTM AST +
hand-crafted structural features, fused via cross-attention, plus a spectral
stream with softmax gating. Per-method losses (HierTree, SupCon, IB, etc.)
are defined in the exp_NN_*.py files; the model itself stays identical so the
paper-efficiency claim holds across methods.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModel

from _common import SpectralConfig
from _features import (
    STRUCTURAL_FEATURE_DIM,
    SPECTRAL_FEATURE_DIM,
    extract_ast_sequence,
    extract_structural_features,
    extract_spectral_features,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Canonical source-id mapping consumed by exp_02 / exp_08 / exp_14 / exp_18.
# exp_14 expects GH = 1 (see _gh_consistency_loss gh_label default).
# Unknown / missing source → -1 sentinel; loss functions already treat None
# or invalid as "skip this sample" so -1 routes through the same guard once
# filtered.
SOURCE_ID_MAP = {"cf": 0, "gh": 1, "lc": 2}


def _source_to_id(src) -> int:
    if src is None:
        return -1
    s = str(src).strip().lower()
    return SOURCE_ID_MAP.get(s, -1)


class AICDDataset(TorchDataset):
    """Generic {code, label, source?} dataset. Works for AICD, Droid, CoDET-M4
    after each benchmark's loader normalises rows to {"code": str, "label": int,
    "source": str (optional)}."""

    def __init__(self, data, tokenizer, max_length: int = 512, ast_seq_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ast_seq_len = ast_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item["code"]
        label = item["label"]
        encoding = self.tokenizer(
            code, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        ast_seq = extract_ast_sequence(code, self.ast_seq_len)
        struct_feat = extract_structural_features(code)
        source_id = _source_to_id(item.get("source")) if isinstance(item, dict) else -1
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ast_seq": torch.tensor(ast_seq, dtype=torch.long),
            "struct_feat": torch.tensor(struct_feat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "source": torch.tensor(source_id, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Sub-encoders
# ---------------------------------------------------------------------------

class ASTEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, ast_seq: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ast_seq)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.proj(h)


class StructuralEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, token_dim: int, ast_dim: int, struct_dim: int, output_dim: int):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, output_dim)
        self.ast_proj = nn.Linear(ast_dim, output_dim)
        self.struct_proj = nn.Linear(struct_dim, output_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        self.gate = nn.Linear(output_dim * 3, output_dim)

    def forward(self, token_repr, ast_repr, struct_repr):
        t = self.token_proj(token_repr)
        a = self.ast_proj(ast_repr)
        s = self.struct_proj(struct_repr)
        seq = torch.stack([t, a, s], dim=1)
        attn_out, _ = self.cross_attn(seq, seq, seq)
        attn_out = self.norm(attn_out + seq)
        concat = torch.cat([attn_out[:, 0], attn_out[:, 1], attn_out[:, 2]], dim=-1)
        return self.gate(concat)


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------

class SpectralCode(nn.Module):
    """Multi-granularity encoder with softmax-gated neural + spectral heads.

    Inputs produced by AICDDataset; outputs a dict with:
      - logits         : gated final prediction
      - neural_logits  : fusion-head prediction (for aux loss)
      - spectral_logits: spectral-head prediction (for aux loss)
      - embeddings     : fused representation (for contrastive / hier losses)
      - gate_weights   : [B, 2] softmax gate for analysis
    """

    def __init__(self, config: SpectralConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.token_encoder = AutoModel.from_pretrained(config.encoder_name, attn_implementation="sdpa")
        token_hidden_size = self.token_encoder.config.hidden_size

        self.ast_encoder = ASTEncoder(config.num_ast_node_types, config.ast_embed_dim, config.gnn_hidden_dim)
        self.struct_encoder = StructuralEncoder(STRUCTURAL_FEATURE_DIM, config.gnn_hidden_dim)

        fusion_dim = config.z_style_dim + config.z_content_dim
        self.fusion = CrossAttentionFusion(token_hidden_size, config.gnn_hidden_dim, config.gnn_hidden_dim, fusion_dim)

        self.spectral_encoder = nn.Sequential(
            nn.Linear(SPECTRAL_FEATURE_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
        )
        spectral_dim = 128

        self.neural_head = nn.Linear(fusion_dim, num_classes)
        self.spectral_head = nn.Linear(spectral_dim, num_classes)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim + spectral_dim, 64), nn.GELU(), nn.Linear(64, 2),
        )

    def forward(self, input_ids, attention_mask, ast_seq, struct_feat, labels=None) -> Dict[str, Any]:
        token_out = self.token_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_repr = token_out.last_hidden_state[:, 0, :]
        ast_repr = self.ast_encoder(ast_seq)
        struct_repr = self.struct_encoder(struct_feat)
        h_neural = self.fusion(token_repr, ast_repr, struct_repr)

        spectral_feats = extract_spectral_features(input_ids, attention_mask)
        h_spectral = self.spectral_encoder(spectral_feats)

        neural_logits = self.neural_head(h_neural)
        spectral_logits = self.spectral_head(h_spectral)
        gate_input = torch.cat([h_neural, h_spectral], dim=-1)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)
        logits = gate_weights[:, 0:1] * neural_logits + gate_weights[:, 1:2] * spectral_logits

        return {
            "logits": logits,
            "neural_logits": neural_logits,
            "spectral_logits": spectral_logits,
            "gate_weights": gate_weights,
            "embeddings": h_neural,
        }
