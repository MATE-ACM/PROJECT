import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attentive Pooling
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Tanh()
        )
        self.last_attn = None  # 可选：给分析脚本用

    def forward(self, x, mask=None):
        """
        x: [B, T, D]
        mask: [B, T], 1=valid, 0=pad
        """
        attn_scores = self.attention(x).squeeze(-1)  # [B, T]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=-1)  # [B, T]
        self.last_attn = attn_scores

        pooled = torch.sum(x * attn_scores.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, attn_scores


class UniversalAudioExpert(nn.Module):
    """
    通用音频专家骨架：支持 transformer / lstm / cnn 三种 method
    输入：帧级特征 x=[B,T,D] + mask=[B,T]
    输出：{'logits': [B,C], 'pooled': [B,H]}
    """
    def __init__(self,
                 input_dim,
                 hidden_dim=256,
                 method='cnn',
                 num_layers=2,
                 dropout=0.3,
                 num_classes=6,
                 kernel_size=3):
        super().__init__()
        self.method = method

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if method == 'transformer':
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=num_layers)

        elif method == 'lstm':
            self.backbone = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False
            )

        elif method == 'cnn':
            self.backbone = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.pooler = AttentionPooling(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask=None):
        # x: [B,T,D]
        x = self.proj(x)
        x = self.dropout(x)

        if self.method == 'transformer':
            if mask is not None:
                key_padding_mask = (mask == 0)
            else:
                key_padding_mask = None
            feat_seq = self.backbone(x, src_key_padding_mask=key_padding_mask)

        elif self.method == 'lstm':
            feat_seq, _ = self.backbone(x)

        elif self.method == 'cnn':
            x_cnn = x.transpose(1, 2)            # [B,H,T]
            feat_seq = self.backbone(x_cnn)      # [B,H,T]
            feat_seq = feat_seq.transpose(1, 2)  # [B,T,H]
            if mask is not None:
                feat_seq = feat_seq.masked_fill(mask.unsqueeze(-1) == 0, 0.0)

        feat_pooled, attn_scores = self.pooler(feat_seq, mask)
        logits = self.classifier(feat_pooled)

        return {'logits': logits, 'pooled': feat_pooled}
