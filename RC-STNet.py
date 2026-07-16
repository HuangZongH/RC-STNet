import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict

# =========================
# Multi-scale Temporal Stem
# =========================
class MultiScaleTemporalStem(nn.Module):
    def __init__(self, in_ch=63, hidden=64, kernels=( 16, 32, 64,128)):
        super().__init__()
        self.kernels = kernels
        hidden_per = hidden // len(kernels)

        self.temporal_convs = nn.ModuleList([
            nn.Conv2d(1, hidden_per, kernel_size=(1, k),
                      padding=(0, k // 2), bias=False)
            for k in kernels
        ])

        self.hidden_total = hidden_per * len(kernels)

        self.spatial = nn.Conv2d(
            self.hidden_total, self.hidden_total,
            kernel_size=(in_ch, 1),
            groups=self.hidden_total,
            bias=False
        )

        self.bn = nn.BatchNorm2d(self.hidden_total)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.hidden_total, self.hidden_total // 4, 1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_total // 4, self.hidden_total, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T]
        x = x.unsqueeze(1)  # [B,1,C,T]
        feats = [conv(x) for conv in self.temporal_convs]
        feat = torch.cat(feats, dim=1)  # [B,H,C,T]

        # per-channel tokens
        per_channel = feat.mean(dim=-1).permute(0, 2, 1)

        # spatial aggregation
        sp = self.spatial(feat)
        sp = self.bn(sp)
        sp = sp * self.se(sp)
        sp = sp.squeeze(2)  # [B,H,T]

        return per_channel, sp


# =========================
# EARFM & NSATM
# =========================
class EARFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Conv1d(dim, dim // 2, 3, padding=1)
        self.dec = nn.Conv1d(dim // 2, dim, 3, padding=1)
        self.gate = nn.Sequential(nn.Conv1d(dim, 1, 1), nn.Sigmoid())

    def forward(self, x):
        z = F.relu(self.enc(x))
        rec = self.dec(z)
        g = self.gate(x - rec)
        return g * rec + (1 - g) * x, g.squeeze(1)


class NSATM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.c2 = nn.Conv1d(dim, dim, 3, dilation=2, padding=2)
        self.c3 = nn.Conv1d(dim, dim, 3, dilation=4, padding=4)
        self.gate = nn.Sequential(nn.Conv1d(dim, 1, 1), nn.Sigmoid())

    def forward(self, x):
        ns = F.relu(self.c1(x) + self.c2(x) + self.c3(x))
        g = self.gate(ns)
        return g * ns + (1 - g) * x


# =========================
# Temporal Transformer
# =========================
class TemporalTransformer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B,D,T] → [B,T,D]
        x = x.permute(0, 2, 1)
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        h = self.ffn(x)
        x = self.norm2(x + h)
        return x.permute(0, 2, 1)


# =========================
# RegionConditioned SAFE
# =========================
class RegionConditionedSAFE(nn.Module):
    def __init__(self, dim, channel_names, channel_to_region, num_classes=2):
        super().__init__()
        self.channel_names = channel_names
        self.channel_to_region = channel_to_region

        self.regions = sorted(set(channel_to_region.values()))
        self.region_idx = {r: i for i, r in enumerate(self.regions)}

        self.groups = defaultdict(list)
        for i, ch in enumerate(channel_names):
            self.groups[channel_to_region[ch]].append(i)

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x_node, labels=None, logits=None, alpha=0.5):
        B, N, D = x_node.shape
        device = x_node.device

        if logits is not None:
            pred = logits.argmax(1)
        else:
            pred = torch.zeros(B, dtype=torch.long, device=device)

        g_pred = torch.stack([self.gates[pred[i]](x_node[i]).squeeze(-1)
                              for i in range(B)])

        if labels is not None:
            g_gt = torch.stack([self.gates[labels[i]](x_node[i]).squeeze(-1)
                                for i in range(B)])
            g = alpha * g_gt + (1 - alpha) * g_pred
        else:
            g = g_pred

        xw = x_node * g.unsqueeze(-1)

        region_feat = torch.zeros(B, len(self.regions), D, device=device)
        for r, idxs in self.groups.items():
            region_feat[:, self.region_idx[r]] = xw[:, idxs].mean(1)

        q, k, v = self.qkv(region_feat).chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(D), dim=-1)
        region_att = attn @ v

        feat = (region_feat.mean(1) + region_att.mean(1)) / 2
        conn = F.normalize(region_feat, dim=-1) @ F.normalize(region_feat, dim=-1).transpose(1, 2)

        return {
            "feat_space": self.out(feat),
            "channel_gate": g,
            "region_feat": region_feat,
            "region_conn": conn
        }

def entropy_loss(logits):
    p = torch.softmax(logits, dim=1)
    return - (p * torch.log(p + 1e-8)).sum(dim=1).mean()


def subject_consistency_loss(feats, subject_ids):
    """
    feats: [B, D]
    subject_ids: [B]
    """
    loss = 0.0
    uniq = torch.unique(subject_ids)
    for sid in uniq:
        idx = (subject_ids == sid)
        if idx.sum() > 1:
            f = feats[idx]
            loss += torch.var(f, dim=0).mean()
    return loss / max(len(uniq), 1)

class MarginClassifier(nn.Module):
    def __init__(self, in_dim, n_classes=2, s=20.0, m=0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes, bias=False)
        self.s = s
        self.m = m


    def forward(self, x, labels=None):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.fc.weight, dim=1)
        logits = F.linear(x, w)
        if labels is not None and self.m > 0:
            oh = F.one_hot(labels, logits.size(1)).float()
            logits = logits - oh * self.m
        return logits * self.s


# =========================
# EGI 128-channel Model
# =========================

CHANNEL_TO_REGION = {
    # ========== Prefrontal ==========
    'E2': 'Prefrontal',
    'E3': 'Prefrontal',
    'E8': 'Prefrontal',
    'E9': 'Prefrontal',
    'E10': 'Prefrontal',
    'E14': 'Prefrontal',
    'E15': 'Prefrontal',
    'E16': 'Prefrontal',
    'E17': 'Prefrontal',
    'E18': 'Prefrontal',
    'E21': 'Prefrontal',
    'E22': 'Prefrontal',
    'E23': 'Prefrontal',
    'E25': 'Prefrontal',
    'E26': 'Prefrontal',
    'E126': 'Prefrontal',
    'E127': 'Prefrontal',

    # ========== Frontal ==========
    'E1': 'Frontal',
    'E4': 'Frontal',
    'E5': 'Frontal',
    'E11': 'Frontal',
    'E12': 'Frontal',
    'E19': 'Frontal',
    'E20': 'Frontal',
    'E24': 'Frontal',
    'E27': 'Frontal',
    'E32': 'Frontal',
    'E33': 'Frontal',
    'E38': 'Frontal',
    'E118': 'Frontal',
    'E121': 'Frontal',
    'E122': 'Frontal',
    'E123': 'Frontal',
    'E124': 'Frontal',
    'E125': 'Frontal',
    'E128': 'Frontal',

    # ========== Fronto-Central ==========
    'E6': 'FrontoCentral',
    'E7': 'FrontoCentral',
    'E13': 'FrontoCentral',
    'E28': 'FrontoCentral',
    'E29': 'FrontoCentral',
    'E30': 'FrontoCentral',
    'E34': 'FrontoCentral',
    'E35': 'FrontoCentral',
    'E36': 'FrontoCentral',
    'E39': 'FrontoCentral',
    'E40': 'FrontoCentral',
    'E43': 'FrontoCentral',
    'E44': 'FrontoCentral',
    'E48': 'FrontoCentral',
    'E49': 'FrontoCentral',
    'E104': 'FrontoCentral',
    'E105': 'FrontoCentral',
    'E106': 'FrontoCentral',
    'E109': 'FrontoCentral',
    'E110': 'FrontoCentral',
    'E111': 'FrontoCentral',
    'E112': 'FrontoCentral',
    'E113': 'FrontoCentral',
    'E114': 'FrontoCentral',
    'E115': 'FrontoCentral',
    'E116': 'FrontoCentral',
    'E117': 'FrontoCentral',
    'E119': 'FrontoCentral',
    'E120': 'FrontoCentral',

    # ========== Central ==========
    'E31': 'Central',
    'E37': 'Central',
    'E41': 'Central',
    'E45': 'Central',
    'E56': 'Central',
    'E63': 'Central',
    'E80': 'Central',
    'E87': 'Central',
    'E99': 'Central',
    'E103': 'Central',
    'E107': 'Central',
    'E108': 'Central',

    # ========== Centro-Parietal ==========
    'E42': 'CentroParietal',
    'E46': 'CentroParietal',
    'E47': 'CentroParietal',
    'E50': 'CentroParietal',
    'E51': 'CentroParietal',
    'E52': 'CentroParietal',
    'E53': 'CentroParietal',
    'E54': 'CentroParietal',
    'E55': 'CentroParietal',
    'E57': 'CentroParietal',
    'E61': 'CentroParietal',
    'E78': 'CentroParietal',
    'E79': 'CentroParietal',
    'E86': 'CentroParietal',
    'E92': 'CentroParietal',
    'E93': 'CentroParietal',
    'E97': 'CentroParietal',
    'E98': 'CentroParietal',
    'E100': 'CentroParietal',
    'E101': 'CentroParietal',
    'E102': 'CentroParietal',

    # ========== Parietal ==========
    'E58': 'Parietal',
    'E59': 'Parietal',
    'E60': 'Parietal',
    'E62': 'Parietal',
    'E64': 'Parietal',
    'E85': 'Parietal',
    'E91': 'Parietal',
    'E95': 'Parietal',
    'E96': 'Parietal',

    # ========== Parieto-Occipital ==========
    'E65': 'ParietoOccipital',
    'E66': 'ParietoOccipital',
    'E67': 'ParietoOccipital',
    'E68': 'ParietoOccipital',
    'E69': 'ParietoOccipital',
    'E71': 'ParietoOccipital',
    'E72': 'ParietoOccipital',
    'E73': 'ParietoOccipital',
    'E76': 'ParietoOccipital',
    'E77': 'ParietoOccipital',
    'E84': 'ParietoOccipital',
    'E88': 'ParietoOccipital',
    'E89': 'ParietoOccipital',
    'E90': 'ParietoOccipital',
    'E94': 'ParietoOccipital',

    # ========== Occipital ==========
    'E70': 'Occipital',
    'E74': 'Occipital',
    'E75': 'Occipital',
    'E81': 'Occipital',
    'E82': 'Occipital',
    'E83': 'Occipital',
}


CHANNEL_NAMES = [
    'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8',
    'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16',
    'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24',
    'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E32',
    'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40',
    'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E48',
    'E49', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E56',
    'E57', 'E58', 'E59', 'E60', 'E61', 'E62', 'E63', 'E64',
    'E65', 'E66', 'E67', 'E68', 'E69', 'E70', 'E71', 'E72',
    'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80',
    'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E88',
    'E89', 'E90', 'E91', 'E92', 'E93', 'E94', 'E95', 'E96',
    'E97', 'E98', 'E99', 'E100', 'E101', 'E102', 'E103', 'E104',
    'E105', 'E106', 'E107', 'E108', 'E109', 'E110', 'E111', 'E112',
    'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120',
    'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128'
]

class EEG_MDD_Model(nn.Module):
    def __init__(self, n_ch=63, n_classes=2, label_ratio=0.0):
        super().__init__()
        self.label_ratio = label_ratio

        self.stem = MultiScaleTemporalStem(n_ch, 64)
        self.earfm = EARFM(self.stem.hidden_total)
        self.nsatm = NSATM(self.stem.hidden_total)

        self.proj = nn.Conv1d(self.stem.hidden_total, 64, 1)
        self.temporal = TemporalTransformer(64)

        self.node_proj = nn.Linear(self.stem.hidden_total, 64)
        self.graph = RegionConditionedSAFE(64, CHANNEL_NAMES, CHANNEL_TO_REGION)

        self.classifier = MarginClassifier(128, n_classes, s=20.0, m=0.1)

    def forward(self, x, labels=None):
        x = x.permute(0, 2, 1)
        per_ch, sp = self.stem(x)

        sp, heat = self.earfm(sp)
        sp = self.nsatm(sp)

        sp = self.proj(sp)
        sp = self.temporal(sp)
        feat_t = sp.mean(-1)

        x_node = self.node_proj(per_ch)

        glabel = labels if (self.training and torch.rand(1).item() < self.label_ratio) else None
        g = self.graph(x_node, labels=glabel)

        feat = torch.cat([feat_t, g["feat_space"]], dim=1)
        logits = self.classifier(feat, labels if self.training else None)

        return {
            "logits": logits,
            "feat": feat,
            "node_gate": g["channel_gate"],
            "region_conn": g["region_conn"],
            "region_feat": g["region_feat"],
            "earfm_gate": heat
        }
