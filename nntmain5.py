import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve, cohen_kappa_score)
from typing import Any, Dict, Optional, Tuple
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn

import numpy as np
import math
from torch.utils.data import DataLoader
from nnt5 import (
EEG_MDD_Model,
entropy_loss,
subject_consistency_loss
)



def intra_class_compactness(feat, labels):
    loss = 0.0
    count = 0
    D = feat.size(1)
    for c in labels.unique():
        f = feat[labels == c]
        if f.size(0) > 1:
            var = torch.var(f, dim=0).mean() / D  # 规范化
            loss += var
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=feat.device)
    return loss / count

# 更新 train_one_epoch：添加分解损失和阶段控制
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, warmup_epochs=20, aux_start_epoch=10):  # 延长到20
    model.train()
    model.label_ratio = min(0.5, epoch / warmup_epochs)

    awl = AutomaticWeightedLoss(num=3).to(device)  # 新增

    total_loss, correct, total = 0, 0, 0

    for x, y, s in loader:
        x, y, s = x.to(device), y.to(device), s.to(device)
        optimizer.zero_grad()

        out = model(x, labels=y)
        logits = out["logits"]
        feat = out["feat"]

        loss_ce = criterion(logits, y)
        loss_cons = subject_consistency_loss(feat, s)
        loss_comp = intra_class_compactness(feat, y)

        if epoch >= aux_start_epoch:
            loss = awl(loss_ce, loss_cons, loss_comp)  # uncertainty 平衡
        else:
            loss = loss_ce

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total  # 简化日志，实际运行可加分解

# 更新 evaluate：类似，但无 awl.backward (只计算)
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    awl = AutomaticWeightedLoss(num=3).to(device)  # 一致性

    total_loss, correct, total = 0, 0, 0

    for x, y, s in loader:
        x, y, s = x.to(device), y.to(device), s.to(device)

        out = model(x, labels=None)
        logits = out["logits"]
        feat = out["feat"]

        loss_ce = criterion(logits, y)
        loss_cons = subject_consistency_loss(feat, s)
        loss_comp = intra_class_compactness(feat, y)
        loss = awl(loss_ce, loss_cons, loss_comp)  # 计算

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=3):  # 3 losses: ce, cons, comp
        super().__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            weighted_loss = 0.5 / (self.params[i] ** 2) * loss
            regularization = torch.log(1 + self.params[i] ** 2)
            loss_sum += weighted_loss + regularization
        return loss_sum

def _split_path(processed_dir: str, name: str) -> Path:
    path = Path(processed_dir) / f"{name}_normalized.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate it first with eegtransformer_modma_erp.py --prepare_only."
        )
    return path
class CachedERPDataset(Dataset):
    """Cached MODMA dataset returning the reference ``(x, y, s)`` triple.

    ``nnt5.EEG_MDD_Model`` expects x in ``(batch, time, channel)`` order, so
    ``input_format='btc'`` is used below. ``s`` is an integer subject index,
    which is required by ``subject_consistency_loss``.
    """

    def __init__(self, X_ntc: np.ndarray, y: np.ndarray, s: np.ndarray, input_format: str):
        self.X = X_ntc
        self.y = y.astype(np.int64)
        self.input_format = input_format
        self.s = np.asarray(s, dtype=np.int64)
        if len(self.X) != len(self.y) or len(self.X) != len(self.s):
            raise ValueError("X, y and subject-id arrays must have the same length.")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x_ntc = torch.from_numpy(self.X[index]).contiguous()
        if self.input_format == "b1ct":
            x = x_ntc.transpose(0, 1).contiguous().unsqueeze(0)
        elif self.input_format == "bct":
            x = x_ntc.transpose(0, 1).contiguous()
        elif self.input_format == "btc":
            x = x_ntc
        else:
            raise ValueError(f"Unsupported input_format={self.input_format}")
        y = torch.tensor(self.y[index], dtype=torch.long)
        s = torch.tensor(self.s[index], dtype=torch.long)
        return x, y, s


def encode_subject_ids(*subject_arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Encode string subject IDs consistently across train/val/test."""
    values = np.concatenate([np.asarray(item).astype(str) for item in subject_arrays])
    lookup = {name: index for index, name in enumerate(np.unique(values))}
    return tuple(
        np.asarray([lookup[str(value)] for value in np.asarray(item)], dtype=np.int64)
        for item in subject_arrays
    )

def load_cached_split(
    processed_dir: str,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    path = _split_path(processed_dir, name)
    with np.load(path, allow_pickle=False) as split:
        if "X" not in split or "y" not in split:
            raise ValueError(f"{path} must contain X and y arrays.")
        X = split["X"].astype(np.float32)
        y = split["y"].astype(np.int64)
        subjects = split["subject_ids"].astype(str) if "subject_ids" in split else None
        event_types = split["event_types"].astype(str) if "event_types" in split else None

    if X.ndim != 3:
        raise ValueError(f"{path}: expected X=(trials,time,channels), got {X.shape}.")
    if len(X) != len(y):
        raise ValueError(f"{path}: X and y lengths differ: {len(X)} vs {len(y)}.")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: X contains NaN or infinite values.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{path}: expected labels 0/1, got {np.unique(y)}.")
    return X, y, subjects, event_types


if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = 200
    patience = 50
    processed_dir = r"/home/huangzh/MODMA/EEG_128channels_ERP_processed_nos"
    dim = 32
    X_train, y_train, sid_train_raw, _ = load_cached_split(processed_dir, "train")
    X_val, y_val, sid_val_raw, _ = load_cached_split(processed_dir, "val")
    X_test, y_test, sid_test_raw, _ = load_cached_split(processed_dir, "test")

    if X_train.shape[1:] != X_val.shape[1:] or X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(f"Split shapes differ: {X_train.shape}, {X_val.shape}, {X_test.shape}")
    n_time, n_channels = X_train.shape[1], X_train.shape[2]
    sid_train, sid_val, sid_test = encode_subject_ids(
        sid_train_raw, sid_val_raw, sid_test_raw
    )
    model = EEG_MDD_Model(
        n_ch=n_channels,
        n_classes=2,
    )
    model.to(device)

    train_loader = DataLoader(CachedERPDataset(X_train, y_train, sid_train, "btc"), batch_size=16, shuffle=True, num_workers=4,
                              pin_memory=True)  # Larger batch for speed
    val_loader = DataLoader(CachedERPDataset(X_val, y_val, sid_val, "btc"), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(CachedERPDataset(X_test, y_test, sid_test, "btc"), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-3)  # AdamW for better regularization
    scheduler = CosineAnnealingLR(optimizer, 15)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)  # 新增
    best_val_acc = 0
    counter = 0

    # for epoch in range(1, epochs + 1):
    #     train_loss,  train_acc = train_one_epoch(
    #         model, train_loader, optimizer, criterion, device, epoch,
    #         warmup_epochs=20, aux_start_epoch=10
    #     )
    #
    #     val_loss, val_acc = evaluate(
    #         model, val_loader, criterion, device
    #     )
    #
    #     print(
    #         f"Epoch {epoch:03d} | "
    #         f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
    #         f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
    #         f"label_ratio={model.label_ratio:.2f}"
    #     )
    #
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), "best_model_m64.pth")
    #         counter = 0
    #     else:
    #         counter += 1
    #         if counter >= patience:
    #             print("Early stopping!")
    #             break
    #     scheduler.step()
    model.load_state_dict(torch.load(r"best_model_m64.pth", map_location=device))

    model.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for x, y, s in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x, labels= None)
            ll = logits['logits']
            probs.extend(torch.softmax(ll, dim=1).cpu().numpy())
            pred = ll.argmax(1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    preds = np.array(preds)
    probs = np.array(probs)
    labels = np.array(labels)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs[:, 1])
    kappa = cohen_kappa_score(labels, preds)

    print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Kappa: {kappa:.4f}")
