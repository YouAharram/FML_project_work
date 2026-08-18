import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def masked_bce_loss(logits, y, pos_weight=None):
    mask = ~torch.isnan(y)
    # i NaN vanno azzerati prima della BCE: 0 * NaN resta NaN
    target = torch.where(mask, y, torch.zeros_like(y))
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    n = mask.sum()
    if n == 0:
        return (logits * 0.0).sum()
    return (loss * mask).sum() / n


def compute_pos_weight(y):
    mask = ~torch.isnan(y)
    pos = ((y == 1) & mask).sum(0).float()
    neg = ((y == 0) & mask).sum(0).float()
    return neg / pos.clamp(min=1)


def _to_numpy(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def _per_task(y_true, y_score, fn):
    y_true, y_score = _to_numpy(y_true), _to_numpy(y_score)
    out = []
    for t in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, t])
        yt = y_true[m, t]
        # un task senza entrambe le classi non ha AUC definita
        if yt.size == 0 or np.unique(yt).size < 2:
            out.append(np.nan)
        else:
            out.append(fn(yt, y_score[m, t]))
    return np.array(out, dtype=float)


def masked_roc_auc(y_true, y_score):
    return _per_task(y_true, y_score, roc_auc_score)


def masked_average_precision(y_true, y_score):
    return _per_task(y_true, y_score, average_precision_score)


def _safe_mean(a):
    return float(np.nanmean(a)) if np.any(~np.isnan(a)) else float("nan")


def summarize(y_true, y_score):
    auc = masked_roc_auc(y_true, y_score)
    ap = masked_average_precision(y_true, y_score)
    return {
        "roc_auc": _safe_mean(auc),
        "ap": _safe_mean(ap),
        "roc_auc_per_task": auc,
        "ap_per_task": ap,
        "n_task_valutati": int(np.sum(~np.isnan(auc))),
    }
