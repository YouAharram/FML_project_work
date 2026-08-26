"""Loss e metriche per il problema multi-task con etichette mancanti.

Su Tox21 il 17% delle celle della matrice delle etichette e' NaN: una molecola non e'
stata testata su tutti e 12 gli assay. Ogni funzione di questo modulo ignora quelle celle
invece di sostituirle con zeri, che le trasformerebbe in negativi inventati. La ROC-AUC e
l'average precision sono calcolate per task e poi mediate, escludendo i task che nello
split in esame hanno una sola classe e per cui l'AUC non e' definita.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def masked_bce_loss(logits, y, pos_weight=None):
    """BCE con logit mediata sulle sole etichette osservate.

    ``y`` ha shape ``[batch, 12]`` e contiene NaN dove l'etichetta manca. La media e' sul
    numero di celle valide, non sul totale, cosi' che batch con quantita' diverse di
    etichette mancanti restino confrontabili. ``pos_weight`` (opzionale, shape ``[12]``)
    riequilibra le classi ed e' usato nell'esperimento E7.
    """
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
    """Rapporto negativi/positivi per task, da passare a :func:`masked_bce_loss`.

    Va calcolato sul solo split di training. Su Tox21 va da ~5.7 (SR-MMP) a ~38.6
    (NR-PPAR-gamma).
    """
    mask = ~torch.isnan(y)
    pos = ((y == 1) & mask).sum(0).float()
    neg = ((y == 0) & mask).sum(0).float()
    return neg / pos.clamp(min=1)


def _to_numpy(a):
    """Converte tensori torch o array-like in ``numpy.ndarray``."""
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def _per_task(y_true, y_score, fn):
    """Applica la metrica ``fn`` colonna per colonna, ignorando le etichette mancanti.

    Restituisce un array di 12 valori, con NaN nelle posizioni dei task non valutabili
    (nessuna etichetta, oppure una sola classe presente).
    """
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
    """ROC-AUC per task (array di 12 valori, NaN dove non definita)."""
    return _per_task(y_true, y_score, roc_auc_score)


def masked_average_precision(y_true, y_score):
    """Average precision per task (array di 12 valori, NaN dove non definita)."""
    return _per_task(y_true, y_score, average_precision_score)


def _safe_mean(a):
    """Media ignorando i NaN; NaN se non resta nessun valore."""
    return float(np.nanmean(a)) if np.any(~np.isnan(a)) else float("nan")


def summarize(y_true, y_score):
    """Riassume le predizioni di uno split.

    Restituisce un dizionario con ROC-AUC e AP mediate sui task, i valori per task e
    ``n_task_valutati``, cioe' quanti dei 12 task erano effettivamente valutabili.
    """
    auc = masked_roc_auc(y_true, y_score)
    ap = masked_average_precision(y_true, y_score)
    return {
        "roc_auc": _safe_mean(auc),
        "ap": _safe_mean(ap),
        "roc_auc_per_task": auc,
        "ap_per_task": ap,
        "n_task_valutati": int(np.sum(~np.isnan(auc))),
    }
