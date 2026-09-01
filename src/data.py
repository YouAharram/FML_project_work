"""Caricamento del dataset Tox21. """
from pathlib import Path
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from ogb.graphproppred import PygGraphPropPredDataset

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

TASK_NAMES = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def _allow_pyg_globals():
    """Autorizza le classi PyG per ``torch.load``.

    ogb 1.3.6 chiama ``torch.load`` senza ``weights_only=False``; da torch 2.6 il default
    e' cambiato e il caricamento fallisce se le classi non sono in allowlist.
    """
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])


def load_tox21(root=DATA_ROOT):
    """Scarica (se serve) e apre ogbg-moltox21.
    Restituisce ``(dataset, split_idx)`` dove ``split_idx`` e' lo split scaffold ufficiale
    con chiavi ``train`` / ``valid`` / ``test``.  """
    _allow_pyg_globals()

    dataset = PygGraphPropPredDataset(name="ogbg-moltox21", root=str(root))
    return dataset, dataset.get_idx_split()


def random_split(dataset, seed=0):
    scaffold = dataset.get_idx_split()
    n_train, n_valid = len(scaffold["train"]), len(scaffold["valid"])

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g)
    return {
        "train": perm[:n_train],
        "valid": perm[n_train:n_train + n_valid],
        "test": perm[n_train + n_valid:],
    }


def get_split(dataset, tipo="scaffold", seed=0):
    """Sceglie il criterio di partizione: ``"scaffold"`` (default) o ``"random"``."""
    if tipo == "scaffold":
        return dataset.get_idx_split()
    if tipo == "random":
        return random_split(dataset, seed)
    raise ValueError(f"split sconosciuto: {tipo}")


def dataset_stats(dataset, split_idx=None):
    y = torch.cat([d.y for d in dataset], dim=0)  # [N, 12]
    valid = ~torch.isnan(y)

    rows = []
    for t, name in enumerate(TASK_NAMES):
        col = y[:, t]
        n_valid = int(valid[:, t].sum())
        n_pos = int((col == 1).sum())
        rows.append({
            "task": name,
            "n_misurate": n_valid,
            "perc_mancanti": 100.0 * (1 - n_valid / len(y)),
            "n_positivi": n_pos,
            "perc_positivi": 100.0 * n_pos / max(n_valid, 1),
        })

    overall = {
        "n_molecole": len(y),
        "n_task": y.shape[1],
        "perc_etichette_mancanti": 100.0 * float((~valid).sum()) / valid.numel(),
    }
    if split_idx is not None:
        overall.update({f"n_{k}": len(v) for k, v in split_idx.items()})
    return overall, rows


if __name__ == "__main__":
    dataset, split_idx = load_tox21()
    overall, rows = dataset_stats(dataset, split_idx)

    print("\n=== Tox21 (ogbg-moltox21) ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n  primo grafo: {dataset[0]}")
    print(f"  feature nodo (x): {dataset[0].x.shape[1]} attributi categorici per atomo")
    print(f"  feature arco (edge_attr): {dataset[0].edge_attr.shape[1]} attributi per legame")

    print(f"\n{'task':<15}{'misurate':>10}{'% mancanti':>13}{'positivi':>10}{'% positivi':>13}")
    print("-" * 61)
    for r in rows:
        print(f"{r['task']:<15}{r['n_misurate']:>10}{r['perc_mancanti']:>12.1f}%"
              f"{r['n_positivi']:>10}{r['perc_positivi']:>12.1f}%")
