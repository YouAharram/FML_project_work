"""Addestramento di una GNN su Tox21: un run, un file di risultati.

Il modello si seleziona sulla ROC-AUC di validation (early stopping con pazienza
configurabile) e il test si valuta una sola volta, alla fine, con i pesi migliori. Ogni run
scrive ``results/runs/<tag>_seed<N>.json`` con configurazione, storia per epoca e metriche
finali: e' quel file che ``aggregate.py`` rilegge per costruire la tabella.

    python src/train.py                            # GIN 5 strati, mean, seed 0
    python src/train.py --conv gine --pooling sum  # varianti per le ablation
    python src/train.py --help                     # elenco completo delle opzioni
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data import DATA_ROOT, get_split, load_tox21
from metrics import compute_pos_weight, masked_bce_loss, summarize
from models import GNN, count_parameters

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"


# nota: questo fissa l'inizializzazione e l'ordine dei batch, non l'aritmetica. su GPU gli
# scatter del message passing usano somme atomiche in ordine non deterministico, quindi due
# run con lo stesso seed divergono (su CPU sono identiche). le medie su piu' seed misurano
# quindi la variabilita' complessiva run-to-run, non la sola inizializzazione
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(dataset, split_idx, batch_size, seed, num_workers=0):
    """DataLoader PyG per i tre split; solo il training viene mescolato.

    PyG impacchetta i grafi del batch in un unico grafo disconnesso, con ``batch`` che dice
    a quale molecola appartiene ogni atomo: e' l'informazione che il pooling globale usa.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return {
        name: DataLoader(
            dataset[split_idx[name]],
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            generator=g if name == "train" else None,
        )
        for name in ("train", "valid", "test")
    }


def train_epoch(model, loader, optimizer, device, pos_weight=None):
    """Una epoca di addestramento; restituisce la loss media per molecola."""
    model.train()
    totale, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = masked_bce_loss(model(batch), batch.y.float(), pos_weight)
        loss.backward()
        optimizer.step()
        totale += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return totale / n


@torch.no_grad()
def evaluate(model, loader, device):
    """Valuta uno split e restituisce il dizionario prodotto da :func:`summarize`."""
    model.eval()
    y_true, y_score = [], []
    for batch in loader:
        batch = batch.to(device)
        y_score.append(torch.sigmoid(model(batch)).cpu())
        y_true.append(batch.y.float().cpu())
    return summarize(torch.cat(y_true), torch.cat(y_score))


def run(args):
    """Esegue un run completo e salva il record dei risultati su disco."""
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset, _ = load_tox21(Path(args.data_root))
    # con lo split casuale il seed cambia anche la partizione: la deviazione standard
    # riportata su E6 include quindi la variabilita' del sorteggio, non solo quella dell'init
    split_idx = get_split(dataset, args.split, args.seed)
    loaders = build_loaders(dataset, split_idx, args.batch_size, args.seed, args.num_workers)

    pos_weight = None
    if args.pos_weight:
        y_train = torch.cat([d.y for d in dataset[split_idx["train"]]], dim=0).float()
        pos_weight = compute_pos_weight(y_train).to(device)

    model = GNN(args.conv, args.layers, args.hidden, args.dropout,
                args.pooling, dataset.num_tasks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"  {args.conv} x{args.layers}, hidden {args.hidden}, pooling {args.pooling}, "
          f"dropout {args.dropout}, pos_weight {bool(args.pos_weight)}, "
          f"split {args.split}, seed {args.seed}")
    print(f"  {count_parameters(model):,} parametri su {device}\n")

    storia = []
    best = {"valid_auc": -1.0, "epoca": -1}
    best_state = None
    t0 = time.time()

    for epoca in range(1, args.epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, device, pos_weight)
        val = evaluate(model, loaders["valid"], device)
        storia.append({"epoca": epoca, "loss": loss,
                       "valid_auc": val["roc_auc"], "valid_ap": val["ap"]})

        migliorato = val["roc_auc"] > best["valid_auc"]
        if migliorato:
            best = {"valid_auc": val["roc_auc"], "valid_ap": val["ap"], "epoca": epoca}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoca % args.log_every == 0 or migliorato:
            print(f"  epoca {epoca:3d}  loss {loss:.4f}  valid AUC {val['roc_auc']:.4f}"
                  f"{'  *' if migliorato else ''}")

        if epoca - best["epoca"] >= args.patience:
            print(f"\n  early stopping: nessun miglioramento da {args.patience} epoche")
            break

    if best_state is None:
        # nessuna epoca ha prodotto un'AUC valida (tutti NaN): non c'e' modello da
        # selezionare, meglio fallire subito che valutare pesi arbitrari
        raise RuntimeError("nessuna epoca con AUC di validation valida: run interrotto")
    model.load_state_dict(best_state)
    finale = {k: evaluate(model, loaders[k], device) for k in ("valid", "test")}
    durata = time.time() - t0

    print(f"\n  migliore epoca {best['epoca']}  ({durata / 60:.1f} min, {len(storia)} epoche)")
    print(f"  valid  ROC-AUC {finale['valid']['roc_auc']:.4f}  AP {finale['valid']['ap']:.4f}")
    print(f"  test   ROC-AUC {finale['test']['roc_auc']:.4f}  AP {finale['test']['ap']:.4f}")

    record = {
        "config": {
            "conv": args.conv, "layers": args.layers, "hidden": args.hidden,
            "dropout": args.dropout, "pooling": args.pooling, "lr": args.lr,
            "batch_size": args.batch_size, "epochs": args.epochs,
            "patience": args.patience, "pos_weight": bool(args.pos_weight),
            "seed": args.seed, "split": args.split, "n_parametri": count_parameters(model),
        },
        "migliore_epoca": best["epoca"],
        "epoche_eseguite": len(storia),
        "durata_min": durata / 60,
        "storia": storia,
        "valid": finale["valid"],
        "test": finale["test"],
    }
    for split in ("valid", "test"):
        for campo in ("roc_auc_per_task", "ap_per_task"):
            record[split][campo] = np.asarray(record[split][campo]).tolist()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    nome = args.tag or f"{args.conv}_{args.layers}l_{args.pooling}"
    out = RUNS_DIR / f"{nome}_seed{args.seed}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\n  run salvato in {out}")

    if args.save_checkpoint:
        torch.save(best_state, RUNS_DIR / f"{nome}_seed{args.seed}.pt")

    return record


def get_args(argv=None):
    """Interfaccia da riga di comando; i default sono la configurazione di riferimento."""
    p = argparse.ArgumentParser()
    p.add_argument("--conv", default="gin", choices=["gin", "gine", "gcn"])
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--hidden", type=int, default=300)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--pooling", default="mean", choices=["mean", "sum", "max"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--pos-weight", action="store_true")
    p.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--tag", default=None)
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(get_args())
