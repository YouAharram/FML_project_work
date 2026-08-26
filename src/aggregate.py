"""Tabella riassuntiva dei run della GNN: media e deviazione standard sui seed.

Legge tutti i ``results/runs/*.json``, li raggruppa per tag (il nome del file senza il
suffisso ``_seed<N>``) e stampa una riga per configurazione, preceduta dalla baseline E1 se
``results/baseline_rf.json`` esiste.

    python src/aggregate.py              # tabella principale
    python src/aggregate.py --per-task   # anche il dettaglio per i 12 task
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import TASK_NAMES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"

# ordine di lettura della tabella: prima il riferimento, poi un esperimento alla volta
ETICHETTE = {
    "gin_base": ("--", "GIN 5 strati, mean (riferimento)"),
    "gcn_base": ("E2", "GCN al posto di GIN"),
    "gine_base": ("E3", "GINE: feature dei legami"),
    "gin_sum": ("E4", "pooling sum"),
    "gin_max": ("E4", "pooling max"),
    "gin_2l": ("E5", "2 strati"),
    "gin_3l": ("E5", "3 strati"),
    "gin_random": ("E6", "split casuale"),
    "gin_posw": ("E7", "pos_weight nella loss"),
}


def _ordine(tag):
    """Chiave di ordinamento: l'ordine di ETICHETTE, poi i tag sconosciuti in fondo."""
    return (list(ETICHETTE).index(tag), tag) if tag in ETICHETTE else (len(ETICHETTE), tag)


def load_runs(runs_dir=RUNS_DIR, pattern="*.json"):
    """Raggruppa i json dei run per tag, cioe' per configurazione."""
    gruppi = defaultdict(list)
    for f in sorted(Path(runs_dir).glob(pattern)):
        r = json.loads(f.read_text())
        # il tag e' il nome del file senza il suffisso del seed
        gruppi[f.stem.rsplit("_seed", 1)[0]].append(r)
    return gruppi


def _ms(valori):
    """Media e deviazione standard (popolazione) lungo l'asse dei seed."""
    a = np.asarray(valori, dtype=float)
    return a.mean(0), a.std(0, ddof=0)


def riassumi(runs, split="test"):
    """Aggrega i run di una stessa configurazione su uno split.

    La deviazione standard e' quella fra seed: sulle run GPU misura la variabilita'
    complessiva run-to-run, non la sola inizializzazione (vedi la nota in ``train.py``).
    """
    auc_m, auc_s = _ms([r[split]["roc_auc"] for r in runs])
    ap_m, ap_s = _ms([r[split]["ap"] for r in runs])
    task_m, task_s = _ms([r[split]["roc_auc_per_task"] for r in runs])
    return {
        "n_seed": len(runs),
        "roc_auc_media": float(auc_m), "roc_auc_std": float(auc_s),
        "ap_media": float(ap_m), "ap_std": float(ap_s),
        "roc_auc_per_task_media": task_m.tolist(),
        "roc_auc_per_task_std": task_s.tolist(),
        "epoche_medie": float(np.mean([r["epoche_eseguite"] for r in runs])),
        "durata_media_min": float(np.mean([r["durata_min"] for r in runs])),
    }


def baseline():
    """Risultati aggregati della baseline E1, o ``None`` se non e' stata ancora eseguita."""
    f = RESULTS_DIR / "baseline_rf.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["aggregato"]


def main():
    """Stampa la tabella di confronto fra tutte le configurazioni disponibili."""
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="*.json")
    p.add_argument("--per-task", action="store_true")
    args = p.parse_args()

    gruppi = load_runs(pattern=args.pattern)
    if not gruppi:
        print("nessun run in results/runs/")
        return

    righe = []
    b = baseline()
    if b:
        righe.append(("E1", "fingerprint + Random Forest", 3,
                      b["valid"]["roc_auc_media"], b["valid"]["roc_auc_std"],
                      b["test"]["roc_auc_media"], b["test"]["roc_auc_std"],
                      b["test"]["ap_media"], b["test"]["ap_std"]))
    for tag in sorted(gruppi, key=_ordine):
        runs = gruppi[tag]
        v, t = riassumi(runs, "valid"), riassumi(runs, "test")
        exp, descrizione = ETICHETTE.get(tag, ("", tag))
        righe.append((exp, descrizione, len(runs),
                      v["roc_auc_media"], v["roc_auc_std"],
                      t["roc_auc_media"], t["roc_auc_std"],
                      t["ap_media"], t["ap_std"]))

    print(f"\n{'':<4}{'configurazione':<32}{'n':>3}{'AUC valid':>19}"
          f"{'AUC test':>19}{'AP test':>19}")
    print("-" * 96)
    for exp, nome, n, vm, vs, tm, ts, am, asd in righe:
        print(f"{exp:<4}{nome:<32}{n:>3}{vm:>12.4f} ±{vs:.4f}"
              f"{tm:>12.4f} ±{ts:.4f}{am:>12.4f} ±{asd:.4f}")

    if args.per_task:
        for tag in sorted(gruppi, key=_ordine):
            runs = gruppi[tag]
            t = riassumi(runs, "test")
            print(f"\n  {tag} — AUC test per task")
            for nome, m, s in zip(TASK_NAMES, t["roc_auc_per_task_media"],
                                  t["roc_auc_per_task_std"]):
                print(f"    {nome:<16}{m:.4f} ±{s:.4f}")


if __name__ == "__main__":
    main()
