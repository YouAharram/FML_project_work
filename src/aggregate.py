import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import TASK_NAMES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"


def load_runs(runs_dir=RUNS_DIR, pattern="*.json"):
    gruppi = defaultdict(list)
    for f in sorted(Path(runs_dir).glob(pattern)):
        r = json.loads(f.read_text())
        # il tag e' il nome del file senza il suffisso del seed
        gruppi[f.stem.rsplit("_seed", 1)[0]].append(r)
    return gruppi


def _ms(valori):
    a = np.asarray(valori, dtype=float)
    return a.mean(0), a.std(0, ddof=0)


def riassumi(runs, split="test"):
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
    f = RESULTS_DIR / "baseline_rf.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["aggregato"]


def main():
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
        righe.append(("fingerprint+RF (E1)", 3,
                      b["valid"]["roc_auc_media"], b["valid"]["roc_auc_std"],
                      b["test"]["roc_auc_media"], b["test"]["roc_auc_std"]))
    for tag, runs in sorted(gruppi.items()):
        v, t = riassumi(runs, "valid"), riassumi(runs, "test")
        righe.append((tag, len(runs), v["roc_auc_media"], v["roc_auc_std"],
                      t["roc_auc_media"], t["roc_auc_std"]))

    print(f"\n{'configurazione':<24}{'seed':>5}{'AUC valid':>20}{'AUC test':>20}")
    print("-" * 69)
    for nome, n, vm, vs, tm, ts in righe:
        print(f"{nome:<24}{n:>5}{vm:>13.4f} ±{vs:.4f}{tm:>13.4f} ±{ts:.4f}")

    if args.per_task:
        for tag, runs in sorted(gruppi.items()):
            t = riassumi(runs, "test")
            print(f"\n  {tag} — AUC test per task")
            for nome, m, s in zip(TASK_NAMES, t["roc_auc_per_task_media"],
                                  t["roc_auc_per_task_std"]):
                print(f"    {nome:<16}{m:.4f} ±{s:.4f}")


if __name__ == "__main__":
    main()
