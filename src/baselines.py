"""Baseline E1: fingerprint Morgan/ECFP4 piu' Random Forest, un modello per task.

E' il termine di paragone non neurale del progetto: al posto di imparare la
rappresentazione dal grafo si usano i fingerprint circolari di rdkit, che codificano quali
sottostrutture di raggio 2 sono presenti nella molecola. Per ogni task viene addestrata una
foresta separata sulle sole molecole effettivamente misurate su quel task.

Gli SMILES e le etichette arrivano da ``mapping/mol.csv.gz`` del dataset OGB e lo split e'
quello scaffold ufficiale, letto dagli stessi file: la baseline e la GNN vedono quindi
esattamente le stesse molecole in train, validation e test (verificato da
``tests/test_baselines.py``).

    python src/baselines.py                      # 3 seed
    python src/baselines.py --class-weight none  # confronto senza riequilibrio
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

from data import DATA_ROOT, TASK_NAMES
from metrics import summarize

RDLogger.DisableLog("rdApp.*")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MOL_CSV = DATA_ROOT / "ogbg_moltox21" / "mapping" / "mol.csv.gz"
SPLIT_DIR = DATA_ROOT / "ogbg_moltox21" / "split" / "scaffold"


def load_smiles(path=MOL_CSV):
    """SMILES ed etichette del dataset, nell'ordine dei grafi PyG."""
    df = pd.read_csv(path)
    # la riga i-esima di mol.csv.gz e' il grafo i-esimo del dataset PyG
    return df["smiles"].tolist(), df[TASK_NAMES].to_numpy(dtype=float)


def load_split(path=SPLIT_DIR):
    """Indici dello split scaffold ufficiale, letti dai csv di OGB."""
    return {
        k: pd.read_csv(path / f"{k}.csv.gz", header=None).to_numpy().ravel()
        for k in ("train", "valid", "test")
    }


SANITIZE_RILASSATA = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES


def _parse(s):
    """Converte uno SMILES in molecola rdkit, con un secondo tentativo piu' permissivo.

    Otto composti di alluminio non superano il controllo di valenza standard; vengono
    riletti disattivando la sola sanitizzazione delle proprieta', cosi' nessuna molecola
    resta senza fingerprint. Restituisce ``(mol, rilassato)``, con ``mol`` a ``None`` se
    entrambi i tentativi falliscono.
    """
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        return mol, False
    mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol is None:
        return None, False
    if Chem.SanitizeMol(mol, sanitizeOps=SANITIZE_RILASSATA, catchErrors=True):
        return None, False
    return mol, True


def morgan_fingerprints(smiles, radius=2, n_bits=2048):
    """Matrice ``[n_molecole, n_bits]`` di fingerprint ECFP binari.

    Restituisce ``(X, falliti, rilassati)``: gli indici delle molecole non parsate (riga di
    zeri) e di quelle rilette con sanitizzazione rilassata.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    falliti, rilassati = [], []
    for i, s in enumerate(smiles):
        mol, rilassato = _parse(s)
        if mol is None:
            falliti.append(i)
            continue
        if rilassato:
            rilassati.append(i)
        X[i] = gen.GetFingerprintAsNumPy(mol)
    return X, falliti, rilassati


def fit_predict(X, y, split, seed, n_estimators=500, class_weight="balanced"):
    """Addestra una foresta per task e restituisce gli score di valid e test.

    Ogni foresta vede solo le molecole con etichetta osservata su quel task, il che replica
    sui dati tabellari la stessa mascheratura che :func:`metrics.masked_bce_loss` applica
    alla GNN.
    """
    scores = {k: np.zeros((len(split[k]), len(TASK_NAMES))) for k in ("valid", "test")}
    for t in range(len(TASK_NAMES)):
        tr = split["train"]
        m = ~np.isnan(y[tr, t])
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X[tr][m], y[tr, t][m])
        for k in ("valid", "test"):
            proba = clf.predict_proba(X[split[k]])
            # con una sola classe nel train predict_proba ha una colonna sola:
            # lo score e' costante e il task finisce fra i non valutabili
            scores[k][:, t] = (proba[:, 1] if proba.shape[1] > 1
                               else float(clf.classes_[0]))
    return scores


def run(seeds=(0, 1, 2), n_estimators=500, class_weight="balanced", n_bits=2048, radius=2):
    """Ripete l'esperimento su piu' seed e raccoglie le metriche di ciascuno.

    I fingerprint si calcolano una volta sola: fra un seed e l'altro cambia solo la foresta,
    quindi la deviazione standard riportata per E1 misura la sola varianza del modello, su
    dati e split identici. Non e' confrontabile con quella delle run GPU della GNN.
    """
    smiles, y = load_smiles()
    split = load_split()
    X, falliti, rilassati = morgan_fingerprints(smiles, radius=radius, n_bits=n_bits)

    per_seed = []
    for seed in seeds:
        t0 = time.time()
        scores = fit_predict(X, y, split, seed, n_estimators, class_weight)
        out = {k: summarize(y[split[k]], scores[k]) for k in ("valid", "test")}
        per_seed.append(out)
        print(f"  seed {seed}: valid AUC {out['valid']['roc_auc']:.4f}  "
              f"test AUC {out['test']['roc_auc']:.4f}  ({time.time() - t0:.0f}s)")

    return {
        "config": {
            "modello": "Morgan/ECFP4 + RandomForest",
            "radius": radius,
            "n_bits": n_bits,
            "n_estimators": n_estimators,
            "class_weight": class_weight,
            "seeds": list(seeds),
            "smiles_non_parsati": falliti,
            "smiles_sanitizzazione_rilassata": rilassati,
        },
        "per_seed": per_seed,
    }


def aggregate(per_seed, split):
    """Media e deviazione standard sui seed, nel formato usato da ``aggregate.py``."""
    auc = np.array([s[split]["roc_auc"] for s in per_seed])
    ap = np.array([s[split]["ap"] for s in per_seed])
    auc_task = np.array([s[split]["roc_auc_per_task"] for s in per_seed])
    return {
        "roc_auc_media": float(auc.mean()),
        "roc_auc_std": float(auc.std(ddof=0)),
        "ap_media": float(ap.mean()),
        "ap_std": float(ap.std(ddof=0)),
        "roc_auc_per_task_media": auc_task.mean(0).tolist(),
        "roc_auc_per_task_std": auc_task.std(0, ddof=0).tolist(),
    }


def main():
    """Esegue la baseline, stampa il riepilogo e salva ``results/baseline_rf.json``."""
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cw = None if args.class_weight == "none" else args.class_weight
    print(f"\nbaseline fingerprint: ECFP{2 * args.radius}, {args.n_bits} bit, "
          f"RF {args.n_estimators} alberi, class_weight={args.class_weight}")

    res = run(tuple(args.seeds), args.n_estimators, cw, args.n_bits, args.radius)
    res["aggregato"] = {k: aggregate(res["per_seed"], k) for k in ("valid", "test")}

    n_ril = len(res["config"]["smiles_sanitizzazione_rilassata"])
    n_falliti = len(res["config"]["smiles_non_parsati"])
    if n_ril or n_falliti:
        print(f"\n  SMILES con sanitizzazione rilassata: {n_ril}   non parsati: {n_falliti}")

    a = res["aggregato"]
    print(f"\n  valid  ROC-AUC {a['valid']['roc_auc_media']:.4f} +/- {a['valid']['roc_auc_std']:.4f}"
          f"   AP {a['valid']['ap_media']:.4f} +/- {a['valid']['ap_std']:.4f}")
    print(f"  test   ROC-AUC {a['test']['roc_auc_media']:.4f} +/- {a['test']['roc_auc_std']:.4f}"
          f"   AP {a['test']['ap_media']:.4f} +/- {a['test']['ap_std']:.4f}")

    print(f"\n{'task':<15}{'AUC valid':>12}{'AUC test':>12}{'std test':>11}")
    print("-" * 50)
    for t, name in enumerate(TASK_NAMES):
        print(f"{name:<15}{a['valid']['roc_auc_per_task_media'][t]:>12.4f}"
              f"{a['test']['roc_auc_per_task_media'][t]:>12.4f}"
              f"{a['test']['roc_auc_per_task_std'][t]:>11.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / "baseline_rf.json"
    for s in res["per_seed"]:
        for k in s:
            for campo in ("roc_auc_per_task", "ap_per_task"):
                s[k][campo] = np.asarray(s[k][campo]).tolist()
    out.write_text(json.dumps(res, indent=2))
    print(f"\n  risultati salvati in {out}")


if __name__ == "__main__":
    main()
