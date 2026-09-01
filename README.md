# Previsione della tossicità molecolare con Graph Isomorphism Networks

Project work per *Fundamentals of Machine Learning* (3 CFU) — Youness Aharram, mat. 7198778.

## Descrizione

Le molecole del dataset **Tox21** sono rappresentate come grafi (nodi = atomi, archi =
legami) e classificate con una **Graph Isomorphism Network**, che tramite *message passing*
costruisce una rappresentazione dell'intera molecola e la usa per predire 12 tipi di
tossicità contemporaneamente: un problema multi-task e multi-label, con etichette
sbilanciate. Come termine di paragone è stata usata una **Random
Forest su fingerprint ECFP4**.

Gli esperimenti confrontano la baseline (E1), convoluzioni diverse (E2 GCN, E3 GINE),
pooling globale (E4), profondità (E5), split casuale contro scaffold (E6) e riequilibrio
delle classi nella loss (E7).

## Requisiti

Python 3.13, PyTorch 2.11 (CUDA opzionale). Le versioni in `requirements.txt` sono quelle
usate per produrre i risultati.

```bash
conda create -n fml python=3.13 -y
conda activate fml
pip install -r requirements.txt
```


## Replicare gli esperimenti

Tutti i comandi vanno lanciati dalla radice del progetto con l'ambiente `fml` attivo. Ogni
passo è indipendente dai successivi e può essere ripetuto da solo.

### 1. Dati

```bash
python src/data.py
```

Alla prima esecuzione scarica `ogbg-moltox21` dai server OGB in `data/` e stampa le statistiche per ogni task.

### 2. Baseline ECFP4 + Random Forest

```bash
python src/baselines.py
```

Calcola i fingerprint con RDKit e addestra una foresta separata per ciascun task
(500 alberi, `class_weight="balanced"`), su 3 seed: qualche minuto su CPU. Scrive
`results/baseline_rf.json` con i risultati per seed e aggregati.

### 3. GIN di riferimento e ablation

```bash
# riferimento (E1), 3 seed
for s in 0 1 2; do python src/train.py --tag gin_base --seed $s --epochs 200; done

# ablation E2-E7, 3 seed per configurazione
bash scripts/run_ablation.sh

# seed aggiuntivi: 3-4 sul riferimento e su pooling max, 3-5 su pooling sum
bash scripts/run_extra_seed.sh
```

Ogni run scrive `results/runs/<tag>_seed<N>.json` con configurazione, storia delle epoche e
metriche finali su validation e test. Le configurazioni sono queste:

| Esperimento | Tag | Comando |
|---|---|---|
| E1 riferimento | `gin_base` | `--tag gin_base` |
| E2 convoluzione GCN | `gcn_base` | `--tag gcn_base --conv gcn` |
| E3 convoluzione GINE | `gine_base` | `--tag gine_base --conv gine` |
| E4 pooling sum | `gin_sum` | `--tag gin_sum --pooling sum` |
| E4 pooling max | `gin_max` | `--tag gin_max --pooling max` |
| E5 profondità 2 | `gin_2l` | `--tag gin_2l --layers 2` |
| E5 profondità 3 | `gin_3l` | `--tag gin_3l --layers 3` |
| E6 split casuale | `gin_random` | `--tag gin_random --split random` |
| E7 riequilibrio nella loss | `gin_posw` | `--tag gin_posw --pos-weight` |

Tutto il resto resta ai valori di default: GIN a 5 strati, 300 unità nascoste, dropout 0,5,
pooling media, Adam con learning rate 1e-3, batch da 32, split scaffold. L'elenco completo
delle opzioni è in `python src/train.py --help`; per una singola configurazione basta
`python src/train.py --conv gine --pooling max --seed 1`.

`run_ablation.sh` salta i run già presenti in `results/runs/`, quindi si può interrompere e
rilanciare; per rigenerare tutto da zero va prima svuotata quella cartella. Gli script usano
`python` dal `PATH`, sovrascrivibile con `PYTHON=/percorso/a/python bash scripts/...`.
`scripts/run_troncati.sh` è solo storico: rilanciava con tetto 200 i run partiti quando il
default era 100, e su un repository pulito non serve.

### 4. Tabella dei risultati

```bash
python src/aggregate.py --per-task
```

Rilegge tutti i file in `results/runs/` più `results/baseline_rf.json` e stampa media e
deviazione standard sui seed di ROC-AUC e average precision — la Tabella 3 della relazione.
Senza `--per-task` mostra solo le medie sui 12 task.
