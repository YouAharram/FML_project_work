#!/bin/bash
# Ablation E2-E7: 3 seed per configurazione, split scaffold salvo dove indicato.
# Riprende da dove si era interrotto: i run gia' presenti in results/runs/ vengono saltati.
#
# Uso:   bash scripts/run_ablation.sh
#        PYTHON=/percorso/a/python bash scripts/run_ablation.sh
set -euo pipefail

RADICE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$RADICE/src"

esegui() {  # esegui <tag> <argomenti di train.py...>
  tag=$1; shift
  for s in 0 1 2; do
    if [ -f "$RADICE/results/runs/${tag}_seed${s}.json" ]; then
      echo "== salto ${tag} seed ${s} (gia' presente)"; continue
    fi
    echo "== ${tag} seed ${s}  $(date +%H:%M:%S)"
    "$PY" train.py --tag "$tag" --seed "$s" --epochs 200 --log-every 25 "$@"
  done
}

esegui gcn_base    --conv gcn                 # E2: GCN al posto di GIN
esegui gine_base   --conv gine                # E3: GINE, feature dei legami
esegui gin_sum     --pooling sum              # E4: pooling sum
esegui gin_max     --pooling max              # E4: pooling max
esegui gin_2l      --layers 2                 # E5: profondita' 2
esegui gin_3l      --layers 3                 # E5: profondita' 3
esegui gin_random  --split random             # E6: split casuale
esegui gin_posw    --pos-weight               # E7: riequilibrio nella loss
echo "== fine $(date +%H:%M:%S)"
