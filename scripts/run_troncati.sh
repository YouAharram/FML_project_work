#!/bin/bash
# Run che avevano esaurito il budget di 100 epoche senza far scattare l'early stopping,
# rilanciate con tetto 200: il confronto fra configurazioni deve fermarsi sulla pazienza,
# non sul budget. Storico: run_ablation.sh ora usa gia' --epochs 200, quindi su un repo
# pulito questo script non serve piu'.
#
# Uso:   bash scripts/run_troncati.sh
set -euo pipefail

RADICE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$RADICE/src"

for s in 0 1 2; do "$PY" train.py --tag gin_sum --seed "$s" --pooling sum --epochs 200 --log-every 50; done
"$PY" train.py --tag gin_random --seed 0 --split random --epochs 200 --log-every 50
"$PY" train.py --tag gin_posw   --seed 2 --pos-weight    --epochs 200 --log-every 50
"$PY" train.py --tag gine_base  --seed 2 --conv gine     --epochs 200 --log-every 50
echo "== fine troncati $(date +%H:%M:%S)"
