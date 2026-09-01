#!/bin/bash
# Seed aggiuntivi (3-5) per il riferimento, per la configurazione finale e per quella
# piu' instabile. Le run su GPU non sono riproducibili a parita' di seed: su queste tre
# tre ripetizioni sono troppo poche per una media affidabile.
#
# Uso:   bash scripts/run_extra_seed.sh
set -euo pipefail

RADICE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$RADICE/src"

for s in 3 4 5; do "$PY" train.py --tag gin_sum  --seed "$s" --pooling sum --epochs 200 --log-every 50; done
for s in 3 4;   do "$PY" train.py --tag gin_base --seed "$s"               --epochs 200 --log-every 50; done
for s in 3 4;   do "$PY" train.py --tag gin_max  --seed "$s" --pooling max --epochs 200 --log-every 50; done
echo "== fine extra $(date +%H:%M:%S)"
