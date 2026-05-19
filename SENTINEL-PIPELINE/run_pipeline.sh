#!/usr/bin/env bash
# Executa o pipeline Sentinel completo na ordem correta (inclui busca de índices — passo 02).
# Uso: bash run_pipeline.sh   (a partir desta pasta ou com caminho completo)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python 01_extrai_banco.py
python 02_busca_indices.py
python 03_monta_dataset.py
python 04_modelo_rf.py

echo "Pipeline concluído."
