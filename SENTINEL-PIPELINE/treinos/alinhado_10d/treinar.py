"""Treino RF — alinhado_10d. Não altera 04_modelo_rf.py."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_lib"))

import config_treino as cfg  # noqa: E402
from treino_rf import executar_treino  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    if not (cfg.TREINO_DIR / "dados" / cfg.DATASET_ARQUIVO).exists():
        print("Execute primeiro: python monta_dataset.py")
        raise SystemExit(1)
    executar_treino(cfg)
