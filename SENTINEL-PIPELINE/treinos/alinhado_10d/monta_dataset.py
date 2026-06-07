"""Gera dataset local com limpeza e filtro ≤10 dias. Não altera 03_monta_dataset.py."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_lib"))

import config_treino as cfg  # noqa: E402
from monta_dataset_qualidade import montar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if __name__ == "__main__":
    montar(cfg)
