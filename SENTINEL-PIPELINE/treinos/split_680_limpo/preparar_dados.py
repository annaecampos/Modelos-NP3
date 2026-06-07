"""Copia o dataset corrigido do pipeline principal para esta pasta de treino."""

import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_lib"))

import config_treino as cfg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    origem = cfg.PIPELINE_DADOS / cfg.DATASET_ORIGEM
    destino_dir = cfg.TREINO_DIR / "dados"
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / cfg.DATASET_ARQUIVO

    if not origem.exists():
        log.error("Arquivo não encontrado: %s", origem)
        raise SystemExit(1)

    shutil.copy2(origem, destino)
    log.info("Copiado: %s → %s", origem.name, destino)


if __name__ == "__main__":
    main()
