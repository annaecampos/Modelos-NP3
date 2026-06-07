"""Configuração — dataset limpo + alinhamento ≤5 dias (experimento restritivo)."""

from pathlib import Path

TREINO_DIR = Path(__file__).resolve().parent
PIPELINE_DADOS = TREINO_DIR.parents[1] / "dados"

TIPO_TREINO = "alinhado_5d"
DESCRICAO = "Dedup + outlier + apenas pares com ≤5 dias entre medição e imagem."

ALVO = "mstotal"
DATASET_ARQUIVO = "dataset.csv"

MAX_DIAS_CAMPO_IMAGEM = 5
MSTOTAL_MAX_VALIDO = 25_000
DEDUP_CHAVES = ["data", "idpotreiro", "idsubarea", "dentrofora"]

SPLIT_TREINO_FIM = "2018-08-20"
SPLIT_VALID_FIM = "2019-01-14"

N_ESTIMATORS = 500
RUN_NOTAS = "Experimento restritivo ≤5 dias — comparar com alinhado_10d."
