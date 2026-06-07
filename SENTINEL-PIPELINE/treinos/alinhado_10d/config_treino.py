"""Configuração — dataset limpo + alinhamento ≤10 dias campo↔satélite."""

from pathlib import Path

TREINO_DIR = Path(__file__).resolve().parent
PIPELINE_DADOS = TREINO_DIR.parents[1] / "dados"

TIPO_TREINO = "alinhado_10d"
DESCRICAO = "Dedup + outlier + apenas pares com ≤10 dias entre medição e imagem Sentinel."

ALVO = "mstotal"
DATASET_ARQUIVO = "dataset.csv"

MAX_DIAS_CAMPO_IMAGEM = 10
MSTOTAL_MAX_VALIDO = 25_000
DEDUP_CHAVES = ["data", "idpotreiro", "idsubarea", "dentrofora"]

SPLIT_TREINO_FIM = "2018-08-20"
SPLIT_VALID_FIM = "2019-01-14"

N_ESTIMATORS = 500
RUN_NOTAS = "Estratégia alinhamento ≤10 dias — menos ruído temporal."
