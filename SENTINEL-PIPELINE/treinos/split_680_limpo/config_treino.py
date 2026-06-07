"""Configuração — baseline split temporal, dataset corrigido 680 linhas."""

from pathlib import Path

TREINO_DIR = Path(__file__).resolve().parent
PIPELINE_DADOS = TREINO_DIR.parents[1] / "dados"

TIPO_TREINO = "split_680_limpo"
DESCRICAO = "680 linhas limpas (dedup+outlier manual), split treino/val/teste, até ±15 dias satélite."

ALVO = "mstotal"
DATASET_ARQUIVO = "dataset.csv"
DATASET_ORIGEM = "dataset_final_corrigido_sem_outlier.csv"  # em PIPELINE_DADOS

SPLIT_TREINO_FIM = "2018-08-20"
SPLIT_VALID_FIM = "2019-01-14"
MAX_DIAS_CAMPO_IMAGEM = 15  # dataset já existente; sem filtro extra na origem

N_ESTIMATORS = 500
RUN_NOTAS = "Baseline dissertação — dataset corrigido 680 linhas."
