"""
Monta o dataset final unindo dados de campo (TouceiraTech) e
índices de vegetação (Sentinel-2 / Copernicus).

Também realiza:
  - Engenharia de features temporais (mês, estação, dias desde última medição)
  - Remoção de linhas sem índices (data sem cobertura de satélite válida)
  - Relatório de cobertura e estatísticas básicas

Saída:
    dados/dataset_final.csv  ← tabela de treino completa

Uso:
    python 03_monta_dataset.py
"""

import logging

import numpy as np
import pandas as pd

from config import ALVO, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def estacao_sul(mes: int) -> str:
    """Estação do ano para o hemisfério sul."""
    if mes in (12, 1, 2):
        return "verao"
    elif mes in (3, 4, 5):
        return "outono"
    elif mes in (6, 7, 8):
        return "inverno"
    return "primavera"


def main() -> None:
    f_campo   = DATA_DIR / "dados_campo.csv"
    f_indices = DATA_DIR / "indices_sentinel2.csv"
    f_saida   = DATA_DIR / "dataset_final.csv"

    if not f_campo.exists():
        log.error(
            "Arquivo não encontrado: %s\nExecute primeiro: python 01_extrai_banco.py",
            f_campo,
        )
        raise SystemExit(1)
    if not f_indices.exists():
        log.error(
            "Arquivo não encontrado: %s\nExecute antes: python 02_busca_indices.py "
            "(gera índices Sentinel-2 a partir de dados_campo.csv)",
            f_indices,
        )
        raise SystemExit(1)

    # ── Carrega ───────────────────────────────────────────────────────────────
    df_campo = pd.read_csv(f_campo,   sep=";", parse_dates=["data"])
    df_idx   = pd.read_csv(f_indices, sep=";", parse_dates=["data"])

    log.info("Campo   : %d linhas", len(df_campo))
    log.info("Índices : %d linhas", len(df_idx))

    # ── Join por (data, idpotreiro, idsubarea) ────────────────────────────────
    chave = ["data", "idpotreiro", "idsubarea"]
    df = df_campo.merge(
        df_idx.drop(columns=["cloud_cover"], errors="ignore"),
        on=chave,
        how="inner",
    )
    log.info("Após join inner: %d linhas", len(df))

    if len(df) == 0:
        log.error(
            "Join resultou em 0 linhas. Verifique se dados_campo.csv e "
            "indices_sentinel2.csv têm datas/IDs compatíveis."
        )
        raise SystemExit(1)

    # ── Remove linhas sem índice principal ───────────────────────────────────
    antes = len(df)
    df = df.dropna(subset=["ndvi"])
    log.info("Removidas sem NDVI: %d linhas (restam %d)", antes - len(df), len(df))

    # ── Remove linhas sem alvo ────────────────────────────────────────────────
    antes = len(df)
    df = df.dropna(subset=[ALVO])
    log.info("Removidas sem '%s': %d linhas (restam %d)", ALVO, antes - len(df), len(df))

    # ── Engenharia de features ────────────────────────────────────────────────
    df["mes"]          = df["data"].dt.month
    df["ano"]          = df["data"].dt.year
    df["estacao"]      = df["mes"].apply(estacao_sul)

    # Dias desde a última medição para o mesmo piquete (proxy do intervalo de pastejo)
    df = df.sort_values(["idpotreiro", "idsubarea", "data"])
    df["dias_desde_ult_medicao"] = (
        df.groupby(["idpotreiro", "idsubarea"])["data"]
        .diff()
        .dt.days
        .fillna(0)
        .astype(int)
    )

    # One-hot encoding da estação (sem dropar nenhuma — evita multicolinearidade implícita)
    df = pd.get_dummies(df, columns=["estacao"], prefix="est", drop_first=False)
    # dentrofora: D=1, F=0
    df["dentrofora"] = (df["dentrofora"] == "D").astype(int)

    # ── Ordena colunas ────────────────────────────────────────────────────────
    colunas_id = ["data", "idpotreiro", "idsubarea", "data_imagem"]
    colunas_alvo = ["mstotal", "msanoni", "msoutras", "media"]
    colunas_sat  = [c for c in df.columns if c.startswith("s2_") or c in ("ndvi", "evi", "ndre", "savi")]
    colunas_clim = ["tmin", "tmed", "tmax", "umidade", "velocidadevento",
                    "radiacaosolar", "chuva", "somatermica", "def", "exc"]
    colunas_feat = ["dentrofora", "mes", "ano", "dias_desde_ult_medicao"] + \
                   [c for c in df.columns if c.startswith("est_")]

    outras = [c for c in df.columns
              if c not in colunas_id + colunas_alvo + colunas_sat + colunas_clim + colunas_feat + ["geometria"]]

    ordem = colunas_id + colunas_alvo + colunas_sat + colunas_clim + colunas_feat + outras
    df = df[[c for c in ordem if c in df.columns]]

    # ── Salva ─────────────────────────────────────────────────────────────────
    df.to_csv(f_saida, index=False, sep=";")

    # ── Relatório ─────────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("Dataset final: %s", f_saida)
    log.info("  Linhas       : %d", len(df))
    log.info("  Colunas      : %d", len(df.columns))
    log.info("  Piquetes     : %s", sorted(df["idpotreiro"].unique()))
    log.info("  Período      : %s → %s", df["data"].min().date(), df["data"].max().date())
    log.info("  Alvo (%s)    :", ALVO)
    log.info("    Média      : %.1f", df[ALVO].mean())
    log.info("    Desvio     : %.1f", df[ALVO].std())
    log.info("    Min / Max  : %.1f / %.1f", df[ALVO].min(), df[ALVO].max())
    log.info("  NDVI médio   : %.3f ± %.3f", df["ndvi"].mean(), df["ndvi"].std())
    log.info("=" * 55)

    # Cobertura de satélite por piquete
    for pid in sorted(df["idpotreiro"].unique()):
        sub = df[df["idpotreiro"] == pid]
        log.info("  Piquete %d: %d registros | NDVI %.3f ± %.3f",
                 pid, len(sub), sub["ndvi"].mean(), sub["ndvi"].std())


if __name__ == "__main__":
    main()
