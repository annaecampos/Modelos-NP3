"""Monta dataset limpo + filtro de alinhamento campo↔satélite."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def estacao_sul(mes: int) -> str:
    if mes in (12, 1, 2):
        return "verao"
    if mes in (3, 4, 5):
        return "outono"
    if mes in (6, 7, 8):
        return "inverno"
    return "primavera"


def montar(cfg) -> Path:
    entrada = cfg.PIPELINE_DADOS
    saida_dir = Path(cfg.TREINO_DIR) / "dados"
    saida_dir.mkdir(parents=True, exist_ok=True)
    saida = saida_dir / cfg.DATASET_ARQUIVO

    campo = pd.read_csv(entrada / "dados_campo.csv", sep=";", parse_dates=["data"])
    idx = pd.read_csv(entrada / "indices_sentinel2.csv", sep=";", parse_dates=["data"])

    df = campo.merge(
        idx.drop(columns=["cloud_cover"], errors="ignore"),
        on=["data", "idpotreiro", "idsubarea"],
        how="inner",
    )
    df = df.dropna(subset=["ndvi", cfg.ALVO])
    df["data_imagem"] = pd.to_datetime(df["data_imagem"])
    df["dias_campo_imagem"] = (df["data_imagem"] - df["data"]).dt.days.abs().astype(int)

    antes = len(df)
    df = df.sort_values("dias_campo_imagem").drop_duplicates(cfg.DEDUP_CHAVES, keep="first")
    log.info("Dedup: %d → %d", antes, len(df))

    df = df[df[cfg.ALVO] <= cfg.MSTOTAL_MAX_VALIDO]
    log.info("Sem outlier > %d: %d linhas", cfg.MSTOTAL_MAX_VALIDO, len(df))

    df = df[df["dias_campo_imagem"] <= cfg.MAX_DIAS_CAMPO_IMAGEM]
    log.info("Alinhamento ≤ %d dias: %d linhas", cfg.MAX_DIAS_CAMPO_IMAGEM, len(df))

    df["mes"] = df["data"].dt.month
    df["ano"] = df["data"].dt.year
    df["estacao"] = df["mes"].map(estacao_sul)
    df = df.sort_values(["idpotreiro", "idsubarea", "data"])
    df["dias_desde_ult_medicao"] = (
        df.groupby(["idpotreiro", "idsubarea"])["data"].diff().dt.days.fillna(0).astype(int)
    )
    df = pd.get_dummies(df, columns=["estacao"], prefix="est", drop_first=False)
    df["dentrofora"] = (df["dentrofora"] == "D").astype(int)

    df.to_csv(saida, index=False, sep=";")
    log.info("Salvo: %s (%d linhas, mediana %d dias campo↔imagem)",
             saida, len(df), int(df["dias_campo_imagem"].median()))
    return saida
