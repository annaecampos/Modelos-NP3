"""
Extrai dados de campo do banco TouceiraTech em nível de subárea.

Diferença em relação aos scripts existentes em MANIPULAÇÃO DE DADOS/:
  - Mantém granularidade de subárea (não agrega por potreiro)
  - Inclui geometria de cada subárea em formato GeoJSON (WGS84)
  - Filtra datas a partir de julho/2015 (início do Sentinel-2)
  - Considera os 4 piquetes experimentais

Saída:
    dados/dados_campo.csv — uma linha por (data, potreiro, subárea)

Uso:
    python 01_extrai_banco.py
"""

import csv
import logging
from pathlib import Path

import psycopg2
import pandas as pd

from config import DB_CONFIG, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Geometrias precisam estar em WGS84 para o STAC / rasterio.
# ST_Transform converte do CRS nativo do banco para EPSG:4326.
QUERY = """
SELECT
    m.data,
    p.id                                         AS idpotreiro,
    s.id                                         AS idsubarea,
    pa.dentrofora,
    pa.media,
    pa.mstotal,
    pa.msanoni,
    pa.msoutras,
    c.tmin,
    c.tmed,
    c.tmax,
    c.umidade,
    c.velocidadevento,
    c.radiacaosolar,
    c.chuva,
    c.somatermica,
    c.def,
    c.exc,
    ST_AsGeoJSON(ST_Transform(s.subpoligono, 4326)) AS geometria
FROM pastagem pa
JOIN subarea  s ON s.id      = pa.idsubarea
JOIN potreiro p ON p.id      = s.idpotreiro
JOIN medicao  m ON m.id      = pa.idmedicao
LEFT JOIN clima_evapo c ON c.data = m.data
WHERE m.data      >= '2015-07-01'
  AND pa.mstotal  IS NOT NULL
  AND pa.mstotal  >  0
  AND pa.msanoni  IS NOT NULL
  AND pa.msoutras IS NOT NULL
ORDER BY m.data, p.id, s.id;
"""

COLUNAS = [
    "data", "idpotreiro", "idsubarea", "dentrofora",
    "media", "mstotal", "msanoni", "msoutras",
    "tmin", "tmed", "tmax",
    "umidade", "velocidadevento", "radiacaosolar",
    "chuva", "somatermica", "def", "exc",
    "geometria",
]


def main() -> None:
    saida = DATA_DIR / "dados_campo.csv"

    log.info("Conectando ao banco %s...", DB_CONFIG["dbname"])
    try:
        con = psycopg2.connect(**DB_CONFIG)
        cur = con.cursor()
    except Exception as exc:
        log.error("Falha na conexão: %s", exc)
        raise SystemExit(1) from exc

    log.info("Executando query...")
    cur.execute(QUERY)
    linhas = cur.fetchall()
    con.close()

    if not linhas:
        log.error("Nenhum registro retornado. Verifique o banco e as datas.")
        raise SystemExit(1)

    df = pd.DataFrame(linhas, columns=COLUNAS)
    df["data"] = pd.to_datetime(df["data"]).dt.date

    # Relatório rápido
    log.info("Registros extraídos : %d", len(df))
    log.info("Datas únicas        : %d", df["data"].nunique())
    log.info("Piquetes            : %s", sorted(df["idpotreiro"].unique()))
    log.info("Subáreas            : %d", df["idsubarea"].nunique())
    log.info("Período             : %s → %s", df["data"].min(), df["data"].max())

    sem_geometria = df["geometria"].isna().sum()
    if sem_geometria:
        log.warning("%d registros sem geometria — serão ignorados no passo 02.", sem_geometria)

    df.to_csv(saida, index=False, sep=";")
    log.info("CSV salvo em: %s", saida)


if __name__ == "__main__":
    main()
