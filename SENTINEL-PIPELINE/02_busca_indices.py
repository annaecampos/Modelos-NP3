"""
Busca imagens Sentinel-2 L2A no Copernicus Data Space Ecosystem (CDSE)
e extrai valores médios de banda por subárea para cada data de medição.

Fluxo por data de medição:
  1. API OData do CDSE: busca cenas S2 L2A na janela de ±JANELA_DIAS dias
  2. Seleciona a cena com menos nuvens e mais próxima da data
  3. Navega a árvore de arquivos do produto (OData Nodes) para achar cada banda
  4. Baixa cada JP2 via OData (sem Range — vsicurl não funciona) + média na AOI (rasterio MemoryFile)
  5. Calcula NDVI, EVI, NDRE, SAVI + mantém reflectâncias brutas

Saída:
    dados/indices_sentinel2.csv

Uso:
    python 02_busca_indices.py
"""

import json
import logging
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from config import (
    BANDAS,
    CDSE_PASSWORD,
    CDSE_S3_BASE,
    CDSE_TOKEN_URL,
    CDSE_USER,
    DATA_DIR,
    ESCALA_S2,
    JANELA_DIAS,
    MAX_CLOUD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ODATA_SEARCH   = "https://catalogue.dataspace.copernicus.eu/odata/v1"
ODATA_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"


# ── Autenticação ──────────────────────────────────────────────────────────────

def obter_token(usuario: str, senha: str) -> str:
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": usuario,
            "password": senha,
            "grant_type": "password",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def configurar_gdal_auth(token: str) -> None:
    os.environ["GDAL_HTTP_HEADERS"]                  = f"Authorization: Bearer {token}"
    os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"]  = "YES"
    os.environ["GDAL_HTTP_MULTIPLEX"]                = "YES"
    os.environ["GDAL_HTTP_VERSION"]                  = "2"
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"]       = "EMPTY_DIR"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"]   = ".jp2,.tif,.tiff"


# ── Busca OData ───────────────────────────────────────────────────────────────

def buscar_melhor_cena(
    session: requests.Session,
    bbox: list[float],
    data_medicao: pd.Timestamp,
    janela: int = JANELA_DIAS,
    max_cloud: float = MAX_CLOUD,
) -> dict | None:
    """Retorna dict com id, name, cloud_cover, datetime do melhor produto."""
    d0 = (data_medicao - timedelta(days=janela)).strftime("%Y-%m-%dT00:00:00.000Z")
    d1 = (data_medicao + timedelta(days=janela)).strftime("%Y-%m-%dT23:59:59.000Z")
    w, s, e, n = bbox
    poly = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"

    filtro = (
        f"Collection/Name eq 'SENTINEL-2'"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{poly}')"
        f" and ContentDate/Start gt {d0}"
        f" and ContentDate/Start lt {d1}"
        f" and Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover'"
        f" and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"
        f" and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType'"
        f" and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
    )

    try:
        resp = session.get(
            f"{ODATA_SEARCH}/Products",
            params={"$filter": filtro, "$orderby": "ContentDate/Start asc", "$top": 20},
            timeout=30,
        )
        resp.raise_for_status()
        produtos = resp.json().get("value", [])
    except Exception as exc:
        log.warning("Falha OData para %s: %s", data_medicao.date(), exc)
        return None

    if not produtos:
        log.warning("Sem cena L2A (≤%.0f%% nuvens) para %s", max_cloud, data_medicao.date())
        return None

    def chave(p):
        dt = pd.Timestamp(p["ContentDate"]["Start"]).tz_localize(None)
        cloud = next(
            (a["Value"] for a in p.get("Attributes", []) if a.get("Name") == "cloudCover"),
            100,
        )
        return (abs((dt - data_medicao).days), cloud)

    produtos.sort(key=chave)
    p = produtos[0]
    cloud = next(
        (a["Value"] for a in p.get("Attributes", []) if a.get("Name") == "cloudCover"),
        None,
    )
    return {"id": p["Id"], "name": p["Name"], "cloud_cover": cloud,
            "datetime": p["ContentDate"]["Start"]}


# ── Navegação de Nodes OData ──────────────────────────────────────────────────

def _diagnosticar_produto(produto_id: str, produto_name: str, token: str) -> None:
    """Inspeciona o produto diretamente para entender a estrutura disponível."""
    log.info("  [DIAG] Inspecionando produto %s", produto_id)

    # 1. Metadados do produto (verifica campo Online)
    url_meta = f"{ODATA_SEARCH}/Products({produto_id})"
    resp = requests.get(url_meta, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    log.info("  [DIAG] Metadados HTTP %d", resp.status_code)
    if resp.ok:
        meta = resp.json()
        log.info("  [DIAG] Online: %s | ContentLength: %s",
                 meta.get("Online"), meta.get("ContentLength"))

    # 2. Nodes raiz via domínio de download
    url_nodes = f"{ODATA_DOWNLOAD}/Products({produto_id})/Nodes"
    resp2 = requests.get(url_nodes, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    log.info("  [DIAG] Nodes HTTP %d | body: %s", resp2.status_code, resp2.text[:300])

    # 3. Tenta URL S3 direto para o produto
    partes = produto_name.split("_")
    sensing_dt = partes[2]
    year, month, day = sensing_dt[:4], sensing_dt[4:6], sensing_dt[6:8]
    url_s3 = (
        f"{CDSE_S3_BASE}/eodata/Sentinel-2/MSI/L2A"
        f"/{year}/{month}/{day}/{produto_name}.SAFE/GRANULE/"
    )
    resp3 = requests.get(url_s3, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    log.info("  [DIAG] S3 GRANULE HTTP %d | body: %s", resp3.status_code, resp3.text[:300])


def _get_nodes(url: str, token: str) -> list[dict]:
    """Faz GET num endpoint de Nodes e retorna a lista de itens.
    O CDSE usa 'result' como chave (não 'value' como no OData padrão).
    """
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if not resp.ok:
        log.warning("  Nodes HTTP %d: %s", resp.status_code, url[:100])
        return []
    data = resp.json()
    # CDSE usa "result" no domínio download e "value" no catálogo
    return data.get("result", data.get("value", []))


def achar_urls_bandas(
    produto_id: str, produto_name: str, bandas: list[str], token: str
) -> dict[str, str]:
    """
    Navega a árvore OData e retorna URL de download /$value para cada banda.

    O CDSE usa ``Nodes(NOME)/Nodes`` (parênteses, sem aspas no nome). O formato
    ``Nodes('NOME')`` provoca HTTP 403 nas listagens aninhadas.
    """
    base = f"{ODATA_DOWNLOAD}/Products({produto_id})"
    urls: dict[str, str] = {}

    def _nodes_path(*segmentos: str) -> str:
        """Monta ``.../Nodes(seg1)/Nodes(seg2)/...`` conforme documentação CDSE."""
        url = f"{base}/Nodes"
        for seg in segmentos:
            url = f"{url}({seg})/Nodes"
        return url

    # Nível 1: lista nodes raiz do produto
    nodes_raiz = _get_nodes(f"{base}/Nodes", token)
    if not nodes_raiz:
        log.warning("  Nodes raiz vazios para %s", produto_name)
        return urls

    # Encontra o nó .SAFE (pode ter nome ligeiramente diferente)
    safe_node = next(
        (n for n in nodes_raiz if n["Id"].endswith(".SAFE")),
        nodes_raiz[0] if nodes_raiz else None,
    )
    if not safe_node:
        log.warning("  Nó .SAFE não encontrado em %s", produto_name)
        return urls
    safe_id = safe_node["Id"]

    # Nível 2: dentro do SAFE, busca pasta GRANULE
    nodes_safe = _get_nodes(_nodes_path(safe_id), token)
    granule_node = next((n for n in nodes_safe if n["Id"] == "GRANULE"), None)
    if not granule_node:
        log.warning("  Pasta GRANULE não encontrada em %s | nodes: %s",
                    safe_id, [n["Id"] for n in nodes_safe])
        return urls

    # Nível 3: lista granules (geralmente só 1)
    nodes_granule = _get_nodes(_nodes_path(safe_id, "GRANULE"), token)
    if not nodes_granule:
        log.warning("  GRANULE vazio para %s", produto_name)
        return urls
    granule_id = nodes_granule[0]["Id"]
    log.debug("  Granule: %s", granule_id)

    # Nível 4: lista IMG_DATA
    nodes_img = _get_nodes(
        _nodes_path(safe_id, "GRANULE", granule_id, "IMG_DATA"),
        token,
    )
    resolucoes_disponiveis = [n["Id"] for n in nodes_img]
    log.debug("  Resoluções: %s", resolucoes_disponiveis)

    for banda in bandas:
        res_preferida = "R10m" if banda in ("B02", "B04", "B08") else "R20m"
        ordem = [res_preferida] + [r for r in resolucoes_disponiveis if r != res_preferida]

        for res in ordem:
            if res not in resolucoes_disponiveis:
                continue
            node_res = _get_nodes(
                _nodes_path(safe_id, "GRANULE", granule_id, "IMG_DATA", res),
                token,
            )
            for arq in node_res:
                nome = arq["Id"]
                if f"_{banda}_" in nome and nome.endswith(".jp2"):
                    urls[banda] = (
                        f"{base}/Nodes({safe_id})/Nodes(GRANULE)"
                        f"/Nodes({granule_id})/Nodes(IMG_DATA)"
                        f"/Nodes({res})/Nodes({nome})/$value"
                    )
                    break
            if banda in urls:
                break

    log.debug("  Bandas encontradas: %s", list(urls.keys()))
    return urls


# ── Extração de pixels ────────────────────────────────────────────────────────

def _bytes_jp2_url(
    url: str, session: requests.Session, cache: dict[str, bytes]
) -> bytes | None:
    """GET completo do JP2 (endpoint OData $value não suporta HTTP Range para /vsicurl/)."""
    if url in cache:
        return cache[url]
    try:
        r = session.get(url, timeout=600)
        r.raise_for_status()
    except Exception as exc:
        log.warning("  Download JP2 falhou (%s…): %s", url[:70], exc)
        return None
    cache[url] = r.content
    return r.content


def extrair_media_banda(
    url: str,
    geojson_wgs84: dict,
    session: requests.Session,
    jp2_cache: dict[str, bytes],
) -> float | None:
    """Abre o JP2 em memória e retorna a reflectância média na AOI (WGS84)."""
    try:
        blob = _bytes_jp2_url(url, session, jp2_cache)
        if blob is None:
            return None
        geom = shape(geojson_wgs84)
        with MemoryFile(blob) as mem:
            with mem.open() as src:
                geom_img = shape(
                    transform_geom("EPSG:4326", src.crs.to_string(), mapping(geom))
                )
                out, _ = rio_mask(src, [mapping(geom_img)], crop=True, nodata=0)
                pixels = out[0]
                validos = pixels[(pixels > 0) & (pixels < 65535)]
                if len(validos) == 0:
                    return None
                return float(validos.mean()) / ESCALA_S2
    except Exception as exc:
        log.debug("Erro ao ler banda (%s...): %s", url[:60], exc)
        return None


# ── Índices de vegetação ──────────────────────────────────────────────────────

def calcular_indices(b: dict) -> dict:
    b2, b4, b5, b8, b11, b12 = (
        b.get("B02"), b.get("B04"), b.get("B05"),
        b.get("B08"), b.get("B11"), b.get("B12"),
    )

    def _safe(num, den):
        return (num / den) if den and den != 0 else None

    idx = {}
    if b4 is not None and b8 is not None:
        idx["ndvi"] = _safe(b8 - b4, b8 + b4)
    if b2 is not None and b4 is not None and b8 is not None:
        den = b8 + 6 * b4 - 7.5 * b2 + 1
        idx["evi"] = _safe(2.5 * (b8 - b4), den)
    if b5 is not None and b8 is not None:
        idx["ndre"] = _safe(b8 - b5, b8 + b5)
    if b4 is not None and b8 is not None:
        idx["savi"] = _safe(1.5 * (b8 - b4), b8 + b4 + 0.5)
    return idx


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    entrada = DATA_DIR / "dados_campo.csv"
    saida   = DATA_DIR / "indices_sentinel2.csv"

    if not entrada.exists():
        log.error("Execute primeiro: python 01_extrai_banco.py")
        raise SystemExit(1)
    if not CDSE_USER or not CDSE_PASSWORD:
        log.error("Preencha CDSE_USER e CDSE_PASSWORD em config.py")
        raise SystemExit(1)

    df = pd.read_csv(entrada, sep=";", parse_dates=["data"])
    df = df.dropna(subset=["geometria"])

    geometrias = [shape(json.loads(g)) for g in df["geometria"]]
    xs = [c for g in geometrias for c in [g.bounds[0], g.bounds[2]]]
    ys = [c for g in geometrias for c in [g.bounds[1], g.bounds[3]]]
    bbox = [min(xs) - 0.01, min(ys) - 0.01, max(xs) + 0.01, max(ys) + 0.01]
    log.info("Bbox: %.4f, %.4f, %.4f, %.4f", *bbox)

    log.info("Autenticando no CDSE...")
    token = obter_token(CDSE_USER, CDSE_PASSWORD)
    configurar_gdal_auth(token)
    token_ts = time.time()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    datas = sorted(df["data"].unique())
    resultados = []

    for i, data_raw in enumerate(datas):
        data = pd.Timestamp(data_raw)
        subareas = df[df["data"] == data_raw]
        log.info("[%d/%d] %s | %d subáreas", i + 1, len(datas), data.date(), len(subareas))

        # Renova token antes de expirar
        if time.time() - token_ts > 550:
            log.info("Renovando token...")
            token = obter_token(CDSE_USER, CDSE_PASSWORD)
            configurar_gdal_auth(token)
            session.headers.update({"Authorization": f"Bearer {token}"})
            token_ts = time.time()

        produto = buscar_melhor_cena(session, bbox, data)
        if produto is None:
            continue

        data_img = pd.Timestamp(produto["datetime"]).strftime("%Y-%m-%d")
        log.info("  → %s | data: %s | nuvens: %.1f%%",
                 produto["name"], data_img, produto["cloud_cover"] or 0)

        # Diagnóstico completo apenas na primeira cena encontrada
        if not resultados:
            _diagnosticar_produto(produto["id"], produto["name"], token)

        # Busca URLs das bandas (uma vez por produto, compartilhado entre subáreas)
        urls_bandas = achar_urls_bandas(produto["id"], produto["name"], BANDAS, token)
        if not urls_bandas:
            log.warning("  Não foi possível obter URLs das bandas para %s", produto["name"])
            continue
        if urls_bandas:
            log.info("  Bandas encontradas: %s", list(urls_bandas.keys()))
        else:
            log.warning("  Nenhuma banda encontrada — pulando data %s", data.date())

        jp2_por_url: dict[str, bytes] = {}
        for _, row in subareas.iterrows():
            geom_json = json.loads(row["geometria"])
            bandas_vals: dict[str, float | None] = {}

            for banda, url in urls_bandas.items():
                bandas_vals[banda] = extrair_media_banda(
                    url, geom_json, session, jp2_por_url
                )

            indices = calcular_indices(bandas_vals)
            resultados.append({
                "data":        data_raw,
                "idpotreiro":  row["idpotreiro"],
                "idsubarea":   row["idsubarea"],
                "data_imagem": data_img,
                "cloud_cover": produto["cloud_cover"],
                **{f"s2_{k.lower()}": v for k, v in bandas_vals.items()},
                **indices,
            })

    if not resultados:
        log.error("Nenhum resultado gerado.")
        raise SystemExit(1)

    df_out = pd.DataFrame(resultados)
    df_out.to_csv(saida, index=False, sep=";")

    total = len(df_out)
    com_ndvi = int(df_out["ndvi"].notna().sum()) if "ndvi" in df_out.columns else 0
    log.info("Salvo: %s", saida)
    log.info(
        "Linhas: %d | com NDVI: %d (%.0f%%)",
        total,
        com_ndvi,
        100 * com_ndvi / total if total else 0,
    )


if __name__ == "__main__":
    main()
