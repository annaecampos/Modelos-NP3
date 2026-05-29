"""
Busca imagens Sentinel-2 L2A no Copernicus Data Space Ecosystem (CDSE),
baixa o produto .zip inteiro, extrai localmente e calcula médias de bandas
por subárea usando os polígonos do banco.

Saída:
    dados/indices_sentinel2.csv

Uso:
    python3 02_busca_indices.py
"""

import json
import logging
import zipfile
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from config import (
    BANDAS,
    CDSE_PASSWORD,
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

ODATA_SEARCH = "https://catalogue.dataspace.copernicus.eu/odata/v1"
ODATA_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"

PRODUTOS_DIR = DATA_DIR / "produtos_sentinel"
PRODUTOS_DIR.mkdir(parents=True, exist_ok=True)

# Para testar sem baixar muitos GB, processa só 5 datas com resultado.
# Depois que validar tudo, mude para None para processar todas.
MAX_DATAS_COM_RESULTADO = None

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


def buscar_melhor_cena(
    session: requests.Session,
    bbox: list[float],
    data_medicao: pd.Timestamp,
    janela: int = JANELA_DIAS,
    max_cloud: float = MAX_CLOUD,
) -> dict | None:
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
            params={
                "$filter": filtro,
                "$orderby": "ContentDate/Start asc",
                "$top": 20,
            },
            timeout=30,
        )

        resp.raise_for_status()
        produtos = resp.json().get("value", [])

    except Exception as exc:
        log.warning("Falha OData para %s: %s", data_medicao.date(), exc)
        return None

    if not produtos:
        log.warning(
            "Sem cena L2A (≤%.0f%% nuvens) para %s",
            max_cloud,
            data_medicao.date(),
        )
        return None

    def chave(p):
        dt = pd.Timestamp(p["ContentDate"]["Start"]).tz_localize(None)

        cloud = next(
            (
                a["Value"]
                for a in p.get("Attributes", [])
                if a.get("Name") == "cloudCover"
            ),
            100,
        )

        return (abs((dt - data_medicao).days), cloud)

    produtos.sort(key=chave)

    p = produtos[0]

    cloud = next(
        (
            a["Value"]
            for a in p.get("Attributes", [])
            if a.get("Name") == "cloudCover"
        ),
        None,
    )

    return {
        "id": p["Id"],
        "name": p["Name"],
        "cloud_cover": cloud,
        "datetime": p["ContentDate"]["Start"],
    }


def baixar_produto(produto_id: str, produto_name: str, token: str) -> Path | None:
    """
    Baixa o produto inteiro via /$value e extrai localmente.
    Retorna o caminho da pasta .SAFE extraída.
    """

    safe_name = produto_name

    if not safe_name.endswith(".SAFE"):
        safe_name = f"{safe_name}.SAFE"

    safe_dir = PRODUTOS_DIR / safe_name
    zip_path = PRODUTOS_DIR / f"{safe_name}.zip"

    if safe_dir.exists():
        jp2s = list(safe_dir.rglob("*.jp2"))

        if jp2s:
            log.info("  Produto já extraído: %s", safe_dir)
            return safe_dir

    if not zip_path.exists():
        url = f"{ODATA_DOWNLOAD}/Products({produto_id})/$value"

        log.info("  Baixando produto completo: %s", safe_name)
        log.info("  Arquivo local: %s", zip_path)

        headers = {"Authorization": f"Bearer {token}"}

        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    log.warning("  Falha no download: HTTP %d", resp.status_code)
                    log.warning("  Resposta: %s", resp.text[:500])
                    return None

                total = int(resp.headers.get("Content-Length", 0))
                baixado = 0
                ultimo_pct = -1

                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            baixado += len(chunk)

                            if total > 0:
                                pct = int(100 * baixado / total)

                                if pct >= ultimo_pct + 10:
                                    ultimo_pct = pct
                                    log.info("  Download %d%%", pct)

        except Exception as exc:
            log.warning("  Erro baixando produto: %s", exc)
            return None

    else:
        log.info("  ZIP já existe: %s", zip_path)

    try:
        if not zipfile.is_zipfile(zip_path):
            log.warning("  Arquivo baixado não é ZIP válido: %s", zip_path)
            return None

        log.info("  Extraindo ZIP...")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(PRODUTOS_DIR)

    except Exception as exc:
        log.warning("  Erro extraindo ZIP: %s", exc)
        return None

    if safe_dir.exists():
        return safe_dir

    possiveis = list(PRODUTOS_DIR.glob("*.SAFE"))

    for p in possiveis:
        if p.name == safe_name:
            return p

    log.warning("  Pasta SAFE não encontrada após extração: %s", safe_name)
    return None


def encontrar_bandas_locais(safe_dir: Path, bandas: list[str]) -> dict[str, Path]:
    """
    Procura arquivos .jp2 das bandas dentro do produto Sentinel-2 extraído.
    """

    bandas_encontradas: dict[str, Path] = {}

    todos_jp2 = list(safe_dir.rglob("*.jp2"))

    if not todos_jp2:
        log.warning("  Nenhum .jp2 encontrado em %s", safe_dir)
        return bandas_encontradas

    for banda in bandas:
        res_preferida = "R10m" if banda in ("B02", "B04", "B08") else "R20m"

        candidatos = [
            p
            for p in todos_jp2
            if f"_{banda}_" in p.name and res_preferida in str(p)
        ]

        if not candidatos:
            candidatos = [
                p
                for p in todos_jp2
                if f"_{banda}_" in p.name
            ]

        if candidatos:
            bandas_encontradas[banda] = candidatos[0]

    log.info("  Bandas locais encontradas: %s", list(bandas_encontradas.keys()))

    return bandas_encontradas


def extrair_media_banda(caminho_banda: Path, geojson_wgs84: dict) -> float | None:
    """
    Lê a banda local, recorta pelo polígono da subárea e retorna média da reflectância.
    """

    try:
        geom = shape(geojson_wgs84)

        with rasterio.open(caminho_banda) as src:
            geom_img = shape(
                transform_geom(
                    "EPSG:4326",
                    src.crs.to_string(),
                    mapping(geom),
                )
            )

            out, _ = rio_mask(
                src,
                [mapping(geom_img)],
                crop=True,
                nodata=0,
            )

            pixels = out[0]
            validos = pixels[(pixels > 0) & (pixels < 65535)]

            if len(validos) == 0:
                return None

            return float(validos.mean()) / ESCALA_S2

    except Exception as exc:
        log.warning("    Erro ao processar banda %s: %s", caminho_banda.name, exc)
        return None


def calcular_indices(b: dict) -> dict:
    b2 = b.get("B02")
    b4 = b.get("B04")
    b5 = b.get("B05")
    b8 = b.get("B08")
    b11 = b.get("B11")
    b12 = b.get("B12")

    def _safe(num, den):
        if den is None or den == 0:
            return None

        return num / den

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

    if b8 is not None and b12 is not None:
        idx["nbr"] = _safe(b8 - b12, b8 + b12)

    if b8 is not None and b11 is not None:
        idx["ndmi"] = _safe(b8 - b11, b8 + b11)

    return idx


def main() -> None:
    entrada = DATA_DIR / "dados_campo.csv"
    saida = DATA_DIR / "indices_sentinel2.csv"

    if not entrada.exists():
        log.error("Execute primeiro: python3 01_extrai_banco.py")
        raise SystemExit(1)

    if not CDSE_USER or not CDSE_PASSWORD:
        log.error("Preencha CDSE_USER e CDSE_PASSWORD em config.py")
        raise SystemExit(1)

    df = pd.read_csv(entrada, sep=";", parse_dates=["data"])
    df = df.dropna(subset=["geometria"])

    geometrias = [shape(json.loads(g)) for g in df["geometria"]]

    xs = [c for g in geometrias for c in [g.bounds[0], g.bounds[2]]]
    ys = [c for g in geometrias for c in [g.bounds[1], g.bounds[3]]]

    bbox = [
        min(xs) - 0.01,
        min(ys) - 0.01,
        max(xs) + 0.01,
        max(ys) + 0.01,
    ]

    log.info("Bbox: %.4f, %.4f, %.4f, %.4f", *bbox)

    log.info("Autenticando no CDSE...")
    token = obter_token(CDSE_USER, CDSE_PASSWORD)
    token_ts = time.time()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    datas = sorted(df["data"].unique())

    resultados = []
    datas_com_resultado = 0

    log.info(
        "Processando datas. Limite com resultado: %s",
        MAX_DATAS_COM_RESULTADO if MAX_DATAS_COM_RESULTADO is not None else "todas",
    )

    for i, data_raw in enumerate(datas):
        if MAX_DATAS_COM_RESULTADO is not None and datas_com_resultado >= MAX_DATAS_COM_RESULTADO:
            log.info("Limite de %d datas com resultado atingido.", MAX_DATAS_COM_RESULTADO)
            break

        data = pd.Timestamp(data_raw)
        subareas = df[df["data"] == data_raw]

        log.info(
            "[%d/%d] %s | %d subáreas",
            i + 1,
            len(datas),
            data.date(),
            len(subareas),
        )

        if time.time() - token_ts > 550:
            log.info("Renovando token CDSE...")
            token = obter_token(CDSE_USER, CDSE_PASSWORD)
            session.headers.update({"Authorization": f"Bearer {token}"})
            token_ts = time.time()

        produto = buscar_melhor_cena(session, bbox, data)

        if produto is None:
            continue

        data_img = pd.Timestamp(produto["datetime"]).strftime("%Y-%m-%d")

        log.info(
            "  → %s | data: %s | nuvens: %.1f%%",
            produto["name"],
            data_img,
            produto["cloud_cover"] or 0,
        )

        safe_dir = baixar_produto(produto["id"], produto["name"], token)

        if safe_dir is None:
            log.warning("  Produto não pôde ser baixado/extraído.")
            continue

        bandas_locais = encontrar_bandas_locais(safe_dir, BANDAS)

        if not bandas_locais:
            log.warning("  Nenhuma banda local encontrada.")
            continue

        resultados_antes = len(resultados)

        for _, row in subareas.iterrows():
            geom_json = json.loads(row["geometria"])
            bandas_vals: dict[str, float | None] = {}

            for banda, caminho in bandas_locais.items():
                bandas_vals[banda] = extrair_media_banda(caminho, geom_json)

            indices = calcular_indices(bandas_vals)

            resultados.append(
                {
                    "data": data_raw,
                    "idpotreiro": row["idpotreiro"],
                    "idsubarea": row["idsubarea"],
                    "data_imagem": data_img,
                    "cloud_cover": produto["cloud_cover"],
                    **{f"s2_{k.lower()}": v for k, v in bandas_vals.items()},
                    **indices,
                }
            )

        resultados_data = len(resultados) - resultados_antes

        if resultados_data > 0:
            datas_com_resultado += 1
            log.info(
                "  Data processada com sucesso: %d linhas geradas.",
                resultados_data,
            )

    if not resultados:
        log.error("Nenhum resultado gerado.")
        raise SystemExit(1)

    df_out = pd.DataFrame(resultados)
    df_out.to_csv(saida, index=False, sep=";")

    total = len(df_out)
    com_ndvi = df_out["ndvi"].notna().sum() if "ndvi" in df_out.columns else 0

    log.info("Salvo: %s", saida)
    log.info(
        "Linhas: %d | com NDVI: %d (%.0f%%)",
        total,
        com_ndvi,
        100 * com_ndvi / total,
    )


if __name__ == "__main__":
    main()