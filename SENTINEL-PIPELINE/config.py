"""
Configurações globais do pipeline Sentinel-2 / TouceiraTech.
"""

from pathlib import Path

# ── Diretórios ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dados"
DATA_DIR.mkdir(exist_ok=True)

# ── Banco TouceiraTech ────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "db_2019",
    "user": "postgres",
    "password": "12345",
}

# ── Copernicus Data Space Ecosystem (CDSE) ────────────────────────────────────
# Crie sua conta gratuita em: https://dataspace.copernicus.eu
CDSE_USER = "annaecampos@gmail.com"      # seu e-mail cadastrado no CDSE
CDSE_PASSWORD = "ModeloNP3@2026"  # sua senha CDSE

# ── Parâmetros de busca de imagens ────────────────────────────────────────────
# Janela de busca: busca imagens em ±JANELA_DIAS em torno da data da medição
JANELA_DIAS = 15

# Descarta imagens com cobertura de nuvens acima desse valor (%)
MAX_CLOUD = 20

# ── Alvo do modelo ────────────────────────────────────────────────────────────
# Opções: "mstotal" | "msanoni" | "msoutras" | "media"
ALVO = "mstotal"

# ── Bandas Sentinel-2 L2A a extrair ──────────────────────────────────────────
# Resolução nativa: B02/B04/B08 = 10 m  |  B05/B06/B07/B11/B12 = 20 m
BANDAS = ["B02", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]

# Fator de escala S2 L2A (reflectância = DN / 10000)
ESCALA_S2 = 10_000.0

# ── Endpoints CDSE ────────────────────────────────────────────────────────────
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
CDSE_S3_BASE  = "https://eodata.dataspace.copernicus.eu"
