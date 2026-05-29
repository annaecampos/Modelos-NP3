# SENTINEL-PIPELINE

Pipeline para estimativa de forragem combinando dados de campo do **TouceiraTech**
com imagens **Sentinel-2** do **Copernicus Data Space Ecosystem (CDSE)**.

## Ideia central

```
TouceiraTech (verdade de campo)    Sentinel-2 (Copernicus)    clima_evapo
       ↓                                    ↓                      ↓
mstotal, msanoni, media          NDVI, EVI, NDRE, SAVI       tmin, chuva...
          bandas brutas: B02, B04, B05, B06, B07, B08, B11, B12
                              ↘           ↓           ↙
                                 dataset → modelo ML
```

O TouceiraTech fornece a **verdade de campo**: medições reais de massa seca
georreferenciadas por subárea. O Sentinel-2 fornece o sinal espectral dessas
mesmas subáreas nas mesmas datas. O modelo aprende a relação entre os dois.

## Pré-requisitos

### 1. Conta gratuita no CDSE

Registre-se em: <https://dataspace.copernicus.eu>

Após registrar, preencha `config.py`:
```python
CDSE_USER     = "seu_email@exemplo.com"
CDSE_PASSWORD = "sua_senha"
```

### 2. Banco TouceiraTech

O banco `db_2019` deve estar rodando em `localhost:5432`.
Restaure o backup se necessário:

```bash
createdb db_2019
pg_restore -d db_2019 "../RESTORE DO BANCO/db_2020.backup"
```

### 3. Dependências Python

A partir da raiz do projeto (com o `.venv` ativado):

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Dependências novas do pipeline:

| Pacote          | Finalidade                                              |
|-----------------|---------------------------------------------------------|
| `pystac-client` | Busca no catálogo STAC do CDSE                          |
| `rasterio`      | Leitura de bandas via `/vsicurl/` (sem download total)  |
| `shapely`       | Manipulação de geometrias vetoriais                     |
| `pyproj`        | Reprojeção de coordenadas (WGS84 ↔ UTM)                 |
| `requests`      | Autenticação OAuth2 no CDSE                             |
| `joblib`        | Serialização do modelo treinado                         |

## Execução

Rode os scripts em ordem:

```bash
cd SENTINEL-PIPELINE

# 1. Extrai dados do banco com geometrias das subáreas
python 01_extrai_banco.py

# 2. Busca imagens Sentinel-2 no CDSE e calcula índices (pode demorar ~1 min/data)
python 02_busca_indices.py

# 3. Monta a tabela final de treino
python 03_monta_dataset.py

# 4. Treina Random Forest e gera métricas e gráficos
python 04_modelo_rf.py
```

## Saídas

```
SENTINEL-PIPELINE/
└── dados/
    ├── dados_campo.csv       ← dados TouceiraTech por (data, potreiro, subárea)
    ├── indices_sentinel2.csv ← bandas S2 + NDVI/EVI/NDRE/SAVI por (data, subárea)
    └── dataset_final.csv     ← tabela de treino completa
└── resultados_rf/
    └── YYYYMMDD_HHMMSS/
        ├── metricas.txt
        ├── importancia_features.png
        ├── real_vs_predito.png
        ├── residuos.png
        └── modelo_rf.joblib
```

## Bandas extraídas do Sentinel-2 L2A

| Band  | Comprimento de onda | Resolução | Uso                         |
|-------|---------------------|-----------|-----------------------------|
| B02   | 490 nm (Azul)       | 10 m      | EVI (correção de solo)      |
| B04   | 665 nm (Vermelho)   | 10 m      | NDVI, EVI, SAVI             |
| B05   | 705 nm (Red Edge 1) | 20 m      | NDRE — melhor proxy de biomassa |
| B06   | 740 nm (Red Edge 2) | 20 m      | Sensível à clorofila        |
| B07   | 783 nm (Red Edge 3) | 20 m      | Sensível à clorofila        |
| B08   | 842 nm (NIR)        | 10 m      | Base de todos os índices    |
| B11   | 1610 nm (SWIR1)     | 20 m      | Conteúdo de água nas folhas |
| B12   | 2190 nm (SWIR2)     | 20 m      | Matéria seca — 3ª feature mais importante (Fernandes 2024) |

## Por que Copernicus CDSE e não GEE?

| Critério             | CDSE (Copernicus)          | Google Earth Engine        |
|----------------------|----------------------------|----------------------------|
| Conta gratuita       | Sim, sem aprovação          | Sim, requer aprovação      |
| Acervo S2            | Completo desde 2015         | Completo desde 2015        |
| Acesso programático  | API REST / STAC / S3        | Python SDK (ee)            |
| Leitura sem download | Sim, via `/vsicurl/` COG    | Sim, processamento na nuvem|
| Dependência externa  | Só `pystac-client`+`rasterio` | SDK proprietário Google  |
| Dados na Europa      | Servidores EU               | Servidores Google          |

## Índices calculados

| Índice | Fórmula                                              | Quando é melhor          |
|--------|------------------------------------------------------|--------------------------|
| NDVI   | (B08 − B04) / (B08 + B04)                           | Pasto em crescimento médio|
| EVI    | 2.5 × (B08 − B04) / (B08 + 6·B04 − 7.5·B02 + 1)   | Pasto denso (anti-saturação)|
| NDRE   | (B08 − B05) / (B08 + B05)                           | Estimativa de massa seca |
| SAVI   | 1.5 × (B08 − B04) / (B08 + B04 + 0.5)              | Pós-pastejo (solo exposto)|

## Limitação temporal

O Sentinel-2A foi lançado em junho de 2015. Medições de campo anteriores
a `2015-07-01` **não serão incluídas** no pipeline (filtro aplicado em
`01_extrai_banco.py`).

## Próximos passos (evolução do modelo)

1. **XGBoost** — mesmo dataset, modelo mais poderoso
2. **LSTM/GRU** — incorpora a série temporal por piquete (janela deslizante)
3. **CNN** — usa recortes de imagem (patches) em vez de médias por AOI
4. **Integração com TouceiraTech** — inferência em tempo real para novas datas

## Referências

- Fernandes et al. (2024). *Forage mass estimation in tropical pastures with Sentinel-2 and ML*.
  Scientific Reports. <https://pmc.ncbi.nlm.nih.gov/articles/PMC11018762/>
- Peng et al. (2023). *Forage Biomass Estimation at High Latitudes with Sentinel-2*.
  Remote Sensing. <https://www.mdpi.com/2072-4292/15/9/2350>
- CDSE STAC API: <https://catalogue.dataspace.copernicus.eu/stac>
