# Dados disponíveis no banco de dados
_Exported on 12/05/2026 at 22:20:07 BRT from Cursor (2.6.20)_

---

**User**

olhando para o projeto quais dados exsitem no banco?

---

**Cursor**

Aqui está um resumo do que existe no banco de dados do projeto:

---

## Banco de Dados

**PostgreSQL 12** com extensão **PostGIS 3**, banco chamado `db_2019` (rodando em `localhost:5432`).

---

## Tabelas identificadas (pelos arquivos SQL do projeto)

| Tabela | Colunas / Função |
|---|---|
| **`potreiro`** | `id`, `poligono` (geometria PostGIS) — representa os piquetes/potros |
| **`subarea`** | `id`, `idpotreiro`, `subpoligono` (geometria) — subdivisões dos piquetes |
| **`medicao`** | `id`, `data` — datas das campanhas de medição |
| **`pastagem`** | `idsubarea`, `idmedicao`, `dentrofora` (`'D'`/`'F'`), `media`, `mstotal`, `msanoni`, `msoutras` — dados de altura e massa seca da pastagem |
| **`clima_evapo`** | `data`, `tmin`, `tmed`, `tmax`, `umidade`, `velocidadevento`, `radiacaosolar`, `chuva`, `somatermica`, `def`, `exc` — variáveis climáticas e de evapotranspiração |

---

## Relacionamentos

```
potreiro (1) ──< subarea (N)
subarea  (1) ──< pastagem (N)
medicao  (1) ──< pastagem (N)
clima_evapo ── alinhada por datas de medicao
```

---

## Estrutura conceitual dos dados

1. **Unidades espaciais**: piquetes (`potreiro`) divididos em subareas com geometrias PostGIS
2. **Medições de pastagem**: altura, massa seca total, fração forrageira (`msanoni`) vs outras (`msoutras`), com flag de dentro/fora de limiar
3. **Dados climáticos**: séries diárias de temperatura, umidade, vento, radiação, chuva, saldo térmico, déficit e excedente hídrico
4. **Piquetes experimentais**: os IDs 1 a 4 correspondem a parcelas experimentais distintas (p20/p21, infestado/mirapasto)

---

## Observacoes importantes

- **Sem migrations ou DDL no repositório** — o schema só existe no dump externo do banco (mencionado no `README.md` mas não incluso)
- Os scripts Python (em `MANIPULAÇÃO DE DADOS/`) consultam o banco e geram CSVs de treinamento (`e1_Treino.csv`, `teste.csv`) para os modelos LSTM/GRU
- O schema completo com tipos e constraints só pode ser visto no dump ou numa instância ativa do `db_2019`

---

**User**

existem imagens?

---

**Cursor**

Não, o projeto não contém nenhuma imagem. Nenhum arquivo `.png`, `.jpg`, `.svg`, `.gif` ou qualquer outro formato de imagem foi encontrado no repositório.

---

**User**

nem acessa base com imagens?

---

**Cursor**

Não. O projeto não acessa nenhuma base com imagens. As únicas referências encontradas são um import comentado (`#from folium.plugins import FloatImage`) em todos os arquivos de modelo — ou seja, está desativado e nunca é usado.

O projeto trabalha exclusivamente com **dados tabulares**: medições de campo (altura, massa seca) e variáveis climáticas, sem nenhuma fonte de imagens satelitais, rasters, ou similares.

---

**User**

Ideia central

Usar o TouceiraTech como base real de campo e o Sentinel-2 como fonte de imagem.

Ou seja:

subarea/potreiro → imagem Sentinel-2 → índices de vegetação
pastagem → massa seca real
clima_evapo → contexto climático
modelo → estimar forragem
O que cada tabela vira no projeto
potreiro
→ unidade maior da área experimental

subarea
→ unidade espacial usada para extrair imagem de satélite

medicao
→ data em que houve coleta de campo

pastagem
→ variável alvo do modelo
   principais alvos:
   - mstotal
   - msanoni
   - msoutras
   - media

clima_evapo
→ variáveis explicativas climáticas
Dataset final que precisamos montar

A tabela final de treino deve ficar assim:

data
idpotreiro
idsubarea
dentrofora
media
mstotal
msanoni
msoutras
ndvi
evi
ndre
savi
tmin
tmed
tmax
umidade
velocidadevento
radiacaosolar
chuva
somatermica
def
exc
Query base para extrair dados do TouceiraTech
SELECT
    m.data,
    p.id AS idpotreiro,
    s.id AS idsubarea,
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
    ST_AsGeoJSON(s.subpoligono) AS geometria
FROM pastagem pa
JOIN subarea s ON s.id = pa.idsubarea
JOIN potreiro p ON p.id = s.idpotreiro
JOIN medicao m ON m.id = pa.idmedicao
LEFT JOIN clima_evapo c ON c.data = m.data
ORDER BY m.data, p.id, s.id;
O que fazer depois dessa query
Exportar esse resultado para CSV ou GeoJSON.
Para cada subarea + data, buscar imagem Sentinel-2 próxima da data da medição.
Calcular índices:
NDVI
EVI
NDRE
SAVI
Juntar esses índices com os dados reais de campo.
Treinar o modelo.
Primeiro modelo recomendado

Começar simples:

Entrada:
NDVI, EVI, NDRE, SAVI, clima

Saída:
msanoni ou mstotal

Modelo inicial:

Random Forest Regressor

Depois evoluir para:

XGBoost
LSTM/GRU
CNN se usar imagem em recortes
Como explicar para o orientador

O TouceiraTech entra desde o início como fonte de verdade de campo. As tabelas pastagem, medicao, subarea e clima_evapo permitem montar um dataset georreferenciado e temporal, onde cada subárea tem uma data de medição, valores reais de massa seca e variáveis climáticas.

A partir das geometrias das subáreas, vamos extrair índices de vegetação do Sentinel-2 para as mesmas datas das medições. Assim, o modelo não vai mais aprender apenas “vigor por NDVI”, mas sim tentar estimar variáveis reais de forragem, como mstotal e msanoni.

Isso conecta imagem, clima e dado real de campo, antes de qualquer integração com a interface do TouceiraTech.


https://chatgpt.com/share/69975cf2-fa64-800b-a87e-c7afffa7b921 
esse é o link de todo o chat desde o inicio da ideia do projeto


---

**Cursor**

Entendi perfeitamente a proposta. Agora o cenário está muito mais claro:

---

## O que você quer construir

Um pipeline que conecta **3 fontes de dados** para estimar forragem:

```
TouceiraTech (campo real)    +    Sentinel-2 (imagem)    +    clima
       ↓                                ↓                       ↓
  mstotal, msanoni, media          NDVI, EVI, NDRE, SAVI     tmin, tmax...
                         ↘              ↓              ↙
                           dataset final → modelo ML
```

---

## O que precisa ser feito (etapas)

**Etapa 1 — Extração do banco (TouceiraTech)**
Já temos a query pronta. Ela exporta dados de campo com geometria de cada subarea.

**Etapa 2 — Busca de imagens Sentinel-2**
Para cada `(idsubarea, data)`, buscar a cena Sentinel-2 mais próxima da data da medição via **Google Earth Engine** (Python API) ou **Copernicus Data Space / STAC**.

**Etapa 3 — Cálculo de índices por subarea**
Recortar a geometria de cada `subarea` na imagem e calcular:
- **NDVI** = (NIR - Red) / (NIR + Red)
- **EVI** = 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
- **NDRE** = (NIR - RedEdge) / (NIR + RedEdge)
- **SAVI** = 1.5 × (NIR - Red) / (NIR + Red + 0.5)

**Etapa 4 — Montagem do dataset final**
Juntar campo + índices + clima numa tabela única.

**Etapa 5 — Modelo**
Começar com **Random Forest**, depois evoluir para XGBoost / LSTM.

---

## O que precisa ser decidido antes de começar

Tenho algumas perguntas importantes para estruturar o trabalho:

> **Qual fonte você quer usar para buscar as imagens Sentinel-2?**
> - Google Earth Engine (Python API — ee)
> - Copernicus Data Space (STAC / sentinelsat)
> - Ainda não sei, preciso de recomendação ✔

> **Por onde você quer começar agora?**
> - Criar o script de extração do banco (a query + exportação)
> - Criar o script de busca + cálculo dos índices Sentinel-2
> - Criar o pipeline completo de uma vez ✔
> - Só o modelo (já tenho os dados prontos)

> **Como você prefere trabalhar?**
> - Jupyter Notebook
> - Scripts Python (.py) ✔
> - Tanto faz

Ótimo. Vou recomendar o **Google Earth Engine** e criar o pipeline completo.

**Por que GEE?**
- Gratuito para pesquisa/academia
- Tem a coleção Sentinel-2 Surface Reflectance pronta (`COPERNICUS/S2_SR_HARMONIZED`)
- Filtra por data e geometria direto na nuvem — sem baixar imagens
- Calcula os índices sem precisar de disco local
- API Python (`earthengine-api`) bem documentada

Pipeline completo criado em `SENTINEL-PIPELINE/`. Aqui o resumo do que foi gerado:

---

## Estrutura criada

```
SENTINEL-PIPELINE/
├── 01_extrai_banco.py        ← query no TouceiraTech → dados_campo.csv
├── 02_indices_sentinel2.py   ← GEE: busca S2 + calcula NDVI/EVI/NDRE/SAVI → indices_sentinel2.csv
├── 03_monta_dataset.py       ← junta campo + índices → dataset_final.csv
├── 04_modelo_rf.py           ← treina Random Forest → resultados_rf/
└── README.md                 ← instruções completas
```

---

## Como começar

**1. Instalar dependências:**
```bash
pip install psycopg2-binary pandas scikit-learn matplotlib joblib earthengine-api geojson
```

**2. Criar conta no GEE e autenticar:**
```bash
earthengine authenticate
```

**3. Rodar em ordem:**
```bash
python 01_extrai_banco.py
python 02_indices_sentinel2.py   # pode demorar — faz chamadas ao GEE por subarea/data
python 03_monta_dataset.py
python 04_modelo_rf.py
```

---

## Pontos de atenção

- O script 02 é o mais lento — para cada par `(subarea, data)` ele consulta o GEE. Se o banco tiver muitas datas/subareas, pode ser necessário paralelizar ou usar batch export do GEE.
- A janela de busca de imagens é **±15 dias** em torno da data da medição (ajustável em `JANELA_DIAS`). Imagens com mais de **20% de nuvem** são descartadas.
- O alvo do modelo é definido pela variável `ALVO` no script 04 — basta trocar para `msanoni`, `msoutras` ou `media`.

---

**User**

me explica o que você fez até agora de forma bem simples

---

**Cursor**

Criei uma pasta nova chamada `SENTINEL-PIPELINE` com **4 scripts** que formam um caminho do banco até o modelo. Funciona assim:

---

**Script 1 — Pega os dados do banco**
Acessa o banco `db_2019` (o TouceiraTech) e exporta tudo para um arquivo CSV: data da medição, qual piquete, qual subárea, altura da pastagem, massa seca, e os dados de clima.

---

**Script 2 — Busca as imagens de satélite**
Para cada medição de campo, vai até o Google Earth Engine e busca uma imagem do Sentinel-2 próxima daquela data. A partir da imagem, calcula 4 índices de vegetação:
- **NDVI** — o mais clássico, mede verde da vegetação
- **EVI** — versão melhorada do NDVI
- **NDRE** — usa a banda red-edge, boa para biomassa
- **SAVI** — NDVI ajustado para áreas com solo exposto

Salva esses índices em outro CSV.

---

**Script 3 — Monta a tabela de treino**
Junta os dois CSVs anteriores — dados de campo + índices — em uma tabela única e organizada, pronta para alimentar um modelo.

---

**Script 4 — Treina o modelo**
Usa essa tabela para treinar um **Random Forest** (modelo de machine learning mais simples e robusto para dados tabulares) que aprende a estimar a massa seca da pastagem a partir dos índices e do clima. Gera gráficos e métricas de avaliação automaticamente.

---

**Resumo visual:**

```
banco TouceiraTech  +  Sentinel-2 (satélite)  +  clima
         ↓                      ↓                  ↓
    dados de campo          NDVI, EVI...     tmin, chuva...
                  ↘              ↓          ↙
                      tabela de treino
                             ↓
                       modelo → prediz mstotal
```

---

O próximo passo seria rodar o script 1 para ver se o banco está acessível, e depois criar uma conta no Google Earth Engine para conseguir rodar o script 2.

---

**User**

agora monta uma doc com os dados disponiveis, tudo que usaremos e  porque. e também a solução

---

**Cursor**

Agora tenho todos os detalhes. Vou criar um documento completo de documentação do projeto.

Documento criado em `DOCUMENTACAO.md` na raiz do projeto. Aqui o que ele cobre:

---

**Seção 1 — Contexto:** explica o problema (medir forragem é caro e manual) e o objetivo.

**Seção 2 — Dados disponíveis:** descreve cada tabela do banco com suas colunas, tipos e o papel de cada uma — incluindo a lógica de `dentrofora`, as 30 features climáticas e as bandas do Sentinel-2.

**Seção 3 — Por que cada dado é usado:** tabela direta relacionando cada dado ao seu papel e justificativa científica.

**Seção 4 — Dataset final:** mostra o formato da tabela de treino.

**Seção 5 — Solução:** inclui diagrama da arquitetura, tabela dos scripts, evolução planejada dos modelos (RF → XGBoost → LSTM → CNN), métricas e uma comparação direta com os modelos anteriores do projeto.

**Seção 6 e 7 — Execução:** dependências, conta GEE e ordem dos scripts.

**Seção 8 — Referências** dos índices de vegetação.

---

**User**

em algum momento conseguimos pelo satelite saber se uma area atual quanto de capim/forragem tem ?

---

**Cursor**

Sim, e esse é exatamente o ponto central do projeto. Deixa eu explicar como isso funciona:

---

## O que o satélite consegue "ver"

O Sentinel-2 mede a **reflexão da luz** pela vegetação. Plantas com mais biomassa, mais clorofila e mais cobertura verde refletem a luz de forma diferente de solo exposto ou pasto seco.

Os índices calculam isso:

```
NDVI alto  →  muita vegetação verde, densa
NDVI baixo →  pouco verde, solo exposto, pasto seco ou colhido
```

---

## O que o satélite sozinho **não consegue** dizer

O satélite vê a **superfície**, não a massa. Ele não sabe se aquele verde é:
- capim alto com muita biomassa
- capim baixo e denso com pouca biomassa
- outra espécie que não é forragem

Por isso **sozinho ele erra muito**.

---

## O que o projeto faz para resolver isso

É exatamente aqui que o TouceiraTech entra:

```
Satélite vê:   NDVI = 0.72  (muito verde)
Campo mediu:   mstotal = 3.200 kg/ha

O modelo aprende:
"quando NDVI ≈ 0.72 + esse clima + essa época → mstotal ≈ 3.200 kg/ha"
```

Com muitos pares desses (subárea × data de medição × índice), o modelo aprende a **traduzir o sinal do satélite em kg/ha reais** para aquele tipo específico de pastagem.

---

## O que fica possível depois do modelo treinado

```
Nova data (sem ninguém ir ao campo)
         ↓
Sentinel-2 tira foto da área
         ↓
Calcula NDVI, EVI, NDRE, SAVI da geometria de cada piquete
         ↓
Passa pelo modelo + clima do período
         ↓
"Piquete 2: ~2.800 kg/ha de forragem disponível"
```

Ou seja: **sim**, após o treinamento é possível estimar quanto de forragem existe em qualquer área que tenha geometria cadastrada, a cada ~5 dias (revisita do Sentinel-2), sem coleta manual.

---

## Uma limitação real que vale mencionar

O modelo vai ser tão bom quanto os dados de campo que o TouceiraTech tem. Se os dados são de uma espécie e uma região específica (p20/p21, campo experimental), ele vai funcionar melhor para **condições parecidas**. Para generalizar para outras fazendas ou espécies, precisaria de mais dados de campo de outros locais. Isso é algo para discutir com o orientador como passo futuro.

---

**User**

consegue buscar artigos relevantes que fazem isso ou algo parecido e como fazem? Me passando o link deles

---

**Cursor**

Encontrei artigos muito relevantes e diretamente relacionados ao projeto. Aqui estão organizados por proximidade com o que você está fazendo:

---

## Artigos mais próximos do seu projeto

### Pastagem tropical + Sentinel-2 + ML

**[1] Fernandes et al. (2024) — Scientific Reports / Nature**
Usa Sentinel-2 + variáveis meteorológicas + Random Forest e SVR para estimar **massa de forragem, proteína bruta e fibra** em pastagens tropicais de *Urochloa brizantha* (Brachiaria) no Brasil.
Quase idêntico ao que você está construindo.
[https://www.nature.com/articles/s41598-024-59160-x](https://www.nature.com/articles/s41598-024-59160-x)

---

**[2] Chen et al. (2021) — Remote Sensing (MDPI)**
Usa Sentinel-2 em série temporal + dados de campo + clima para estimar biomassa aérea de pastagens via **rede neural sequencial**. R² ≈ 0.6. Muito próximo da fase 3 do seu pipeline (LSTM).
[https://www.mdpi.com/2072-4292/13/4/603](https://www.mdpi.com/2072-4292/13/4/603)

---

**[3] Gestão sustentável de pastagens tropicais — ScienceDirect (2026)**
Usa Sentinel-2 com bandas NIR e Red Edge (as mesmas do NDRE que você usa) para estimar biomassa tropical no Brasil — contexto Cerrado/Amazônia.
[https://www.sciencedirect.com/article/pii/S295042522600085X](https://www.sciencedirect.com/article/pii/S295042522600085X)

---

**[4] PMC / NIH — versão aberta do artigo Fernandes et al.**
Versão gratuita e completa do artigo [1] com todos os detalhes metodológicos.
[https://pmc.ncbi.nlm.nih.gov/articles/PMC11018762/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11018762/)

---

### Qualidade da forragem via satélite

**[5] Irisarri et al. (2025) — Frontiers in Veterinary Science**
Random Forest com Sentinel-2 para prever **qualidade da forragem** (proteína, fibra ADF/NDF) — evolução natural do modelo de massa seca.
[https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2025.1678123/full](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2025.1678123/full)

---

### Série temporal + LSTM + satélite

**[6] LSTM para estimativa de produção agrícola com satélite + clima — ScienceDirect**
Usa LSTM integrando NDVI, EVI e dados meteorológicos — base metodológica para a fase 3 do seu pipeline.
[https://www.sciencedirect.com/article/pii/S0168192321003154](https://www.sciencedirect.com/article/pii/S0168192321003154)

---

### Biomassa em região tropical do Brasil (ANN)

**[7] Biomass Prediction com ANN no Cerrado/Amazônia — MDPI Forests**
Usa Sentinel-2 + redes neurais artificiais para prever biomassa aérea em zona de transição Cerrado-Amazônia no Brasil. Usa EVI e GNDVI.
[https://www.mdpi.com/1999-4907/15/9/1599](https://www.mdpi.com/1999-4907/15/9/1599)

---

### Alta latitude — base de comparação metodológica

**[8] Peng et al. (2023) — Forage Biomass Estimation at High Latitudes — Remote Sensing (MDPI)**
Compara RF, SVM e regressão para estimar produção de forragem com Sentinel-2. Random Forest foi o mais robusto — confirma a escolha como modelo base.
[https://www.mdpi.com/2072-4292/15/9/2350](https://www.mdpi.com/2072-4292/15/9/2350)

---

## O que esses artigos confirmam para o seu projeto

| Escolha do seu projeto | Confirmação na literatura |
|---|---|
| Sentinel-2 como fonte de imagem | Todos os artigos usam |
| NDVI + EVI + NDRE + SAVI | Fernandes [1], Peng [8], Irisarri [5] |
| Random Forest como modelo base | Fernandes [1], Wang, Irisarri [5], Peng [8] |
| Clima como feature complementar | Fernandes [1], Chen [2], LSTM [6] |
| Evolução para LSTM | Chen [2], artigo LSTM+clima [6] |
| Contexto tropical brasileiro | Fernandes [1], ScienceDirect [3], ANN Cerrado [7] |

O artigo **[1] (Fernandes et al., 2024)** é o mais relevante — vale ler completo, pois a metodologia é praticamente a mesma do que você está construindo, só que eles não tinham o TouceiraTech como fonte de campo estruturada como você tem.

---

**User**

Calcula NDVI, EVI, NDRE, SAVI da geometria de cada piquete
porque vamos usar esses?

---

**Cursor**

Boa pergunta. Cada índice foi escolhido porque captura um aspecto diferente da vegetação. Veja:

---

## NDVI — o mais clássico

**Fórmula:** (NIR − Red) / (NIR + Red)

A vegetação verde absorve muito vermelho (para fotossíntese) e reflete muito infravermelho. O NDVI mede exatamente essa diferença.

```
Pasto denso e verde  →  NDVI próximo de 1
Solo exposto         →  NDVI próximo de 0
Água / área morta    →  NDVI negativo
```

**Por que usar:** é o índice mais estudado do mundo, tem décadas de validação em pastagens. Funciona bem para saber se tem verde ou não.

**Limitação:** satura em vegetação muito densa — quando o capim está alto demais, o NDVI "trava" e não distingue bem as diferenças de biomassa.

---

## EVI — NDVI melhorado

**Fórmula:** 2.5 × (NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1)

Criado pela NASA justamente para corrigir os dois maiores problemas do NDVI:
- **Saturação** em vegetação densa
- **Interferência do solo** quando a cobertura é incompleta

```
Pastagem tropical rala  →  EVI é mais preciso que NDVI
Pastagem muito densa    →  EVI ainda diferencia, NDVI não
```

**Por que usar:** pastagens tropicais alternam entre estado ralo (após pastejo) e denso (rebrota). O EVI cobre os dois extremos melhor.

---

## NDRE — o mais sensível à biomassa

**Fórmula:** (NIR − RedEdge) / (NIR + RedEdge)

Usa a banda **Red Edge** (705 nm) — uma faixa de luz que o NDVI ignora mas que é muito sensível à **concentração de clorofila** dentro das folhas.

```
Planta com muito nitrogênio / clorofila  →  NDRE alto
Planta sob estresse ou com pouca clorofila →  NDRE baixo
```

**Por que usar:** é o índice que mais se correlaciona com **massa seca real** em pastagens tropicais — exatamente o `mstotal` que queremos prever. O artigo de Fernandes (2024) que encontramos confirma isso especificamente para Brachiaria.

---

## SAVI — NDVI para solo exposto

**Fórmula:** 1.5 × (NIR − Red) / (NIR + Red + 0.5)

Adiciona um fator de correção (0.5) que reduz a interferência do solo na leitura.

```
Pasto logo após pastejo (muito solo visível)  →  NDVI erra, SAVI acerta
Cobertura vegetal < 40%  →  SAVI é mais confiável
```

**Por que usar:** pastagens experimentais como as do TouceiraTech têm ciclos de pastejo onde o solo fica exposto. Nesses momentos o NDVI superestima o verde que existe — o SAVI corrige isso.

---

## Por que usar os 4 juntos?

Porque cada um "enxerga" um estado diferente da pastagem:

| Situação no piquete | Melhor índice |
|---|---|
| Pasto em rebrota inicial | SAVI (muito solo exposto) |
| Pasto em crescimento médio | NDVI |
| Pasto denso / próximo do ponto de colheita | EVI |
| Estimar massa seca real | NDRE |

Sozinho, nenhum cobre todos os estados. **Juntos**, o modelo Random Forest consegue aprender qual peso dar para cada um dependendo do momento — e isso é exatamente o que a literatura confirma nos artigos que encontramos.

---

**User**

o que significa esses resoltuados no primeiro artigo que você buscou?


---

**Cursor**

Esse gráfico é um **mapa de importância das variáveis** do modelo Random Forest do artigo — ele mostra quais dados mais influenciaram na predição da forragem. Veja o que cada sigla significa:

---

## Traduzindo as variáveis

**Bandas espectrais do Sentinel-2:**

| Sigla no artigo | Banda S2 | O que é |
|---|---|---|
| **RE3** | B7 (783 nm) | Red Edge 3 — infravermelho próximo da borda |
| **RE1** | B5 (705 nm) | Red Edge 1 — sensível à clorofila |
| **RE2** | B6 (740 nm) | Red Edge 2 |
| **NIR** | B8 (842 nm) | Infravermelho próximo |
| **NIR8A** | B8A (865 nm) | NIR estreito |
| **SWIR2** | B12 (2190 nm) | Infravermelho de ondas curtas 2 |
| **SWIR1** | B11 (1610 nm) | Infravermelho de ondas curtas 1 |
| **RED** | B4 (665 nm) | Vermelho |
| **BLUE** | B2 (490 nm) | Azul |
| **GREEN** | B3 (560 nm) | Verde |
| **ND** | — | NDVI (o índice clássico) |

**Variáveis climáticas:**

| Sigla | Significado |
|---|---|
| **Rainfall** | Precipitação (chuva) |
| **RH_p** | Umidade relativa |
| **Isolat** | Radiação solar |
| **Tmax / Tavg / Tmin** | Temperatura máx / média / mín |

---

## O que o resultado diz

**RE3 e RE1 são as variáveis mais importantes** — as duas bandas Red Edge dominam com ~11% cada. Isso confirma diretamente por que o **NDRE** é tão valioso no seu projeto: ele é calculado com exatamente essas bandas.

**SWIR2** em terceiro lugar (~8.5%) é uma banda que **você ainda não está usando**. Ela é sensível à água dentro das folhas e ao teor de matéria seca — muito relevante para `mstotal`.

**Chuva (Rainfall)** em quarto (~8%) — confirma que o clima importa muito, como você já incluiu.

**NDVI (ND) ficou quase no último lugar** (~3%) — o índice mais famoso é um dos menos úteis aqui. Isso acontece porque as **bandas brutas** carregam mais informação do que o índice calculado a partir delas.

---

## O que isso muda no seu projeto

Isso sugere **duas melhorias importantes**:

**1. Incluir as bandas SWIR no script 02**
A banda B11 e B12 do Sentinel-2 podem aumentar bastante a precisão — elas não estão no pipeline atual.

**2. Considerar usar as bandas brutas além dos índices**
Em vez de só NDVI/EVI/NDRE/SAVI, passar também as reflectâncias de B5, B7 e B12 diretamente para o modelo.

Quer que eu atualize o script `02_indices_sentinel2.py` para incluir também as bandas SWIR e as reflectâncias Red Edge brutas?

---

**User**

os dados são antigos, vou conseuir pegar do satelite

---

**Cursor**

Boa notícia — e esse é um ponto importante que precisa ser esclarecido. Deixa eu explicar a situação exata:

---

## O que está disponível no arquivo histórico do Sentinel-2

| Período | Sentinel-2 disponível? |
|---|---|
| 2013 — Jun/2015 | **Não** — o satélite ainda não existia |
| Jul/2015 — Mar/2017 | **Sim** — só o Sentinel-2A (revisita ~10 dias) |
| Mar/2017 — hoje | **Sim** — Sentinel-2A + 2B (revisita ~5 dias) |

O Sentinel-2A foi lançado em **junho de 2015**. Os dados mais antigos que você vai conseguir do GEE são de meados de 2015.

---

## Cruzando com os dados do TouceiraTech (2013–2018)

```
2013 ████████████████░░░░░░░░░░░░░░  sem satélite
2014 ████████████████░░░░░░░░░░░░░░  sem satélite
2015 ████████░░░░░░░░████████████░░  S2A a partir de jul/2015
2016 ░░░░░░░░░░░░░░░░████████████░░  S2A completo
2017 ░░░░░░░░░░░░░░░░████████████░░  S2A + S2B
2018 ░░░░░░░░░░░░░░░░████████████░░  S2A + S2B
```

Ou seja: **você vai perder as medições de 2013 e 2014** mas vai conseguir cruzar tudo de meados de 2015 até julho de 2018 — que deve ser a maior parte dos dados.

---

## O que isso muda no pipeline

O script `01_extrai_banco.py` atualmente puxa dados de `2013-12-20` até `2018-07-13`. Basta ajustar o filtro de data para começar em `2015-07-01`, e o script `02_indices_sentinel2.py` vai encontrar imagens para todos os registros.

Se quiser, já faço esse ajuste agora. Quer também que eu adicione no script uma verificação automática que avisa quando uma data de medição é anterior a julho de 2015, em vez de travar?

---

**User**

verifica se o projeto precisa de atualizações
porqe ele está ha um tempo parado

---

**Cursor**

Agora tenho um diagnóstico completo. O projeto está com dois problemas sérios:

Agora vou corrigir os imports obsoletos nos modelos. O maior problema é que o TensorFlow 2.16+ mudou como o Keras é importado:

Os 5 modelos têm o mesmo padrão de imports. Vou corrigir todos de uma vez — os problemas são os mesmos em todos:

Nenhum import antigo restante. Agora verifico também se ficou algum uso de `RandomSearch` pelo import antigo nos arquivos:

Tudo atualizado. Aqui está o diagnóstico completo do que foi encontrado e corrigido:

---

## O que estava desatualizado / quebrado

### 1. Nada instalado
O projeto não tinha nenhuma dependência instalada — `tensorflow`, `keras`, `numpy`, `pandas`, tudo zerado. **Não rodaria nada.**

### 2. Imports do Keras obsoletos (todos os 5 modelos)
O Keras foi incorporado dentro do TensorFlow e o import antigo parou de funcionar:

| Import antigo (quebrado) | Import correto |
|---|---|
| `from keras.models import Sequential` | `from tensorflow.keras.models import Sequential` |
| `from keras.layers import LSTM` | `from tensorflow.keras.layers import LSTM` |
| `from keras.utils import plot_model` | `from tensorflow.keras.utils import plot_model` |
| `from keras.optimizers import Adam` | `from tensorflow.keras.optimizers import Adam` |
| `from keras_tuner.tuners import RandomSearch` | `import keras_tuner as kt` + `kt.RandomSearch(...)` |

### 3. Sem `requirements.txt`
O projeto não documentava nenhuma dependência. Qualquer pessoa que clonasse o repo não saberia o que instalar.

---

## O que foi criado / corrigido

| Arquivo | O que fez |
|---|---|
| `requirements.txt` | Lista todas as dependências com versões compatíveis |
| `setup_ambiente.sh` | Script que instala tudo automaticamente em um ambiente virtual |
| 5 arquivos em `MODELOS/` | Imports corrigidos para funcionar com TensorFlow 2.16+ |

---

## Para colocar o projeto para rodar agora

```bash
cd /home/ana.campos/Documentos/ana/Modelos-NP3
bash setup_ambiente.sh
source .venv/bin/activate
```

---

**User**

@/home/ana.campos/.cursor/projects/home-ana-campos-Documentos-ana-Modelos-NP3/terminals/5.txt:106-118 

---

**Cursor**

O problema é que o arquivo foi salvo com **quebras de linha do Windows** (`\r\n`) em vez do Linux (`\n`). Vou corrigir isso:

Dois pacotes do sistema precisam ser instalados antes. Rode no terminal:

O `sudo` precisa de senha digitada no terminal. Copie e cole esse comando diretamente no terminal do seu sistema:

```bash
sudo apt install -y python3-pip python3.12-venv
```

Depois que terminar, rode:

```bash
cd /home/ana.campos/Documentos/ana/Modelos-NP3
bash setup_ambiente.sh
```

E por último, para ativar o ambiente sempre que for trabalhar no projeto:

```bash
source .venv/bin/activate
```

---

O que cada comando instala:
- **`python3-pip`** — o gerenciador de pacotes Python (estava faltando)
- **`python3.12-venv`** — permite criar ambientes virtuais isolados (evita conflito com outros projetos do sistema)

---

**User**

@/home/ana.campos/.cursor/projects/home-ana-campos-Documentos-ana-Modelos-NP3/terminals/5.txt:229-236 

---

**Cursor**

O `.venv` ficou com a criação incompleta da tentativa anterior. Vou apagar e recriar:

Tudo instalado com sucesso. Resumo do que está no ambiente agora:

| Pacote | Versão instalada |
|---|---|
| TensorFlow | 2.20.0 |
| Keras Tuner | 1.4.8 |
| NumPy | 2.4.4 |
| Pandas | 3.0.2 |
| Scikit-learn | 1.8.0 |
| SciPy | 1.17.1 |
| Matplotlib | 3.10.9 |
| psycopg2 | 2.9.12 |
| earthengine-api | 1.7.24 |
| joblib | 1.5.3 |

---

As mensagens sobre CUDA são normais — significam apenas que não há GPU, o modelo vai rodar na CPU, o que está ótimo para desenvolvimento.

O próximo passo antes de rodar qualquer script é ativar o ambiente e autenticar no GEE:

```bash
source .venv/bin/activate
earthengine authenticate
```

