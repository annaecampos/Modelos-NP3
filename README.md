# Modelos NP3 — Predição de Forragem com RNN (LSTM / GRU)

Conjunto de modelos de redes neurais recorrentes para predição de disponibilidade de forragem em pastagens (agricultura digital / pecuária de precisão), utilizando dados climáticos extraídos de um banco PostgreSQL/PostGIS.

## Trabalhos relacionados

- [TOUCEIRATECH: FMIS para Pecuária de Precisão](https://repositorio.unipampa.edu.br/bitstreams/e888f087-4356-4ef2-aef9-d04e07c8affb/download)
- [Impactos do KerasTuner em LSTM para Pecuária Sustentável](https://sol.sbc.org.br/index.php/eradrs/article/view/28012/27822)
- [Análise Instrumental de Modelos RNN para Computação Verde](https://ei.unipampa.edu.br/uploads/evt/arq_trabalhos/29643/etp1_resumo_expandido_29643.pdf)
- [Árvores de Decisão para Ajuste de Modelo de Predição](http://www2.bage.ifsul.edu.br/encif/inscricao/pdf/2024110523590338801.pdf)
- [Instrumentação AD e KerasTuner para GRU — ERADRS 2025](https://sol.sbc.org.br/index.php/eradrs/article/view/35364/35153)

---

## Pré-requisitos do sistema

| Software        | Versão recomendada |
|-----------------|--------------------|
| Ubuntu          | 22.04 LTS          |
| **Python**      | **3.10 – 3.12**¹   |
| PostgreSQL      | 12                 |
| PostGIS         | 3                  |

> ¹ TensorFlow 2.19 suporta Python 3.10–3.12. Python 3.13 ainda **não** é suportado oficialmente pelo TensorFlow.

---

## Guia passo a passo

### Parte 1 — Banco de dados (PostgreSQL + PostGIS)

```bash
# Adicionar repositório PostgreSQL
sudo sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
  > /etc/apt/sources.list.d/pgdg.list'
wget -qO - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install postgresql-12

# Definir senha do postgres
sudo -u postgres psql -c "\password postgres"
# → Insira: 12345

# Instalar PostGIS
sudo apt install postgis postgresql-12-postgis-3

# Instalar pgAdmin4 (opcional)
curl -fsS https://www.pgadmin.org/static/packages_pgadmin_org.pub \
  | sudo gpg --dearmor -o /usr/share/keyrings/packages-pgadmin-org.gpg
sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/packages-pgadmin-org.gpg] \
  https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" \
  > /etc/apt/sources.list.d/pgadmin4.list && apt update'
sudo apt install pgadmin4
```

**Restaurar banco:** Crie uma database `db_2019` e restaure o arquivo `RESTORE DO BANCO/db_2020.backup` via pgAdmin4 ou psql.

---

### Parte 2 — Ambiente Python

```bash
# Criar e ativar virtualenv
python3.12 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# (Opcional) Para plot_model funcionar
sudo apt install graphviz
```

---

### Parte 3 — Extração de dados

```bash
# Modelo Original
cd "MANIPULAÇÃO DE DADOS/MODELO ORIGINAL"
python3 consolida_dados.py

# Modelo Ajustado
cd "MANIPULAÇÃO DE DADOS/MODELO AJUSTADO"
python3 consolida_dados.py
```

**Selecionar potreiro:** altere `t.id` nos arquivos `selectent.sql`:
| Valor | Potreiro         |
|-------|-----------------|
| 1     | P20 Infestado   |
| 2     | P20 Mirapasto   |
| 3     | P21 Infestado   |
| 4     | P21 Mirapasto   |

---

### Parte 4 — Execução dos modelos

```bash
cd MODELOS

# Modelo Original (execução única)
python3 "MODELO ORIGINAL.py"

# Modelo Ajustado (execução única)
python3 "MODELO AJUSTADO.py"

# Modelos com loop — informe o número de rodadas
python3 "MODELO AUTOAJUSTADO COM AD.py" 5
python3 "MODELO AUTOAJUSTADO COM KT.py" 3
python3 "MODELO AJUSTADO E AUTOAJUSTADO COM KT - GRU.py" 3
```

Cada execução cria um subdiretório com timestamp contendo:
- `Log_SKT.txt` / `Log_CKT.txt` — RMSE, R², desvios
- `GBoxP_*.png` — boxplot Real × Predito × Erro
- `GDisp_*.png` — gráfico de dispersão
- `model_plot_*.png` — arquitetura da rede
- `modelsummary.txt` — resumo do modelo (modelos com KT)

---

## Descrição dos modelos

| Arquivo | Descrição |
|---------|-----------|
| `MODELO ORIGINAL.py` | LSTM dupla, hiperparâmetros fixos |
| `MODELO AJUSTADO.py` | LSTM dupla, dados ajustados (split treino/teste diferente) |
| `MODELO AUTOAJUSTADO COM AD.py` | LSTM com busca aleatória filtrada por Árvore de Decisão |
| `MODELO AUTOAJUSTADO COM KT.py` | LSTM com Keras Tuner (Random Search) |
| `MODELO AJUSTADO E AUTOAJUSTADO COM KT - GRU.py` | GRU ajustada + GRU autoajustada via Keras Tuner |

---

## Mudanças recentes (atualização 2026)

- Imports padronizados para **Keras 3** (`import keras`)
- `from keras_tuner import RandomSearch` (API atual; `keras_tuner.tuners` depreciado)
- Caminhos hardcoded substituídos por `pathlib.Path(__file__).parent`
- Variável `range` renomeada para `plot_range` (evita sombra do builtin)
- Loops `while` com contador manual substituídos por `for aux in range(...)`
- Imports não utilizados removidos
- `requirements.txt` criado com versões pinadas
- SQL formatado com `JOIN` explícito (legibilidade)

---

## Suporte

- **Bianca Durgante** — biancadurgante.aluno@unipampa.edu.br
- **Davi Lemos** — davilemos.aluno@unipampa.edu.br
- **Leonardo Pinho** — leonardopinho@unipampa.edu.br
